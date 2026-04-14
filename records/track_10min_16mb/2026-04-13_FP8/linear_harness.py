#!/usr/bin/env python3
"""Benchmark one train_gpt linear layer against local Hopper CuTe dense GEMMs.

This script mirrors the default microbatch geometry from `train_gpt.py`:

- `TRAIN_BATCH_TOKENS=786432`
- `TRAIN_SEQ_LEN=2048`
- `grad_accum_steps=8 // WORLD_SIZE`
- default layer: MLP FC (`512 -> 2048`)

The baseline path uses a compiled `CastedLinear` and times forward/backward under
BF16 autocast. The custom paths wrap the local `dense_gemm.py` Hopper kernel as a
full linear layer with forward, dgrad, and wgrad GEMMs.

Example:

```bash
python3 records/track_10min_16mb/2026-04-13_FP8/linear_harness.py \
  --warmup-iters 10 \
  --iters 50 \
  --autotune-warmup-iters 3 \
  --autotune-iters 10
```
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


THIS_DIR = Path(__file__).resolve().parent
DENSE_GEMM_PATH = THIS_DIR / "dense_gemm.py"

LAYER_CHOICES = (
    "mlp_fc",
    "mlp_proj",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_proj",
)

TILE_SHAPE_CHOICES = ((128, 128), (128, 256), (128, 64), (64, 64))
CLUSTER_SHAPE_CHOICES = ((1, 1), (2, 1), (1, 2), (2, 2))

FP8_INFO = {
    "e4m3fn": {"cutlass_name": "Float8E4M3FN", "max": 448.0},
    "e5m2": {"cutlass_name": "Float8E5M2", "max": 57344.0},
}

_CUTLASS_STACK: dict[str, Any] | None = None


def parse_pair(text: str) -> tuple[int, int]:
    parts = tuple(int(p.strip()) for p in text.split(","))
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated integers")
    return parts


def load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_cutlass_stack() -> dict[str, Any]:
    global _CUTLASS_STACK
    if _CUTLASS_STACK is not None:
        return _CUTLASS_STACK
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    import cutlass.cute.testing as cute_testing
    import cutlass.torch as cutlass_torch

    dense_gemm = load_local_module("record_dense_gemm", DENSE_GEMM_PATH)
    _CUTLASS_STACK = {
        "cuda_driver": cuda_driver,
        "cutlass": cutlass,
        "cute": cute,
        "cute_testing": cute_testing,
        "cutlass_torch": cutlass_torch,
        "dense_gemm": dense_gemm,
    }
    return _CUTLASS_STACK


@dataclass(frozen=True)
class TrainLikeDefaults:
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 786432))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    embedding_dim: int = int(os.environ.get("EMBEDDING_DIM", 512))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: float = float(os.environ.get("MLP_MULT", 4.0))
    world_size: int = int(os.environ.get("WORLD_SIZE", "1"))

    @property
    def grad_accum_steps(self) -> int:
        steps = 8 // self.world_size
        if steps <= 0:
            raise ValueError(f"WORLD_SIZE={self.world_size} is incompatible with 8-way accumulation")
        return steps

    @property
    def device_batch_tokens(self) -> int:
        if self.train_batch_tokens % self.grad_accum_steps != 0:
            raise ValueError(
                f"TRAIN_BATCH_TOKENS={self.train_batch_tokens} must be divisible by"
                f" grad_accum_steps={self.grad_accum_steps}"
            )
        return self.train_batch_tokens // self.grad_accum_steps

    @property
    def micro_batch_size(self) -> int:
        if self.device_batch_tokens % self.train_seq_len != 0:
            raise ValueError(
                f"TRAIN_BATCH_TOKENS={self.train_batch_tokens} is not divisible into whole"
                f" sequences for TRAIN_SEQ_LEN={self.train_seq_len} and grad_accum_steps={self.grad_accum_steps}"
            )
        return self.device_batch_tokens // self.train_seq_len


@dataclass(frozen=True)
class LinearProblem:
    layer: str
    batch_size: int
    seq_len: int
    m: int
    k: int
    n: int

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.batch_size, self.seq_len, self.k)

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return (self.batch_size, self.seq_len, self.n)


@dataclass(frozen=True)
class DenseGemmConfig:
    tile_shape_mn: tuple[int, int]
    cluster_shape_mn: tuple[int, int]

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "tile_shape_mn": list(self.tile_shape_mn),
            "cluster_shape_mn": list(self.cluster_shape_mn),
        }

    def label(self) -> str:
        tm, tn = self.tile_shape_mn
        cm, cn = self.cluster_shape_mn
        return f"tile={tm}x{tn} cluster={cm}x{cn}"


@dataclass
class TimingSummary:
    mean_us: float
    min_us: float
    max_us: float


@dataclass
class ErrorSummary:
    max_abs: float
    mean_abs: float
    rmse: float
    rel_l2: float


class CastedLinear(nn.Linear):
    def forward(self, x):
        w = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def restore_fp32_params(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()


class BaselineLinearModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, weight: torch.Tensor):
        super().__init__()
        self.linear = CastedLinear(in_features, out_features, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
        restore_fp32_params(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class CuteDenseGemmOp:
    def __init__(
        self,
        *,
        name: str,
        m: int,
        n: int,
        k: int,
        a_dtype: Any,
        b_dtype: Any,
        c_dtype: Any,
        acc_dtype: Any,
        a_major: str,
        b_major: str,
        c_major: str,
        config: DenseGemmConfig,
    ) -> None:
        stack = load_cutlass_stack()
        self.cutlass = stack["cutlass"]
        self.cute = stack["cute"]
        self.cute_testing = stack["cute_testing"]
        self.cutlass_torch = stack["cutlass_torch"]
        self.cuda_driver = stack["cuda_driver"]
        self.dense_gemm = stack["dense_gemm"]
        self.name = name
        self.m = m
        self.n = n
        self.k = k
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.c_dtype = c_dtype
        self.acc_dtype = acc_dtype
        self.a_major = a_major
        self.b_major = b_major
        self.c_major = c_major
        self.config = config

        if not self.dense_gemm.HopperWgmmaGemmKernel.is_valid_dtypes(
            a_dtype, b_dtype, acc_dtype, c_dtype, a_major, b_major
        ):
            raise ValueError(
                f"{name}: unsupported dtype/layout combo"
                f" a={a_dtype} b={b_dtype} c={c_dtype} acc={acc_dtype}"
                f" a_major={a_major} b_major={b_major}"
            )
        if not self.dense_gemm.HopperWgmmaGemmKernel.is_valid_tensor_alignment(
            m, n, k, 1, a_dtype, c_dtype, a_major, b_major, c_major
        ):
            raise ValueError(
                f"{name}: shape (m={m}, n={n}, k={k}) violates dense_gemm 16B alignment"
            )

        self.stream = self.cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
        self.a_source, self.a_storage, self.a_tensor = self._make_operand_buffers(
            m, k, a_major == "m", a_dtype
        )
        self.b_source, self.b_storage, self.b_tensor = self._make_operand_buffers(
            n, k, b_major == "n", b_dtype
        )
        self.c_source, self.c_storage, self.c_tensor = self._make_operand_buffers(
            m, n, c_major == "m", c_dtype
        )
        self.c_view = self.c_storage[..., 0]

        gemm = self.dense_gemm.HopperWgmmaGemmKernel(
            acc_dtype, config.tile_shape_mn, config.cluster_shape_mn
        )
        t0 = time.perf_counter()
        self.compiled = self.cute.compile(
            gemm, self.a_tensor, self.b_tensor, self.c_tensor, self.stream
        )
        self.compile_ms = 1e3 * (time.perf_counter() - t0)

    def _make_operand_buffers(
        self,
        mode0: int,
        mode1: int,
        is_mode0_major: bool,
        cutlass_dtype: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        source_dtype = (
            self.cutlass.Float32
            if cutlass_dtype.is_float and cutlass_dtype.width <= 8
            else cutlass_dtype
        )
        source = self.cutlass_torch.matrix(
            1,
            mode0,
            mode1,
            is_mode0_major,
            source_dtype,
            init_type=self.cutlass_torch.TensorInitType.SKIP,
            device=torch.device("cuda"),
        )
        source.zero_()
        cute_tensor, storage = self.cutlass_torch.cute_tensor_like(
            source,
            cutlass_dtype,
            is_dynamic_layout=True,
            assumed_align=16,
        )
        storage.zero_()
        return source, storage, cute_tensor

    def _load_operand(
        self,
        source: torch.Tensor,
        storage: torch.Tensor,
        cute_tensor: Any,
        cutlass_dtype: Any,
        value: torch.Tensor,
    ) -> None:
        if value.ndim != 2:
            raise ValueError(f"{self.name}: expected a 2D operand, got shape {tuple(value.shape)}")
        source[..., 0].copy_(value.to(dtype=source.dtype))
        if cutlass_dtype.is_float and cutlass_dtype.width <= 8:
            self.cutlass_torch.convert_cute_tensor(source, cute_tensor, cutlass_dtype, True)
        else:
            storage.copy_(source)

    def load_operands(self, a: torch.Tensor, b: torch.Tensor) -> None:
        self._load_operand(self.a_source, self.a_storage, self.a_tensor, self.a_dtype, a)
        self._load_operand(self.b_source, self.b_storage, self.b_tensor, self.b_dtype, b)

    def run(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        self.load_operands(a, b)
        self.compiled(self.a_tensor, self.b_tensor, self.c_tensor, self.stream)
        return self.c_view

    def benchmark_kernel_us(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        warmup_iterations: int,
        iterations: int,
    ) -> float:
        self.load_operands(a, b)
        kernel_args = self.cute_testing.JitArguments(
            self.a_tensor, self.b_tensor, self.c_tensor, self.stream
        )
        return self.cute_testing.benchmark(
            self.compiled,
            kernel_arguments=kernel_args,
            stream=self.stream,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
        )


class CuTeLinearBackend:
    def __init__(
        self,
        *,
        problem: LinearProblem,
        variant: str,
        fwd_op: CuteDenseGemmOp,
        dgrad_op: CuteDenseGemmOp,
        wgrad_op: CuteDenseGemmOp,
        fp8_format: str,
        fp8_target_max: float,
    ) -> None:
        self.problem = problem
        self.variant = variant
        self.fwd_op = fwd_op
        self.dgrad_op = dgrad_op
        self.wgrad_op = wgrad_op
        self.fp8_format = fp8_format
        self.fp8_target_max = fp8_target_max

    def _scale_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x.float().abs().amax() / self.fp8_target_max, min=1e-8)

    def _forward_2d(self, x2d: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self.variant == "fp16":
            return self.fwd_op.run(x2d, weight).float()
        sx = self._scale_tensor(x2d)
        sw = self._scale_tensor(weight)
        y = self.fwd_op.run(x2d.float() / sx, weight.float() / sw).float()
        return y * (sx * sw)

    def _dgrad_2d(self, grad_out_2d: torch.Tensor, weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        weight_t = weight.transpose(0, 1).contiguous()
        if self.variant == "fp16":
            dx = self.dgrad_op.run(grad_out_2d, weight_t).float()
        else:
            sg = self._scale_tensor(grad_out_2d)
            sw = self._scale_tensor(weight_t)
            dx = self.dgrad_op.run(
                grad_out_2d.float() / sg,
                weight_t.float() / sw,
            ).float()
            dx.mul_(sg * sw)
        return dx.to(dtype=dtype)

    def _wgrad(self, x2d: torch.Tensor, grad_out_2d: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        x_t = x2d.transpose(0, 1).contiguous()
        grad_t = grad_out_2d.transpose(0, 1).contiguous()
        if self.variant == "fp16":
            dw_t = self.wgrad_op.run(x_t, grad_t).float()
        else:
            sx = self._scale_tensor(x_t)
            sg = self._scale_tensor(grad_t)
            dw_t = self.wgrad_op.run(
                x_t.float() / sx,
                grad_t.float() / sg,
            ).float()
            dw_t.mul_(sx * sg)
        return dw_t.transpose(0, 1).contiguous().to(dtype=dtype)

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x2d = x.reshape(-1, x.shape[-1])
        y2d = self._forward_2d(x2d, weight)
        return y2d.view(*x.shape[:-1], weight.shape[0]).to(dtype=x.dtype)

    def backward(self, x: torch.Tensor, weight: torch.Tensor, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x2d = x.reshape(-1, x.shape[-1])
        grad_out_2d = grad_out.reshape(-1, grad_out.shape[-1])
        grad_x = self._dgrad_2d(grad_out_2d, weight, x.dtype).view_as(x)
        grad_w = self._wgrad(x2d, grad_out_2d, weight.dtype)
        return grad_x, grad_w


class CuTeLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, backend: CuTeLinearBackend):
        ctx.backend = backend
        ctx.save_for_backward(x.detach(), weight.detach())
        return backend.forward(x, weight)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight = ctx.saved_tensors
        grad_x, grad_w = ctx.backend.backward(x, weight, grad_out)
        return grad_x, grad_w, None


class CuTeLinearModule(nn.Module):
    def __init__(self, weight: torch.Tensor, backend: CuTeLinearBackend):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.backend = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return CuTeLinearFn.apply(x, self.weight, self.backend)


def build_problem(args: argparse.Namespace, defaults: TrainLikeDefaults) -> LinearProblem:
    batch_size = args.batch_size or defaults.micro_batch_size
    seq_len = args.seq_len or defaults.train_seq_len
    kv_dim = defaults.num_kv_heads * (defaults.model_dim // defaults.num_heads)
    hidden = int(defaults.mlp_mult * defaults.model_dim)
    layer_map = {
        "mlp_fc": (defaults.model_dim, hidden),
        "mlp_proj": (hidden, defaults.model_dim),
        "attn_q": (defaults.model_dim, defaults.model_dim),
        "attn_k": (defaults.model_dim, kv_dim),
        "attn_v": (defaults.model_dim, kv_dim),
        "attn_proj": (defaults.model_dim, defaults.model_dim),
    }
    default_k, default_n = layer_map[args.layer]
    k = args.in_features or default_k
    n = args.out_features or default_n
    m = batch_size * seq_len
    return LinearProblem(args.layer, batch_size, seq_len, m, k, n)


def configure_runtime(device_index: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark harness")
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return device


def make_master_tensors(
    problem: LinearProblem,
    *,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = torch.randn(problem.input_shape, device=device, dtype=torch.float32)
    x = F.rms_norm(x, (problem.k,)).to(torch.bfloat16)
    weight = torch.empty(problem.n, problem.k, device=device, dtype=torch.float32)
    if weight.shape[0] >= 64 and weight.shape[1] >= 64:
        nn.init.orthogonal_(weight, gain=1.0)
    else:
        nn.init.normal_(weight, mean=0.0, std=0.02)
    grad_out = torch.randn(problem.output_shape, device=device, dtype=torch.float32).to(torch.bfloat16)
    return x, weight, grad_out


def summarize_times(times_us: list[float]) -> TimingSummary:
    return TimingSummary(
        mean_us=sum(times_us) / len(times_us),
        min_us=min(times_us),
        max_us=max(times_us),
    )


def summarize_error(actual: torch.Tensor, reference: torch.Tensor) -> ErrorSummary:
    diff = actual.float() - reference.float()
    ref_norm = reference.float().norm().clamp_min(1e-12)
    return ErrorSummary(
        max_abs=float(diff.abs().max().item()),
        mean_abs=float(diff.abs().mean().item()),
        rmse=float(diff.square().mean().sqrt().item()),
        rel_l2=float((diff.norm() / ref_norm).item()),
    )


def make_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_out: torch.Tensor,
) -> dict[str, torch.Tensor]:
    x_ref = x.detach().float().requires_grad_(True)
    w_ref = weight.detach().float().requires_grad_(True)
    y_ref = F.linear(x_ref, w_ref)
    y_ref.backward(grad_out.detach().float())
    return {
        "output": y_ref.detach(),
        "grad_input": x_ref.grad.detach(),
        "grad_weight": w_ref.grad.detach(),
    }


def unwrap_module(module: nn.Module) -> nn.Module:
    return getattr(module, "_orig_mod", module)


def get_weight_param(module: nn.Module) -> torch.Tensor:
    raw = unwrap_module(module)
    if hasattr(raw, "linear"):
        return raw.linear.weight
    if hasattr(raw, "weight"):
        return raw.weight
    raise AttributeError(f"could not locate weight parameter on {type(raw).__name__}")


def benchmark_autograd_module(
    module: nn.Module,
    x_master: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    warmup_iters: int,
    iters: int,
    use_autocast: bool,
) -> tuple[dict[str, TimingSummary], dict[str, torch.Tensor]]:
    module.train()
    raw_module = unwrap_module(module)
    x = x_master.detach().clone().requires_grad_(True)
    fwd_times_us: list[float] = []
    bwd_times_us: list[float] = []
    total_times_us: list[float] = []
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)
    bwd_start = torch.cuda.Event(enable_timing=True)
    bwd_end = torch.cuda.Event(enable_timing=True)
    last_y = None

    torch.cuda.synchronize()
    for step in range(warmup_iters + iters):
        raw_module.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        total_start.record()
        fwd_start.record()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
            y = module(x)
        fwd_end.record()
        bwd_start.record()
        y.backward(grad_out)
        bwd_end.record()
        total_end.record()
        torch.cuda.synchronize()
        if step >= warmup_iters:
            fwd_times_us.append(1e3 * fwd_start.elapsed_time(fwd_end))
            bwd_times_us.append(1e3 * bwd_start.elapsed_time(bwd_end))
            total_times_us.append(1e3 * total_start.elapsed_time(total_end))
        last_y = y.detach()

    weight = get_weight_param(module)
    outputs = {
        "output": last_y.detach().clone(),
        "grad_input": x.grad.detach().clone(),
        "grad_weight": weight.grad.detach().clone(),
    }
    summaries = {
        "forward": summarize_times(fwd_times_us),
        "backward": summarize_times(bwd_times_us),
        "total": summarize_times(total_times_us),
    }
    return summaries, outputs


def representative_operands(
    problem: LinearProblem,
    weight: torch.Tensor,
    x: torch.Tensor,
    grad_out: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    x2d = x.reshape(problem.m, problem.k).float()
    grad2d = grad_out.reshape(problem.m, problem.n).float()
    return {
        "fwd": (x2d, weight.float()),
        "dgrad": (grad2d, weight.transpose(0, 1).contiguous().float()),
        "wgrad": (
            x2d.transpose(0, 1).contiguous().float(),
            grad2d.transpose(0, 1).contiguous().float(),
        ),
    }


def make_cutlass_dtypes(variant: str, fp8_format: str) -> dict[str, Any]:
    cutlass = load_cutlass_stack()["cutlass"]
    if variant == "fp16":
        ab_dtype = cutlass.Float16
    else:
        ab_dtype = getattr(cutlass, FP8_INFO[fp8_format]["cutlass_name"])
    return {
        "ab_dtype": ab_dtype,
        "acc_dtype": cutlass.Float32,
        "fwd_c_dtype": cutlass.Float16,
        "dgrad_c_dtype": cutlass.Float16,
        "wgrad_c_dtype": cutlass.Float32,
    }


def build_ops_for_config(
    problem: LinearProblem,
    variant: str,
    fp8_format: str,
    config_map: dict[str, DenseGemmConfig],
) -> tuple[CuteDenseGemmOp, CuteDenseGemmOp, CuteDenseGemmOp]:
    dtypes = make_cutlass_dtypes(variant, fp8_format)
    fwd = build_single_op(problem, variant, fp8_format, "fwd", config_map["fwd"], dtypes)
    dgrad = build_single_op(problem, variant, fp8_format, "dgrad", config_map["dgrad"], dtypes)
    wgrad = build_single_op(problem, variant, fp8_format, "wgrad", config_map["wgrad"], dtypes)
    return fwd, dgrad, wgrad


def build_single_op(
    problem: LinearProblem,
    variant: str,
    fp8_format: str,
    op_name: str,
    config: DenseGemmConfig,
    dtypes: dict[str, Any] | None = None,
) -> CuteDenseGemmOp:
    if dtypes is None:
        dtypes = make_cutlass_dtypes(variant, fp8_format)
    if op_name == "fwd":
        return CuteDenseGemmOp(
            name=f"{variant}_fwd",
            m=problem.m,
            n=problem.n,
            k=problem.k,
            a_dtype=dtypes["ab_dtype"],
            b_dtype=dtypes["ab_dtype"],
            c_dtype=dtypes["fwd_c_dtype"],
            acc_dtype=dtypes["acc_dtype"],
            a_major="k",
            b_major="k",
            c_major="n",
            config=config,
        )
    if op_name == "dgrad":
        return CuteDenseGemmOp(
            name=f"{variant}_dgrad",
            m=problem.m,
            n=problem.k,
            k=problem.n,
            a_dtype=dtypes["ab_dtype"],
            b_dtype=dtypes["ab_dtype"],
            c_dtype=dtypes["dgrad_c_dtype"],
            acc_dtype=dtypes["acc_dtype"],
            a_major="k",
            b_major="k",
            c_major="n",
            config=config,
        )
    if op_name == "wgrad":
        return CuteDenseGemmOp(
            name=f"{variant}_wgrad",
            m=problem.k,
            n=problem.n,
            k=problem.m,
            a_dtype=dtypes["ab_dtype"],
            b_dtype=dtypes["ab_dtype"],
            c_dtype=dtypes["wgrad_c_dtype"],
            acc_dtype=dtypes["acc_dtype"],
            a_major="k",
            b_major="k",
            c_major="n",
            config=config,
        )
    raise ValueError(f"unknown op_name={op_name}")


def autotune_ops(
    problem: LinearProblem,
    variant: str,
    fp8_format: str,
    operands: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    warmup_iterations: int,
    iterations: int,
) -> tuple[dict[str, DenseGemmConfig], dict[str, Any]]:
    summaries: dict[str, Any] = {}
    best_configs: dict[str, DenseGemmConfig] = {}
    dtypes = make_cutlass_dtypes(variant, fp8_format)
    for op_name in ("fwd", "dgrad", "wgrad"):
        a, b = operands[op_name]
        trials = []
        best_time_us = float("inf")
        best_config = None
        for tile in TILE_SHAPE_CHOICES:
            for cluster in CLUSTER_SHAPE_CHOICES:
                config = DenseGemmConfig(tile, cluster)
                try:
                    op = build_single_op(problem, variant, fp8_format, op_name, config, dtypes)
                    kernel_us = op.benchmark_kernel_us(
                        a,
                        b,
                        warmup_iterations=warmup_iterations,
                        iterations=iterations,
                    )
                    trial = {
                        "config": op.config.as_dict(),
                        "kernel_us": kernel_us,
                        "compile_ms": op.compile_ms,
                    }
                except Exception as exc:
                    trial = {
                        "config": config.as_dict(),
                        "error": type(exc).__name__,
                        "message": str(exc),
                    }
                trials.append(trial)
                if "kernel_us" in trial and kernel_us < best_time_us:
                    best_time_us = kernel_us
                    best_config = config
        if best_config is None:
            raise RuntimeError(f"autotune produced no valid config for {variant} {op_name}")
        trials.sort(key=lambda t: t.get("kernel_us", float("inf")))
        summaries[op_name] = {
            "best": trials[0],
            "all_trials": trials,
        }
        best_configs[op_name] = best_config
    return best_configs, summaries


def fixed_config_map(args: argparse.Namespace) -> dict[str, DenseGemmConfig]:
    config = DenseGemmConfig(args.tile_shape_mn, args.cluster_shape_mn)
    return {"fwd": config, "dgrad": config, "wgrad": config}


def build_custom_module(
    problem: LinearProblem,
    variant: str,
    fp8_format: str,
    weight: torch.Tensor,
    config_map: dict[str, DenseGemmConfig],
) -> tuple[CuTeLinearModule, dict[str, Any]]:
    fwd_op, dgrad_op, wgrad_op = build_ops_for_config(problem, variant, fp8_format, config_map)
    backend = CuTeLinearBackend(
        problem=problem,
        variant=variant,
        fwd_op=fwd_op,
        dgrad_op=dgrad_op,
        wgrad_op=wgrad_op,
        fp8_format=fp8_format,
        fp8_target_max=FP8_INFO[fp8_format]["max"],
    )
    meta = {
        "variant": variant,
        "fp8_format": fp8_format if variant == "fp8" else None,
        "configs": {name: cfg.as_dict() for name, cfg in config_map.items()},
        "compile_ms": {
            "fwd": fwd_op.compile_ms,
            "dgrad": dgrad_op.compile_ms,
            "wgrad": wgrad_op.compile_ms,
        },
    }
    return CuTeLinearModule(weight, backend).cuda(), meta


def benchmark_baseline(
    problem: LinearProblem,
    weight: torch.Tensor,
    x: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    compile_baseline: bool,
    warmup_iters: int,
    iters: int,
    reference: dict[str, torch.Tensor],
) -> dict[str, Any]:
    module = BaselineLinearModule(problem.k, problem.n, weight).cuda()
    if compile_baseline:
        module = torch.compile(module, dynamic=False, fullgraph=True)
    timings, outputs = benchmark_autograd_module(
        module,
        x,
        grad_out,
        warmup_iters=warmup_iters,
        iters=iters,
        use_autocast=True,
    )
    return {
        "timings": {name: asdict(summary) for name, summary in timings.items()},
        "errors": {
            name: asdict(summarize_error(outputs[name], reference[name]))
            for name in ("output", "grad_input", "grad_weight")
        },
        "compiled": compile_baseline,
    }


def benchmark_custom_variant(
    problem: LinearProblem,
    variant: str,
    fp8_format: str,
    weight: torch.Tensor,
    x: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    warmup_iters: int,
    iters: int,
    autotune: bool,
    autotune_warmup_iters: int,
    autotune_iters: int,
    fixed_configs: dict[str, DenseGemmConfig],
    reference: dict[str, torch.Tensor],
) -> dict[str, Any]:
    operands = representative_operands(problem, weight, x, grad_out)
    autotune_summary = None
    if autotune:
        config_map, autotune_summary = autotune_ops(
            problem,
            variant,
            fp8_format,
            operands,
            warmup_iterations=autotune_warmup_iters,
            iterations=autotune_iters,
        )
    else:
        config_map = fixed_configs
    module, meta = build_custom_module(problem, variant, fp8_format, weight, config_map)
    timings, outputs = benchmark_autograd_module(
        module,
        x,
        grad_out,
        warmup_iters=warmup_iters,
        iters=iters,
        use_autocast=False,
    )
    return {
        "timings": {name: asdict(summary) for name, summary in timings.items()},
        "errors": {
            name: asdict(summarize_error(outputs[name], reference[name]))
            for name in ("output", "grad_input", "grad_weight")
        },
        "meta": meta,
        "autotune": autotune_summary,
    }


def print_problem(problem: LinearProblem) -> None:
    print(
        f"Problem: layer={problem.layer} input_shape={problem.input_shape}"
        f" output_shape={problem.output_shape} flattened_m={problem.m} k={problem.k} n={problem.n}"
    )


def print_result(label: str, result: dict[str, Any]) -> None:
    timings = result["timings"]
    print(
        f"{label}:"
        f" fwd={timings['forward']['mean_us']:.1f}us"
        f" bwd={timings['backward']['mean_us']:.1f}us"
        f" total={timings['total']['mean_us']:.1f}us"
    )
    errs = result["errors"]
    print(
        "  numerics:"
        f" out.rel_l2={errs['output']['rel_l2']:.3e}"
        f" dx.rel_l2={errs['grad_input']['rel_l2']:.3e}"
        f" dw.rel_l2={errs['grad_weight']['rel_l2']:.3e}"
    )
    meta = result.get("meta")
    if meta is not None:
        cfgs = meta["configs"]
        print(
            "  configs:"
            f" fwd={cfgs['fwd']}"
            f" dgrad={cfgs['dgrad']}"
            f" wgrad={cfgs['wgrad']}"
        )
    autotune = result.get("autotune")
    if autotune:
        for name in ("fwd", "dgrad", "wgrad"):
            best = autotune[name]["best"]
            print(
                f"  autotune[{name}]:"
                f" kernel={best['kernel_us']:.1f}us"
                f" compile={best['compile_ms']:.1f}ms"
                f" config={best['config']}"
            )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark one train_gpt linear layer with CuTe dense GEMMs")
    parser.add_argument("--layer", choices=LAYER_CHOICES, default="mlp_fc")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--in-features", type=int, default=None)
    parser.add_argument("--out-features", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--autotune-warmup-iters", type=int, default=3)
    parser.add_argument("--autotune-iters", type=int, default=10)
    parser.add_argument("--fp8-format", choices=tuple(FP8_INFO), default="e4m3fn")
    parser.add_argument("--tile-shape-mn", type=parse_pair, default=(128, 128))
    parser.add_argument("--cluster-shape-mn", type=parse_pair, default=(1, 1))
    parser.add_argument("--no-compile-baseline", action="store_true")
    parser.add_argument("--no-autotune", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-fp16", action="store_true")
    parser.add_argument("--skip-fp8", action="store_true")
    parser.add_argument("--results-json", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    if args.warmup_iters < 0 or args.iters <= 0:
        raise ValueError("--warmup-iters must be >= 0 and --iters must be > 0")
    if args.autotune_warmup_iters < 0 or args.autotune_iters <= 0:
        raise ValueError("--autotune-warmup-iters must be >= 0 and --autotune-iters must be > 0")

    device = configure_runtime(args.device)
    defaults = TrainLikeDefaults()
    problem = build_problem(args, defaults)
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    print(
        f"CUDA device: {torch.cuda.get_device_name(device_index)}"
        f" cc={torch.cuda.get_device_capability(device_index)}"
    )
    print_problem(problem)

    x, weight, grad_out = make_master_tensors(problem, seed=args.seed, device=device)
    reference = make_reference(x, weight, grad_out)

    results: dict[str, Any] = {
        "device": {
            "name": torch.cuda.get_device_name(device_index),
            "capability": list(torch.cuda.get_device_capability(device_index)),
            "torch": torch.__version__,
        },
        "problem": asdict(problem),
        "defaults": {
            **asdict(defaults),
            "grad_accum_steps": defaults.grad_accum_steps,
            "device_batch_tokens": defaults.device_batch_tokens,
            "micro_batch_size": defaults.micro_batch_size,
        },
        "results": {},
    }

    fixed_configs = fixed_config_map(args)
    if not args.skip_baseline:
        baseline = benchmark_baseline(
            problem,
            weight,
            x,
            grad_out,
            compile_baseline=not args.no_compile_baseline,
            warmup_iters=args.warmup_iters,
            iters=args.iters,
            reference=reference,
        )
        results["results"]["baseline"] = baseline
        print_result("baseline", baseline)

    if not args.skip_fp16:
        fp16 = benchmark_custom_variant(
            problem,
            "fp16",
            args.fp8_format,
            weight,
            x,
            grad_out,
            warmup_iters=args.warmup_iters,
            iters=args.iters,
            autotune=not args.no_autotune,
            autotune_warmup_iters=args.autotune_warmup_iters,
            autotune_iters=args.autotune_iters,
            fixed_configs=fixed_configs,
            reference=reference,
        )
        results["results"]["cute_fp16"] = fp16
        print_result("cute_fp16", fp16)

    if not args.skip_fp8:
        fp8 = benchmark_custom_variant(
            problem,
            "fp8",
            args.fp8_format,
            weight,
            x,
            grad_out,
            warmup_iters=args.warmup_iters,
            iters=args.iters,
            autotune=not args.no_autotune,
            autotune_warmup_iters=args.autotune_warmup_iters,
            autotune_iters=args.autotune_iters,
            fixed_configs=fixed_configs,
            reference=reference,
        )
        results["results"]["cute_fp8"] = fp8
        print_result("cute_fp8", fp8)

    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        with args.results_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote results to {args.results_json}")


if __name__ == "__main__":
    main()
