#!/usr/bin/env python3
"""Probe train_gpt-style MLP linears with simple PyTorch implementations.

This script mirrors the default microbatch geometry from `train_gpt.py`:

- `TRAIN_BATCH_TOKENS=786432`
- `TRAIN_SEQ_LEN=2048`
- `grad_accum_steps=8 // WORLD_SIZE`
- default layer: MLP FC (`512 -> 2048`)

It benchmarks a compiled `CastedLinear` under BF16 autocast.

Each implementation is measured in two modes:

- inference: forward only
- training: forward + backward
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


LAYER_CHOICES = (
    "mlp_fc",
    "mlp_proj",
)


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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


def restore_fp32_params(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()


class CastedLinearModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, weight: torch.Tensor):
        super().__init__()
        self.linear = CastedLinear(in_features, out_features, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
        restore_fp32_params(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def build_problem(args: argparse.Namespace, defaults: TrainLikeDefaults) -> LinearProblem:
    batch_size = defaults.micro_batch_size
    seq_len = defaults.train_seq_len
    hidden = int(defaults.mlp_mult * defaults.model_dim)
    layer_map = {
        "mlp_fc": (defaults.model_dim, hidden),
        "mlp_proj": (hidden, defaults.model_dim),
    }
    default_k, default_n = layer_map[args.layer]
    k = default_k
    n = default_n
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


def benchmark_inference_module(
    module: nn.Module,
    x_master: torch.Tensor,
    *,
    warmup_iters: int,
    iters: int,
    use_autocast: bool,
) -> tuple[TimingSummary, torch.Tensor]:
    module.eval()
    times_us: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    last_y = None

    torch.cuda.synchronize()
    with torch.no_grad():
        for step in range(warmup_iters + iters):
            start.record()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                y = module(x_master)
            end.record()
            torch.cuda.synchronize()
            if step >= warmup_iters:
                times_us.append(1e3 * start.elapsed_time(end))
            last_y = y.detach()

    return summarize_times(times_us), last_y.detach().clone()


def benchmark_train_module(
    label: str,
    module: nn.Module,
    x: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    warmup_iters: int,
    iters: int,
    use_autocast: bool,
    reference: dict[str, torch.Tensor],
) -> dict[str, Any]:
    timings, outputs = benchmark_autograd_module(
        module,
        x,
        grad_out,
        warmup_iters=warmup_iters,
        iters=iters,
        use_autocast=use_autocast,
    )
    return {
        "label": label,
        "mode": "train",
        "timings": {name: asdict(summary) for name, summary in timings.items()},
        "errors": {
            name: asdict(summarize_error(outputs[name], reference[name]))
            for name in ("output", "grad_input", "grad_weight")
        },
    }


def benchmark_eval_module(
    label: str,
    module: nn.Module,
    x: torch.Tensor,
    *,
    warmup_iters: int,
    iters: int,
    use_autocast: bool,
    reference_output: torch.Tensor,
) -> dict[str, Any]:
    timing, output = benchmark_inference_module(
        module,
        x,
        warmup_iters=warmup_iters,
        iters=iters,
        use_autocast=use_autocast,
    )
    return {
        "label": label,
        "mode": "inference",
        "timings": {"forward": asdict(timing)},
        "errors": {"output": asdict(summarize_error(output, reference_output))},
    }


def print_problem(problem: LinearProblem) -> None:
    print(
        f"Problem: layer={problem.layer} input_shape={problem.input_shape}"
        f" output_shape={problem.output_shape} flattened_m={problem.m} k={problem.k} n={problem.n}"
    )


def print_result(label: str, result: dict[str, Any]) -> None:
    timings = result["timings"]
    if result["mode"] == "inference":
        print(f"{label}: fwd={timings['forward']['mean_us']:.1f}us")
        errs = result["errors"]
        print(f"  numerics: out.rel_l2={errs['output']['rel_l2']:.3e}")
        return
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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe train_gpt MLP linears with compiled CastedLinear"
    )
    parser.add_argument("--layer", choices=LAYER_CHOICES, default="mlp_fc")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--results-json", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    if args.warmup_iters < 0 or args.iters <= 0:
        raise ValueError("--warmup-iters must be >= 0 and --iters must be > 0")

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

    module = CastedLinearModule(problem.k, problem.n, weight).cuda()
    module = torch.compile(module, dynamic=False, fullgraph=True)
    if not args.skip_inference:
        inference_result = benchmark_eval_module(
            "casted_linear.inference",
            module,
            x,
            warmup_iters=args.warmup_iters,
            iters=args.iters,
            use_autocast=True,
            reference_output=reference["output"],
        )
        results["results"]["casted_linear.inference"] = inference_result
        print_result("casted_linear.inference", inference_result)
    if not args.skip_training:
        train_result = benchmark_train_module(
            "casted_linear.train",
            module,
            x,
            grad_out,
            warmup_iters=args.warmup_iters,
            iters=args.iters,
            use_autocast=True,
            reference=reference,
        )
        results["results"]["casted_linear.train"] = train_result
        print_result("casted_linear.train", train_result)

    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        with args.results_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote results to {args.results_json}")


if __name__ == "__main__":
    main()
