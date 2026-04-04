import copy
import glob
import io
import lzma
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor, nn

from flash_attn_interface import flash_attn_func as flash_attn_3_func
from codebook.latticee8_padded12 import E8P12_codebook, decode_e8p_indices, quantize_blocks_to_e8p

try:
    from fast_hadamard_transform import hadamard_transform as _fast_hadamard_transform
    HADAMARD_BACKEND = "fast_hadamard_transform"
except Exception:
    _fast_hadamard_transform = None
    HADAMARD_BACKEND = "torch_fallback"


def hadamard_transform(x: Tensor, scale: float = 1.0) -> Tensor:
    if _fast_hadamard_transform is not None and x.is_cuda:
        return _fast_hadamard_transform(x, scale=scale)
    original_shape = x.shape
    dim = x.shape[-1]
    padded_dim = 1 << (max(dim, 1) - 1).bit_length()
    out = x.reshape(-1, dim)
    if padded_dim != dim:
        out = F.pad(out, (0, padded_dim - dim))
    h = 1
    while h < padded_dim:
        out = out.view(-1, padded_dim // (2 * h), 2, h)
        a = out[:, :, 0, :]
        b = out[:, :, 1, :]
        out = torch.stack((a + b, a - b), dim=2).reshape(-1, padded_dim)
        h *= 2
    out = out[:, :dim]
    return (out * scale).reshape(*original_shape)

# ----------------------------------------
# Hyperparameters
# ----------------------------------------

class Hyperparameters():
    # Experiment settings
    data_dir = os.environ.get('DATA_DIR', './data/')
    seed = int(os.environ.get('SEED', 1337))
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))

    # Training length
    iterations = int(os.environ.get('ITERATIONS', 20000))
    warmdown_frac = float(os.environ.get('WARMDOWN_FRAC', 0.667))
    warmup_steps = int(os.environ.get('WARMUP_STEPS', 20))
    train_batch_tokens = int(os.environ.get('TRAIN_BATCH_TOKENS', 2048 * 48 * 8))
    train_seq_len = int(os.environ.get('TRAIN_SEQ_LEN', 2048))
    eval_seq_len = int(os.environ.get('EVAL_SEQ_LEN', 2048))
    max_wallclock_seconds = float(os.environ.get('MAX_WALLCLOCK_SECONDS', 600.0))
    train_log_every = int(os.environ.get('TRAIN_LOG_EVERY', 500))

    # Validation/Evals
    val_batch_tokens = int(os.environ.get('VAL_BATCH_TOKENS', 2048 * 32 * 8))
    val_loss_every = int(os.environ.get('VAL_LOSS_EVERY', 4000))
    sliding_window_enabled = bool(int(os.environ.get('SLIDING_WINDOW_ENABLED', '1')))

    # Model architecture
    vocab_size = int(os.environ.get('VOCAB_SIZE', 4096))
    num_layers = int(os.environ.get('NUM_LAYERS', 11))
    xsa_last_n = int(os.environ.get('XSA_LAST_N', 11))
    num_kv_heads = int(os.environ.get('NUM_KV_HEADS', 4))
    model_dim = int(os.environ.get('MODEL_DIM', 512))
    embedding_dim = int(os.environ.get('EMBEDDING_DIM', 512))
    num_heads = int(os.environ.get('NUM_HEADS', 8))
    mlp_mult = float(os.environ.get('MLP_MULT', 4.0))
    skip_gates_enabled = bool(int(os.environ.get('SKIP_GATES_ENABLED', '1')))
    tie_embeddings = bool(int(os.environ.get('TIE_EMBEDDINGS', '1')))
    logit_softcap = float(os.environ.get('LOGIT_SOFTCAP', 30.0))
    rope_base = float(os.environ.get('ROPE_BASE', 10000.0))
    rope_dims = int(os.environ.get('ROPE_DIMS', 16))
    rope_train_seq_len = int(os.environ.get('ROPE_TRAIN_SEQ_LEN', 2048))
    ln_scale = bool(int(os.environ.get('LN_SCALE', '1')))
    ve_enabled = bool(int(os.environ.get('VE_ENABLED', '1')))
    ve_dim = int(os.environ.get('VE_DIM', 128))
    ve_layers = os.environ.get('VE_LAYERS', '9,10')
    qk_gain_init = float(os.environ.get('QK_GAIN_INIT', 4.0))

    # Optimizer
    min_lr = float(os.environ.get('MIN_LR', 0.0))
    embed_lr = float(os.environ.get('EMBED_LR', 0.6))
    head_lr = float(os.environ.get('HEAD_LR', 0.008))
    tied_embed_lr = float(os.environ.get('TIED_EMBED_LR', 0.03))
    tied_embed_init_std = float(os.environ.get('TIED_EMBED_INIT_STD', 0.005))
    matrix_lr = float(os.environ.get('MATRIX_LR', 0.02))
    scalar_lr = float(os.environ.get('SCALAR_LR', 0.02))
    muon_momentum = float(os.environ.get('MUON_MOMENTUM', 0.99))
    muon_backend_steps = int(os.environ.get('MUON_BACKEND_STEPS', 5))
    muon_momentum_warmup_start = float(os.environ.get('MUON_MOMENTUM_WARMUP_START', 0.92))
    muon_momentum_warmup_steps = int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS', 1500))
    beta1 = float(os.environ.get('BETA1', 0.9))
    beta2 = float(os.environ.get('BETA2', 0.95))
    adam_eps = float(os.environ.get('ADAM_EPS', 1e-8))
    grad_clip_norm = float(os.environ.get('GRAD_CLIP_NORM', 0.3))
    eval_stride = int(os.environ.get('EVAL_STRIDE', 64))
    muon_beta2 = float(os.environ.get('MUON_BETA2', 0.95))
    adam_wd = float(os.environ.get('ADAM_WD', 0.02))
    muon_wd = float(os.environ.get('MUON_WD', 0.085))
    embed_wd = float(os.environ.get('EMBED_WD', 0.085))

    # Compression
    compressor = os.environ.get('COMPRESSOR', 'brotli')  #(lzma or brotli)
    codebook_block_dim = int(os.environ.get("CODEBOOK_BLOCK_DIM", 8))
    codebook_use_hadamard = bool(int(os.environ.get("CODEBOOK_USE_HADAMARD", "1")))
    codebook_use_residual = bool(int(os.environ.get("CODEBOOK_USE_RESIDUAL", "1")))
    codebook_use_outliers = bool(int(os.environ.get("CODEBOOK_USE_OUTLIERS", "1")))
    codebook_residual_k = int(os.environ.get("CODEBOOK_RESIDUAL_K", 16))
    codebook_scale_bits = int(os.environ.get("CODEBOOK_SCALE_BITS", 6))
    codebook_init_kmeans_iters = int(os.environ.get("CODEBOOK_INIT_KMEANS_ITERS", 8))
    codebook_refinement_iters = int(os.environ.get("CODEBOOK_REFINEMENT_ITERS", 2))
    codebook_calibration_batches = int(os.environ.get("CODEBOOK_CALIBRATION_BATCHES", 64))
    codebook_reserve_seconds = float(os.environ.get("CODEBOOK_RESERVE_SECONDS", 10.0))
    codebook_hessian_damp = float(os.environ.get("CODEBOOK_HESSIAN_DAMP", 0.01))
    codebook_lattice_scale = float(os.environ.get("CODEBOOK_LATTICE_SCALE", 1.03))
    codebook_ldlq_iters = int(os.environ.get("CODEBOOK_LDLQ_ITERS", 2))
    codebook_ldlq_topk = int(os.environ.get("CODEBOOK_LDLQ_TOPK", 4))
    codebook_outlier_frac = float(os.environ.get("CODEBOOK_OUTLIER_FRAC", 0.01))
    codebook_outlier_max_blocks = int(os.environ.get("CODEBOOK_OUTLIER_MAX_BLOCKS", 4096))
    codebook_debug_topk = int(os.environ.get("CODEBOOK_DEBUG_TOPK", 4))

    # Distributed setup
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_process = rank == 0
    grad_accum_steps = 8 // world_size

    # Data paths
    datasets_dir = os.path.join(data_dir, 'datasets', f'fineweb10B_sp{vocab_size}')
    train_files = os.path.join(datasets_dir, 'fineweb_train_*.bin')
    val_files = os.path.join(datasets_dir, 'fineweb_val_*.bin')
    tokenizer_path = os.path.join(data_dir, 'tokenizers', f'fineweb_{vocab_size}_bpe.model')

    # Experiment files
    logfile = f"logs/{run_id}.txt"
    model_path = "final_model.pt"
    quantized_model_path = "final_model.codebook.ptz"

# ----------------------------------------
# Global Logging Function
# ----------------------------------------

_logger_hparams = None


def set_logging_hparams(h: Hyperparameters) -> None:
    global _logger_hparams
    _logger_hparams = h


def log(msg, console: bool = True) -> None:
    if _logger_hparams is None:
        print(msg)
    if _logger_hparams.is_main_process:
        if console:
            print(msg)
        if _logger_hparams.logfile is not None:
            with open(_logger_hparams.logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

# ----------------------------------------
# Data Loading
# ----------------------------------------

class ValidationData:
    def __init__(self, h: Hyperparameters, device: torch.device):
        if not h.tokenizer_path.endswith(".model"):
            raise ValueError(f"Script only setup for SentencePiece .model file: {h.tokenizer_path}")
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        if int(self.sp.vocab_size()) != h.vocab_size:
            raise ValueError(
                f"VOCAB_SIZE={h.vocab_size} does not match tokenizer vocab_size={int(self.sp.vocab_size())}"
            )

        self.val_tokens = load_validation_tokens(h.val_files, h.eval_seq_len)
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = (
            build_sentencepiece_luts(self.sp, h.vocab_size, device))


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    # The BPB calculation assumes "▁" is its own token so that leading-space bytes
    # are counted correctly. See https://github.com/openai/parameter-golf/issues/897
    assert sp.piece_to_id("\u2581") != sp.unk_id(), \
        "Tokenizer must have '▁' (space) as its own token for correct BPB byte counting"
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


_SHARD_HEADER_BYTES = 256 * np.dtype("<i4").itemsize
_SHARD_NTOKENS_CACHE: dict[str, int] = {}
_MMAP_CACHE: dict[str, np.memmap] = {}


def _read_num_tokens(file: Path) -> int:
    key = str(file)
    cached = _SHARD_NTOKENS_CACHE.get(key)
    if cached is not None:
        return cached
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    n = int(header[2])
    _SHARD_NTOKENS_CACHE[key] = n
    return n


def _get_shard_memmap(file: Path) -> np.memmap:
    key = str(file)
    mm = _MMAP_CACHE.get(key)
    if mm is not None:
        return mm
    n = _read_num_tokens(file)
    mm = np.memmap(file, mode="r", dtype="<u2", offset=_SHARD_HEADER_BYTES, shape=(n,))
    _MMAP_CACHE[key] = mm
    return mm


class DistributedTokenLoader:
    """Coprime-stride multi-shard loader. Samples windows across shards with
    increasing diversity over training, using coprime strides for coverage."""

    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self._num_tokens = np.array([_read_num_tokens(f) for f in self.files], dtype=np.int64)
        seed = 0
        for f in self.files:
            for b in str(f).encode():
                seed = ((seed ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._cfg: tuple[int, int, int, int] | None = None
        self._eligible_shards: np.ndarray | None = None
        self._base_block_counts: np.ndarray | None = None
        n = len(self.files)
        self._cursor_phase = np.zeros(n, dtype=np.int64)
        self._cursor_block_count = np.zeros(n, dtype=np.int64)
        self._cursor_next = np.zeros(n, dtype=np.int64)
        self._cursor_start = np.zeros(n, dtype=np.int64)
        self._cursor_stride = np.ones(n, dtype=np.int64)
        self._cursor_init = np.zeros(n, dtype=np.bool_)
        self._batches_built = 0

    def _pick_coprime_stride(self, n: int) -> int:
        if n <= 1:
            return 1
        while True:
            s = int(self._rng.integers(1, n))
            if math.gcd(s, n) == 1:
                return s

    def _reset_cursor(self, si: int, seq_len: int) -> None:
        nt = int(self._num_tokens[si])
        max_phase = min(seq_len - 1, max(0, nt - seq_len - 1))
        phase = int(self._rng.integers(max_phase + 1)) if max_phase > 0 else 0
        bc = (nt - 1 - phase) // seq_len
        self._cursor_phase[si] = phase
        self._cursor_block_count[si] = bc
        self._cursor_next[si] = 0
        self._cursor_start[si] = int(self._rng.integers(bc)) if bc > 1 else 0
        self._cursor_stride[si] = self._pick_coprime_stride(bc)
        self._cursor_init[si] = True

    def _ensure_cursor(self, si: int, seq_len: int) -> None:
        if not self._cursor_init[si] or self._cursor_next[si] >= self._cursor_block_count[si]:
            self._reset_cursor(si, seq_len)

    def _take_from_shard(self, si: int, seq_len: int, count: int, out: list[tuple[int, int]]) -> None:
        rem = count
        while rem > 0:
            self._ensure_cursor(si, seq_len)
            bc = int(self._cursor_block_count[si])
            ni = int(self._cursor_next[si])
            take = min(rem, bc - ni)
            phase = int(self._cursor_phase[si])
            start = int(self._cursor_start[si])
            stride = int(self._cursor_stride[si])
            for j in range(take):
                bi = (start + (ni + j) * stride) % bc
                out.append((si, phase + bi * seq_len))
            self._cursor_next[si] = ni + take
            rem -= take

    def _init_pipeline(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> None:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        num_seqs = local_tokens // seq_len
        global_num_seqs = num_seqs * self.world_size
        self._cfg = (local_tokens, seq_len, num_seqs, global_num_seqs)
        bbc = (self._num_tokens - 1) // seq_len
        eligible = bbc > 0
        self._eligible_shards = np.nonzero(eligible)[0].astype(np.int64)
        self._base_block_counts = bbc[self._eligible_shards].astype(np.int64)

    def _sample_global_windows(self) -> list[tuple[int, int]]:
        assert self._cfg is not None and self._eligible_shards is not None
        _, seq_len, _, gns = self._cfg
        ec = int(self._eligible_shards.size)
        progress = min(self._batches_built / 1800.0, 1.0)
        remaining = np.empty(ec, dtype=np.float64)
        for i, si in enumerate(self._eligible_shards.tolist()):
            if self._cursor_init[si]:
                r = int(self._cursor_block_count[si]) - int(self._cursor_next[si])
                remaining[i] = float(max(r, 1))
            else:
                remaining[i] = float(self._base_block_counts[i])
        alpha = 0.90 - 0.40 * progress
        weights = np.power(remaining, alpha)
        ws = float(weights.sum())
        if not np.isfinite(ws) or ws <= 0.0:
            weights = np.ones(ec, dtype=np.float64)
            ws = float(weights.sum())
        probs = weights / ws
        low = min(max(8, self.world_size), ec, gns)
        high = min(max(32, self.world_size * 8), ec, gns)
        mix = max(1, min(int(round(low + progress * (high - low))), ec, gns))
        cp = self._rng.choice(ec, size=mix, replace=False, p=probs)
        cs = self._eligible_shards[cp]
        cpr = probs[cp].copy()
        cpr /= cpr.sum()
        counts = np.ones(mix, dtype=np.int64)
        extra = gns - mix
        if extra > 0:
            counts += self._rng.multinomial(extra, cpr).astype(np.int64)
        perm = self._rng.permutation(mix)
        cs, counts = cs[perm], counts[perm]
        buckets: list[list[tuple[int, int]]] = []
        for si, cnt in zip(cs.tolist(), counts.tolist()):
            b: list[tuple[int, int]] = []
            self._take_from_shard(int(si), seq_len, int(cnt), b)
            if b:
                if len(b) > 1:
                    bp = self._rng.permutation(len(b))
                    b = [b[int(k)] for k in bp.tolist()]
                buckets.append(b)
        windows: list[tuple[int, int]] = []
        active = [i for i, bk in enumerate(buckets) if bk]
        while active:
            order = self._rng.permutation(len(active))
            new_active: list[int] = []
            for oi in order.tolist():
                bi = active[oi]
                if buckets[bi]:
                    windows.append(buckets[bi].pop())
                if buckets[bi]:
                    new_active.append(bi)
            active = new_active
        return windows

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        if self._cfg is None:
            self._init_pipeline(global_tokens, seq_len, grad_accum_steps)
        _, _, num_seqs, _ = self._cfg
        gw = self._sample_global_windows()
        local_w = gw[self.rank::self.world_size]
        x = torch.empty((num_seqs, seq_len), dtype=torch.int64)
        y = torch.empty((num_seqs, seq_len), dtype=torch.int64)
        for slot, (si, pos) in enumerate(local_w):
            mm = _get_shard_memmap(self.files[si])
            window = torch.as_tensor(np.array(mm[pos:pos + seq_len + 1], dtype=np.int64))
            x[slot] = window[:-1]
            y[slot] = window[1:]
        self._batches_built += 1
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# ----------------------------------------
# Model Architecture
# ----------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float, train_seq_len: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len)
        self.use_xsa = False

    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x: Tensor, v_embed: Tensor | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x)
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)


class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size: int, ve_dim: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float, train_seq_len: int,
                 layer_idx: int = 0, ln_scale: bool = False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(self, x: Tensor, x0: Tensor, v_embed: Tensor | None = None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, v_embed=v_embed)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        return x_out


class GPT(nn.Module):
    def __init__(self, h: Hyperparameters):
        super().__init__()
        self._ve_target_dim = h.num_kv_heads * (h.model_dim // h.num_heads)
        if h.logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {h.logit_softcap}")
        self.tie_embeddings = h.tie_embeddings
        self.tied_embed_init_std = h.tied_embed_init_std
        self.logit_softcap = h.logit_softcap
        self.tok_emb = nn.Embedding(h.vocab_size, h.embedding_dim)
        if h.embedding_dim != h.model_dim:
            self.embed_proj = CastedLinear(h.embedding_dim, h.model_dim, bias=False)
            self.head_proj = CastedLinear(h.model_dim, h.embedding_dim, bias=False)
        else:
            self.embed_proj = None
            self.head_proj = None
        self.num_encoder_layers = h.num_layers // 2
        self.num_decoder_layers = h.num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, h.model_dim, dtype=torch.float32))
        self.skip_gates = nn.Parameter(torch.zeros(self.num_skip_weights, h.model_dim, dtype=torch.float32)) if h.skip_gates_enabled else None
        self.blocks = nn.ModuleList([
            Block(h.model_dim, h.num_heads, h.num_kv_heads, h.mlp_mult, h.rope_base,
                  h.qk_gain_init, h.train_seq_len, layer_idx=i, ln_scale=h.ln_scale)
            for i in range(h.num_layers)
        ])
        if h.rope_dims > 0:
            head_dim = h.model_dim // h.num_heads
            for block in self.blocks:
                block.attn.rope_dims = h.rope_dims
                block.attn.rotary = Rotary(head_dim, base=h.rope_base, train_seq_len=h.train_seq_len, rope_dims=h.rope_dims)
        self.ve_layer_indices = [int(x) for x in h.ve_layers.split(",") if x.strip()] if h.ve_enabled else []
        kv_dim = self._ve_target_dim
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(h.vocab_size, h.ve_dim, kv_dim)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices]
            )
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        self.value_embeds = nn.ModuleList()
        self.final_norm = RMSNorm()
        self.lm_head = None if h.tie_embeddings else CastedLinear(h.embedding_dim, h.vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        if h.xsa_last_n > 0:
            for i in range(max(0, h.num_layers - h.xsa_last_n), h.num_layers):
                self.blocks[i].attn.use_xsa = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def _get_ve(self, layer_idx: int, input_ids: Tensor, ve_cache: dict | None = None) -> Tensor | None:
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if ve_cache is not None and 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = ve_cache['ve'] if ve_cache is not None else self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        x0 = x
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0, v_embed=ve)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                scaled_skip = self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                if self.skip_gates is not None:
                    g = torch.sigmoid(self.skip_gates[i].to(dtype=x.dtype))[None, None, :]
                    x = torch.lerp(scaled_skip, x, g)
                else:
                    x = x + scaled_skip
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0, v_embed=ve)
        x = self.final_norm(x)
        if self.head_proj is not None:
            x = self.head_proj(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.forward_logits(input_ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1), reduction="mean")

# ----------------------------------------
# Optimization
# ----------------------------------------

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


class Optimizers():
    def __init__(self, h: Hyperparameters, base_model: GPT):
        block_named_params = list(base_model.blocks.named_parameters())
        matrix_params = [
            p
            for name, p in block_named_params
            if p.ndim == 2
            and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        scalar_params = [
            p
            for name, p in block_named_params
            if (p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS))
        ]
        if base_model.skip_weights.numel() > 0:
            scalar_params.append(base_model.skip_weights)
        if base_model.skip_gates is not None and base_model.skip_gates.numel() > 0:
            scalar_params.append(base_model.skip_gates)

        token_lr = h.tied_embed_lr if h.tie_embeddings else h.embed_lr
        tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
        if base_model.ve_shared is not None:
            tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
            if base_model.ve_shared.proj is not None:
                matrix_params.append(base_model.ve_shared.proj.weight)
            scalar_params.append(base_model.ve_shared.scale)
            for s in base_model.ve_layer_scales:
                scalar_params.append(s)

        self.optimizer_tok = torch.optim.AdamW(
            tok_params,
            betas=(h.beta1, h.beta2),
            eps=h.adam_eps,
            weight_decay=h.embed_wd,
            fused=True,
        )
        self.optimizer_muon = Muon(
            matrix_params,
            lr=h.matrix_lr,
            momentum=h.muon_momentum,
            backend_steps=h.muon_backend_steps,
            weight_decay=h.muon_wd,
        )
        for group in self.optimizer_muon.param_groups:
            group["base_lr"] = h.matrix_lr
        self.optimizer_scalar = torch.optim.AdamW(
            [{"params": scalar_params, "lr": h.scalar_lr, "base_lr": h.scalar_lr}],
            betas=(h.beta1, h.beta2),
            eps=h.adam_eps,
            weight_decay=h.adam_wd,
            fused=True,
        )
        self.optimizers: list[torch.optim.Optimizer] = [self.optimizer_tok, self.optimizer_muon, self.optimizer_scalar]
        if base_model.lm_head is not None:
            self.optimizer_head = torch.optim.Adam(
                [{"params": [base_model.lm_head.weight], "lr": h.head_lr, "base_lr": h.head_lr}],
                betas=(h.beta1, h.beta2),
                eps=h.adam_eps,
                fused=True,
            )
            self.optimizers.insert(1, self.optimizer_head)
        else:
            self.optimizer_head = None

    def __iter__(self):
        return iter(self.optimizers)

    def zero_grad_all(self) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def step(self):
        for opt in self.optimizers:
            opt.step()
        self.zero_grad_all()

# ----------------------------------------
# Quantization
# ----------------------------------------

def restore_fp32_params(model: nn.Module) -> None:
    """After .bfloat16(), restore CastedLinear weights and control params to FP32."""
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    for name, param in model.named_parameters():
        if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
            param.data = param.data.float()


def collect_hessians(
    model: nn.Module,
    train_loader: DistributedTokenLoader,
    h: Hyperparameters,
    device: torch.device,
    target_names: set[str],
    n_calibration_batches: int,
) -> dict[str, Tensor]:
    """Collect X^T X activation covariances for the target linear weights."""
    if n_calibration_batches <= 0 or not target_names:
        return {}
    hessians: dict[str, Tensor] = {}
    hooks = []
    was_training = model.training

    def make_hook(name: str):
        def hook_fn(module, inputs, output):
            x = inputs[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            if name not in hessians:
                hessians[name] = torch.zeros(
                    x.shape[1],
                    x.shape[1],
                    dtype=torch.float32,
                    device=device,
                )
            hessians[name].addmm_(x.t(), x)
        return hook_fn

    for module_name, module in model.named_modules():
        if not isinstance(module, CastedLinear):
            continue
        weight_name = f"{module_name}.weight"
        if weight_name in target_names:
            hooks.append(module.register_forward_hook(make_hook(weight_name)))

    model.eval()
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            x, _ = train_loader.next_batch(
                h.train_batch_tokens,
                h.train_seq_len,
                h.grad_accum_steps,
            )
            model.forward_logits(x)

    for hook in hooks:
        hook.remove()

    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    for name, hessian in hessians.items():
        if world > 1:
            dist.all_reduce(hessian, op=dist.ReduceOp.SUM)
        hessians[name] = hessian.cpu() / float(max(world * n_calibration_batches, 1))

    if was_training:
        model.train()
    return hessians


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates,ve_layer_scales,ve_shared.scale",
    ).split(",")
    if pattern
)
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

CODEBOOK_TARGET_PATTERNS = (
    "attn.c_q.weight",
    "attn.c_k.weight",
    "attn.c_v.weight",
    "attn.proj.weight",
    "mlp.fc.weight",
    "mlp.proj.weight",
)


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32, device=t32.device)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32, device=t32.device)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def _blockify_weight(t: Tensor, block_dim: int) -> tuple[Tensor, tuple[int, ...]]:
    if t.ndim != 2:
        raise ValueError(f"Codebook block quantization expects a 2D tensor, got shape {tuple(t.shape)}")
    if t.shape[1] % block_dim != 0:
        raise ValueError(
            f"Tensor shape {tuple(t.shape)} is not compatible with CODEBOOK_BLOCK_DIM={block_dim}"
        )
    return t.contiguous().view(-1, block_dim), tuple(t.shape)


def _unblockify_weight(blocks: Tensor, original_shape: tuple[int, ...]) -> Tensor:
    return blocks.contiguous().view(original_shape)


def _hadamard_sign_vector(name: str, block_dim: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    rng = random.Random(f"hadamard::{name}::{block_dim}")
    return torch.tensor(
        [1.0 if rng.getrandbits(1) else -1.0 for _ in range(block_dim)],
        device=device,
        dtype=dtype,
    )


def hadamard_rotate_blocks(blocks: Tensor, sign_vec: Tensor, *, enabled: bool = True) -> Tensor:
    if not enabled:
        return blocks
    scale = blocks.shape[-1] ** -0.5
    return hadamard_transform(blocks * sign_vec, scale=scale)


def hadamard_unrotate_blocks(blocks: Tensor, sign_vec: Tensor, *, enabled: bool = True) -> Tensor:
    if not enabled:
        return blocks
    scale = blocks.shape[-1] ** -0.5
    return hadamard_transform(blocks, scale=scale) * sign_vec


def _should_codebook_quantize(name: str, t: Tensor, h: Hyperparameters) -> bool:
    return (
        t.is_floating_point()
        and t.ndim == 2
        and t.shape[1] % h.codebook_block_dim == 0
        and name.startswith("blocks.")
        and any(name.endswith(pattern) for pattern in CODEBOOK_TARGET_PATTERNS)
    )


def normalize_blocks(blocks: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
    scales = blocks.norm(dim=-1, keepdim=True).clamp_min(eps)
    return blocks / scales, scales


def nearest_codeword(x: Tensor, codebook: Tensor) -> Tensor:
    dists = (
        x.square().sum(dim=-1, keepdim=True)
        - 2 * x @ codebook.t()
        + codebook.square().sum(dim=-1).unsqueeze(0)
    )
    return dists.argmin(dim=-1)


def codebook_lookup(indices: Tensor, codebook: Tensor) -> Tensor:
    return codebook.index_select(0, indices.long())


def _bits_for_size(size: int) -> int:
    if size <= 0:
        raise ValueError(f"Expected positive size, got {size}")
    return max(1, math.ceil(math.log2(max(size, 2))))


def _pack_codes(codes: Tensor, bits: int) -> Tensor:
    if not 1 <= bits <= 32:
        raise ValueError(f"Only 1..32 bit packing is supported, got {bits}")
    codes_np = codes.detach().view(-1).cpu().numpy().astype(np.uint64, copy=False)
    bitplanes = ((codes_np[:, None] >> np.arange(bits, dtype=np.uint64)) & 1).astype(np.uint8, copy=False)
    packed = np.packbits(bitplanes.reshape(-1), bitorder="little")
    return torch.from_numpy(packed.copy())


def _unpack_codes(packed: Tensor, bits: int, count: int) -> Tensor:
    if not 1 <= bits <= 32:
        raise ValueError(f"Only 1..32 bit packing is supported, got {bits}")
    if count == 0:
        return torch.empty((0,), dtype=torch.long)
    packed_np = packed.detach().view(-1).cpu().numpy().astype(np.uint8, copy=False)
    bits_np = np.unpackbits(packed_np, bitorder="little")[: count * bits].reshape(count, bits)
    weights = (1 << np.arange(bits, dtype=np.uint64)).reshape(1, bits)
    codes_np = (bits_np.astype(np.uint64, copy=False) * weights).sum(axis=1)
    return torch.from_numpy(codes_np.astype(np.int64, copy=False))


def quantize_log_scales(scales: Tensor, bits: int, eps: float = 1e-8) -> tuple[Tensor, dict[str, float]]:
    if not 1 <= bits <= 8:
        raise ValueError(f"Only 1..8 scale bits are supported, got {bits}")
    flat = scales.detach().view(-1).float().cpu().clamp_min(eps)
    log_scales = flat.log()
    log_min = float(log_scales.min().item())
    log_max = float(log_scales.max().item())
    levels = 1 << bits
    if levels <= 1 or log_max - log_min < 1e-12:
        codes = torch.zeros_like(flat, dtype=torch.long)
        return codes, {"log_min": log_min, "log_max": log_min}
    step = (log_max - log_min) / (levels - 1)
    codes = torch.clamp(torch.round((log_scales - log_min) / step), 0, levels - 1).to(torch.long)
    return codes, {"log_min": log_min, "log_max": log_max}


def dequantize_log_scales(
    codes: Tensor,
    bits: int,
    log_min: float,
    log_max: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if not 1 <= bits <= 8:
        raise ValueError(f"Only 1..8 scale bits are supported, got {bits}")
    levels = 1 << bits
    codes_f = codes.to(device=device, dtype=torch.float32)
    if levels <= 1 or log_max - log_min < 1e-12:
        logs = torch.full_like(codes_f, log_min)
    else:
        step = (log_max - log_min) / (levels - 1)
        logs = log_min + codes_f * step
    return logs.exp().to(dtype=dtype).unsqueeze(-1)


_E8P_CODEBOOK_CACHE: dict[str, nn.Module] = {}


def _e8p_cache_key(device: torch.device | None) -> str:
    if device is None:
        return "cpu"
    return f"{device.type}:{device.index}"


def _get_e8p_codebook(device: torch.device | None) -> nn.Module:
    key = _e8p_cache_key(device)
    cached = _E8P_CODEBOOK_CACHE.get(key)
    if cached is not None:
        return cached
    codebook = E8P12_codebook(inference=False)
    if device is not None:
        codebook = codebook.to(device=device, dtype=torch.float32)
    else:
        codebook = codebook.to(dtype=torch.float32)
    codebook.eval()
    _E8P_CODEBOOK_CACHE[key] = codebook
    return codebook


@torch.no_grad()
def _quantize_e8p_blocks(blocks: Tensor, lattice_scale: float) -> tuple[Tensor, Tensor]:
    codebook = _get_e8p_codebook(blocks.device)
    vals, idxs = quantize_blocks_to_e8p(
        blocks,
        codebook=codebook,
        scale=lattice_scale,
        return_idx=True,
    )
    return vals.to(device=blocks.device, dtype=torch.float32), idxs.to(device=blocks.device, dtype=torch.long)


@torch.no_grad()
def _decode_e8p_blocks(
    idxs: Tensor,
    lattice_scale: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    codebook = _get_e8p_codebook(device)
    return decode_e8p_indices(idxs.to(device=device), codebook=codebook, scale=lattice_scale, dtype=dtype)


def _weighted_sample_init_centroids(
    points: Tensor,
    weights: Tensor,
    k: int,
    name: str,
    stage: str,
) -> Tensor:
    n = points.shape[0]
    if n <= 0:
        raise ValueError(f"Cannot initialize codebook for {name}: no points")
    probs = weights.detach().float().cpu().clamp_min(0)
    if float(probs.sum().item()) <= 0.0:
        probs = torch.ones_like(probs)
    probs = probs / probs.sum()
    seed = random.Random(f"codebook::{name}::{stage}").randrange(1 << 31)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    sampled = torch.multinomial(probs, k, replacement=n < k, generator=generator)
    return points.index_select(0, sampled.to(device=points.device))


def weighted_kmeans_update(
    points: Tensor,
    weights: Tensor,
    indices: Tensor,
    k: int,
    prev_centroids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    d = points.shape[1]
    centroids = torch.zeros(k, d, device=points.device, dtype=points.dtype)
    counts = torch.zeros(k, device=points.device, dtype=torch.float32)
    centroids.index_add_(0, indices, points * weights.unsqueeze(-1))
    counts.index_add_(0, indices, weights)
    mask = counts > 0
    if mask.any():
        centroids[mask] /= counts[mask].unsqueeze(-1)
    if prev_centroids is not None and (~mask).any():
        centroids[~mask] = prev_centroids[~mask]
    return centroids, counts


def run_weighted_lloyd_kmeans(
    points: Tensor,
    weights: Tensor,
    k: int,
    iters: int,
    name: str,
    stage: str,
) -> Tensor:
    centroids = _weighted_sample_init_centroids(points, weights, k, name, stage)
    for _ in range(max(iters, 1)):
        indices = nearest_codeword(points, centroids)
        centroids, _ = weighted_kmeans_update(points, weights, indices, k, prev_centroids=centroids)
    return centroids


def dequantize_int8_rows(q: Tensor, s: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    return (q.to(device=device, dtype=torch.float32) * s.to(device=device, dtype=torch.float32).view(-1, 1)).to(dtype=dtype)


def _validate_codebook_hparams(h: Hyperparameters) -> None:
    if not _is_power_of_two(h.codebook_block_dim):
        raise ValueError(f"CODEBOOK_BLOCK_DIM must be a power of 2, got {h.codebook_block_dim}")
    if h.codebook_block_dim != 8:
        raise ValueError(
            f"This hybrid E8P path requires CODEBOOK_BLOCK_DIM=8, got {h.codebook_block_dim}"
        )
    if h.codebook_residual_k <= 0 or h.codebook_residual_k > 256:
        raise ValueError(f"CODEBOOK_RESIDUAL_K must be in [1, 256], got {h.codebook_residual_k}")
    if not 1 <= h.codebook_scale_bits <= 8:
        raise ValueError(f"CODEBOOK_SCALE_BITS must be in [1, 8], got {h.codebook_scale_bits}")
    if h.codebook_calibration_batches < 0:
        raise ValueError(
            f"CODEBOOK_CALIBRATION_BATCHES must be non-negative, got {h.codebook_calibration_batches}"
        )
    if h.codebook_reserve_seconds < 0:
        raise ValueError(
            f"CODEBOOK_RESERVE_SECONDS must be non-negative, got {h.codebook_reserve_seconds}"
        )
    if h.codebook_hessian_damp < 0:
        raise ValueError(
            f"CODEBOOK_HESSIAN_DAMP must be non-negative, got {h.codebook_hessian_damp}"
        )
    if h.codebook_lattice_scale <= 0:
        raise ValueError(f"CODEBOOK_LATTICE_SCALE must be positive, got {h.codebook_lattice_scale}")
    if h.codebook_ldlq_iters <= 0:
        raise ValueError(f"CODEBOOK_LDLQ_ITERS must be positive, got {h.codebook_ldlq_iters}")
    if h.codebook_ldlq_topk <= 0:
        raise ValueError(f"CODEBOOK_LDLQ_TOPK must be positive, got {h.codebook_ldlq_topk}")
    if not 0.0 <= h.codebook_outlier_frac <= 1.0:
        raise ValueError(f"CODEBOOK_OUTLIER_FRAC must be in [0, 1], got {h.codebook_outlier_frac}")
    if h.codebook_outlier_max_blocks < 0:
        raise ValueError(
            f"CODEBOOK_OUTLIER_MAX_BLOCKS must be non-negative, got {h.codebook_outlier_max_blocks}"
        )
    if h.codebook_debug_topk < 0:
        raise ValueError(f"CODEBOOK_DEBUG_TOPK must be non-negative, got {h.codebook_debug_topk}")


def _hybrid_base_raw_bits(num_blocks: int, h: Hyperparameters) -> int:
    return (
        _hybrid_fixed_raw_bits(num_blocks)
        + _hybrid_scale_raw_bits(num_blocks, h)
        + _hybrid_residual_raw_bits(num_blocks, h)
    )


def _hybrid_fixed_raw_bits(num_blocks: int) -> int:
    return num_blocks * 16


def _hybrid_scale_raw_bits(num_blocks: int, h: Hyperparameters) -> int:
    return num_blocks * h.codebook_scale_bits


def _hybrid_residual_raw_bits(num_blocks: int, h: Hyperparameters) -> int:
    if not h.codebook_use_residual:
        return 0
    residual_idx_bits = _bits_for_size(h.codebook_residual_k)
    residual_codebook_bits = h.codebook_residual_k * h.codebook_block_dim * 16
    return num_blocks * residual_idx_bits + residual_codebook_bits


class Codebook:
    def __init__(self, h: Hyperparameters, model: nn.Module):
        _validate_codebook_hparams(h)
        self.h = h
        self.states: dict[str, dict[str, object]] = {}
        self.target_modules: list[tuple[str, CastedLinear]] = []
        self.target_names: set[str] = set()
        for module_name, module in model.named_modules():
            if not isinstance(module, CastedLinear):
                continue
            weight_name = f"{module_name}.weight"
            if _should_codebook_quantize(weight_name, module.weight, self.h):
                self.target_modules.append((weight_name, module))
                self.target_names.add(weight_name)
        if self.target_modules:
            _get_e8p_codebook(self.target_modules[0][1].weight.device)

    def _active_residual_k(self) -> int:
        return self.h.codebook_residual_k if self.h.codebook_use_residual else 0

    @staticmethod
    def _tensor_group(name: str) -> str | None:
        if ".attn." in name:
            return "attn"
        if ".mlp." in name:
            return "mlp"
        return None

    @staticmethod
    def _metric_stage_summary(
        bucket: dict[str, float],
        key: str,
    ) -> tuple[float, float]:
        mse = bucket[key] / max(bucket["num_weights"], 1.0)
        return mse, math.sqrt(mse)

    def _aggregate_group_stats(self, items: list[tuple[str, dict[str, object]]]) -> dict[str, dict[str, float]]:
        grouped: dict[str, dict[str, float]] = {}
        for name, stats in items:
            group = self._tensor_group(name)
            if group is None:
                continue
            bucket = grouped.setdefault(
                group,
                {
                    "fixed_sse": 0.0,
                    "residual_sse": 0.0,
                    "final_sse": 0.0,
                    "num_weights": 0.0,
                    "num_blocks": 0.0,
                    "outlier_blocks": 0.0,
                    "used_residual_frac_sum": 0.0,
                    "num_tensors": 0.0,
                },
            )
            num_weights = float(stats["num_weights"])
            bucket["fixed_sse"] += float(stats["fixed_mse"]) * num_weights
            bucket["residual_sse"] += float(stats["residual_mse"]) * num_weights
            bucket["final_sse"] += float(stats["mse"]) * num_weights
            bucket["num_weights"] += num_weights
            bucket["num_blocks"] += float(stats["num_blocks"])
            bucket["outlier_blocks"] += float(stats["outlier_blocks"])
            bucket["used_residual_frac_sum"] += float(stats["used_residual"]) / max(float(self._active_residual_k()), 1.0)
            bucket["num_tensors"] += 1.0
        summary: dict[str, dict[str, float]] = {}
        for group, bucket in grouped.items():
            fixed_mse = bucket["fixed_sse"] / max(bucket["num_weights"], 1.0)
            residual_mse = bucket["residual_sse"] / max(bucket["num_weights"], 1.0)
            final_mse = bucket["final_sse"] / max(bucket["num_weights"], 1.0)
            summary[group] = {
                "fixed_mse": fixed_mse,
                "fixed_rmse": math.sqrt(fixed_mse),
                "residual_mse": residual_mse,
                "residual_rmse": math.sqrt(residual_mse),
                "mse": final_mse,
                "rmse": math.sqrt(final_mse),
                "outlier_frac": bucket["outlier_blocks"] / max(bucket["num_blocks"], 1.0),
                "used_residual_frac": bucket["used_residual_frac_sum"] / max(bucket["num_tensors"], 1.0),
                "num_tensors": bucket["num_tensors"],
            }
        return summary

    def _log_group_stats(self, prefix: str, items: list[tuple[str, dict[str, object]]]) -> None:
        grouped = self._aggregate_group_stats(items)
        parts = [prefix]
        for group in ("attn", "mlp"):
            stats = grouped.get(group)
            if stats is None:
                continue
            parts.append(
                f"{group}_fixed_mse:{stats['fixed_mse']:.6e} {group}_fixed_rmse:{stats['fixed_rmse']:.6e} "
                f"{group}_residual_mse:{stats['residual_mse']:.6e} {group}_residual_rmse:{stats['residual_rmse']:.6e} "
                f"{group}_final_mse:{stats['mse']:.6e} {group}_final_rmse:{stats['rmse']:.6e} "
                f"{group}_outlier_frac:{stats['outlier_frac']:.4%} "
                f"{group}_used_residual_frac:{stats['used_residual_frac']:.2%} "
                f"{group}_tensors:{int(stats['num_tensors'])}"
            )
        if len(parts) > 1:
            log(" ".join(parts))

    def _log_top_tensor_stats(
        self,
        prefix: str,
        items: list[tuple[str, dict[str, object]]],
        *,
        sort_key: str = "rel_mse",
    ) -> None:
        if self.h.codebook_debug_topk <= 0 or not items:
            return
        ranked = sorted(items, key=lambda item: float(item[1].get(sort_key, 0.0)), reverse=True)
        for name, stats in ranked[: self.h.codebook_debug_topk]:
            parts = [
                prefix,
                f"name:{name}",
                f"fixed_rel_mse:{float(stats.get('fixed_rel_mse', 0.0)):.6e}",
                f"residual_rel_mse:{float(stats.get('residual_rel_mse', 0.0)):.6e}",
                f"final_rel_mse:{float(stats.get('rel_mse', 0.0)):.6e}",
                f"outliers:{int(stats.get('outlier_blocks', 0))}/{int(stats.get('num_blocks', 1))}",
                f"used_residual:{int(stats.get('used_residual', 0))}/{self._active_residual_k()}",
                f"fixed_raw_bpw:{float(stats.get('fixed_raw_bpw', 0.0)):.4f}",
                f"scale_raw_bpw:{float(stats.get('scale_raw_bpw', 0.0)):.4f}",
                f"residual_raw_bpw:{float(stats.get('residual_raw_bpw', 0.0)):.4f}",
                f"outlier_raw_bpw:{float(stats.get('outlier_raw_bpw', 0.0)):.4f}",
                f"scale_min:{float(stats.get('scale_min', 0.0)):.3e}",
                f"scale_mean:{float(stats.get('scale_mean', 0.0)):.3e}",
                f"scale_max:{float(stats.get('scale_max', 0.0)):.3e}",
                f"metric_diag_mean:{float(stats.get('metric_diag_mean', 0.0)):.3e}",
            ]
            if "payload_bytes" in stats:
                parts.extend(
                    [
                        f"payload_bytes:{int(stats['payload_bytes'])}",
                        f"fixed_payload:{int(stats.get('fixed_payload_bytes', 0))}",
                        f"scale_payload:{int(stats.get('scale_payload_bytes', 0))}",
                        f"residual_payload:{int(stats.get('residual_payload_bytes', 0))}",
                        f"outlier_payload:{int(stats.get('outlier_payload_bytes', 0))}",
                    ]
                )
            log(" ".join(parts))

    def _new_state(self, name: str, weight: Tensor) -> dict[str, object]:
        weight_shape = tuple(weight.shape)
        block_dim = self.h.codebook_block_dim
        num_rows, num_cols = weight_shape
        num_positions = num_cols // block_dim
        num_blocks = num_rows * num_positions
        device = weight.device
        return {
            "name": name,
            "shape": weight_shape,
            "num_rows": num_rows,
            "num_positions": num_positions,
            "block_dim": block_dim,
            "sign_vec": _hadamard_sign_vector(name, block_dim, device=device, dtype=torch.float32),
            "fixed_idx": torch.zeros(num_blocks, dtype=torch.long, device=device),
            "residual_codebook": torch.zeros(
                self.h.codebook_residual_k,
                block_dim,
                dtype=torch.float32,
                device=device,
            ),
            "residual_idx": torch.zeros(num_blocks, dtype=torch.long, device=device),
            "log_scales": torch.zeros(num_blocks, 1, dtype=torch.float32, device=device),
            "outlier_positions": torch.empty((0,), dtype=torch.long, device=device),
            "outlier_q": torch.empty((0, block_dim), dtype=torch.int8, device=device),
            "outlier_scale": torch.empty((0,), dtype=INT8_PER_ROW_SCALE_DTYPE, device=device),
            "stats": {},
        }

    @staticmethod
    def _scales(state: dict[str, object], device: torch.device, dtype: torch.dtype) -> Tensor:
        return state["log_scales"].to(device=device, dtype=torch.float32).exp().clamp_min(1e-8).to(dtype=dtype)

    @staticmethod
    def _blockified_weight(weight: Tensor, block_dim: int) -> Tensor:
        blocks, _ = _blockify_weight(weight.float(), block_dim)
        return blocks

    @staticmethod
    def _block_grid(state: dict[str, object], blocks: Tensor) -> Tensor:
        return blocks.view(int(state["num_rows"]), int(state["num_positions"]), int(state["block_dim"]))

    @staticmethod
    def _flatten_blocks(blocks: Tensor) -> Tensor:
        return blocks.reshape(-1, blocks.shape[-1])

    @staticmethod
    def _scale_grid(state: dict[str, object], scales: Tensor) -> Tensor:
        return scales.view(int(state["num_rows"]), int(state["num_positions"]), 1)

    def _rotated_blocks(self, state: dict[str, object], weight: Tensor) -> Tensor:
        blocks = self._blockified_weight(weight, int(state["block_dim"]))
        sign_vec = state["sign_vec"].to(device=blocks.device, dtype=blocks.dtype)
        return hadamard_rotate_blocks(blocks, sign_vec, enabled=self.h.codebook_use_hadamard)

    def _decode_fixed_blocks(self, state: dict[str, object], *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return _decode_e8p_blocks(
            state["fixed_idx"].to(device=device),
            self.h.codebook_lattice_scale,
            device=device,
            dtype=dtype,
        )

    def _outlier_delta(self, state: dict[str, object], *, device: torch.device, dtype: torch.dtype) -> Tensor:
        num_blocks = int(state["num_rows"]) * int(state["num_positions"])
        block_dim = int(state["block_dim"])
        out = torch.zeros(num_blocks, block_dim, device=device, dtype=dtype)
        positions = state["outlier_positions"].to(device=device)
        if positions.numel() == 0:
            return out
        delta = dequantize_int8_rows(
            state["outlier_q"],
            state["outlier_scale"],
            device=device,
            dtype=dtype,
        )
        out.index_copy_(0, positions, delta)
        return out

    def _reconstructed_stage_blocks(
        self,
        state: dict[str, object],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor]:
        fixed_blocks = self._decode_fixed_blocks(state, device=device, dtype=dtype)
        if self.h.codebook_use_residual:
            residual_codebook = state["residual_codebook"].to(device=device, dtype=dtype)
            residual_idx = state["residual_idx"].to(device=device)
            residual_blocks = codebook_lookup(residual_idx, residual_codebook)
        else:
            residual_blocks = torch.zeros_like(fixed_blocks)
        scales = self._scales(state, device=device, dtype=dtype)
        fixed_rotated = fixed_blocks * scales
        residual_rotated = (fixed_blocks + residual_blocks) * scales
        final_rotated = residual_rotated + self._outlier_delta(state, device=device, dtype=dtype)
        return fixed_rotated, residual_rotated, final_rotated

    def _reconstructed_weight(self, state: dict[str, object], device: torch.device, dtype: torch.dtype) -> Tensor:
        _, _, rotated_blocks = self._reconstructed_stage_blocks(state, device=device, dtype=torch.float32)
        sign_vec = state["sign_vec"].to(device=device, dtype=rotated_blocks.dtype)
        blocks = hadamard_unrotate_blocks(rotated_blocks, sign_vec, enabled=self.h.codebook_use_hadamard)
        return _unblockify_weight(blocks, tuple(state["shape"])).to(dtype=dtype)

    @staticmethod
    def _weighted_nearest_codeword(x: Tensor, codebook: Tensor, metric: Tensor) -> Tensor:
        x_metric = x @ metric
        codebook_metric = codebook @ metric
        dists = (
            (x_metric * x).sum(dim=-1, keepdim=True)
            - 2.0 * x_metric @ codebook.t()
            + (codebook_metric * codebook).sum(dim=-1).unsqueeze(0)
        )
        return dists.argmin(dim=-1)

    def _metric_topk_codewords(
        self,
        x: Tensor,
        codebook: Tensor,
        metric: Tensor,
        topk: int,
    ) -> Tensor:
        x_metric = x @ metric
        codebook_metric = codebook @ metric
        dists = (
            (x_metric * x).sum(dim=-1, keepdim=True)
            - 2.0 * x_metric @ codebook.t()
            + (codebook_metric * codebook).sum(dim=-1).unsqueeze(0)
        )
        return torch.topk(dists, k=min(topk, codebook.shape[0]), largest=False, dim=-1).indices

    def _identity_block_metrics(self, state: dict[str, object], device: torch.device) -> Tensor:
        eye = torch.eye(int(state["block_dim"]), device=device, dtype=torch.float32)
        return eye.unsqueeze(0).repeat(int(state["num_positions"]), 1, 1)

    def _block_metrics_from_hessian(
        self,
        state: dict[str, object],
        hessian: Tensor | None,
        *,
        device: torch.device,
    ) -> Tensor:
        if hessian is None:
            return self._identity_block_metrics(state, device)
        expected_dim = int(state["shape"][1])
        if hessian.ndim != 2 or hessian.shape != (expected_dim, expected_dim):
            return self._identity_block_metrics(state, device)
        hessian_work = hessian.to(device=device, dtype=torch.float32).clone()
        diag = hessian_work.diag()
        dead = diag == 0
        if dead.any():
            hessian_work[dead, dead] = 1.0
        damp = float(self.h.codebook_hessian_damp * hessian_work.diag().mean().clamp_min(1e-8).item())
        hessian_work.diagonal().add_(damp)
        block_dim = int(state["block_dim"])
        num_positions = int(state["num_positions"])
        sign_vec = state["sign_vec"].to(device=device, dtype=torch.float32)
        unrotate = hadamard_unrotate_blocks(
            torch.eye(block_dim, device=device, dtype=torch.float32),
            sign_vec,
            enabled=self.h.codebook_use_hadamard,
        )
        reshaped = hessian_work.view(num_positions, block_dim, num_positions, block_dim)
        block_indices = torch.arange(num_positions, device=device)
        block_hessian = reshaped[block_indices, :, block_indices, :]
        metrics = torch.matmul(unrotate.unsqueeze(0), torch.matmul(block_hessian, unrotate.t().unsqueeze(0)))
        metrics = 0.5 * (metrics + metrics.transpose(-1, -2))
        metrics.diagonal(dim1=-2, dim2=-1).add_(1e-6)
        return metrics

    def _metric_point_weights(self, metrics: Tensor, num_rows: int) -> Tensor:
        pos_weights = metrics.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-8)
        return pos_weights.unsqueeze(0).expand(num_rows, -1).reshape(-1)

    @torch.no_grad()
    def _initialize_residual_codebook(
        self,
        state: dict[str, object],
        residual_blocks: Tensor,
        metrics: Tensor,
    ) -> None:
        points = self._flatten_blocks(residual_blocks)
        weights = self._metric_point_weights(metrics, int(state["num_rows"])).to(device=points.device, dtype=torch.float32)
        codebook = run_weighted_lloyd_kmeans(
            points,
            weights,
            self.h.codebook_residual_k,
            self.h.codebook_init_kmeans_iters,
            str(state["name"]),
            "residual",
        )
        state["residual_codebook"].copy_(
            codebook.to(device=state["residual_codebook"].device, dtype=state["residual_codebook"].dtype)
        )

    @torch.no_grad()
    def _assign_residual_with_scale(
        self,
        normalized_blocks: Tensor,
        rotated_blocks: Tensor,
        fixed_blocks: Tensor,
        residual_codebook: Tensor,
        metrics: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        topk = min(self.h.codebook_ldlq_topk, residual_codebook.shape[0])
        target_norm = normalized_blocks - fixed_blocks
        target_norm_metric = torch.einsum("npd,pde->npe", target_norm, metrics)
        codebook_metric = torch.einsum("kd,pde->pke", residual_codebook, metrics)
        dists = (
            (target_norm_metric * target_norm).sum(dim=-1, keepdim=True)
            - 2.0 * torch.matmul(target_norm_metric, residual_codebook.t())
            + (codebook_metric * residual_codebook.unsqueeze(0)).sum(dim=-1).unsqueeze(0)
        )
        topk_idx = torch.topk(dists, k=topk, largest=False, dim=-1).indices
        block_dim = normalized_blocks.shape[-1]
        candidates = residual_codebook[topk_idx]
        base = fixed_blocks.unsqueeze(2) + candidates
        target_metric = torch.einsum("npd,pde->npe", rotated_blocks, metrics)
        base_metric = torch.einsum("nptd,pde->npte", base, metrics)
        numer = (target_metric.unsqueeze(2) * base).sum(dim=-1)
        denom = (base_metric * base).sum(dim=-1).clamp_min(1e-8)
        candidate_scales = (numer / denom).clamp_min(1e-8)
        target_quad = (target_metric * rotated_blocks).sum(dim=-1, keepdim=True)
        errors = target_quad - 2.0 * candidate_scales * numer + candidate_scales.square() * denom
        best = errors.argmin(dim=-1)
        gather_idx = best.unsqueeze(-1)
        residual_idx = topk_idx.gather(2, gather_idx).squeeze(-1)
        residual_vals = candidates.gather(
            2,
            gather_idx.unsqueeze(-1).expand(-1, -1, 1, block_dim),
        ).squeeze(2)
        best_numer = numer.gather(2, gather_idx).squeeze(-1)
        best_denom = denom.gather(2, gather_idx).squeeze(-1)
        scales = (best_numer / best_denom).clamp_min(1e-8)
        return residual_idx, residual_vals, scales.unsqueeze(-1)

    @torch.no_grad()
    def _weighted_codebook_update(
        self,
        targets: Tensor,
        assignments: Tensor,
        metrics: Tensor,
        prev_codebook: Tensor,
    ) -> Tensor:
        k, block_dim = prev_codebook.shape
        eye = torch.eye(block_dim, device=targets.device, dtype=torch.float32)
        one_hot = F.one_hot(assignments.long(), num_classes=k).to(dtype=torch.float32)
        counts_by_pos = one_hot.sum(dim=0)
        counts = counts_by_pos.sum(dim=0)
        system = torch.einsum("pk,pde->kde", counts_by_pos, metrics)
        system = system + eye.unsqueeze(0) * 1e-6
        metric_targets = torch.einsum("npd,pde->npe", targets, metrics)
        rhs = torch.einsum("npk,npe->ke", one_hot, metric_targets)
        updated = prev_codebook.to(device=targets.device, dtype=torch.float32).clone()
        mask = counts > 0
        if mask.any():
            updated[mask] = torch.linalg.solve(system[mask], rhs[mask])
        return updated

    @torch.no_grad()
    def _local_assign_blocks(
        self,
        state: dict[str, object],
        rotated_blocks: Tensor,
        metrics: Tensor,
        residual_codebook: Tensor,
        initial_scales: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        scales = initial_scales
        residual_vals = torch.zeros_like(rotated_blocks)
        fixed_idx = torch.zeros(
            int(state["num_rows"]),
            int(state["num_positions"]),
            device=rotated_blocks.device,
            dtype=torch.long,
        )
        fixed_vals = torch.zeros_like(rotated_blocks)
        residual_idx = torch.zeros_like(fixed_idx)
        for _ in range(self.h.codebook_ldlq_iters):
            normalized_adjusted = rotated_blocks / scales - residual_vals
            fixed_flat, fixed_idx_flat = _quantize_e8p_blocks(
                self._flatten_blocks(normalized_adjusted),
                self.h.codebook_lattice_scale,
            )
            fixed_vals = self._block_grid(state, fixed_flat)
            fixed_idx = fixed_idx_flat.view(int(state["num_rows"]), int(state["num_positions"]))
            normalized_blocks = rotated_blocks / scales
            residual_idx, residual_vals, scales = self._assign_residual_with_scale(
                normalized_blocks,
                rotated_blocks,
                fixed_vals,
                residual_codebook,
                metrics,
            )
        normalized_adjusted = rotated_blocks / scales - residual_vals
        fixed_flat, fixed_idx_flat = _quantize_e8p_blocks(
            self._flatten_blocks(normalized_adjusted),
            self.h.codebook_lattice_scale,
        )
        fixed_vals = self._block_grid(state, fixed_flat)
        fixed_idx = fixed_idx_flat.view(int(state["num_rows"]), int(state["num_positions"]))
        normalized_blocks = rotated_blocks / scales
        residual_idx, residual_vals, scales = self._assign_residual_with_scale(
            normalized_blocks,
            rotated_blocks,
            fixed_vals,
            residual_codebook,
            metrics,
        )
        return fixed_idx, fixed_vals, residual_idx, residual_vals, scales

    @staticmethod
    def _metric_scores(diff_blocks: Tensor, metrics: Tensor) -> Tensor:
        return torch.einsum("npd,pde,npe->np", diff_blocks, metrics, diff_blocks)

    @torch.no_grad()
    def _select_outliers(
        self,
        state: dict[str, object],
        rotated_blocks: Tensor,
        residual_rotated: Tensor,
        metrics: Tensor,
    ) -> None:
        if not self.h.codebook_use_outliers:
            state["outlier_positions"] = torch.empty((0,), dtype=torch.long, device=rotated_blocks.device)
            state["outlier_q"] = torch.empty((0, int(state["block_dim"])), dtype=torch.int8, device=rotated_blocks.device)
            state["outlier_scale"] = torch.empty((0,), dtype=INT8_PER_ROW_SCALE_DTYPE, device=rotated_blocks.device)
            return
        num_blocks = int(state["num_rows"]) * int(state["num_positions"])
        outlier_count = min(
            self.h.codebook_outlier_max_blocks,
            int(round(self.h.codebook_outlier_frac * num_blocks)),
        )
        if self.h.codebook_outlier_frac > 0.0 and outlier_count == 0 and self.h.codebook_outlier_max_blocks > 0:
            outlier_count = 1
        if outlier_count <= 0:
            state["outlier_positions"] = torch.empty((0,), dtype=torch.long, device=rotated_blocks.device)
            state["outlier_q"] = torch.empty((0, int(state["block_dim"])), dtype=torch.int8, device=rotated_blocks.device)
            state["outlier_scale"] = torch.empty((0,), dtype=INT8_PER_ROW_SCALE_DTYPE, device=rotated_blocks.device)
            return
        diff_blocks = rotated_blocks - residual_rotated
        scores = self._metric_scores(diff_blocks, metrics).reshape(-1)
        topk = min(outlier_count, scores.numel())
        positions = torch.topk(scores, k=topk, largest=True).indices
        delta = self._flatten_blocks(diff_blocks).index_select(0, positions)
        q, s = quantize_float_tensor(delta)
        state["outlier_positions"] = positions.to(device=rotated_blocks.device)
        state["outlier_q"] = q.to(device=rotated_blocks.device)
        state["outlier_scale"] = s.view(-1).to(device=rotated_blocks.device, dtype=INT8_PER_ROW_SCALE_DTYPE)

    def _outlier_raw_bits(self, state: dict[str, object]) -> int:
        count = int(state["outlier_positions"].numel())
        if count == 0:
            return 0
        num_blocks = int(state["num_rows"]) * int(state["num_positions"])
        position_bits = _bits_for_size(num_blocks)
        return count * (position_bits + int(state["block_dim"]) * 8 + 16)

    def _state_raw_bits(self, state: dict[str, object]) -> int:
        num_blocks = int(state["num_rows"]) * int(state["num_positions"])
        return _hybrid_base_raw_bits(num_blocks, self.h) + self._outlier_raw_bits(state)

    @torch.no_grad()
    def _snapshot_stats(self, state: dict[str, object], weight: Tensor) -> dict[str, object]:
        rotated_flat = self._rotated_blocks(state, weight)
        rotated_blocks = self._block_grid(state, rotated_flat)
        fixed_rotated, residual_rotated, final_rotated = self._reconstructed_stage_blocks(
            state,
            device=rotated_blocks.device,
            dtype=torch.float32,
        )
        fixed_diff = rotated_blocks - self._block_grid(state, fixed_rotated)
        residual_diff = rotated_blocks - self._block_grid(state, residual_rotated)
        final_diff = rotated_blocks - self._block_grid(state, final_rotated)
        fixed_mse = fixed_diff.square().mean()
        residual_mse = residual_diff.square().mean()
        final_mse = final_diff.square().mean()
        energy = rotated_blocks.square().mean().clamp_min(1e-12)
        fixed_rel_mse = fixed_mse / energy
        residual_rel_mse = residual_mse / energy
        final_rel_mse = final_mse / energy
        scales = self._scales(state, device=rotated_blocks.device, dtype=torch.float32)
        if self.h.codebook_use_residual:
            counts_residual = torch.bincount(
                state["residual_idx"].to(device=rotated_blocks.device),
                minlength=self.h.codebook_residual_k,
            ).to(torch.float32)
            used_residual = int((counts_residual > 0).sum().item())
        else:
            counts_residual = torch.zeros(self.h.codebook_residual_k, device=rotated_blocks.device, dtype=torch.float32)
            used_residual = 0
        num_blocks = int(state["fixed_idx"].numel())
        fixed_raw_bits = _hybrid_fixed_raw_bits(num_blocks)
        scale_raw_bits = _hybrid_scale_raw_bits(num_blocks, self.h)
        residual_raw_bits = _hybrid_residual_raw_bits(num_blocks, self.h)
        outlier_raw_bits = self._outlier_raw_bits(state)
        return {
            "num_weights": int(weight.numel()),
            "num_blocks": num_blocks,
            "raw_bits": int(fixed_raw_bits + scale_raw_bits + residual_raw_bits + outlier_raw_bits),
            "fixed_mse": float(fixed_mse.item()),
            "fixed_rmse": float(fixed_mse.sqrt().item()),
            "fixed_rel_mse": float(fixed_rel_mse.item()),
            "residual_mse": float(residual_mse.item()),
            "residual_rmse": float(residual_mse.sqrt().item()),
            "residual_rel_mse": float(residual_rel_mse.item()),
            "mse": float(final_mse.item()),
            "rmse": float(final_mse.sqrt().item()),
            "rel_mse": float(final_rel_mse.item()),
            "mae": float(final_diff.abs().mean().item()),
            "max_abs": float(final_diff.abs().max().item()),
            "scale_min": float(scales.min().item()),
            "scale_max": float(scales.max().item()),
            "scale_mean": float(scales.mean().item()),
            "used_residual": used_residual,
            "outlier_blocks": int(state["outlier_positions"].numel()),
            "fixed_raw_bits": fixed_raw_bits,
            "fixed_raw_bpw": fixed_raw_bits / max(float(weight.numel()), 1.0),
            "scale_raw_bits": scale_raw_bits,
            "scale_raw_bpw": scale_raw_bits / max(float(weight.numel()), 1.0),
            "residual_raw_bits": residual_raw_bits,
            "residual_raw_bpw": residual_raw_bits / max(float(weight.numel()), 1.0),
            "outlier_raw_bits": outlier_raw_bits,
            "outlier_raw_bpw": outlier_raw_bits / max(float(weight.numel()), 1.0),
            "hist_residual": counts_residual.to(torch.int64).cpu().tolist(),
        }

    @torch.no_grad()
    def _fit_tensor(self, name: str, module: CastedLinear, hessian: Tensor | None) -> tuple[str, dict[str, object]]:
        state = self.states.get(name)
        if state is None:
            state = self._new_state(name, module.weight.detach())
            self.states[name] = state
        rotated_blocks = self._block_grid(state, self._rotated_blocks(state, module.weight.detach()).float())
        metrics = self._block_metrics_from_hessian(state, hessian, device=rotated_blocks.device)
        metric_diag = metrics.diagonal(dim1=-2, dim2=-1)
        initial_scales = rotated_blocks.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        normalized_blocks = rotated_blocks / initial_scales
        fixed_flat, fixed_idx_flat = _quantize_e8p_blocks(
            self._flatten_blocks(normalized_blocks),
            self.h.codebook_lattice_scale,
        )
        fixed_blocks = self._block_grid(state, fixed_flat)
        state["fixed_idx"].copy_(fixed_idx_flat.to(device=state["fixed_idx"].device))
        state["log_scales"].copy_(
            initial_scales.reshape(-1, 1).log().to(device=state["log_scales"].device, dtype=state["log_scales"].dtype)
        )
        if self.h.codebook_use_residual:
            residual_seed = normalized_blocks - fixed_blocks
            self._initialize_residual_codebook(state, residual_seed, metrics)

        for _ in range(max(self.h.codebook_refinement_iters, 1) if self.h.codebook_use_residual else 0):
            residual_codebook = state["residual_codebook"].to(device=rotated_blocks.device, dtype=torch.float32)
            fixed_idx, fixed_blocks, residual_idx, residual_vals, scales = self._local_assign_blocks(
                state,
                rotated_blocks,
                metrics,
                residual_codebook,
                initial_scales,
            )
            residual_targets = rotated_blocks / scales - fixed_blocks
            updated_codebook = self._weighted_codebook_update(
                residual_targets,
                residual_idx,
                metrics,
                residual_codebook,
            )
            state["residual_codebook"].copy_(
                updated_codebook.to(device=state["residual_codebook"].device, dtype=state["residual_codebook"].dtype)
            )
            initial_scales = scales

        if not self.h.codebook_use_residual:
            state["residual_codebook"].zero_()
        residual_codebook = state["residual_codebook"].to(device=rotated_blocks.device, dtype=torch.float32)
        fixed_idx, fixed_blocks, residual_idx, residual_vals, scales = self._local_assign_blocks(
            state,
            rotated_blocks,
            metrics,
            residual_codebook,
            initial_scales,
        )
        state["fixed_idx"].copy_(fixed_idx.reshape(-1).to(device=state["fixed_idx"].device))
        if self.h.codebook_use_residual:
            state["residual_idx"].copy_(residual_idx.reshape(-1).to(device=state["residual_idx"].device))
        else:
            state["residual_idx"].zero_()
            residual_vals = torch.zeros_like(fixed_blocks)
        state["log_scales"].copy_(
            scales.reshape(-1, 1).log().to(device=state["log_scales"].device, dtype=state["log_scales"].dtype)
        )
        residual_rotated = (fixed_blocks + residual_vals) * scales
        self._select_outliers(state, rotated_blocks, residual_rotated, metrics)

        stats = self._snapshot_stats(state, module.weight.detach())
        stats.update(
            {
                "metric_diag_min": float(metric_diag.min().item()),
                "metric_diag_mean": float(metric_diag.mean().item()),
                "metric_diag_max": float(metric_diag.max().item()),
                "hessian_missing": float(hessian is None),
            }
        )
        state["stats"] = stats
        return name, stats

    @torch.no_grad()
    def fit(self, model: nn.Module, hessians: dict[str, Tensor]) -> None:
        if not self.target_modules:
            if self.h.is_main_process:
                log("codebook:no eligible tensors found")
            return
        target_weights = sum(int(module.weight.numel()) for _, module in self.target_modules)
        total_fp_weights = sum(
            int(t.numel())
            for _, t in model.state_dict().items()
            if t.is_floating_point()
        )
        coverage = target_weights / max(total_fp_weights, 1)
        if self.h.is_main_process:
            log(
                f"codebook:fit target_tensors:{len(self.target_modules)} target_weights:{target_weights} "
                f"coverage:{coverage:.4%} calibration_batches:{self.h.codebook_calibration_batches} "
                f"hadamard:{self.h.codebook_use_hadamard} residual:{self.h.codebook_use_residual} "
                f"outliers:{self.h.codebook_use_outliers} hadamard_backend:{HADAMARD_BACKEND} lattice:e8p12"
            )
        missing = [name for name, _ in self.target_modules if name not in hessians]
        if missing and self.h.is_main_process:
            log(f"codebook:missing_hessians count:{len(missing)} fallback:identity_block_metric")
        fitted_items = [self._fit_tensor(name, module, hessians.get(name)) for name, module in self.target_modules]
        total_target_weights = sum(int(stats["num_weights"]) for _, stats in fitted_items)
        weighted_fixed_rel_mse = sum(float(stats["fixed_rel_mse"]) * int(stats["num_weights"]) for _, stats in fitted_items)
        weighted_residual_rel_mse = sum(float(stats["residual_rel_mse"]) * int(stats["num_weights"]) for _, stats in fitted_items)
        weighted_rel_mse = sum(float(stats["rel_mse"]) * int(stats["num_weights"]) for _, stats in fitted_items)
        total_raw_bits = sum(int(stats["raw_bits"]) for _, stats in fitted_items)
        total_fixed_raw_bits = sum(int(stats["fixed_raw_bits"]) for _, stats in fitted_items)
        total_scale_raw_bits = sum(int(stats["scale_raw_bits"]) for _, stats in fitted_items)
        total_residual_raw_bits = sum(int(stats["residual_raw_bits"]) for _, stats in fitted_items)
        total_outlier_raw_bits = sum(int(stats["outlier_raw_bits"]) for _, stats in fitted_items)
        total_outlier_blocks = sum(int(stats["outlier_blocks"]) for _, stats in fitted_items)
        total_blocks = sum(int(stats["num_blocks"]) for _, stats in fitted_items)
        if self.h.is_main_process:
            log(
                f"codebook:fit_summary fixed_rel_mse:{weighted_fixed_rel_mse / max(total_target_weights, 1):.6e} "
                f"residual_rel_mse:{weighted_residual_rel_mse / max(total_target_weights, 1):.6e} "
                f"final_rel_mse:{weighted_rel_mse / max(total_target_weights, 1):.6e} "
                f"target_raw_bpw:{total_raw_bits / max(total_target_weights, 1):.4f} "
                f"fixed_raw_bpw:{total_fixed_raw_bits / max(total_target_weights, 1):.4f} "
                f"scale_raw_bpw:{total_scale_raw_bits / max(total_target_weights, 1):.4f} "
                f"residual_raw_bpw:{total_residual_raw_bits / max(total_target_weights, 1):.4f} "
                f"outlier_raw_bpw:{total_outlier_raw_bits / max(total_target_weights, 1):.4f} "
                f"outlier_frac:{total_outlier_blocks / max(total_blocks, 1):.4%}"
            )
            self._log_group_stats("codebook:fit_groups", fitted_items)
            self._log_top_tensor_stats("codebook:fit_top", fitted_items, sort_key="rel_mse")

    def build_export(
        self,
        state_dict: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, object], dict[str, object]]:
        result: dict[str, Tensor] = {}
        meta: dict[str, object] = {}
        export_stats: dict[str, object] = {"tensors": {}}
        total_raw_bits = 0
        total_target_weights = 0
        total_payload_bytes = 0
        fixed_payload_bytes = 0
        scale_payload_bytes = 0
        residual_payload_bytes = 0
        outlier_payload_bytes = 0
        int8_fallback_payload_bytes = 0
        int8_fallback_weights = 0
        passthrough_payload_bytes = 0
        passthrough_weights = 0
        total_fp_weights = sum(
            int(t.numel())
            for _, t in state_dict.items()
            if t.is_floating_point()
        )
        for name, tensor in state_dict.items():
            t = tensor.detach()
            if name in self.states:
                state = self.states[name]
                fixed_idx_cpu = state["fixed_idx"].long().cpu()
                residual_idx_cpu = state["residual_idx"].long().cpu()
                residual_codebook_cpu = state["residual_codebook"].to(torch.float16).cpu().contiguous()
                scales_cpu = self._scales(state, device=torch.device("cpu"), dtype=torch.float32)
                scale_codes, scale_meta = quantize_log_scales(scales_cpu, self.h.codebook_scale_bits)
                packed_fixed_idx = _pack_codes(fixed_idx_cpu, 16).cpu().contiguous()
                packed_scales = _pack_codes(scale_codes, self.h.codebook_scale_bits).cpu().contiguous()
                result[name + ".fi"] = packed_fixed_idx
                result[name + ".s"] = packed_scales
                if self.h.codebook_use_residual:
                    packed_residual_idx = _pack_codes(
                        residual_idx_cpu,
                        _bits_for_size(self.h.codebook_residual_k),
                    ).cpu().contiguous()
                    result[name + ".rc"] = residual_codebook_cpu
                    result[name + ".ri"] = packed_residual_idx

                outlier_count = int(state["outlier_positions"].numel())
                outlier_position_bits = _bits_for_size(int(state["fixed_idx"].numel()))
                if outlier_count > 0:
                    result[name + ".op"] = _pack_codes(state["outlier_positions"].long().cpu(), outlier_position_bits).cpu().contiguous()
                    result[name + ".oq"] = state["outlier_q"].cpu().contiguous()
                    result[name + ".os"] = state["outlier_scale"].cpu().contiguous()

                meta[name] = {
                    "type": "codebook_hybrid",
                    "shape": list(state["shape"]),
                    "block_dim": int(state["block_dim"]),
                    "fixed_bits": 16,
                    "fixed_codebook": "E8P12",
                    "lattice_scale": float(self.h.codebook_lattice_scale),
                    "hadamard": bool(self.h.codebook_use_hadamard),
                    "residual_enabled": bool(self.h.codebook_use_residual),
                    "residual_k": self.h.codebook_residual_k if self.h.codebook_use_residual else 0,
                    "residual_bits": _bits_for_size(self.h.codebook_residual_k) if self.h.codebook_use_residual else 0,
                    "scale_bits": self.h.codebook_scale_bits,
                    "scale_log_min": scale_meta["log_min"],
                    "scale_log_max": scale_meta["log_max"],
                    "outlier_enabled": bool(self.h.codebook_use_outliers),
                    "outlier_count": outlier_count,
                    "outlier_position_bits": outlier_position_bits,
                }

                quantized_scales = dequantize_log_scales(
                    scale_codes,
                    self.h.codebook_scale_bits,
                    scale_meta["log_min"],
                    scale_meta["log_max"],
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                fixed_blocks_cpu = _decode_e8p_blocks(
                    fixed_idx_cpu,
                    self.h.codebook_lattice_scale,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                if self.h.codebook_use_residual:
                    residual_blocks_cpu = codebook_lookup(residual_idx_cpu, residual_codebook_cpu.float())
                else:
                    residual_blocks_cpu = torch.zeros_like(fixed_blocks_cpu)
                fixed_rotated = fixed_blocks_cpu * quantized_scales
                residual_rotated = (fixed_blocks_cpu + residual_blocks_cpu) * quantized_scales
                final_rotated = residual_rotated.clone()
                if outlier_count > 0:
                    outlier_delta = dequantize_int8_rows(
                        result[name + ".oq"],
                        result[name + ".os"],
                        device=torch.device("cpu"),
                        dtype=torch.float32,
                    )
                    final_rotated.index_add_(0, state["outlier_positions"].long().cpu(), outlier_delta)
                target_rotated = self._rotated_blocks(state, t).float().cpu()
                fixed_diff = target_rotated - fixed_rotated
                residual_diff = target_rotated - residual_rotated
                final_diff = target_rotated - final_rotated
                fixed_mse = float(fixed_diff.square().mean().item())
                residual_mse = float(residual_diff.square().mean().item())
                final_mse = float(final_diff.square().mean().item())
                final_rel_mse = final_mse / max(float(target_rotated.square().mean().item()), 1e-12)

                payload_keys = [name + ".fi", name + ".s"]
                if self.h.codebook_use_residual:
                    payload_keys.extend([name + ".rc", name + ".ri"])
                if outlier_count > 0:
                    payload_keys.extend([name + ".op", name + ".oq", name + ".os"])
                payload_bytes = sum(result[key].numel() * result[key].element_size() for key in payload_keys)
                fixed_bytes = result[name + ".fi"].numel() * result[name + ".fi"].element_size()
                scale_bytes = result[name + ".s"].numel() * result[name + ".s"].element_size()
                residual_bytes = 0
                if self.h.codebook_use_residual:
                    residual_bytes = (
                        result[name + ".rc"].numel() * result[name + ".rc"].element_size()
                        + result[name + ".ri"].numel() * result[name + ".ri"].element_size()
                    )
                outlier_bytes = 0
                if outlier_count > 0:
                    outlier_bytes = sum(result[key].numel() * result[key].element_size() for key in (name + ".op", name + ".oq", name + ".os"))
                total_payload_bytes += payload_bytes
                fixed_payload_bytes += fixed_bytes
                scale_payload_bytes += scale_bytes
                residual_payload_bytes += residual_bytes
                outlier_payload_bytes += outlier_bytes
                state_export_stats = dict(state["stats"])
                state_export_stats.update(
                    {
                        "fixed_mse": fixed_mse,
                        "fixed_rmse": math.sqrt(fixed_mse),
                        "residual_mse": residual_mse,
                        "residual_rmse": math.sqrt(residual_mse),
                        "mse": final_mse,
                        "rmse": math.sqrt(final_mse),
                        "rel_mse": final_rel_mse,
                        "payload_bytes": payload_bytes,
                        "fixed_payload_bytes": fixed_bytes,
                        "scale_payload_bytes": scale_bytes,
                        "residual_payload_bytes": residual_bytes,
                        "outlier_payload_bytes": outlier_bytes,
                        "raw_bpw": float(self._state_raw_bits(state)) / max(float(state["fixed_idx"].numel() * int(state["block_dim"])), 1.0),
                        "scale_log_min": scale_meta["log_min"],
                        "scale_log_max": scale_meta["log_max"],
                    }
                )
                total_raw_bits += int(state_export_stats["raw_bits"])
                total_target_weights += int(state_export_stats["num_weights"])
                export_stats["tensors"][name] = state_export_stats
                continue
            if not t.is_floating_point() or t.numel() <= 65536:
                result[name] = t.to(torch.float16).cpu().contiguous() if t.is_floating_point() else t.cpu().contiguous()
                meta[name] = "passthrough"
                total_payload_bytes += result[name].numel() * result[name].element_size()
                passthrough_payload_bytes += result[name].numel() * result[name].element_size()
                if t.is_floating_point():
                    passthrough_weights += int(t.numel())
                continue
            if any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
                result[name] = t.float().cpu().contiguous()
                meta[name] = "passthrough_ctrl"
                total_payload_bytes += result[name].numel() * result[name].element_size()
                passthrough_payload_bytes += result[name].numel() * result[name].element_size()
                passthrough_weights += int(t.numel())
                continue
            if t.is_floating_point():
                q, s = quantize_float_tensor(t)
                result[name + ".q"] = q.cpu().contiguous()
                result[name + ".scale"] = s.cpu().contiguous()
                meta[name] = {"type": "int8"}
                payload_bytes = (
                    result[name + ".q"].numel() * result[name + ".q"].element_size()
                    + result[name + ".scale"].numel() * result[name + ".scale"].element_size()
                )
                total_payload_bytes += payload_bytes
                int8_fallback_payload_bytes += payload_bytes
                int8_fallback_weights += int(t.numel())
                continue
            result[name] = t.cpu().contiguous()
            meta[name] = "passthrough"
            total_payload_bytes += result[name].numel() * result[name].element_size()
        export_stats["summary"] = {
            "target_tensors": len(self.states),
            "target_weights": total_target_weights,
            "target_raw_bits": total_raw_bits,
            "target_raw_bpw": total_raw_bits / max(total_target_weights, 1),
            "model_fp_weights": total_fp_weights,
            "coverage": total_target_weights / max(total_fp_weights, 1),
            "fixed_payload_bytes": fixed_payload_bytes,
            "scale_payload_bytes": scale_payload_bytes,
            "residual_payload_bytes": residual_payload_bytes,
            "outlier_payload_bytes": outlier_payload_bytes,
            "int8_fallback_payload_bytes": int8_fallback_payload_bytes,
            "int8_fallback_weights": int8_fallback_weights,
            "passthrough_weights": passthrough_weights,
            "passthrough_payload_bytes": passthrough_payload_bytes,
            "payload_bytes_before_torchsave": total_payload_bytes,
            "effective_payload_bpw_all_weights": (8.0 * total_payload_bytes) / max(total_fp_weights, 1),
        }
        export_stats["groups"] = self._aggregate_group_stats(
            [(name, stats) for name, stats in export_stats["tensors"].items()]
        )
        return result, meta, export_stats


def quantize_state_dict_codebook(
    state_dict: dict[str, Tensor],
    h: Hyperparameters,
    quantizer: Codebook,
) -> tuple[dict[str, Tensor], dict[str, object], dict[str, object]]:
    _validate_codebook_hparams(h)
    return quantizer.build_export(state_dict)


def dequantize_state_dict_codebook(
    result: dict[str, Tensor],
    meta: dict[str, object],
    template_sd: dict[str, Tensor],
) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        if info in ("passthrough", "passthrough_ctrl"):
            t = result[name]
            if t.dtype == torch.float16 and orig.dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig.dtype)
            out[name] = t
            continue
        if not isinstance(info, dict):
            raise ValueError(f"Unsupported compression metadata for {name}: {info!r}")
        if info.get("type") == "int8":
            q = result[name + ".q"]
            s = result[name + ".scale"]
            if s.ndim > 0:
                out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig.dtype)
            else:
                out[name] = (q.float() * float(s.item())).to(orig.dtype)
            continue
        if info.get("type") != "codebook_hybrid":
            raise ValueError(f"Unsupported compression metadata for {name}: {info!r}")
        shape = tuple(int(x) for x in info["shape"])
        block_dim = int(info["block_dim"])
        num_blocks = math.prod(shape) // block_dim
        fixed_idx = _unpack_codes(result[name + ".fi"], int(info["fixed_bits"]), num_blocks)
        scale_codes = _unpack_codes(result[name + ".s"], int(info["scale_bits"]), num_blocks)
        scales = dequantize_log_scales(
            scale_codes,
            int(info["scale_bits"]),
            float(info["scale_log_min"]),
            float(info["scale_log_max"]),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        fixed_blocks = _decode_e8p_blocks(
            fixed_idx,
            float(info["lattice_scale"]),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        if bool(info.get("residual_enabled", True)):
            residual_codebook = result[name + ".rc"].to(dtype=torch.float32)
            residual_idx = _unpack_codes(result[name + ".ri"], int(info["residual_bits"]), num_blocks)
            residual_blocks = codebook_lookup(residual_idx, residual_codebook)
        else:
            residual_blocks = torch.zeros_like(fixed_blocks)
        rotated_blocks = (fixed_blocks + residual_blocks) * scales
        outlier_count = int(info.get("outlier_count", 0))
        if outlier_count > 0:
            positions = _unpack_codes(result[name + ".op"], int(info["outlier_position_bits"]), outlier_count)
            delta = dequantize_int8_rows(
                result[name + ".oq"],
                result[name + ".os"],
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            rotated_blocks.index_add_(0, positions.long(), delta)
        sign_vec = _hadamard_sign_vector(name, block_dim, device=torch.device("cpu"), dtype=rotated_blocks.dtype)
        blocks = hadamard_unrotate_blocks(
            rotated_blocks,
            sign_vec,
            enabled=bool(info.get("hadamard", True)),
        )
        out[name] = _unblockify_weight(blocks, shape).to(orig.dtype)
    return out


_BSHF_MAGIC = b"BSHF"


def _byte_shuffle(data: bytes, stride: int = 2) -> bytes:
    """Transpose byte stream by stride position for better compression."""
    if stride <= 1 or len(data) < stride:
        return data
    src = np.frombuffer(data, dtype=np.uint8)
    n = len(src)
    out = np.empty(n, dtype=np.uint8)
    dest_off = 0
    for pos in range(stride):
        chunk = src[pos::stride]
        out[dest_off:dest_off + len(chunk)] = chunk
        dest_off += len(chunk)
    return _BSHF_MAGIC + bytes([stride]) + out.tobytes()


def _byte_unshuffle(data: bytes) -> bytes:
    """Inverse of _byte_shuffle. Auto-detects BSHF magic header."""
    if len(data) < 5 or data[:4] != _BSHF_MAGIC:
        return data
    stride = data[4]
    if stride < 2:
        return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    n = len(payload)
    out = np.empty(n, dtype=np.uint8)
    src_off = 0
    for pos in range(stride):
        chunk_len = n // stride + (1 if pos < n % stride else 0)
        out[pos::stride][:chunk_len] = payload[src_off:src_off + chunk_len]
        src_off += chunk_len
    return out.tobytes()


def _compress(data: bytes, compressor: str, byte_shuffle: bool = True) -> bytes:
    if byte_shuffle:
        data = _byte_shuffle(data)
    if compressor == "lzma":
        return lzma.compress(data, preset=6)
    elif compressor == "brotli":
        import brotli
        return brotli.compress(data, quality=11)
    raise ValueError(f"Unknown compressor: {compressor!r}")


def _decompress(data: bytes, compressor: str, byte_shuffle: bool = True) -> bytes:
    if compressor == "lzma":
        raw = lzma.decompress(data)
    elif compressor == "brotli":
        import brotli
        raw = brotli.decompress(data)
    if byte_shuffle:
        raw = _byte_unshuffle(raw)
    return raw
    raise ValueError(f"Unknown compressor: {compressor!r}")


def serialize(
    h: Hyperparameters,
    base_model: torch.nn.Module,
    code: str,
) -> int:
    quantizer = Codebook(h, base_model)
    device = next(base_model.parameters()).device
    model_bytes = None
    code_bytes = len(code.encode("utf-8"))
    bytes_total = code_bytes
    if h.is_main_process:
        torch.save(base_model.state_dict(), h.model_path)
        model_bytes = os.path.getsize(h.model_path)
        log(f"Serialized model: {model_bytes} bytes")
        log(f"Code size: {code_bytes} bytes")
    hessians: dict[str, Tensor] = {}
    if quantizer.target_names:
        if h.is_main_process:
            log("codebook:collecting calibration hessians...")
        t0 = time.perf_counter()
        calib_loader = DistributedTokenLoader(h.train_files, h.rank, h.world_size, device)
        hessians = collect_hessians(
            base_model,
            calib_loader,
            h,
            device,
            quantizer.target_names,
            h.codebook_calibration_batches,
        )
        if h.is_main_process:
            log(f"codebook:collected {len(hessians)} hessians in {time.perf_counter() - t0:.1f}s")
    if h.is_main_process:
        t0 = time.perf_counter()
        quantizer.fit(base_model, hessians)
        log(f"codebook:fit_time:{time.perf_counter() - t0:.1f}s")
        quant_result, quant_meta, quant_stats = quantize_state_dict_codebook(base_model.state_dict(), h, quantizer)
        quant_buf = io.BytesIO()
        torch.save({"w": quant_result, "m": quant_meta, "s": quant_stats}, quant_buf)
        quant_raw = quant_buf.getvalue()
        quant_blob = _compress(quant_raw, h.compressor)
        quant_file_bytes = len(quant_blob)
        bytes_total = quant_file_bytes + code_bytes
        with open(h.quantized_model_path, "wb") as f:
            f.write(quant_blob)
        summary = quant_stats.get("summary", {})
        log(
            f"Serialized model codebook+{h.compressor}: {quant_file_bytes} bytes "
            f"(payload_before_torchsave:{summary.get('payload_bytes_before_torchsave', 0)} bytes)"
        )
        log(
            f"Codebook target coverage:{summary.get('coverage', 0.0):.4%} "
            f"target_raw_bpw:{summary.get('target_raw_bpw', 0.0):.4f}"
        )
        log(
            f"Codebook payload breakdown fixed_bytes:{summary.get('fixed_payload_bytes', 0)} "
            f"scale_bytes:{summary.get('scale_payload_bytes', 0)} "
            f"residual_bytes:{summary.get('residual_payload_bytes', 0)} "
            f"outlier_bytes:{summary.get('outlier_payload_bytes', 0)} "
            f"int8_fallback_bytes:{summary.get('int8_fallback_payload_bytes', 0)} "
            f"int8_fallback_weights:{summary.get('int8_fallback_weights', 0)} "
            f"passthrough_bytes:{summary.get('passthrough_payload_bytes', 0)} "
            f"passthrough_weights:{summary.get('passthrough_weights', 0)} "
            f"effective_payload_bpw_all_weights:{summary.get('effective_payload_bpw_all_weights', 0.0):.4f} "
            f"effective_compressed_bpw_all_weights:{(8.0 * quant_file_bytes) / max(summary.get('model_fp_weights', 1), 1):.4f}"
        )
        group_stats = quant_stats.get("groups", {})
        group_parts = ["codebook:export_groups"]
        for group in ("attn", "mlp"):
            stats = group_stats.get(group)
            if stats is None:
                continue
            group_parts.append(
                f"{group}_fixed_mse:{stats['fixed_mse']:.6e} {group}_fixed_rmse:{stats['fixed_rmse']:.6e} "
                f"{group}_residual_mse:{stats['residual_mse']:.6e} {group}_residual_rmse:{stats['residual_rmse']:.6e} "
                f"{group}_final_mse:{stats['mse']:.6e} {group}_final_rmse:{stats['rmse']:.6e} "
                f"{group}_outlier_frac:{stats['outlier_frac']:.4%} "
                f"{group}_used_residual_frac:{stats['used_residual_frac']:.2%} "
                f"{group}_tensors:{int(stats['num_tensors'])}"
            )
        if len(group_parts) > 1:
            log(" ".join(group_parts))
        quantizer._log_top_tensor_stats(
            "codebook:export_top_error",
            list(quant_stats.get("tensors", {}).items()),
            sort_key="rel_mse",
        )
        quantizer._log_top_tensor_stats(
            "codebook:export_top_payload",
            list(quant_stats.get("tensors", {}).items()),
            sort_key="payload_bytes",
        )
        log(f"Total submission size codebook+{h.compressor}: {bytes_total} bytes")
    return bytes_total


def deserialize(h: Hyperparameters, device: torch.device) -> GPT:
    eval_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(eval_model)

    template_sd = eval_model.state_dict()

    with open(h.quantized_model_path, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(_decompress(quant_blob_disk, h.compressor)),
        map_location="cpu",
    )
    deq_state = dequantize_state_dict_codebook(quant_state["w"], quant_state["m"], template_sd)
    eval_model.load_state_dict(deq_state, strict=True)

    return eval_model

# ----------------------------------------
# Evaluation
# ----------------------------------------

def _loss_bpb(loss_sum, token_count, byte_count) -> tuple[float, float]:
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    return val_loss, val_bpb


def eval_val(
    h: Hyperparameters,
    device: torch.device,
    val_data: ValidationData,
    model: nn.Module
) -> tuple[float, float]:
    seq_len = h.eval_seq_len
    local_batch_tokens = h.val_batch_tokens // (h.world_size * h.grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={h.val_batch_tokens}, WORLD_SIZE={h.world_size}, "
            f"GRAD_ACCUM_STEPS={h.grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_data.val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * h.rank) // h.world_size
    seq_end = (total_seqs * (h.rank + 1)) // h.world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_data.val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (val_data.has_leading_space_lut[tgt_ids] & ~val_data.is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    model.train()
    return _loss_bpb(val_loss_sum, val_token_count, val_byte_count)


def eval_val_sliding(
    h: Hyperparameters,
    device: torch.device,
    val_data: ValidationData,
    base_model: nn.Module,
    batch_seqs: int = 32
) -> tuple[float, float]:
    """Sliding window evaluation: each token scored with maximum context."""
    base_model.eval()
    logits_fn = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)

    seq_len = h.eval_seq_len
    context_size = seq_len - h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1

    window_starts = [ws for ws in range(0, total_tokens, h.eval_stride)
                     if ws + context_size < total_tokens]

    total_windows = len(window_starts)
    my_s = (total_windows * h.rank) // h.world_size
    my_e = (total_windows * (h.rank + 1)) // h.world_size
    my_windows = window_starts[my_s:my_e]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)

            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []

            for i, ws in enumerate(batch_ws):
                we = min(ws + seq_len, total_tokens)
                wlen = we - ws
                wlens.append(wlen)
                chunk = val_data.val_tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = logits_fn(x_batch)

            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else context_size
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    base_model.train()
    return _loss_bpb(loss_sum, token_count, byte_count)


def timed_eval(label: str, fn, *args, **kwargs) -> tuple[float, float]: 
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    val_loss, val_bpb = fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)
    log(f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms")
    return val_loss, val_bpb


def run_evals(
    h: Hyperparameters,
    device: torch.device,
    val_data: ValidationData,
    eval_model: torch.nn.Module
):
    compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
    timed_eval("final_codebook_roundtrip", eval_val, h, device, val_data, compiled_model)
    if h.sliding_window_enabled:
        timed_eval("final_codebook_sliding_window", eval_val_sliding, h, device, val_data, eval_model)

# -----------------------------
# Training
# -----------------------------

def train_model(
    h: Hyperparameters,
    device: torch.device,
    val_data: ValidationData,
) -> tuple[GPT, nn.Module]:
    # Set up model
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    if h.distributed:
        model = DDP(compiled_model, device_ids=[h.local_rank], broadcast_buffers=False)
    else:
        model = compiled_model
    log(f"model_params:{sum(p.numel() for p in base_model.parameters())}")

    # Set up optimizer and load train data
    optimizers = Optimizers(h, base_model)
    train_loader = DistributedTokenLoader( h.train_files, h.rank, h.world_size, device)

    # Helper functions for training
    max_wallclock_ms = 1000.0 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None
    if max_wallclock_ms is not None and h.codebook_reserve_seconds > 0:
        max_wallclock_ms = max(max_wallclock_ms - h.codebook_reserve_seconds * 1000.0, 0.0)
        log(f"codebook:reserving {h.codebook_reserve_seconds:.0f}s, effective={max_wallclock_ms:.0f}ms")

    def training_frac(step: int, elapsed_ms: float) -> float:
        """Fraction of training completed (0 to 1), using step or wallclock."""
        if max_wallclock_ms is None:
            return step / max(h.iterations, 1)
        return elapsed_ms / max(max_wallclock_ms, 1e-9)

    def lr_mul(frac: float) -> float:
        if h.warmdown_frac <= 0:
            return 1.0
        if frac >= 1.0 - h.warmdown_frac:
            return max((1.0 - frac) / h.warmdown_frac, h.min_lr)
        return 1.0

    def step_fn(step, lr_scale):
        optimizers.zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(h.grad_accum_steps):
            if h.distributed:
                model.require_backward_grad_sync = micro_step == h.grad_accum_steps - 1
            x, y = train_loader.next_batch(h.train_batch_tokens, h.train_seq_len, h.grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss / h.grad_accum_steps).backward()
        train_loss /= h.grad_accum_steps

        muon_frac = min(step / h.muon_momentum_warmup_steps, 1.0) if h.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - muon_frac) * h.muon_momentum_warmup_start + muon_frac * h.muon_momentum
        for group in optimizers.optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * lr_scale

        if h.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)

        optimizers.step()
        return train_loss

    # Model warmup
    if h.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(h.warmup_steps):
            step_fn(warmup_step, 1.0)
            if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == h.warmup_steps:
                log(f"warmup_step: {warmup_step + 1}/{h.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        optimizers.zero_grad_all()
        if h.distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(
            h.train_files, h.rank, h.world_size, device)

    # Training loop
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == h.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (h.val_loss_every > 0 and step % h.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(h, device, val_data, model)
            log(f"{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < h.iterations:
                log(
                    f"stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms "
                    f"step: {step}/{h.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        frac = training_frac(step, elapsed_ms)
        scale = lr_mul(frac)
        train_loss = step_fn(step, scale)

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        should_log_train = (
            h.train_log_every > 0
            and (step <= 5 or step % h.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            tok_per_sec = step * h.train_batch_tokens / (approx_training_time_ms / 1000.0)
            log(
                f"{step}/{h.iterations} train_loss: {train_loss.item():.4f} "
                f"train_time: {approx_training_time_ms / 60000:.1f}m tok/s: {tok_per_sec:.0f}"
            )

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if h.distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    return base_model, compiled_model


def train_and_eval(h: Hyperparameters, device: torch.device) -> None:
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)

    val_data = ValidationData(h, device)
    log(f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}")
    log(f"val_tokens: {val_data.val_tokens.numel() - 1}")

    base_model, compiled_model = train_model(h, device, val_data)
    timed_eval("pre-codebook export fp_model", eval_val, h, device, val_data, compiled_model)

    serialize(h, base_model, Path(__file__).read_text(encoding="utf-8"))
    if h.distributed:
        dist.barrier()
    eval_model = deserialize(h, device)

    run_evals(h, device, val_data, eval_model)


def main():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False

    h = Hyperparameters()
    set_logging_hparams(h)
    if h.is_main_process:
        os.makedirs("logs", exist_ok=True)
        log(100 * "=", console=False)
        log("Hyperparameters:", console=True)
        for k, v in sorted(vars(type(h)).items()):
            if not k.startswith("_"):
                log(f"  {k}: {v}", console=True)
        log(Path(__file__).read_text(encoding="utf-8"), console=False)
        log("=" * 100, console=False)
        log(f"Running Python {sys.version}", console=False)
        log(f"Running PyTorch {torch.__version__}", console=False)
        log(
            subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
            console=False,
        )
        log("=" * 100, console=False)

    train_and_eval(h, device)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()