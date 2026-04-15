"""
Profile a single training step, breaking down time into:
  - data loading
  - forward pass (per micro-step, summed)
  - backward pass (per micro-step, summed)
  - grad clip
  - optimizer: Muon (Newton-Schulz + weight update)
  - optimizer: AdamW tok/scalar
  - EMA update

Run from the FP8 directory with the venv activated, e.g.:
  cd /home/spruce/parameter-golf
  DATA_DIR=./data python records/track_10min_16mb/2026-04-13_FP8/profile_step.py
"""

import copy
import os
import sys
import time

import torch
import torch.nn.functional as F

# Make train_gpt importable
sys.path.insert(0, os.path.dirname(__file__))
import train_gpt as T

# ── helpers ──────────────────────────────────────────────────────────────────

def cuda_timer():
    """Return a pair of CUDA events and a helper to read elapsed ms."""
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    return start, end

def elapsed_ms(start, end):
    return start.elapsed_time(end)

# ── setup ─────────────────────────────────────────────────────────────────────

WARMUP_STEPS  = 4   # let torch.compile finish graph capture
PROFILE_STEPS = 10  # steps to average over

device = torch.device("cuda", 0)
torch.cuda.set_device(device)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

from torch.backends.cuda import (
    enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp,
)
enable_cudnn_sdp(False)
enable_flash_sdp(True)
enable_mem_efficient_sdp(False)
enable_math_sdp(False)

h = T.Hyperparameters()
T.set_logging_hparams(h)

print(f"vocab_size={h.vocab_size}  num_layers={h.num_layers}  model_dim={h.model_dim}")
print(f"grad_accum_steps={h.grad_accum_steps}  train_batch_tokens={h.train_batch_tokens}")

torch.manual_seed(h.seed)
torch.cuda.manual_seed_all(h.seed)

base_model = T.GPT(h).to(device).bfloat16()
T.restore_fp32_params(base_model)
model = torch.compile(base_model, dynamic=False, fullgraph=True)
model.train()
print(f"parameters: {sum(p.numel() for p in base_model.parameters()):,}")

optimizers = T.Optimizers(h, base_model)
train_loader = T.ShuffledSequenceLoader(h, device)

# ── instrumented step ─────────────────────────────────────────────────────────

def profiled_step():
    """Run one full training step and return a dict of phase → ms."""
    timings = {}

    # ── zero grads ────────────────────────────────────────────────────────
    optimizers.zero_grad_all()

    # ── data + forward + backward ─────────────────────────────────────────
    data_ms   = 0.0
    fwd_ms    = 0.0
    bwd_ms    = 0.0
    train_loss = torch.zeros((), device=device)

    for micro_step in range(h.grad_accum_steps):
        s, e = cuda_timer()

        # data loading (CPU work, then H2D copy)
        torch.cuda.synchronize()
        t_data0 = time.perf_counter()
        x, y = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
        torch.cuda.synchronize()          # wait for non-blocking H2D
        data_ms += 1e3 * (time.perf_counter() - t_data0)

        # forward
        s.record()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss = model(x, y)
        e.record()
        torch.cuda.synchronize()
        fwd_ms += elapsed_ms(s, e)

        train_loss += loss.detach()

        # backward
        s.record()
        (loss / h.grad_accum_steps).backward()
        e.record()
        torch.cuda.synchronize()
        bwd_ms += elapsed_ms(s, e)

    timings["data_H2D"]  = data_ms
    timings["forward"]   = fwd_ms
    timings["backward"]  = bwd_ms

    # ── grad clip ─────────────────────────────────────────────────────────
    s, e = cuda_timer()
    s.record()
    if h.grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
    e.record()
    torch.cuda.synchronize()
    timings["grad_clip"] = elapsed_ms(s, e)

    # ── Muon step ─────────────────────────────────────────────────────────
    s, e = cuda_timer()
    s.record()
    optimizers.optimizer_muon.step()
    e.record()
    torch.cuda.synchronize()
    timings["muon"] = elapsed_ms(s, e)

    # ── AdamW tok ─────────────────────────────────────────────────────────
    s, e = cuda_timer()
    s.record()
    optimizers.optimizer_tok.step()
    e.record()
    torch.cuda.synchronize()
    timings["adamw_tok"] = elapsed_ms(s, e)

    # ── AdamW scalar ──────────────────────────────────────────────────────
    s, e = cuda_timer()
    s.record()
    optimizers.optimizer_scalar.step()
    e.record()
    torch.cuda.synchronize()
    timings["adamw_scalar"] = elapsed_ms(s, e)

    # ── EMA update ────────────────────────────────────────────────────────
    s, e = cuda_timer()
    s.record()
    with torch.no_grad():
        for name, t in base_model.state_dict().items():
            pass   # just iterate; actual EMA below
    # real EMA
    ema_decay = h.ema_decay
    s.record()
    with torch.no_grad():
        for name, t in base_model.state_dict().items():
            _ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)
    e.record()
    torch.cuda.synchronize()
    timings["ema"] = elapsed_ms(s, e)

    # zero grads for next step
    optimizers.zero_grad_all()

    return timings, (train_loss / h.grad_accum_steps).item()

# ── build EMA state ───────────────────────────────────────────────────────────

_ema_state = {
    name: t.detach().float().clone()
    for name, t in base_model.state_dict().items()
}

# ── warmup ────────────────────────────────────────────────────────────────────

print(f"\nWarmup ({WARMUP_STEPS} steps) …")
for wi in range(WARMUP_STEPS):
    _, loss_val = profiled_step()
    print(f"  warmup step {wi+1}/{WARMUP_STEPS}  loss={loss_val:.4f}")
torch.cuda.synchronize()

# ── profile ───────────────────────────────────────────────────────────────────

print(f"\nProfiling ({PROFILE_STEPS} steps) …")
accum = {}
wall_times = []

for pi in range(PROFILE_STEPS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    timings, loss_val = profiled_step()
    torch.cuda.synchronize()
    wall_ms = 1e3 * (time.perf_counter() - t0)
    wall_times.append(wall_ms)
    for k, v in timings.items():
        accum[k] = accum.get(k, 0.0) + v
    print(f"  step {pi+1}/{PROFILE_STEPS}  loss={loss_val:.4f}  wall={wall_ms:.0f}ms")

# ── report ────────────────────────────────────────────────────────────────────

avg = {k: v / PROFILE_STEPS for k, v in accum.items()}
avg_wall = sum(wall_times) / len(wall_times)

print("\n" + "=" * 60)
print(f"{'Phase':<20} {'Avg ms':>10}  {'% of wall':>10}")
print("-" * 60)

ordered = ["data_H2D", "forward", "backward", "grad_clip",
           "muon", "adamw_tok", "adamw_scalar", "ema"]

total_accounted = 0.0
for phase in ordered:
    ms = avg.get(phase, 0.0)
    total_accounted += ms
    pct = 100.0 * ms / avg_wall
    print(f"  {phase:<18} {ms:>10.1f}  {pct:>9.1f}%")

print("-" * 60)
print(f"  {'accounted':<18} {total_accounted:>10.1f}  {100*total_accounted/avg_wall:>9.1f}%")
print(f"  {'wall (total)':<18} {avg_wall:>10.1f}  {'100.0':>9}%")
print("=" * 60)

tok_per_sec = h.train_batch_tokens / (avg_wall / 1e3)
print(f"\ntokens/sec ≈ {tok_per_sec:,.0f}")
print(f"peak GPU mem: {torch.cuda.max_memory_allocated()//1024//1024} MiB allocated, "
      f"{torch.cuda.max_memory_reserved()//1024//1024} MiB reserved")
