"""Profile a few training steps on 8 GPUs via torchrun and print a time breakdown."""
import os, sys, time, torch, torch.distributed as dist

sys.path.insert(0, "records/track_10min_16mb/2026-04-09_PrefetchData")
import train_gpt as tg
from torch.nn.parallel import DistributedDataParallel as DDP

# --- setup (mirrors main() in train_gpt.py) ---
world_size = int(os.environ.get("WORLD_SIZE", "1"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
if distributed:
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch._dynamo.config.optimize_ddp = False

h = tg.Hyperparameters()
h.warmup_steps = 0
h.max_wallclock_seconds = 0
tg.set_logging_hparams(h)

is_rank0 = h.is_main_process

def log(msg):
    if is_rank0:
        print(msg, flush=True)

# Build model
base_model = tg.GPT(h).to(device).bfloat16()
tg.restore_fp32_params(base_model)
compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
if distributed:
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
else:
    model = compiled_model

n_params = sum(p.numel() for p in base_model.parameters())
log(f"Model params: {n_params:,}  |  world_size={h.world_size}  grad_accum={h.grad_accum_steps}")

optimizers = tg.Optimizers(h, base_model)
if hasattr(tg, 'PrefetchLoader'):
    train_loader = tg.PrefetchLoader(tg.ShuffledSequenceLoader(h, device), h)
    log("Using PrefetchLoader")
else:
    train_loader = tg.ShuffledSequenceLoader(h, device)
    log("Using ShuffledSequenceLoader (no prefetch)")

def run_step(step_idx):
    """One full training step matching train_gpt.step_fn()."""
    optimizers.zero_grad_all()
    for micro in range(h.grad_accum_steps):
        if distributed:
            model.require_backward_grad_sync = (micro == h.grad_accum_steps - 1)
        x, y = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss = model(x, y)
        (loss / h.grad_accum_steps).backward()
    if h.grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
    optimizers.step()
    return loss

# Warmup: 3 steps to trigger torch.compile
log("Warming up (torch.compile)...")
for i in range(3):
    loss = run_step(i)
log(f"Warmup done. loss={loss.item():.4f}")

# ---- torch.profiler pass ----
NUM_PROFILE_STEPS = 5
log(f"\nProfiling {NUM_PROFILE_STEPS} steps with torch.profiler...")

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    with_stack=False,
    record_shapes=True,
    profile_memory=False,
) as prof:
    for step in range(NUM_PROFILE_STEPS):
        run_step(step)

log("\n" + "="*90)
log("TOP CUDA KERNELS (by total CUDA time)")
log("="*90)
if is_rank0:
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

log("\n" + "="*90)
log("TOP CPU OPERATIONS (by total CPU time)")
log("="*90)
if is_rank0:
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

# ---- Manual phase-level timing ----
log("\n" + "="*90)
log(f"PHASE-LEVEL BREAKDOWN (avg of {NUM_PROFILE_STEPS} steps)")
log("="*90)

phase_times = {"data_load": [], "forward": [], "backward": [], "grad_clip": [], "optimizer": [], "total": []}

for step in range(NUM_PROFILE_STEPS):
    torch.cuda.synchronize()
    t_total = time.perf_counter()

    optimizers.zero_grad_all()
    t_fwd_sum = 0
    t_bwd_sum = 0
    t_data_sum = 0

    for micro in range(h.grad_accum_steps):
        if distributed:
            model.require_backward_grad_sync = (micro == h.grad_accum_steps - 1)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x, y = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
        torch.cuda.synchronize()
        t_data = time.perf_counter()
        t_data_sum += t_data - t0

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss = model(x, y)
        torch.cuda.synchronize()
        t_fwd = time.perf_counter()
        t_fwd_sum += t_fwd - t_data

        (loss / h.grad_accum_steps).backward()
        torch.cuda.synchronize()
        t_bwd = time.perf_counter()
        t_bwd_sum += t_bwd - t_fwd

    torch.cuda.synchronize()
    t_clip_start = time.perf_counter()
    if h.grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
    torch.cuda.synchronize()
    t_clip_end = time.perf_counter()

    optimizers.step()
    torch.cuda.synchronize()
    t_opt_end = time.perf_counter()

    phase_times["data_load"].append(t_data_sum)
    phase_times["forward"].append(t_fwd_sum)
    phase_times["backward"].append(t_bwd_sum)
    phase_times["grad_clip"].append(t_clip_end - t_clip_start)
    phase_times["optimizer"].append(t_opt_end - t_clip_end)
    phase_times["total"].append(t_opt_end - t_total)

if is_rank0:
    print(f"{'Phase':<15} {'Avg (ms)':>10} {'% of step':>10}")
    print("-" * 37)
    avg_total = sum(phase_times["total"]) / NUM_PROFILE_STEPS * 1000
    for phase in ["data_load", "forward", "backward", "grad_clip", "optimizer"]:
        avg_ms = sum(phase_times[phase]) / NUM_PROFILE_STEPS * 1000
        pct = avg_ms / avg_total * 100
        print(f"{phase:<15} {avg_ms:>10.1f} {pct:>9.1f}%")
    print(f"{'TOTAL':<15} {avg_total:>10.1f} {'100.0':>9}%")

    tok_per_sec = h.train_batch_tokens / (avg_total / 1000)
    print(f"\nThroughput: {tok_per_sec:,.0f} tok/s  ({h.train_batch_tokens:,} tokens/step, {avg_total:.1f} ms/step)")
    print(f"Grad accum steps: {h.grad_accum_steps}, seq_len: {h.train_seq_len}, batch_tokens: {h.train_batch_tokens:,}")

if distributed:
    dist.destroy_process_group()
