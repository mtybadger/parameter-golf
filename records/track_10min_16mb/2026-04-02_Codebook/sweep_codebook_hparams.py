#!/usr/bin/env python3
import argparse
import copy
import importlib.util
import io
import itertools
import os
import re
import shlex
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_COMMAND_FILE = THIS_DIR / "command.txt"
DEFAULT_TRAIN_FILE = THIS_DIR / "train_gpt.py"


def parse_latest_command_block(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if "train_gpt.py" in b]
    if not blocks:
        raise ValueError(f"No train_gpt.py command block found in {path}")
    block = blocks[-1].replace("\\\n", " ")
    tokens = shlex.split(block)
    env: dict[str, str] = {}
    for token in tokens:
        if token.startswith("python"):
            break
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if re.fullmatch(r"[A-Z0-9_]+", key):
            env[key] = value
    return env


def load_train_module(train_file: Path):
    spec = importlib.util.spec_from_file_location("codebook_train_mod", train_file)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def apply_legacy_paths(h, env: dict[str, str]) -> None:
    data_path = env.get("DATA_PATH")
    if data_path:
        h.datasets_dir = data_path.rstrip("/")
        h.train_files = str(Path(h.datasets_dir) / "fineweb_train_*.bin")
        h.val_files = str(Path(h.datasets_dir) / "fineweb_val_*.bin")
    tokenizer_path = env.get("TOKENIZER_PATH")
    if tokenizer_path:
        h.tokenizer_path = tokenizer_path


def cast_like(current, raw: str):
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def env_key_to_attr(key: str) -> str:
    return key.lower()


def build_hparam(base_h, env: dict[str, str], overrides: dict[str, str]):
    h = base_h.__class__()
    for key, value in vars(base_h).items():
        setattr(h, key, copy.deepcopy(value))
    apply_legacy_paths(h, env)
    for env_key, raw_value in overrides.items():
        attr = env_key_to_attr(env_key)
        if not hasattr(h, attr):
            raise AttributeError(f"Unknown hyperparameter env key: {env_key}")
        current = getattr(h, attr)
        setattr(h, attr, cast_like(current, raw_value))
    h.model_path = str(Path(base_h.model_path).resolve())
    h.quantized_model_path = str((THIS_DIR / f"tmp_{h.run_id}_{hash(tuple(sorted(overrides.items())))}.codebook.ptz").resolve())
    h.logfile = None
    h.is_main_process = False
    return h


def parse_grid_arg(arg: str) -> tuple[str, list[str]]:
    if "=" not in arg:
        raise ValueError(f"Grid spec must look like KEY=v1,v2,..., got {arg!r}")
    key, raw = arg.split("=", 1)
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError(f"Grid spec has no values: {arg!r}")
    return key.strip(), values


def default_grid(base_env: dict[str, str]) -> dict[str, list[str]]:
    def get_float(key: str, fallback: float) -> float:
        try:
            return float(base_env.get(key, fallback))
        except ValueError:
            return fallback

    def get_int(key: str, fallback: int) -> int:
        try:
            return int(base_env.get(key, fallback))
        except ValueError:
            return fallback

    lattice = get_float("CODEBOOK_LATTICE_SCALE", 1.03)
    damp = get_float("CODEBOOK_HESSIAN_DAMP", 0.01)
    outliers = get_int("CODEBOOK_OUTLIER_MAX_COUNT", 0)
    scale_bits = get_int("CODEBOOK_SCALE_BITS", 8)

    return {
        "CODEBOOK_LATTICE_SCALE": [f"{max(0.8, lattice - 0.02):.3f}", f"{lattice:.3f}", f"{lattice + 0.02:.3f}"],
        "CODEBOOK_HESSIAN_DAMP": [f"{max(1e-4, damp * 0.5):.4f}", f"{damp:.4f}", f"{max(1e-4, damp * 2.0):.4f}"],
        "CODEBOOK_OUTLIER_MAX_COUNT": [str(x) for x in sorted({0, max(256, outliers or 512), max(1024, outliers * 2 or 1024)})],
        "CODEBOOK_SCALE_BITS": [str(x) for x in sorted({8, scale_bits, 12})],
    }


def candidate_overrides(base_env: dict[str, str], grid: dict[str, list[str]], full_grid: bool) -> list[dict[str, str]]:
    baseline = {k: base_env.get(k) for k in grid.keys() if k in base_env}
    if not full_grid:
        out = [dict()]
        for key, values in grid.items():
            base_value = base_env.get(key)
            for value in values:
                if value == base_value:
                    continue
                out.append({key: value})
        return out
    keys = list(grid.keys())
    values_product = list(itertools.product(*[grid[k] for k in keys]))
    combos = []
    for values in values_product:
        combo = {k: v for k, v in zip(keys, values, strict=True)}
        combos.append(combo)
    return combos


def combo_label(combo: dict[str, str]) -> str:
    if not combo:
        return "baseline"
    return ", ".join(f"{k}={v}" for k, v in sorted(combo.items()))


def evaluate_candidate(mod, base_h, base_env, base_model, val_data, hessians, combo):
    h = build_hparam(base_h, base_env, combo)
    quantizer = mod.CodebookCodec(h, base_model)
    quantizer.fit(base_model, hessians)
    quant_result, quant_meta, quant_stats = quantizer.build_export(base_model.state_dict())

    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta, "s": quant_stats}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = mod._compress(quant_raw, h.compressor)
    quant_file_bytes = len(quant_blob)

    eval_model = mod.GPT(h).to(next(base_model.parameters()).device).bfloat16()
    mod.restore_fp32_params(eval_model)
    template_sd = eval_model.state_dict()
    deq_state = mod.dequantize_state_dict_codebook(quant_result, quant_meta, template_sd)
    eval_model.load_state_dict(deq_state, strict=True)
    val_loss, val_bpb = mod.eval_val(h, next(base_model.parameters()).device, val_data, eval_model)
    return {
        "label": combo_label(combo),
        "overrides": combo,
        "val_loss": val_loss,
        "val_bpb": val_bpb,
        "rel_mse": float(quantizer.last_fit_summary.get("rel_mse", float("nan"))),
        "coverage": float(quant_stats["coverage"]),
        "payload_bpw": float(quant_stats["effective_payload_bpw_all_weights"]),
        "quant_bytes": quant_file_bytes,
        "target_bpw": float(quant_stats["target_bpw"]),
        "outlier_weights": int(quant_stats.get("outlier_weights", 0)),
        "outlier_bytes": int(quant_stats.get("outlier_payload_bytes", 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep codebook hyperparameters on an existing trained checkpoint.")
    parser.add_argument("--command-file", type=Path, default=DEFAULT_COMMAND_FILE)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--grid", action="append", default=[], help="Repeatable KEY=v1,v2,... spec")
    parser.add_argument("--full-grid", action="store_true", help="Use full Cartesian product instead of one-at-a-time around baseline")
    parser.add_argument("--output", type=Path, default=THIS_DIR / "codebook_sweep_results.tsv")
    args = parser.parse_args()

    base_env = parse_latest_command_block(args.command_file)
    for key, value in base_env.items():
        os.environ[key] = value

    mod = load_train_module(args.train_file)
    base_h = mod.Hyperparameters()
    apply_legacy_paths(base_h, base_env)
    base_h.logfile = None
    base_h.is_main_process = False

    checkpoint = args.checkpoint or Path(base_h.model_path)
    checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the sweep")
    device_index = 0
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    grid = dict(parse_grid_arg(g) for g in args.grid) if args.grid else default_grid(base_env)
    combos = candidate_overrides(base_env, grid, args.full_grid)

    print(f"Using checkpoint: {checkpoint}")
    print(f"Base command: {args.command_file}")
    print(f"Sweep mode: {'full-grid' if args.full_grid else 'one-at-a-time'}")
    print(f"Candidates: {len(combos)}")
    for key, values in grid.items():
        print(f"  {key}: {', '.join(values)}")

    base_model = mod.GPT(base_h).to(device).bfloat16()
    mod.restore_fp32_params(base_model)
    state = torch.load(checkpoint, map_location="cpu")
    base_model.load_state_dict(state, strict=True)

    val_data = mod.ValidationData(base_h, device)
    quantizer = mod.CodebookCodec(base_h, base_model)
    calib_loader = mod.DistributedTokenLoader(base_h.train_files, base_h.rank, base_h.world_size, device)
    print("Collecting calibration hessians once...")
    t0 = mod.time.perf_counter()
    hessians = mod.collect_hessians(
        base_model,
        calib_loader,
        base_h,
        device,
        quantizer.target_names,
        n_calibration_batches=base_h.codebook_calibration_batches,
    )
    print(f"Collected {len(hessians)} hessians in {mod.time.perf_counter() - t0:.1f}s")

    results = []
    for idx, combo in enumerate(combos, start=1):
        label = combo_label(combo)
        print(f"[{idx}/{len(combos)}] {label}")
        t0 = mod.time.perf_counter()
        row = evaluate_candidate(mod, base_h, base_env, base_model, val_data, hessians, combo)
        row["elapsed_s"] = mod.time.perf_counter() - t0
        results.append(row)
        print(
            f"    val_bpb={row['val_bpb']:.6f} val_loss={row['val_loss']:.6f} rel_mse={row['rel_mse']:.6e} "
            f"bytes={row['quant_bytes']} elapsed={row['elapsed_s']:.1f}s"
        )

    results.sort(key=lambda r: r["val_bpb"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        headers = [
            "rank", "label", "val_bpb", "val_loss", "rel_mse", "coverage",
            "payload_bpw", "target_bpw", "quant_bytes", "outlier_weights", "outlier_bytes", "elapsed_s", "overrides",
        ]
        f.write("\t".join(headers) + "\n")
        for rank, row in enumerate(results, start=1):
            f.write("\t".join([
                str(rank),
                row["label"],
                f"{row['val_bpb']:.8f}",
                f"{row['val_loss']:.8f}",
                f"{row['rel_mse']:.8e}",
                f"{row['coverage']:.8f}",
                f"{row['payload_bpw']:.8f}",
                f"{row['target_bpw']:.8f}",
                str(row["quant_bytes"]),
                str(row["outlier_weights"]),
                str(row["outlier_bytes"]),
                f"{row['elapsed_s']:.3f}",
                ", ".join(f"{k}={v}" for k, v in sorted(row["overrides"].items())) or "baseline",
            ]) + "\n")

    print("\nTop results:")
    for rank, row in enumerate(results[: min(10, len(results))], start=1):
        print(
            f"{rank}. {row['label']} | val_bpb={row['val_bpb']:.6f} rel_mse={row['rel_mse']:.6e} "
            f"bytes={row['quant_bytes']} payload_bpw={row['payload_bpw']:.4f}"
        )
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
