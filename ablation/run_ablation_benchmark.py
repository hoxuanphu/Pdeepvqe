"""Untrained architecture benchmark for DeepVQE ablation variants.

The script writes ``results/ablation_arch_benchmark.csv`` by default.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ablation.ablation_config import (
    TRAIN_CONFIG_PRESETS,
    get_model_config_id,
    get_train_config,
    reproducibility_metadata,
)
from ablation.deepvqe_ablation import (
    ABLATION_CONFIGS,
    DeepVQE_Ablation,
    StreamDeepVQE_Ablation,
    convert_ablation_to_stream,
    count_parameters,
    get_ablation_config,
    stream_sequence,
)


def try_ptflops(model, frames, device):
    try:
        from ptflops import get_model_complexity_info
    except Exception as exc:
        return None, None, f"ptflops import failed: {exc}"

    try:
        macs, params = get_model_complexity_info(
            model,
            (257, frames, 2),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
            backend="pytorch",
        )
        return macs, params, ""
    except Exception as exc:
        return None, None, f"ptflops failed: {exc}"


@torch.no_grad()
def causality_error(model, frames, freq_bins, device):
    prefix = torch.randn(1, freq_bins, frames, 2, device=device)
    suffix_a = torch.randn(1, freq_bins, frames, 2, device=device)
    suffix_b = torch.randn(1, freq_bins, frames, 2, device=device)
    x1 = torch.cat([prefix, suffix_a], dim=2)
    x2 = torch.cat([prefix, suffix_b], dim=2)
    y1 = model(x1)
    y2 = model(x2)
    return (y1[:, :, :frames, :] - y2[:, :, :frames, :]).abs().max().item()


@torch.no_grad()
def streaming_parity_error(model, stream_model, frames, freq_bins, device):
    x = torch.randn(1, freq_bins, frames, 2, device=device)
    offline = model(x)
    streaming, _ = stream_sequence(stream_model, x)
    return (offline - streaming).abs().max().item()


@torch.no_grad()
def stateful_streaming_rtf(stream_model, frames, freq_bins, sample_rate, hop_length, warmup, repeats, device):
    x = torch.randn(1, freq_bins, frames, 2, device=device)
    for _ in range(warmup):
        stream_sequence(stream_model, x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        stream_sequence(stream_model, x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / max(1, repeats)
    audio_seconds = frames * hop_length / sample_rate
    return elapsed / audio_seconds, elapsed


@torch.no_grad()
def stateful_streaming_rtf_per_frame(stream_model, frames, freq_bins, sample_rate, hop_length, warmup, repeats, device):
    x = torch.randn(1, freq_bins, frames, 2, device=device)
    for _ in range(warmup):
        stream_sequence(stream_model, x)
        
    total_processing_time = 0.0
    for _ in range(repeats):
        cache = stream_model.init_cache(x.shape[0], x.shape[1], x.device, x.dtype)
        for frame_idx in range(frames):
            frame_input = x[:, :, frame_idx : frame_idx + 1, :]
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            y, cache = stream_model(frame_input, cache)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            total_processing_time += (t1 - t0)

    elapsed = total_processing_time / max(1, repeats)
    audio_seconds = frames * hop_length / sample_rate
    return elapsed / audio_seconds, elapsed


def passes_budget(row, max_params, max_macs, max_rtf, max_causality_error):
    if row["causality_error"] > max_causality_error:
        return False
    if max_params is not None and row["params"] > max_params:
        return False
    if max_macs is not None and row["macs"] not in (None, "") and row["macs"] > max_macs:
        return False
    if max_rtf is not None and row["streaming_rtf"] > max_rtf:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Benchmark untrained DeepVQE ablation variants")
    parser.add_argument("--output", default="results/ablation_arch_benchmark.csv")
    parser.add_argument("--configs", nargs="*", default=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--frames", type=int, default=63)
    parser.add_argument("--freq-bins", type=int, default=257)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-params", type=int, default=None)
    parser.add_argument("--max-macs", type=float, default=None)
    parser.add_argument("--max-rtf", type=float, default=None)
    parser.add_argument("--max-causality-error", type=float, default=0.0)
    parser.add_argument("--max-streaming-parity-error", type=float, default=1e-5)
    args = parser.parse_args()

    device = torch.device(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for config_id in args.configs:
        model_config_id = get_model_config_id(config_id)
        try:
            get_ablation_config(model_config_id)
        except ValueError as exc:
            valid = ", ".join(list(ABLATION_CONFIGS) + list(TRAIN_CONFIG_PRESETS))
            raise ValueError(f"Unknown config {config_id!r}; valid configs: {valid}") from exc

        model = DeepVQE_Ablation.from_config_id(model_config_id).eval().to(device)
        stream_model = StreamDeepVQE_Ablation.from_config_id(model_config_id).eval().to(device)
        convert_ablation_to_stream(stream_model, model, strict=True)
        params = count_parameters(model)
        macs, ptflops_params, notes = try_ptflops(model, args.frames, device)
        c_error = causality_error(model, args.frames, args.freq_bins, device)
        stream_error = streaming_parity_error(model, stream_model, args.frames, args.freq_bins, device)
        rtf, mean_forward_seconds = stateful_streaming_rtf(
            stream_model,
            args.frames,
            args.freq_bins,
            args.sample_rate,
            args.hop_length,
            args.warmup,
            args.repeats,
            device,
        )
        rtf_per_frame, mean_forward_seconds_per_frame = stateful_streaming_rtf_per_frame(
            stream_model,
            args.frames,
            args.freq_bins,
            args.sample_rate,
            args.hop_length,
            args.warmup,
            args.repeats,
            device,
        )

        row = {
            "config_id": config_id,
            "params": params,
            "ptflops_params": ptflops_params if ptflops_params is not None else "",
            "macs": macs if macs is not None else "",
            "flops": (2 * macs) if macs is not None else "",
            "streaming_rtf": rtf,
            "mean_streaming_ms": mean_forward_seconds * 1000.0,
            "streaming_rtf_per_frame": rtf_per_frame,
            "mean_streaming_ms_per_frame": mean_forward_seconds_per_frame * 1000.0,
            "streaming_parity_error": stream_error,
            "causality_error": c_error,
            "pass": False,
            "notes": notes,
        }
        if model_config_id != config_id:
            row["notes"] = f"architecture={model_config_id}" + (f" | {notes}" if notes else "")
        row["pass"] = (
            passes_budget(row, args.max_params, args.max_macs, args.max_rtf, args.max_causality_error)
            and row["streaming_parity_error"] <= args.max_streaming_parity_error
        )
        row.update(reproducibility_metadata(get_train_config(config_id)))
        rows.append(row)
        print(
            f"{config_id}: params={params} macs={row['macs']} "
            f"stateful_rtf={rtf:.4f} rtf_per_frame={rtf_per_frame:.4f} stream_error={stream_error:.6g} "
            f"causality_error={c_error:.6g} pass={row['pass']}"
        )

    fieldnames = [
        "config_id",
        "git_commit",
        "config_hash",
        "seed",
        "hardware_info",
        "torch_version",
        "onnxruntime_version",
        "num_threads",
        "checkpoint_id",
        "params",
        "ptflops_params",
        "macs",
        "flops",
        "streaming_rtf",
        "mean_streaming_ms",
        "streaming_rtf_per_frame",
        "mean_streaming_ms_per_frame",
        "streaming_parity_error",
        "causality_error",
        "pass",
        "notes",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
