"""RTF benchmark for offline and stateful streaming ablation models.

Examples:
    python ablation/scripts/benchmark_rtf.py --configs Baseline Mamba_b2_h384
    python ablation/scripts/benchmark_rtf.py --devices cpu cuda --frames 63
"""

import argparse
import csv
import os
import sys
import time
import tracemalloc
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from ablation.ablation_config import get_model_config_id, get_train_config, reproducibility_metadata
from ablation.deepvqe_ablation import (
    ABLATION_CONFIGS,
    DeepVQE_Ablation,
    StreamDeepVQE_Ablation,
    convert_ablation_to_stream,
    count_parameters,
    stream_sequence,
)


def process_rss_mb():
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        return ""


def cuda_peak_mb(device):
    if device.type != "cuda":
        return ""
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def benchmark_forward(fn, warmup, repeats, device):
    for _ in range(warmup):
        fn()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tracemalloc.start()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize(device)
    elapsed = (time.perf_counter() - start) / max(1, repeats)
    _, peak_py = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak_py / (1024 ** 2), process_rss_mb(), cuda_peak_mb(device)


def available_devices(requested):
    devices = []
    for name in requested:
        if name == "cuda" and not torch.cuda.is_available():
            continue
        devices.append(torch.device(name))
    return devices


def main():
    parser = argparse.ArgumentParser(description="Benchmark DeepVQE ablation RTF")
    parser.add_argument("--configs", nargs="*", default=["Baseline", "D1b_gru768", "Mamba_b2_h384"])
    parser.add_argument("--devices", nargs="*", default=["cpu", "cuda"])
    parser.add_argument("--frames", type=int, default=63)
    parser.add_argument("--freq-bins", type=int, default=257)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="results/phase2_rtf_benchmark.csv")
    args = parser.parse_args()

    devices = available_devices(args.devices)
    if not devices:
        raise RuntimeError("No requested benchmark devices are available")

    audio_seconds = args.frames * args.hop_length / args.sample_rate
    rows = []
    for config_id in args.configs:
        if config_id not in ABLATION_CONFIGS:
            model_config_id = get_model_config_id(config_id)
        else:
            model_config_id = config_id

        for device in devices:
            x = torch.randn(1, args.freq_bins, args.frames, 2, device=device)
            model = DeepVQE_Ablation.from_config_id(model_config_id).eval().to(device)
            stream_model = StreamDeepVQE_Ablation.from_config_id(model_config_id).eval().to(device)
            convert_ablation_to_stream(stream_model, model, strict=True)

            offline_seconds, offline_py_mb, offline_rss_mb, offline_cuda_mb = benchmark_forward(
                lambda: model(x),
                args.warmup,
                args.repeats,
                device,
            )
            stream_seconds, stream_py_mb, stream_rss_mb, stream_cuda_mb = benchmark_forward(
                lambda: stream_sequence(stream_model, x),
                args.warmup,
                args.repeats,
                device,
            )
            cache = stream_model.init_cache(1, args.freq_bins, device=device, dtype=x.dtype)

            row = {
                "config_id": config_id,
                "architecture": model_config_id,
                "device": str(device),
                "frames": args.frames,
                "freq_bins": args.freq_bins,
                "params": count_parameters(model),
                "offline_rtf": offline_seconds / audio_seconds,
                "offline_ms": offline_seconds * 1000.0,
                "stream_rtf": stream_seconds / audio_seconds,
                "stream_ms": stream_seconds * 1000.0,
                "cache_tensors": len(cache),
                "cache_names": "|".join(stream_model.get_cache_names()),
                "offline_python_peak_mb": offline_py_mb,
                "stream_python_peak_mb": stream_py_mb,
                "offline_rss_mb": offline_rss_mb,
                "stream_rss_mb": stream_rss_mb,
                "offline_cuda_peak_mb": offline_cuda_mb,
                "stream_cuda_peak_mb": stream_cuda_mb,
            }
            row.update(reproducibility_metadata(get_train_config(config_id)))
            rows.append(row)
            print(
                f"{config_id} [{device}]: params={row['params']} "
                f"offline_rtf={row['offline_rtf']:.4f} stream_rtf={row['stream_rtf']:.4f} "
                f"cache_tensors={row['cache_tensors']}"
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config_id",
        "architecture",
        "device",
        "git_commit",
        "config_hash",
        "seed",
        "hardware_info",
        "torch_version",
        "onnxruntime_version",
        "num_threads",
        "checkpoint_id",
        "frames",
        "freq_bins",
        "params",
        "offline_rtf",
        "offline_ms",
        "stream_rtf",
        "stream_ms",
        "cache_tensors",
        "cache_names",
        "offline_python_peak_mb",
        "stream_python_peak_mb",
        "offline_rss_mb",
        "stream_rss_mb",
        "offline_cuda_peak_mb",
        "stream_cuda_peak_mb",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

