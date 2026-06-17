"""Merge ablation CSV artifacts into ``results/ablation_summary.csv``."""

import argparse
import csv
from pathlib import Path


SUMMARY_COLUMNS = [
    "config_id",
    "config_path",
    "config_hash",
    "seed",
    "git_commit",
    "hardware_info",
    "torch_version",
    "onnxruntime_version",
    "num_threads",
    "checkpoint_id",
    "num_eval_items",
    "eval_duration_s",
    "params",
    "params_delta_pct",
    "macs",
    "macs_delta_pct",
    "flops",
    "streaming_rtf",
    "rtf_delta_pct",
    "onnx_latency_ms_per_frame",
    "onnx_latency_delta_pct",
    "causality_error",
    "pesq",
    "pesq_delta",
    "stoi",
    "stoi_delta",
    "si_sdr",
    "si_sdr_delta",
    "dnsmos_ovrl",
    "dnsmos_ovrl_delta",
    "dnsmos_sig",
    "dnsmos_bak",
    "erle",
    "onnx_export_pass",
    "onnx_simplify_pass",
    "onnx_parity_error",
    "onnx_parity_pass",
    "pass",
    "win",
    "notes",
]


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        return {row["config_id"]: row for row in csv.DictReader(f)}


def to_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def pct_delta(value, baseline):
    value = to_float(value)
    baseline = to_float(baseline)
    if value is None or baseline in (None, 0.0):
        return ""
    return 100.0 * (value - baseline) / baseline


def main():
    parser = argparse.ArgumentParser(description="Collect ablation result CSV files")
    parser.add_argument("--arch", default="results/ablation_arch_benchmark.csv")
    parser.add_argument("--quality", default="results/ablation_quality.csv")
    parser.add_argument("--onnx", default="results/ablation_onnx.csv")
    parser.add_argument("--output", default="results/ablation_summary.csv")
    args = parser.parse_args()

    arch = read_csv(args.arch)
    quality = read_csv(args.quality)
    onnx = read_csv(args.onnx)
    config_ids = sorted(set(arch) | set(quality) | set(onnx))

    baseline = arch.get("Baseline", {})
    rows = []
    for config_id in config_ids:
        merged = {column: "" for column in SUMMARY_COLUMNS}
        merged["config_id"] = config_id
        notes = []
        for source in (arch.get(config_id, {}), quality.get(config_id, {}), onnx.get(config_id, {})):
            for key, value in source.items():
                if key in merged and value not in (None, ""):
                    merged[key] = value
            if source.get("notes"):
                notes.append(source["notes"])

        merged["params_delta_pct"] = pct_delta(merged["params"], baseline.get("params"))
        merged["macs_delta_pct"] = pct_delta(merged["macs"], baseline.get("macs"))
        merged["rtf_delta_pct"] = pct_delta(merged["streaming_rtf"], baseline.get("streaming_rtf"))
        quality_baseline = quality.get("Baseline", {})
        for metric, delta_column in (
            ("pesq", "pesq_delta"),
            ("stoi", "stoi_delta"),
            ("si_sdr", "si_sdr_delta"),
            ("dnsmos_ovrl", "dnsmos_ovrl_delta"),
        ):
            value = to_float(merged.get(metric))
            base_value = to_float(quality_baseline.get(metric))
            if value is not None and base_value is not None:
                merged[delta_column] = value - base_value
        if merged["notes"] and merged["notes"] not in notes:
            notes.append(merged["notes"])
        merged["notes"] = " | ".join(note for note in notes if note)
        rows.append(merged)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
