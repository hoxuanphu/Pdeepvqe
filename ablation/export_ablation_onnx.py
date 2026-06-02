"""Export stateful streaming DeepVQE ablation checkpoints to ONNX."""

import argparse
import csv
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ablation.ablation_config import get_train_config, reproducibility_metadata
from ablation.deepvqe_ablation import (
    DeepVQE_Ablation,
    StreamDeepVQE_Ablation,
    StreamDeepVQE_AblationONNXWrapper,
    convert_ablation_to_stream,
    stream_sequence,
)


def load_model(checkpoint_path, config_id, device):
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    cfg = ckpt.get("config", get_train_config(config_id))
    model = DeepVQE_Ablation(**cfg["model"]).eval().to(device)
    model.load_state_dict(ckpt["model"])
    return model, cfg


def append_result(output_path, row):
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
        "onnx_path",
        "onnx_simplified_path",
        "onnx_latency_ms_per_frame",
        "onnx_export_pass",
        "onnx_simplify_pass",
        "onnx_parity_error",
        "onnx_parity_pass",
        "notes",
    ]
    rows = []
    if output_path.exists():
        with output_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    rows = [item for item in rows if item.get("config_id") != row["config_id"]]
    rows.append(row)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cache_feed(cache):
    return {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(StreamDeepVQE_Ablation.cache_names, cache)
    }


def main():
    parser = argparse.ArgumentParser(description="Export stateful DeepVQE ablation to ONNX")
    parser.add_argument("--config-id", default="Baseline")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="onnx_models/ablation")
    parser.add_argument("--results", default="results/ablation_onnx.csv")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-abs-error", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, cfg = load_model(args.checkpoint, args.config_id, device)
    stream_model = StreamDeepVQE_Ablation(**cfg["model"]).eval().to(device)
    convert_ablation_to_stream(stream_model, model, strict=True)
    wrapper = StreamDeepVQE_AblationONNXWrapper(stream_model).eval().to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{args.config_id}_stream.onnx"
    simple_path = output_dir / f"{args.config_id}_stream_simple.onnx"
    dummy_frame = torch.randn(1, 257, 1, 2, device=device)
    dummy_cache = stream_model.init_cache(1, 257, device=device)
    notes = []
    export_pass = False
    simplify_pass = False
    parity_error = ""
    latency_ms = ""

    input_names = ["mix", *StreamDeepVQE_Ablation.cache_names]
    output_names = ["enh", *[f"{name}_out" for name in StreamDeepVQE_Ablation.cache_names]]

    try:
        torch.onnx.export(
            wrapper,
            (dummy_frame, *dummy_cache),
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=args.opset,
            verbose=False,
        )
        export_pass = True
    except Exception as exc:
        notes.append(f"export failed: {exc}")

    run_path = onnx_path
    if export_pass:
        try:
            import onnx
            from onnxsim import simplify

            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            model_simp, check = simplify(onnx_model)
            simplify_pass = bool(check)
            if simplify_pass:
                onnx.save(model_simp, str(simple_path))
                run_path = simple_path
        except Exception as exc:
            notes.append(f"simplify failed: {exc}")

    parity_pass = False
    if export_pass:
        try:
            import onnxruntime

            session = onnxruntime.InferenceSession(str(run_path), None, providers=["CPUExecutionProvider"])
            input_seq = torch.randn(1, 257, args.frames, 2, device=device)
            with torch.no_grad():
                torch_stream, _ = stream_sequence(stream_model, input_seq)

            ort_cache = stream_model.init_cache(1, 257, device=device)
            ort_outputs = []
            input_np = input_seq.detach().cpu().numpy()

            for _ in range(3):
                warm_cache = stream_model.init_cache(1, 257, device=device)
                for frame_idx in range(args.frames):
                    feed = {"mix": input_np[:, :, frame_idx : frame_idx + 1, :], **cache_feed(warm_cache)}
                    values = session.run(output_names, feed)
                    warm_cache = [torch.from_numpy(value) for value in values[1:]]

            start = time.perf_counter()
            for frame_idx in range(args.frames):
                feed = {"mix": input_np[:, :, frame_idx : frame_idx + 1, :], **cache_feed(ort_cache)}
                values = session.run(output_names, feed)
                ort_outputs.append(values[0])
                ort_cache = [torch.from_numpy(value) for value in values[1:]]
            latency_ms = (time.perf_counter() - start) * 1000.0 / args.frames

            ort_out = np.concatenate(ort_outputs, axis=2)
            parity_error = float(np.max(np.abs(torch_stream.detach().cpu().numpy() - ort_out)))
            parity_pass = parity_error <= args.max_abs_error
        except Exception as exc:
            notes.append(f"onnxruntime parity failed: {exc}")

    checkpoint_id = Path(args.checkpoint).stem
    row = {
        "config_id": args.config_id,
        "onnx_path": str(onnx_path) if export_pass else "",
        "onnx_simplified_path": str(simple_path) if simplify_pass else "",
        "onnx_latency_ms_per_frame": latency_ms,
        "onnx_export_pass": export_pass,
        "onnx_simplify_pass": simplify_pass,
        "onnx_parity_error": parity_error,
        "onnx_parity_pass": parity_pass,
        "notes": " | ".join(notes),
    }
    row.update(reproducibility_metadata(cfg, checkpoint_id=checkpoint_id))
    append_result(Path(args.results), row)
    print(row)


if __name__ == "__main__":
    main()
