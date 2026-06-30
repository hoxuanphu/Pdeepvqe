"""Evaluate trained DeepVQE ablation checkpoints on paired waveform manifests."""

import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ablation.ablation_config import get_train_config, reproducibility_metadata
from ablation.checkpoint_utils import (
    apply_notebook_config,
    extract_state_dict,
    load_sidecar_config,
    load_state_dict_flexible,
    torch_load_checkpoint,
)
from ablation.deepvqe_ablation import DeepVQE_Ablation
from ablation.train_ablation import (
    load_audio,
    make_istft,
    make_stft,
    read_json_manifest,
    repeat_to_length,
    resolve_path,
    si_sdr,
)


def optional_metric_modules():
    modules = {}
    try:
        from pesq import pesq
        modules["pesq"] = pesq
    except ImportError:
        modules["pesq"] = None
    try:
        from pystoi import stoi
        modules["stoi"] = stoi
    except ImportError:
        modules["stoi"] = None
    return modules


def pick_path(record, names):
    for name in names:
        if record.get(name):
            return record[name]
    raise KeyError(f"Missing one of {names}: {record}")


def align_length(wav, length):
    if wav.numel() < length:
        return repeat_to_length(wav, length)
    return wav[:length]


def make_window(cfg, device):
    stft_cfg = cfg["stft"]
    name = str(stft_cfg.get("window", "hann")).lower().replace("-", "_")
    window = torch.hann_window(int(stft_cfg["win_length"]), device=device)
    if name in ("hann", "hanning"):
        return window
    if name in ("sqrt_hann", "sqrt_hanning"):
        return window.sqrt()
    raise ValueError(f"Unsupported STFT window: {stft_cfg.get('window')!r}")


def load_checkpoint_model(checkpoint_path, config_id, device):
    ckpt = torch_load_checkpoint(checkpoint_path, "cpu")
    cfg = ckpt.get("config", get_train_config(config_id)) if isinstance(ckpt, dict) else get_train_config(config_id)
    config_notes = []
    if not (isinstance(ckpt, dict) and "config" in ckpt):
        sidecar_cfg, sidecar_path = load_sidecar_config(checkpoint_path)
        if sidecar_cfg is not None:
            cfg = apply_notebook_config(cfg, sidecar_cfg)
            config_notes.append(f"loaded eval config from {sidecar_path.name}")
    model_cfg = cfg.get("model", get_train_config(config_id)["model"])
    model = DeepVQE_Ablation(**model_cfg).to(device).eval()
    load_notes = config_notes + load_state_dict_flexible(model, extract_state_dict(ckpt))
    metadata = ckpt.get("metadata", {}) if isinstance(ckpt, dict) else {}
    return model, cfg, metadata, load_notes


@torch.no_grad()
def enhance(model, mixture, cfg, window, device):
    mixture = mixture.unsqueeze(0).to(device)
    spec = make_stft(mixture, cfg, window)
    enhanced_spec = model(spec)
    enhanced = make_istft(enhanced_spec, cfg, window, mixture.shape[-1])
    return enhanced.squeeze(0).detach().cpu()


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepVQE ablation quality")
    parser.add_argument("--config-id", default="Baseline")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", default="results/ablation_quality.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, cfg, ckpt_metadata, load_notes = load_checkpoint_model(args.checkpoint, args.config_id, device)
    manifest = args.manifest or cfg["data"].get("test_manifest") or cfg["data"].get("valid_manifest")
    records = read_json_manifest(manifest)
    sample_rate = int(cfg["data"]["sample_rate"])
    window = make_window(cfg, device)
    metric_modules = optional_metric_modules()

    values = {"si_sdr": [], "pesq": [], "stoi": []}
    eval_duration_samples = 0
    notes = list(load_notes)
    notes.append(f"STFT window={cfg['stft'].get('window', 'hann')}")
    if metric_modules["pesq"] is None:
        notes.append("pesq package not installed")
    if metric_modules["stoi"] is None:
        notes.append("pystoi package not installed")
    notes.append("DNSMOS and ERLE require external evaluators and are not computed by this script")

    for record in records:
        mixture = load_audio(resolve_path(pick_path(record, ["mixture", "mixture_path", "mix", "mix_path", "noisy", "noisy_wav"]), record, args.data_root), sample_rate)
        target = load_audio(resolve_path(pick_path(record, ["target", "target_path", "clean", "target_wav", "clean_wav"]), record, args.data_root), sample_rate)
        length = min(mixture.numel(), target.numel())
        eval_duration_samples += int(length)
        mixture = align_length(mixture, length)
        target = align_length(target, length)
        enhanced = enhance(model, mixture, cfg, window, device)
        enhanced = align_length(enhanced, target.numel())

        values["si_sdr"].append(float(si_sdr(enhanced.unsqueeze(0), target.unsqueeze(0)).mean()))
        target_np = target.numpy()
        enhanced_np = enhanced.numpy()
        if metric_modules["pesq"] is not None:
            try:
                values["pesq"].append(float(metric_modules["pesq"](sample_rate, target_np, enhanced_np, "wb")))
            except Exception as exc:
                notes.append(f"PESQ failed for one item: {exc}")
        if metric_modules["stoi"] is not None:
            try:
                values["stoi"].append(float(metric_modules["stoi"](target_np, enhanced_np, sample_rate, extended=False)))
            except Exception as exc:
                notes.append(f"STOI failed for one item: {exc}")

    checkpoint_id = Path(args.checkpoint).stem
    row = {
        "config_id": args.config_id,
        "checkpoint_id": checkpoint_id,
        "num_eval_items": len(records),
        "eval_duration_s": eval_duration_samples / sample_rate,
        "pesq": np.mean(values["pesq"]) if values["pesq"] else "",
        "stoi": np.mean(values["stoi"]) if values["stoi"] else "",
        "si_sdr": np.mean(values["si_sdr"]) if values["si_sdr"] else "",
        "dnsmos_ovrl": "",
        "dnsmos_sig": "",
        "dnsmos_bak": "",
        "erle": "",
        "notes": " | ".join(dict.fromkeys(notes)),
    }
    metadata = reproducibility_metadata(cfg, checkpoint_id=checkpoint_id)
    metadata.update({key: value for key, value in ckpt_metadata.items() if value not in (None, "")})
    row.update(metadata)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if output_path.exists():
        with output_path.open("r", newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
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
        "num_eval_items",
        "eval_duration_s",
        "pesq",
        "stoi",
        "si_sdr",
        "dnsmos_ovrl",
        "dnsmos_sig",
        "dnsmos_bak",
        "erle",
        "notes",
    ]
    existing = [item for item in existing if item.get("config_id") != args.config_id]
    existing.append(row)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)

    print(json.dumps(row, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
