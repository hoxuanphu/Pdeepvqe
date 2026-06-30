"""Checkpoint loading helpers shared by evaluation and ONNX export."""

import json
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn


def load_sidecar_config(checkpoint_path):
    config_path = Path(checkpoint_path).parent / "config.json"
    if not config_path.exists():
        return None, None
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f), config_path


def apply_notebook_config(cfg, flat_cfg):
    cfg = deepcopy(cfg)
    stft = cfg.setdefault("stft", {})
    data = cfg.setdefault("data", {})
    experiment = cfg.setdefault("experiment", {})

    for source_key, target_key in (
        ("n_fft", "n_fft"),
        ("hop_length", "hop_length"),
        ("win_length", "win_length"),
    ):
        if source_key in flat_cfg:
            stft[target_key] = flat_cfg[source_key]
    if "stft_window" in flat_cfg:
        stft["window"] = flat_cfg["stft_window"]
    if "sample_rate" in flat_cfg:
        data["sample_rate"] = flat_cfg["sample_rate"]
    for source_key, target_key in (
        ("train_csv", "train_manifest"),
        ("valid_csv", "valid_manifest"),
        ("test_csv", "test_manifest"),
    ):
        if source_key in flat_cfg:
            data[target_key] = flat_cfg[source_key]
    if "seed" in flat_cfg:
        experiment["seed"] = flat_cfg["seed"]
    return cfg


def torch_load_checkpoint(checkpoint_path, device):
    try:
        return torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(checkpoint_path), map_location=device)


def looks_like_state_dict(value):
    return (
        isinstance(value, dict)
        and len(value) > 0
        and all(torch.is_tensor(item) or isinstance(item, nn.Parameter) for item in value.values())
    )


def extract_state_dict(ckpt):
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict()
    if looks_like_state_dict(ckpt):
        return ckpt
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(ckpt).__name__}")

    for key in ("model", "model_state_dict", "state_dict", "net", "network"):
        value = ckpt.get(key)
        if isinstance(value, nn.Module):
            return value.state_dict()
        if looks_like_state_dict(value):
            return value

    keys = sorted(str(key) for key in ckpt.keys())
    raise KeyError(f"Checkpoint does not contain a model state_dict. Available keys: {keys[:20]}")


def state_dict_variants(state_dict):
    yield "original", state_dict
    prefixes = ("module.", "model.", "mods.model.", "module.model.", "hparams.model.")
    for prefix in prefixes:
        if any(key.startswith(prefix) for key in state_dict):
            stripped = {
                key[len(prefix):] if key.startswith(prefix) else key: value
                for key, value in state_dict.items()
            }
            yield f"strip {prefix!r}", stripped


def load_state_dict_flexible(model, state_dict):
    target_keys = set(model.state_dict())
    failures = []
    best = None

    for label, candidate in state_dict_variants(state_dict):
        candidate_keys = set(candidate)
        overlap = len(target_keys & candidate_keys)
        if best is None or overlap > best[0]:
            best = (overlap, label, candidate)
        try:
            model.load_state_dict(candidate, strict=True)
            return [f"loaded checkpoint with {label} keys"]
        except RuntimeError as exc:
            failures.append(f"{label}: {exc}")

    if best is not None and best[0] > 0:
        _, label, candidate = best
        missing, unexpected = model.load_state_dict(candidate, strict=False)
        if not missing:
            note = f"loaded checkpoint with {label} keys; ignored {len(unexpected)} unexpected keys"
            return [note]
        failures.append(
            f"best partial match {label}: missing={list(missing)[:10]}, unexpected={list(unexpected)[:10]}"
        )

    checkpoint_keys = sorted(str(key) for key in state_dict.keys())[:20]
    model_keys = sorted(str(key) for key in model.state_dict().keys())[:20]
    detail = "\n".join(failures[-3:])
    raise RuntimeError(
        "Unable to load checkpoint into DeepVQE_Ablation.\n"
        f"Checkpoint key sample: {checkpoint_keys}\n"
        f"Model key sample: {model_keys}\n"
        f"Recent load errors:\n{detail}"
    )
