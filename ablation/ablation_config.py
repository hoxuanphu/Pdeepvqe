from copy import deepcopy
import hashlib
import json
import os
import platform
import subprocess

import torch


BASE_TRAIN_CONFIG = {
    "experiment": {
        "name": "deepvqe_ablation",
        "config_id": "Baseline",
        "seed": 1337,
        "output_dir": "runs/ablation/Baseline",
        "resume_from": None,
    },
    "model": {
        "prelu_type": None,
        "dw_residual": False,
        "use_eca_f": False,
        "main_block_eca_f": False,
        "gru_hidden": 576,
    },
    "stft": {
        "n_fft": 512,
        "hop_length": 256,
        "win_length": 512,
        "window": "sqrt_hann",
    },
    "data": {
        "sample_rate": 16000,
        "clip_seconds": 3.0,
        "num_workers": 2,
        "pin_memory": True,
        "train_manifest": "data/manifests/train.jsonl",
        "valid_manifest": "data/manifests/valid.jsonl",
        "test_manifest": "data/manifests/test.jsonl",
        "augment": False,
        "aug_gain_range_db": [-6.0, 6.0],
        "aug_snr_remix_range": [0.0, 20.0],
        "aug_prob": 0.5,
    },
    "optimizer": {
        "lr": 1e-3,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "grad_clip_norm": 5.0,
    },
    "scheduler": {
        "mode": "min",
        "factor": 0.5,
        "patience": 5,
        "min_lr": 1e-6,
    },
    "training": {
        "device": "cuda",
        "batch_size": 4,
        "epochs": 100,
        "checkpoint_monitor": "loss",
        "checkpoint_mode": "min",
        "use_amp": True,
    },
    "loss": {
        "lamda_ri": 30.0,
        "lamda_mag": 70.0,
        "compress_factor": 0.3,
    },
}


def deep_update(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def get_train_config(config_id="Baseline"):
    from ablation.deepvqe_ablation import get_ablation_config

    cfg = deepcopy(BASE_TRAIN_CONFIG)
    cfg["experiment"]["config_id"] = config_id
    cfg["experiment"]["name"] = f"deepvqe_ablation_{config_id}"
    cfg["experiment"]["output_dir"] = f"runs/ablation/{config_id}"
    cfg["model"] = get_ablation_config(config_id)
    return cfg


def stable_config_hash(cfg):
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def hardware_info():
    cuda_name = ""
    if torch.cuda.is_available():
        try:
            cuda_name = torch.cuda.get_device_name(0)
        except Exception:
            cuda_name = "cuda"
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": cuda_name,
    }


def reproducibility_metadata(cfg=None, checkpoint_id=""):
    try:
        import onnxruntime

        ort_version = onnxruntime.__version__
    except Exception:
        ort_version = ""

    return {
        "git_commit": git_commit(),
        "config_hash": stable_config_hash(cfg) if cfg is not None else "",
        "seed": cfg.get("experiment", {}).get("seed", "") if cfg is not None else "",
        "hardware_info": json.dumps(hardware_info(), sort_keys=True),
        "torch_version": torch.__version__,
        "onnxruntime_version": ort_version,
        "num_threads": torch.get_num_threads(),
        "checkpoint_id": checkpoint_id,
    }
