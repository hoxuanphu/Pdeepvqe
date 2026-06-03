"""Train a DeepVQE ablation variant from paired waveform manifests.

Manifest records may use keys such as ``mixture``/``target`` or
``mixture_path``/``target_path``. JSON and JSONL manifests are supported.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from ablation.ablation_config import deep_update, reproducibility_metadata, get_train_config
from ablation.deepvqe_ablation import DeepVQE_Ablation


PATH_KEYS = {
    "mixture": ["mixture", "mixture_path", "mix", "mix_path", "noisy", "input", "noisy_wav"],
    "target": ["target", "target_path", "clean", "target_reverb", "target_wav", "clean_wav"],
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json_manifest(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(f"JSON manifest must contain a list: {path}")
    elif path.suffix.lower() == ".csv":
        import csv
        with path.open("r", encoding="utf-8", newline='') as f:
            reader = csv.DictReader(f)
            records = list(reader)
    else:
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    for record in records:
        record["_manifest_dir"] = str(path.parent)
    return records


def pick_key(record, group):
    for key in PATH_KEYS[group]:
        if record.get(key):
            return record[key]
    raise KeyError(f"Manifest item is missing one of {PATH_KEYS[group]}: {record}")


def resolve_path(value, record, data_root=None):
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = []
    if data_root:
        candidates.append(Path(data_root) / path)
    candidates.append(Path(record["_manifest_dir"]) / path)
    candidates.append(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_audio(path, sample_rate):
    wav, sr = torchaudio.load(str(path))
    wav = wav.float()
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.squeeze(0)


def repeat_to_length(wav, length):
    if wav.numel() == 0:
        raise ValueError("Encountered empty waveform")
    if wav.numel() >= length:
        return wav
    repeats = int(np.ceil(length / wav.numel()))
    return wav.repeat(repeats)[:length]


def crop_pair(mixture, target, length, random_crop):
    min_len = min(mixture.numel(), target.numel())
    mixture = mixture[:min_len]
    target = target[:min_len]
    if min_len < length:
        return repeat_to_length(mixture, length), repeat_to_length(target, length)
    start = random.randint(0, min_len - length) if random_crop else max(0, (min_len - length) // 2)
    return mixture[start:start + length], target[start:start + length]


class PairedWaveDataset(Dataset):
    def __init__(self, manifest, cfg, split, data_root=None):
        self.records = read_json_manifest(manifest)
        self.cfg = cfg
        self.split = split
        self.data_root = data_root
        self.sample_rate = int(cfg["data"]["sample_rate"])
        self.clip_samples = int(float(cfg["data"]["clip_seconds"]) * self.sample_rate)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        mixture = load_audio(resolve_path(pick_key(record, "mixture"), record, self.data_root), self.sample_rate)
        target = load_audio(resolve_path(pick_key(record, "target"), record, self.data_root), self.sample_rate)
        mixture, target = crop_pair(mixture, target, self.clip_samples, self.split == "train")
        return {"mixture": mixture, "target": target}


def collate_batch(items):
    return {
        "mixture": torch.stack([item["mixture"] for item in items], dim=0),
        "target": torch.stack([item["target"] for item in items], dim=0),
    }


def make_stft(wav, cfg, window):
    stft_cfg = cfg["stft"]
    spec = torch.stft(
        wav,
        n_fft=int(stft_cfg["n_fft"]),
        hop_length=int(stft_cfg["hop_length"]),
        win_length=int(stft_cfg["win_length"]),
        window=window,
        return_complex=True,
    )
    return torch.view_as_real(spec)


def make_istft(spec, cfg, window, length):
    stft_cfg = cfg["stft"]
    complex_spec = torch.complex(spec[..., 0], spec[..., 1])
    return torch.istft(
        complex_spec,
        n_fft=int(stft_cfg["n_fft"]),
        hop_length=int(stft_cfg["hop_length"]),
        win_length=int(stft_cfg["win_length"]),
        window=window,
        length=length,
    )


def si_sdr(estimate, target, eps=1e-8):
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection = (estimate * target).sum(dim=-1, keepdim=True) * target / (target.pow(2).sum(dim=-1, keepdim=True) + eps)
    noise = estimate - projection
    return 10 * torch.log10((projection.pow(2).sum(dim=-1) + eps) / (noise.pow(2).sum(dim=-1) + eps))


def magnitude_l1(estimate_spec, target_spec):
    estimate_mag = torch.sqrt(estimate_spec[..., 0].pow(2) + estimate_spec[..., 1].pow(2) + 1e-12)
    target_mag = torch.sqrt(target_spec[..., 0].pow(2) + target_spec[..., 1].pow(2) + 1e-12)
    return torch.mean(torch.abs(estimate_mag - target_mag))


def compute_batch(model, batch, cfg, window, device):
    mixture = batch["mixture"].to(device)
    target = batch["target"].to(device)
    mixture_spec = make_stft(mixture, cfg, window)
    target_spec = make_stft(target, cfg, window)
    estimate_spec = model(mixture_spec)
    estimate = make_istft(estimate_spec, cfg, window, target.shape[-1])
    loss = (
        float(cfg["loss"]["si_sdr_weight"]) * (-si_sdr(estimate, target).mean())
        + float(cfg["loss"]["magnitude_l1_weight"]) * magnitude_l1(estimate_spec, target_spec)
    )
    metrics = {
        "si_sdr": float(si_sdr(estimate, target).mean().detach().cpu()),
        "si_sdri": float((si_sdr(estimate, target) - si_sdr(mixture, target)).mean().detach().cpu()),
    }
    return loss, metrics


def average(items):
    keys = sorted({key for item in items for key in item})
    return {key: sum(item[key] for item in items if key in item) / max(1, sum(key in item for item in items)) for key in keys}


def make_loader(manifest, cfg, split, data_root):
    dataset = PairedWaveDataset(manifest, cfg, split, data_root=data_root)
    return DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=split == "train",
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        drop_last=split == "train",
        collate_fn=collate_batch,
    )


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def save_checkpoint(path, model, optimizer, scheduler, cfg, epoch, best_metric, bad_epochs=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = reproducibility_metadata(cfg, checkpoint_id=path.stem)
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "config": cfg,
            "metadata": metadata,
            "epoch": epoch,
            "best_metric": best_metric,
            "bad_epochs": bad_epochs,
        },
        str(path),
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    try:
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(path), map_location=device)
    state = ckpt["model"]
    target = unwrap_model(model)
    try:
        target.load_state_dict(state)
    except RuntimeError:
        if all(key.startswith("module.") for key in state):
            state = {key.replace("module.", "", 1): value for key, value in state.items()}
            target.load_state_dict(state)
        else:
            raise
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("epoch", 0)), ckpt.get("best_metric"), int(ckpt.get("bad_epochs", 0))


def make_model(cfg, device, data_parallel=False):
    model = DeepVQE_Ablation(**cfg["model"]).to(device)
    if data_parallel:
        if device.type != "cuda" or torch.cuda.device_count() < 2:
            print("--data-parallel requested, but fewer than 2 CUDA GPUs are available; using single-device training.", flush=True)
        else:
            print(f"Using DataParallel on {torch.cuda.device_count()} CUDA GPUs", flush=True)
            model = torch.nn.DataParallel(model)
    return model


def make_optimizer_scheduler(model, cfg):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
        betas=tuple(cfg["optimizer"]["betas"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg["scheduler"]["mode"],
        factor=float(cfg["scheduler"]["factor"]),
        patience=int(cfg["scheduler"]["patience"]),
        min_lr=float(cfg["scheduler"]["min_lr"]),
    )
    return optimizer, scheduler


def run_epoch(model, loader, cfg, window, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    values = []
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="train" if train else "valid", dynamic_ncols=True, leave=False, ascii=True)
    for batch in iterator:
        if train:
            optimizer.zero_grad()
        loss, metrics = compute_batch(model, batch, cfg, window, device)
        if train:
            loss.backward()
            grad_clip = cfg["optimizer"].get("grad_clip_norm")
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
        item = {"loss": float(loss.detach().cpu()), **metrics}
        values.append(item)
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix({key: f"{value:.4g}" for key, value in item.items()})
    return average(values)


def main():
    parser = argparse.ArgumentParser(description="Train DeepVQE ablation")
    parser.add_argument("--config-id", default="Baseline")
    parser.add_argument("--config-json", default=None, help="Optional JSON override file")
    parser.add_argument("--config-yaml", default=None, help="Optional YAML override file; requires PyYAML")
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--valid-manifest", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--data-parallel", action="store_true", help="Use torch.nn.DataParallel when multiple CUDA GPUs are available")
    parser.add_argument("--ignore-bad-resume", action="store_true", help="Start from scratch if the resume checkpoint cannot be loaded")
    parser.add_argument("--early-stop-patience", type=int, default=None, help="Stop if the monitored validation metric does not improve for this many epochs")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0, help="Minimum monitored metric improvement required to reset early-stop patience")
    parser.add_argument("--early-stop-min-epochs", type=int, default=0, help="Do not early-stop before this epoch")
    args = parser.parse_args()

    cfg = get_train_config(args.config_id)
    if args.config_json:
        with open(args.config_json, "r", encoding="utf-8") as f:
            cfg = deep_update(cfg, json.load(f))
    if args.config_yaml:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("--config-yaml requires PyYAML. Install pyyaml or use --config-json.") from exc
        with open(args.config_yaml, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        cfg = deep_update(cfg, yaml_cfg)
    if args.train_manifest:
        cfg["data"]["train_manifest"] = args.train_manifest
    if args.valid_manifest:
        cfg["data"]["valid_manifest"] = args.valid_manifest
    if args.output_dir:
        cfg["experiment"]["output_dir"] = args.output_dir
    if args.device:
        cfg["training"]["device"] = args.device
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.resume:
        cfg["experiment"]["resume_from"] = args.resume
    cfg["training"]["data_parallel"] = bool(args.data_parallel)
    if args.early_stop_patience is not None:
        cfg["training"]["early_stop_patience"] = args.early_stop_patience
    cfg["training"]["early_stop_min_delta"] = float(args.early_stop_min_delta)
    cfg["training"]["early_stop_min_epochs"] = int(args.early_stop_min_epochs)

    seed_everything(int(cfg["experiment"]["seed"]))
    requested_device = cfg["training"]["device"]
    device = torch.device(requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    train_loader = make_loader(cfg["data"]["train_manifest"], cfg, "train", args.data_root)
    valid_loader = make_loader(cfg["data"]["valid_manifest"], cfg, "valid", args.data_root)
    model = make_model(cfg, device, data_parallel=args.data_parallel)
    optimizer, scheduler = make_optimizer_scheduler(model, cfg)
    window = torch.hann_window(int(cfg["stft"]["win_length"]), device=device)

    start_epoch = 0
    best_metric = None
    bad_epochs = 0
    if cfg["experiment"].get("resume_from"):
        try:
            start_epoch, best_metric, bad_epochs = load_checkpoint(cfg["experiment"]["resume_from"], model, optimizer, scheduler, device)
            print(
                f"Resumed from {cfg['experiment']['resume_from']} at epoch={start_epoch} "
                f"best_metric={best_metric} bad_epochs={bad_epochs}",
                flush=True,
            )
        except Exception as exc:
            if not args.ignore_bad_resume:
                raise
            print(
                f"WARNING: unable to resume from {cfg['experiment']['resume_from']}; "
                f"starting {args.config_id} from scratch. Error: {exc}",
                flush=True,
            )
            cfg["experiment"]["resume_from"] = None
            model = make_model(cfg, device, data_parallel=args.data_parallel)
            optimizer, scheduler = make_optimizer_scheduler(model, cfg)
            start_epoch = 0
            best_metric = None
            bad_epochs = 0

    monitor = cfg["training"]["checkpoint_monitor"]
    mode = cfg["training"]["checkpoint_mode"]
    early_stop_patience = cfg["training"].get("early_stop_patience")
    early_stop_min_delta = float(cfg["training"].get("early_stop_min_delta", 0.0))
    early_stop_min_epochs = int(cfg["training"].get("early_stop_min_epochs", 0))
    for epoch in range(start_epoch + 1, int(cfg["training"]["epochs"]) + 1):
        start = time.time()
        train_metrics = run_epoch(model, train_loader, cfg, window, device, optimizer)
        with torch.no_grad():
            valid_metrics = run_epoch(model, valid_loader, cfg, window, device)
        monitor_value = valid_metrics.get(monitor, -valid_metrics["loss"])
        scheduler.step(monitor_value)
        previous_best = best_metric
        if mode == "max":
            is_best = previous_best is None or monitor_value > previous_best
            improved_enough = previous_best is None or monitor_value > previous_best + early_stop_min_delta
        else:
            is_best = previous_best is None or monitor_value < previous_best
            improved_enough = previous_best is None or monitor_value < previous_best - early_stop_min_delta
        if is_best:
            best_metric = monitor_value
        if improved_enough:
            bad_epochs = 0
        else:
            bad_epochs += 1
        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, cfg, epoch, best_metric, bad_epochs)
        if is_best:
            save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, cfg, epoch, best_metric, bad_epochs)
        print(
            f"epoch={epoch} time={time.time() - start:.1f}s "
            f"train={train_metrics} valid={valid_metrics} best_{monitor}={best_metric} "
            f"bad_epochs={bad_epochs}",
            flush=True,
        )
        if early_stop_patience is not None and epoch >= early_stop_min_epochs and bad_epochs >= early_stop_patience:
            print(
                f"Early stopping {args.config_id}: {monitor} did not improve by "
                f"{early_stop_min_delta} for {bad_epochs} epochs "
                f"(patience={early_stop_patience}, best_{monitor}={best_metric}).",
                flush=True,
            )
            break


if __name__ == "__main__":
    main()
