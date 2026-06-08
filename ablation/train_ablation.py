"""Train a DeepVQE ablation variant from paired waveform manifests.

Manifest records may use keys such as ``mixture``/``target`` or
``mixture_path``/``target_path``. JSON and JSONL manifests are supported.
"""

import argparse
import gc
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


def compute_batch(model, batch, cfg, window, device):
    mixture = batch["mixture"].to(device)
    target = batch["target"].to(device)
    
    stft_cfg = cfg["stft"]
    loss_cfg = cfg["loss"]
    c = float(loss_cfg["compress_factor"])
    
    mixture_spec = make_stft(mixture, cfg, window)
    
    amp_enabled = bool(cfg["training"].get("use_amp", False)) and device.type == "cuda"
    with torch.amp.autocast("cuda", enabled=amp_enabled):
        estimate_spec = model(mixture_spec)
    
    estimate_spec = estimate_spec.float()
    target_spec = make_stft(target, cfg, window)
    
    min_t = min(estimate_spec.shape[2], target_spec.shape[2])
    estimate_spec = estimate_spec[:, :, :min_t, :]
    target_spec = target_spec[:, :, :min_t, :]
    
    est_real, est_imag = estimate_spec[..., 0], estimate_spec[..., 1]
    tgt_real, tgt_imag = target_spec[..., 0], target_spec[..., 1]
    
    est_mag = torch.sqrt(est_real**2 + est_imag**2 + 1e-12)
    tgt_mag = torch.sqrt(tgt_real**2 + tgt_imag**2 + 1e-12)
    
    est_real_c = est_real / (est_mag**(1 - c))
    est_imag_c = est_imag / (est_mag**(1 - c))
    tgt_real_c = tgt_real / (tgt_mag**(1 - c))
    tgt_imag_c = tgt_imag / (tgt_mag**(1 - c))
    
    real_loss = torch.mean((est_real_c - tgt_real_c)**2)
    imag_loss = torch.mean((est_imag_c - tgt_imag_c)**2)
    mag_loss = torch.mean((est_mag**c - tgt_mag**c)**2)
    
    estimate_wav = make_istft(estimate_spec, cfg, window, target.shape[-1])
    min_wav_len = min(estimate_wav.shape[-1], target.shape[-1])
    estimate_wav = estimate_wav[..., :min_wav_len]
    target_wav = target[..., :min_wav_len]
    
    eps = 1e-8
    true_energy = torch.sum(torch.square(target_wav), dim=-1, keepdim=True)
    y_target = torch.sum(target_wav * estimate_wav, dim=-1, keepdim=True) * target_wav / (true_energy + eps)
    target_energy = torch.sum(torch.square(y_target), dim=-1, keepdim=True)
    noise_energy = torch.sum(torch.square(estimate_wav - y_target), dim=-1, keepdim=True)
    sisnr = -torch.log10((target_energy + eps) / (noise_energy + eps)).mean()
    
    loss = float(loss_cfg["lamda_ri"]) * (real_loss + imag_loss) + float(loss_cfg["lamda_mag"]) * mag_loss + sisnr
    
    metrics = {
        "loss": float(loss.detach().cpu()),
        "ri_loss": float((real_loss + imag_loss).detach().cpu()),
        "mag_loss": float(mag_loss.detach().cpu()),
        "sisnr": float(sisnr.detach().cpu()),
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


def save_checkpoint(path, model, optimizer, scheduler, cfg, epoch, best_metric, bad_epochs=0, scaler=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = reproducibility_metadata(cfg, checkpoint_id=path.stem)
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "config": cfg,
            "metadata": metadata,
            "epoch": epoch,
            "best_metric": best_metric,
            "bad_epochs": bad_epochs,
        },
        str(path),
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu", scaler=None):
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
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
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
    optimizer = torch.optim.Adam(
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


def run_epoch(model, loader, cfg, window, device, optimizer=None, scaler=None, desc_str=""):
    train = optimizer is not None
    model.train(train)
    values = []
    iterator = loader
    
    accum_steps = int(cfg["training"].get("grad_accum_steps", 1)) if train else 1
    valid_accum_batches = 0
    progress_every = int(cfg["training"].get("progress_update_every", 1))

    if tqdm is not None and not cfg["training"].get("disable_tqdm", False):
        iterator = tqdm(loader, desc=desc_str, dynamic_ncols=True, leave=False, ascii=True)
        
    if train:
        optimizer.zero_grad(set_to_none=True)
        
    for batch_idx, batch in enumerate(iterator):
        loss, metrics = compute_batch(model, batch, cfg, window, device)
        
        if not torch.isfinite(loss):
            print(f"  [WARN] Skip batch {batch_idx}: non-finite loss", flush=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            continue
            
        if train:
            loss = loss / accum_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            valid_accum_batches += 1
            if valid_accum_batches % accum_steps == 0:
                grad_clip = cfg["optimizer"].get("grad_clip_norm")
                if scaler is not None:
                    if grad_clip:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
        item = {**metrics}
        values.append(item)
        if hasattr(iterator, "set_postfix") and (batch_idx % progress_every == 0 or batch_idx + 1 == len(loader)):
            iterator.set_postfix({key: f"{value:.4g}" for key, value in item.items()})
            
        del batch, loss, metrics, item
        
    if train and valid_accum_batches % accum_steps != 0:
        grad_clip = cfg["optimizer"].get("grad_clip_norm")
        if scaler is not None:
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
        else:
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
    gc.collect()
    if not values:
        return {"loss": float('nan'), "ri_loss": float('nan'), "mag_loss": float('nan'), "sisnr": float('nan')}
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
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable DataLoader pinned-memory buffers")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--data-parallel", action="store_true", help="Use torch.nn.DataParallel when multiple CUDA GPUs are available")
    parser.add_argument("--ignore-bad-resume", action="store_true", help="Start from scratch if the resume checkpoint cannot be loaded")
    parser.add_argument("--early-stop-patience", type=int, default=None, help="Stop if the monitored validation metric does not improve for this many epochs")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0, help="Minimum monitored metric improvement required to reset early-stop patience")
    parser.add_argument("--early-stop-min-epochs", type=int, default=0, help="Do not early-stop before this epoch")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm progress bars")
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
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.no_pin_memory:
        cfg["data"]["pin_memory"] = False
    if args.resume:
        cfg["experiment"]["resume_from"] = args.resume
    cfg["training"]["data_parallel"] = bool(args.data_parallel)
    if args.early_stop_patience is not None:
        cfg["training"]["early_stop_patience"] = args.early_stop_patience
    cfg["training"]["early_stop_min_delta"] = float(args.early_stop_min_delta)
    cfg["training"]["early_stop_min_epochs"] = int(args.early_stop_min_epochs)
    cfg["training"]["disable_tqdm"] = bool(args.disable_tqdm)

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
    window_name = cfg["stft"].get("window", "hann")
    window = torch.hann_window(int(cfg["stft"]["win_length"]), device=device)
    if window_name == "sqrt_hann":
        window = window.sqrt()

    use_amp = bool(cfg["training"].get("use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\n========================================", flush=True)
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}", flush=True)
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)
            
    total_params = sum(p.numel() for p in unwrap_model(model).parameters())
    trainable_params = sum(p.numel() for p in unwrap_model(model).parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.2f}M | Trainable: {trainable_params / 1e6:.2f}M", flush=True)
    print(f"Mixed Precision (AMP): {'ON' if use_amp else 'OFF'}", flush=True)
    
    aug_cfg = cfg["data"].get("augment", False)
    if aug_cfg:
        aug_prob = cfg["data"].get("aug_prob", 0.5)
        print(f"Augmentation: ON (prob={aug_prob})", flush=True)
    else:
        print("Augmentation: OFF", flush=True)
        
    print(f"Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches", flush=True)
    print(f"Valid: {len(valid_loader.dataset)} samples, {len(valid_loader)} batches", flush=True)
    print(f"========================================\n", flush=True)

    start_epoch = 0
    best_metric = None
    bad_epochs = 0
    if cfg["experiment"].get("resume_from"):
        try:
            start_epoch, best_metric, bad_epochs = load_checkpoint(cfg["experiment"]["resume_from"], model, optimizer, scheduler, device, scaler=scaler)
            print(
                f"Resumed from {cfg['experiment']['resume_from']} at epoch={start_epoch} "
                f"best_metric={best_metric} bad_epochs={bad_epochs}",
                flush=True,
            )
            
            # Tự động chọn LR thấp nhất giữa checkpoint và cấu hình
            config_lr = float(cfg["optimizer"]["lr"])
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
                if config_lr < current_lr:
                    print(f"Bắt buộc hạ LR từ {current_lr} xuống {config_lr} theo cấu hình mới.", flush=True)
                    param_group['lr'] = config_lr

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
        epoch_str_train = f"Epoch {epoch:>2} [Train]"
        epoch_str_valid = f"Epoch {epoch:>2} [Valid]"
        train_metrics = run_epoch(model, train_loader, cfg, window, device, optimizer, scaler, desc_str=epoch_str_train)
        with torch.no_grad():
            valid_metrics = run_epoch(model, valid_loader, cfg, window, device, desc_str=epoch_str_valid)
        monitor_value = valid_metrics.get(monitor, -valid_metrics["loss"])
        
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(monitor_value)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < prev_lr:
            print(f"  >>> Scheduler giảm LR: {prev_lr:.2e} -> {current_lr:.2e}", flush=True)
            
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
        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, cfg, epoch, best_metric, bad_epochs, scaler=scaler)
        if is_best:
            save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, cfg, epoch, best_metric, bad_epochs, scaler=scaler)
            print(f"  >>> Saved best model ({monitor}={best_metric:.6f})", flush=True)
            
        current_lr = optimizer.param_groups[0]['lr']
        log_str = (
            f"Epoch {epoch:>3}/{cfg['training']['epochs']} | "
            f"Train Loss: {train_metrics['loss']:.6f} (ri={train_metrics['ri_loss']:.4f}, mag={train_metrics['mag_loss']:.4f}, sisnr={train_metrics['sisnr']:.4f}) | "
            f"Valid Loss: {valid_metrics['loss']:.6f} (ri={valid_metrics['ri_loss']:.4f}, mag={valid_metrics['mag_loss']:.4f}, sisnr={valid_metrics['sisnr']:.4f}) | "
            f"LR: {current_lr:.2e} | Time: {int(time.time() - start)}s"
        )
        print(log_str, flush=True)
        
        # Ghi log ra file
        with open(output_dir / "train_log.txt", "a", encoding="utf-8") as f:
            f.write(log_str + "\n")
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
