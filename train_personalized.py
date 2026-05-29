import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from deepvqe_personalized import (
    PersonalizedDeepVQE,
    SpeakerEncoder,
    magnitude_l1_loss,
    output_energy_ratio_db,
    si_sdr,
    si_sdr_loss,
    speaker_consistency_loss,
    stft_magnitude,
)
from personalized_train_config import get_config


PATH_KEYS = {
    "mixture": ["mixture", "mixture_path", "mix", "mix_path", "noisy", "input"],
    "target": ["target", "target_path", "clean", "target_reverb", "target_wav"],
    "enrollment": ["enrollment", "enrollment_path", "enroll", "reference", "reference_path"],
    "embedding": ["spk_emb", "spk_emb_path", "speaker_embedding", "embedding", "embedding_path"],
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_manifest(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(f"JSON manifest must contain a list of items: {path}")
        for item in records:
            item["_manifest_dir"] = str(path.parent)
        if not records:
            raise ValueError(f"Manifest is empty: {path}")
        return records

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
            item["_manifest_dir"] = str(path.parent)
            records.append(item)
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


def pick_key(record, group, required=True):
    for key in PATH_KEYS[group]:
        if key in record and record[key] not in (None, ""):
            return record[key]
    if required:
        valid = ", ".join(PATH_KEYS[group])
        raise KeyError(f"Manifest item is missing one of [{valid}]: {record}")
    return None


def resolve_path(value, record, data_root=None):
    path = Path(value)
    if path.is_absolute():
        return path

    candidates = []
    if data_root:
        candidates.append(Path(data_root) / path)
    if "_manifest_dir" in record:
        candidates.append(Path(record["_manifest_dir"]) / path)
    candidates.append(Path(path))

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


def crop_or_pad_pair(mixture, target, length, random_crop):
    min_len = min(mixture.numel(), target.numel())
    mixture = mixture[:min_len]
    target = target[:min_len]
    if min_len < length:
        mixture = repeat_to_length(mixture, length)
        target = repeat_to_length(target, length)
        return mixture, target

    if random_crop:
        start = random.randint(0, min_len - length)
    else:
        start = max(0, (min_len - length) // 2)
    end = start + length
    return mixture[start:end], target[start:end]


def crop_or_pad_single(wav, length, random_crop):
    if wav.numel() < length:
        return repeat_to_length(wav, length)
    if random_crop:
        start = random.randint(0, wav.numel() - length)
    else:
        start = max(0, (wav.numel() - length) // 2)
    return wav[start:start + length]


def is_negative_record(record):
    if bool(record.get("is_negative", False)) or bool(record.get("negative", False)):
        return True
    case = str(record.get("case", "")).lower()
    return case in {"negative", "absent", "absent_speaker", "wrong_speaker"}


class PersonalizedManifestDataset(Dataset):
    def __init__(self, manifest_path, cfg, split="train", data_root=None):
        self.records = read_manifest(manifest_path)
        self.cfg = cfg
        self.split = split
        self.data_root = data_root
        self.sample_rate = int(cfg["data"]["sample_rate"])
        self.clip_samples = int(float(cfg["data"]["clip_seconds"]) * self.sample_rate)
        enroll_max_seconds = float(cfg["data"]["enrollment_seconds"][1])
        self.enrollment_samples = int(enroll_max_seconds * self.sample_rate)
        self.random_crop = split == "train"
        self.use_precomputed_spk_emb = bool(cfg["model"].get("use_precomputed_spk_emb", True))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        mixture_path = resolve_path(pick_key(record, "mixture"), record, self.data_root)
        target_path = resolve_path(pick_key(record, "target"), record, self.data_root)

        mixture = load_audio(mixture_path, self.sample_rate)
        target = load_audio(target_path, self.sample_rate)
        mixture, target = crop_or_pad_pair(mixture, target, self.clip_samples, self.random_crop)

        item = {
            "mixture": mixture,
            "target": target,
            "is_negative": torch.tensor(is_negative_record(record), dtype=torch.bool),
            "id": str(record.get("id", mixture_path.stem)),
        }

        embedding_value = pick_key(record, "embedding", required=False)
        if embedding_value is not None:
            embedding_path = resolve_path(embedding_value, record, self.data_root)
            item["spk_emb"] = torch.from_numpy(np.load(str(embedding_path))).float().view(-1)
        elif self.use_precomputed_spk_emb:
            raise KeyError(
                "Config expects precomputed speaker embeddings, but this manifest item has no embedding path. "
                "Add `embedding_path`/`spk_emb_path`, or run with `--online-ecapa`."
            )
        else:
            enrollment_path = resolve_path(pick_key(record, "enrollment"), record, self.data_root)
            enrollment = load_audio(enrollment_path, self.sample_rate)
            item["enrollment"] = crop_or_pad_single(enrollment, self.enrollment_samples, self.random_crop)

        return item


def collate_batch(items):
    batch = {
        "mixture": torch.stack([item["mixture"] for item in items], dim=0),
        "target": torch.stack([item["target"] for item in items], dim=0),
        "is_negative": torch.stack([item["is_negative"] for item in items], dim=0),
        "id": [item["id"] for item in items],
    }
    if "spk_emb" in items[0]:
        batch["spk_emb"] = torch.stack([item["spk_emb"] for item in items], dim=0)
    if "enrollment" in items[0]:
        batch["enrollment"] = torch.stack([item["enrollment"] for item in items], dim=0)
    return batch


def make_stft(wav, cfg, window):
    stft_cfg = cfg["stft"]
    spec = torch.stft(
        wav,
        n_fft=stft_cfg["n_fft"],
        hop_length=stft_cfg["hop_length"],
        win_length=stft_cfg["win_length"],
        window=window,
        return_complex=True,
    )
    return torch.view_as_real(spec)


def make_istft(spec, cfg, window, length):
    stft_cfg = cfg["stft"]
    complex_spec = torch.complex(spec[..., 0], spec[..., 1])
    return torch.istft(
        complex_spec,
        n_fft=stft_cfg["n_fft"],
        hop_length=stft_cfg["hop_length"],
        win_length=stft_cfg["win_length"],
        window=window,
        length=length,
    )


def make_model(cfg, device):
    model_cfg = cfg["model"]
    model = PersonalizedDeepVQE(
        emb_dim=model_cfg["emb_dim"],
        deepvqe_sr=model_cfg["deepvqe_sr"],
        ecapa_sr=model_cfg["ecapa_sr"],
        speaker_encoder_source=model_cfg["speaker_encoder_source"],
        device=device,
        use_speaker_encoder=model_cfg.get("use_speaker_encoder", False),
    )
    return model.to(device)


def make_eval_ecapa(cfg, device):
    speaker_cfg = cfg["loss"]["speaker_consistency"]
    if not speaker_cfg.get("enabled", False):
        return None
    encoder = SpeakerEncoder(
        source=speaker_cfg["eval_ecapa_source"],
        deepvqe_sr=cfg["data"]["sample_rate"],
        ecapa_sr=cfg["model"]["ecapa_sr"],
        device=device,
    )
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder


def compute_batch(model, batch, cfg, window, device, eval_ecapa=None, batch_index=0, train=True):
    mixture = batch["mixture"].to(device)
    target = batch["target"].to(device)
    is_negative = batch["is_negative"].to(device)
    positive_mask = ~is_negative

    mixture_spec = make_stft(mixture, cfg, window)
    target_spec = make_stft(target, cfg, window)

    spk_emb = batch.get("spk_emb")
    enrollment = batch.get("enrollment")
    if spk_emb is not None:
        spk_emb = spk_emb.to(device)
        estimate_spec = model(mixture_spec, spk_emb=spk_emb)
    else:
        enrollment = enrollment.to(device)
        estimate_spec = model(mixture_spec, enrollment_wav=enrollment)

    estimate = make_istft(estimate_spec, cfg, window, length=target.shape[-1])

    loss = estimate_spec.sum() * 0.0
    loss_parts = {}
    if positive_mask.any():
        rec_cfg = cfg["loss"]["reconstruction"]
        rec_loss = (
            rec_cfg["si_sdr_weight"] * si_sdr_loss(estimate[positive_mask], target[positive_mask])
            + rec_cfg["magnitude_l1_weight"] * magnitude_l1_loss(
                estimate_spec[positive_mask], target_spec[positive_mask]
            )
        )
        loss = loss + rec_loss
        loss_parts["reconstruction"] = float(rec_loss.detach().cpu())

    speaker_cfg = cfg["loss"]["speaker_consistency"]
    apply_every = int(speaker_cfg.get("apply_every_n_batches", 1))
    use_speaker_loss = speaker_cfg.get("enabled", False) and eval_ecapa is not None
    use_speaker_loss = use_speaker_loss and positive_mask.any() and (batch_index % apply_every == 0)
    if use_speaker_loss:
        with torch.no_grad():
            target_emb = eval_ecapa(target[positive_mask])
        estimate_emb = eval_ecapa(estimate[positive_mask])
        spk_loss = speaker_consistency_loss(target_emb, estimate_emb)
        weighted = float(speaker_cfg["alpha"]) * spk_loss
        loss = loss + weighted
        loss_parts["speaker"] = float(weighted.detach().cpu())

    neg_cfg = cfg["loss"]["negative_case"]
    if neg_cfg.get("enabled", False) and is_negative.any():
        neg_wave = estimate[is_negative]
        neg_spec = estimate_spec[is_negative]
        neg_loss = (
            float(neg_cfg["waveform_energy_weight"]) * neg_wave.pow(2).mean()
            + float(neg_cfg["magnitude_suppression_weight"]) * stft_magnitude(neg_spec).mean()
        )
        loss = loss + neg_loss
        loss_parts["negative"] = float(neg_loss.detach().cpu())

    metrics = {}
    if positive_mask.any():
        est_si_sdr = si_sdr(estimate[positive_mask], target[positive_mask])
        mix_si_sdr = si_sdr(mixture[positive_mask], target[positive_mask])
        metrics["si_sdr"] = float(est_si_sdr.mean().detach().cpu())
        metrics["si_sdri"] = float((est_si_sdr - mix_si_sdr).mean().detach().cpu())
    if is_negative.any():
        ratio = output_energy_ratio_db(estimate[is_negative], mixture[is_negative])
        metrics["negative_output_energy_ratio_db"] = float(ratio.mean().detach().cpu())

    return loss, loss_parts, metrics


def average_dict(values):
    if not values:
        return {}
    keys = sorted({key for item in values for key in item})
    return {
        key: sum(item[key] for item in values if key in item) / max(1, sum(key in item for item in values))
        for key in keys
    }


def format_progress(values):
    return {
        key: f"{value:.4g}" if isinstance(value, float) else value
        for key, value in values.items()
    }


def progress_iter(iterable, total, desc, enabled=True, leave=False):
    if enabled and tqdm is not None:
        return tqdm(
            iterable,
            total=total,
            desc=desc,
            dynamic_ncols=True,
            leave=leave,
            file=sys.stdout,
            mininterval=1.0,
            miniters=1,
            ascii=True,
        )
    return iterable


def make_loader(manifest, cfg, split, data_root):
    dataset = PersonalizedManifestDataset(manifest, cfg, split=split, data_root=data_root)
    return DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=split == "train",
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        drop_last=split == "train",
        collate_fn=collate_batch,
    )


def save_checkpoint(path, model, optimizer, scheduler, cfg, epoch, best_metric, bad_epochs=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "config": cfg,
            "epoch": epoch,
            "best_metric": best_metric,
            "bad_epochs": bad_epochs,
        },
        str(path),
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("epoch", 0)), ckpt.get("best_metric", None), int(ckpt.get("bad_epochs", 0))


def train_one_epoch(model, loader, optimizer, cfg, window, device, eval_ecapa, epoch, use_progress=True):
    model.train()
    loss_values = []
    metric_values = []
    start = time.time()
    print(f"Starting epoch {epoch} train: {len(loader)} batches", flush=True)
    iterator = progress_iter(loader, total=len(loader), desc=f"epoch {epoch} train", enabled=use_progress)
    for batch_index, batch in enumerate(iterator):
        optimizer.zero_grad()
        loss, loss_parts, metrics = compute_batch(
            model, batch, cfg, window, device, eval_ecapa=eval_ecapa, batch_index=batch_index, train=True
        )
        loss.backward()
        grad_clip = cfg["optimizer"].get("grad_clip_norm")
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()

        loss_values.append({"loss": float(loss.detach().cpu()), **loss_parts})
        metric_values.append(metrics)
        if use_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                format_progress({
                    "loss": float(loss.detach().cpu()),
                    **loss_parts,
                    **metrics,
                })
            )
        if (batch_index + 1) % 20 == 0:
            avg_loss = average_dict(loss_values[-20:])
            avg_metric = average_dict(metric_values[-20:])
            print(f"epoch={epoch} step={batch_index + 1}/{len(loader)} loss={avg_loss} metrics={avg_metric}", flush=True)

    return average_dict(loss_values), average_dict(metric_values), time.time() - start


@torch.no_grad()
def validate(model, loader, cfg, window, device, eval_ecapa, epoch=None, use_progress=True):
    model.eval()
    loss_values = []
    metric_values = []
    desc = f"epoch {epoch} valid" if epoch is not None else "valid"
    print(f"Starting {desc}: {len(loader)} batches", flush=True)
    iterator = progress_iter(loader, total=len(loader), desc=desc, enabled=use_progress)
    for batch_index, batch in enumerate(iterator):
        loss, loss_parts, metrics = compute_batch(
            model, batch, cfg, window, device, eval_ecapa=eval_ecapa, batch_index=batch_index, train=False
        )
        loss_values.append({"loss": float(loss.detach().cpu()), **loss_parts})
        metric_values.append(metrics)
        if use_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                format_progress({
                    "loss": float(loss.detach().cpu()),
                    **loss_parts,
                    **metrics,
                })
            )
    return average_dict(loss_values), average_dict(metric_values)


def is_better(value, best, mode):
    if best is None:
        return True
    return value > best if mode == "max" else value < best


def main():
    parser = argparse.ArgumentParser(description="Train Personalized DeepVQE")
    parser.add_argument("--phase", default="phase1_reconstruction", choices=[
        "phase1_reconstruction",
        "phase2_speaker_consistency",
        "phase3_negative_case",
    ])
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--valid-manifest", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from output_dir/last.pt when it exists and --resume is not set",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--online-ecapa", action="store_true", help="Load enrollment audio instead of .npy embeddings")
    args = parser.parse_args()

    cfg = get_config(args.phase)
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
    if args.online_ecapa:
        cfg["model"]["use_precomputed_spk_emb"] = False
        cfg["model"]["use_speaker_encoder"] = True

    output_dir = Path(cfg["experiment"]["output_dir"])
    if args.resume:
        cfg["experiment"]["resume_from"] = args.resume
    elif args.auto_resume:
        auto_resume_path = output_dir / "last.pt"
        if auto_resume_path.exists():
            cfg["experiment"]["resume_from"] = str(auto_resume_path)
            print(f"Auto-resume: found checkpoint {auto_resume_path}", flush=True)
        else:
            print(f"Auto-resume: no checkpoint found at {auto_resume_path}; starting from scratch.", flush=True)

    seed_everything(int(cfg["experiment"]["seed"]))
    requested_device = cfg["training"]["device"]
    device = torch.device(requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("Building dataloaders...", flush=True)
    train_loader = make_loader(cfg["data"]["train_manifest"], cfg, "train", args.data_root)
    valid_loader = make_loader(cfg["data"]["valid_manifest"], cfg, "valid", args.data_root)
    print(f"Train batches: {len(train_loader)} | Valid batches: {len(valid_loader)}", flush=True)

    print(f"Building model on {device}...", flush=True)
    model = make_model(cfg, device)
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
    print("Loading eval ECAPA if enabled...", flush=True)
    eval_ecapa = make_eval_ecapa(cfg, device)

    window = torch.hann_window(int(cfg["stft"]["win_length"]), device=device)
    start_epoch = 0
    best_metric = None
    bad_epochs = 0
    if cfg["experiment"].get("resume_from"):
        start_epoch, best_metric, bad_epochs = load_checkpoint(
            cfg["experiment"]["resume_from"], model, optimizer, scheduler, device=device
        )
        print(
            f"Resumed checkpoint from epoch={start_epoch} "
            f"best_metric={best_metric} bad_epochs={bad_epochs}",
            flush=True,
        )

    monitor = cfg["training"]["checkpoint"]["monitor"].split("/")[-1]
    mode = cfg["training"]["checkpoint"]["mode"]
    use_progress = not args.no_progress
    for epoch in range(start_epoch + 1, int(cfg["training"]["epochs"]) + 1):
        train_loss, train_metrics, elapsed = train_one_epoch(
            model, train_loader, optimizer, cfg, window, device, eval_ecapa, epoch, use_progress=use_progress
        )
        valid_loss, valid_metrics = validate(
            model, valid_loader, cfg, window, device, eval_ecapa, epoch=epoch, use_progress=use_progress
        )
        monitor_value = valid_metrics.get(monitor, -valid_loss.get("loss", 0.0))
        scheduler.step(monitor_value)

        print(
            f"epoch={epoch} time={elapsed:.1f}s "
            f"train_loss={train_loss} train_metrics={train_metrics} "
            f"valid_loss={valid_loss} valid_metrics={valid_metrics}",
            flush=True,
        )

        is_current_best = is_better(monitor_value, best_metric, mode)
        if is_current_best:
            best_metric = monitor_value
            bad_epochs = 0
        else:
            bad_epochs += 1

        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, cfg, epoch, best_metric, bad_epochs)
        if is_current_best:
            save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, cfg, epoch, best_metric, bad_epochs)

        early = cfg["training"]["early_stopping"]
        if early.get("enabled", False) and bad_epochs >= int(early["patience"]):
            print(f"Early stopping after {bad_epochs} stale epochs. best_{monitor}={best_metric}", flush=True)
            break


if __name__ == "__main__":
    main()
