import argparse
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

from deepvqe import DeepVQE


def _to_mono(audio: torch.Tensor) -> torch.Tensor:
    """Convert waveform to mono."""
    if audio.ndim == 2:
        audio = audio.mean(dim=1)
    return audio


def _resample_1d(audio: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    """Linear resample for 1D waveform tensor."""
    if src_sr == dst_sr:
        return audio
    x = audio.unsqueeze(0).unsqueeze(0)
    new_len = int(audio.shape[0] * dst_sr / src_sr)
    x = F.interpolate(x, size=new_len, mode="linear", align_corners=False)
    return x.squeeze(0).squeeze(0)


def enhance_wav(
    input_wav: str,
    output_wav: str,
    checkpoint_path: str,
    device: str = "cuda",
    model_sr: int = 16000,
    output_sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256,
    win_length: int = 512,
) -> str:
    """
    Run DeepVQE inference on a wav file and save enhanced result.
    Input is converted to mono and resampled to model_sr (default 16k) for inference.
    Output can be optionally resampled to output_sr (e.g. 24k).
    """
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

    model = DeepVQE().to(dev).eval()
    checkpoint = torch.load(checkpoint_path, map_location=dev)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    wav, sr = sf.read(input_wav, dtype="float32")
    x = torch.from_numpy(wav)
    x = _to_mono(x)
    x = _resample_1d(x, sr, model_sr).to(dev)
    x = x.unsqueeze(0)

    window = torch.hann_window(win_length, device=dev)
    orig_len = x.shape[-1]
    x_spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    x_spec = torch.view_as_real(x_spec)

    with torch.no_grad():
        y_spec = model(x_spec)

    y_spec = torch.complex(y_spec[..., 0], y_spec[..., 1])
    y = torch.istft(
        y_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )
    y = F.pad(y, [0, max(0, orig_len - y.shape[-1])])
    y = y[0].detach().cpu()
    y = _resample_1d(y, model_sr, output_sr).numpy()

    output_path = Path(output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), y, output_sr)
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepVQE wav inference")
    parser.add_argument("--input", required=True, help="Path to input wav")
    parser.add_argument("--output", required=True, help="Path to output wav")
    parser.add_argument("--ckpt", required=True, help="Path to DeepVQE checkpoint (.tar)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument("--output_sr", type=int, default=16000, help="Output sample rate, e.g. 24000")
    args = parser.parse_args()

    out = enhance_wav(
        input_wav=args.input,
        output_wav=args.output,
        checkpoint_path=args.ckpt,
        device=args.device,
        model_sr=16000,
        output_sr=args.output_sr,
    )
    print(f"Saved enhanced audio to: {out}")


if __name__ == "__main__":
    main()
