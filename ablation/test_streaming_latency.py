import argparse
import csv
import time
from pathlib import Path

import torch
import torchaudio

# Chèn thư mục cha vào sys.path nếu cần (tương tự run_ablation_benchmark)
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ablation.deepvqe_ablation import (
    DeepVQE_Ablation,
    StreamDeepVQE_Ablation,
    convert_ablation_to_stream,
)

def main():
    parser = argparse.ArgumentParser(description="Test streaming latency frame by frame on a real audio file.")
    parser.add_argument("--input", type=str, required=True, help="Path to input audio file (.wav)")
    parser.add_argument("--config", type=str, default="Baseline", help="Ablation config ID (e.g., Baseline, C1)")
    parser.add_argument("--output", type=str, default="streaming_metrics.csv", help="CSV output path for per-frame metrics")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--win-length", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 1. Load Audio
    wav, sr = torchaudio.load(args.input)
    # Convert to mono if needed
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.to(device)

    # 2. STFT (Tạo features giống hệt cách FE xử lý trong DeepVQE)
    window = torch.hann_window(args.win_length, device=device)
    spec = torch.stft(
        wav,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        window=window,
        return_complex=True
    )
    spec = torch.view_as_real(spec)  # Shape: (1, F, T, 2)
    frames = spec.shape[2]
    audio_seconds = wav.shape[-1] / sr
    print(f"Audio loaded: {audio_seconds:.2f} seconds -> {frames} frames.")

    # 3. Load Model & Convert to Streaming
    print(f"Loading {args.config} model...")
    offline_model = DeepVQE_Ablation.from_config_id(args.config).eval().to(device)
    stream_model = StreamDeepVQE_Ablation.from_config_id(args.config).eval().to(device)
    convert_ablation_to_stream(stream_model, offline_model, strict=True)

    # 4. Warm-up (Khởi tạo CUDA kernels)
    print("Warming up...")
    with torch.no_grad():
        dummy_input = torch.randn(1, spec.shape[1], 10, 2, device=device)
        cache = stream_model.init_cache(1, spec.shape[1], device, spec.dtype)
        for i in range(10):
            _, cache = stream_model(dummy_input[:, :, i:i+1, :], cache)

    # 5. Measure Streaming (The core test loop)
    print("Starting frame-by-frame measurement...")
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    global_start_time = time.perf_counter()
    
    frame_metrics = []
    total_processing_time = 0.0

    with torch.no_grad():
        cache = stream_model.init_cache(1, spec.shape[1], device, spec.dtype)
        for frame_idx in range(frames):
            frame_input = spec[:, :, frame_idx : frame_idx + 1, :]
            
            # Đồng bộ để đảm bảo kết quả đo thời gian là chính xác cho bước này
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            t0 = time.perf_counter()
            
            # Xử lý 1 frame
            y, cache = stream_model(frame_input, cache)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
                
            t1 = time.perf_counter()
            
            proc_time_sec = t1 - t0
            total_processing_time += proc_time_sec
            
            frame_metrics.append({
                "frame_idx": frame_idx,
                "start_time": t0,
                "end_time": t1,
                "processing_time_ms": proc_time_sec * 1000.0
            })

    if device.type == "cuda":
        torch.cuda.synchronize()
    global_end_time = time.perf_counter()
    
    total_time_wall_clock = global_end_time - global_start_time
    avg_processing_time = total_processing_time / frames if frames > 0 else 0

    # 6. Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "start_time", "end_time", "processing_time_ms"])
        writer.writeheader()
        writer.writerows(frame_metrics)

    # 7. Print Summary
    print("\n" + "="*40)
    print("=== MEASUREMENT SUMMARY ===")
    print("="*40)
    print(f"Total Frames Processed   : {frames}")
    print(f"Audio Duration           : {audio_seconds:.4f} s")
    print("-" * 40)
    print(f"Global Start Time        : {global_start_time:.6f}")
    print(f"Global End Time          : {global_end_time:.6f}")
    print(f"Total Wall Clock Time    : {total_time_wall_clock:.6f} s (Bao gồm overhead Python/Cắt tensor)")
    print(f"Total Processing Time    : {total_processing_time:.6f} s (Thời gian model xử lý thuần)")
    print("-" * 40)
    print(f"Avg Processing Time/Frame: {avg_processing_time * 1000.0:.4f} ms")
    print(f"Real-Time Factor (Pure)  : {(total_processing_time) / audio_seconds:.4f}")
    print(f"Real-Time Factor (Wall)  : {(total_time_wall_clock) / audio_seconds:.4f}")
    print("="*40)
    print(f"Saved per-frame metrics to : {output_path.absolute()}")

if __name__ == "__main__":
    main()
