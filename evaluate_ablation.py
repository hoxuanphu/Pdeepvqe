# Cài đặt thư viện (nếu chưa có)
# !pip install pesq pystoi torchmetrics pandas numpy torchaudio

import time
import json
import torch
import torchaudio
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from pesq import pesq
from pystoi import stoi
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from IPython.display import display, FileLink
import warnings

warnings.filterwarnings("ignore")

# Đảm bảo đã import các hàm từ thư mục ablation
import sys
if "." not in sys.path:
    sys.path.append(".")
from ablation.ablation_config import get_train_config
from ablation.train_ablation import make_model, make_stft, make_istft, pick_key, resolve_path, read_json_manifest

# ==========================================
# 1. CẤU HÌNH ĐÁNH GIÁ (CHỈNH SỬA Ở ĐÂY)
# ==========================================
CONFIG_ID = "B1a"  # Đổi thành B1b hoặc C1 tùy thuộc vào model bạn muốn test
EVAL_MAX_SAMPLES = None  # Đặt thành số nguyên (vd: 50) nếu muốn test nhanh
DATA_ROOT = None # Đặt thành thư mục chứa dataset nếu chạy trên Kaggle (ví dụ: "/kaggle/input/vctk-demand")

cfg = get_train_config(CONFIG_ID)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = Path(cfg["experiment"]["output_dir"])
best_ckpt_path = output_dir / "best.pt"

print(f"Đang chuẩn bị đánh giá mô hình: {CONFIG_ID}")
print(f"Thư mục chứa kết quả: {output_dir}")

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH VÀ LOAD TRỌNG SỐ
# ==========================================
model = make_model(cfg, device, data_parallel=False)

if not Path(best_ckpt_path).exists():
    raise FileNotFoundError(f"Không tìm thấy checkpoint: {best_ckpt_path}")

print(f"Đang tải trọng số từ: {best_ckpt_path}")
checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model"])
model.eval()

# Đếm tham số
total_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {total_params:,}")
print(f"Device: {device}")

# Window cho STFT
window_name = cfg["stft"].get("window", "hann")
window = torch.hann_window(int(cfg["stft"]["win_length"]), device=device)
if window_name == "sqrt_hann":
    window = window.sqrt()

# ==========================================
# 3. CHUẨN BỊ TẬP TEST TỪ MANIFEST
# ==========================================
test_manifest = cfg["data"]["test_manifest"]
if DATA_ROOT:
    test_manifest = str(Path(DATA_ROOT) / test_manifest) if not Path(test_manifest).is_absolute() else test_manifest
    
records = read_json_manifest(test_manifest)

if EVAL_MAX_SAMPLES is not None:
    records = records[:EVAL_MAX_SAMPLES]
print(f"Test samples: {len(records)}")

sisdr_metric = ScaleInvariantSignalDistortionRatio().to(device)
results = []
total_audio_duration = 0.0
total_processing_time = 0.0
EVAL_SR = int(cfg["data"]["sample_rate"])

# ==========================================
# 4. CHẠY ĐÁNH GIÁ
# ==========================================
with torch.no_grad():
    for idx, record in tqdm(enumerate(records), total=len(records), desc=f"Evaluating {CONFIG_ID}"):
        # Resolve đường dẫn (hỗ trợ Kaggle DATA_ROOT)
        noisy_path = resolve_path(pick_key(record, "mixture"), record, data_root=DATA_ROOT)
        clean_path = resolve_path(pick_key(record, "target"), record, data_root=DATA_ROOT)
        
        # Load audio
        noisy_wav, sr_n = torchaudio.load(str(noisy_path))
        clean_wav, sr_c = torchaudio.load(str(clean_path))
        
        # Chuyển thành mono
        if noisy_wav.shape[0] > 1:
            noisy_wav = noisy_wav.mean(dim=0, keepdim=True)
        if clean_wav.shape[0] > 1:
            clean_wav = clean_wav.mean(dim=0, keepdim=True)
            
        # Resample
        if sr_n != EVAL_SR:
            noisy_wav = torchaudio.functional.resample(noisy_wav, sr_n, EVAL_SR)
        if sr_c != EVAL_SR:
            clean_wav = torchaudio.functional.resample(clean_wav, sr_c, EVAL_SR)
            
        noisy_wav = noisy_wav.squeeze(0)  # [T]
        clean_wav = clean_wav.squeeze(0)  # [T]
        
        audio_duration = noisy_wav.shape[0] / EVAL_SR
        total_audio_duration += audio_duration
        
        # --- Inference với đo RTF ---
        noisy_gpu = noisy_wav.unsqueeze(0).to(device)
        clean_gpu = clean_wav.unsqueeze(0).to(device)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        
        # Forward pass (STFT -> Model -> iSTFT)
        noisy_spec = make_stft(noisy_gpu, cfg, window)
        
        amp_enabled = bool(cfg["training"].get("use_amp", False)) and device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            pred_stft = model(noisy_spec)
        
        pred_stft = pred_stft.float()
        enhanced = make_istft(pred_stft, cfg, window, length=noisy_gpu.shape[-1])
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        
        processing_time = t_end - t_start
        total_processing_time += processing_time
        rtf = processing_time / audio_duration
        
        # Cắt cho bằng độ dài với clean_wav
        min_len = min(enhanced.shape[-1], clean_gpu.shape[-1])
        enhanced_gpu = enhanced[..., :min_len]
        clean_gpu = clean_gpu[..., :min_len]
        noisy_gpu = noisy_gpu[..., :min_len]
        
        # Tính SI-SDR (GPU)
        si_sdr_enhanced = sisdr_metric(enhanced_gpu, clean_gpu).item()
        si_sdr_noisy = sisdr_metric(noisy_gpu, clean_gpu).item()
        
        # Chuyển về Numpy (CPU) cho PESQ/STOI
        enhanced_np = enhanced_gpu.squeeze(0).cpu().numpy()
        clean_np = clean_gpu.squeeze(0).cpu().numpy()
        noisy_np = noisy_gpu.squeeze(0).cpu().numpy()
        
        # PESQ
        try:
            pesq_enhanced = pesq(EVAL_SR, clean_np, enhanced_np, 'wb')
            pesq_noisy = pesq(EVAL_SR, clean_np, noisy_np, 'wb')
        except Exception:
            pesq_enhanced = float('nan')
            pesq_noisy = float('nan')
            
        # STOI
        try:
            stoi_enhanced = stoi(clean_np, enhanced_np, EVAL_SR, extended=False)
            stoi_noisy = stoi(clean_np, noisy_np, EVAL_SR, extended=False)
        except Exception:
            stoi_enhanced = float('nan')
            stoi_noisy = float('nan')
            
        results.append({
            'ID': record.get('id', f'Sample_{idx}'),
            'PESQ_enhanced': round(pesq_enhanced, 4),
            'PESQ_noisy': round(pesq_noisy, 4),
            'PESQ_improvement': round(pesq_enhanced - pesq_noisy, 4) if not (np.isnan(pesq_enhanced) or np.isnan(pesq_noisy)) else float('nan'),
            'STOI_enhanced': round(stoi_enhanced, 4),
            'STOI_noisy': round(stoi_noisy, 4),
            'STOI_improvement': round(stoi_enhanced - stoi_noisy, 4) if not (np.isnan(stoi_enhanced) or np.isnan(stoi_noisy)) else float('nan'),
            'SI_SDR_enhanced_dB': round(si_sdr_enhanced, 2),
            'SI_SDR_noisy_dB': round(si_sdr_noisy, 2),
            'SI_SDR_improvement_dB': round(si_sdr_enhanced - si_sdr_noisy, 2),
            'RTF': round(rtf, 6),
            'duration_s': round(audio_duration, 3),
        })

# ==========================================
# 5. TỔNG HỢP VÀ IN KẾT QUẢ
# ==========================================
df_results = pd.DataFrame(results)
overall_rtf = total_processing_time / total_audio_duration

summary = {
    'Metric': ['PESQ (enhanced)', 'PESQ (noisy)', 'PESQ (Δ improvement)',
               'STOI (enhanced)', 'STOI (noisy)', 'STOI (Δ improvement)',
               'SI-SDR enhanced (dB)', 'SI-SDR noisy (dB)', 'SI-SDR Δ (dB)',
               'RTF (mean)', 'RTF (overall)', 'Real-time capable?'],
    'Value': [
        f"{df_results['PESQ_enhanced'].mean():.4f}",
        f"{df_results['PESQ_noisy'].mean():.4f}",
        f"{df_results['PESQ_improvement'].mean():.4f}",
        f"{df_results['STOI_enhanced'].mean():.4f}",
        f"{df_results['STOI_noisy'].mean():.4f}",
        f"{df_results['STOI_improvement'].mean():.4f}",
        f"{df_results['SI_SDR_enhanced_dB'].mean():.2f}",
        f"{df_results['SI_SDR_noisy_dB'].mean():.2f}",
        f"{df_results['SI_SDR_improvement_dB'].mean():.2f}",
        f"{df_results['RTF'].mean():.6f}",
        f"{overall_rtf:.6f}",
        '✅ Có' if overall_rtf < 1.0 else '❌ Không',
    ]
}
df_summary = pd.DataFrame(summary)

print('\n' + '=' * 60)
print(f'  KẾT QUẢ ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH {CONFIG_ID}')
print('=' * 60)
print(f'Test samples: {len(df_results)}')
print(f'Model params: {total_params:,}')
print(f'Total audio: {total_audio_duration:.1f}s | Processing: {total_processing_time:.1f}s\n')
display(df_summary)

# --- Lưu CSV ---
save_dir = output_dir / 'evaluation'
save_dir.mkdir(parents=True, exist_ok=True)
detail_csv = save_dir / f'eval_metrics_per_sample_{CONFIG_ID}.csv'
summary_csv_path = save_dir / f'eval_metrics_summary_{CONFIG_ID}.csv'

df_results.to_csv(detail_csv, index=False)
df_summary.to_csv(summary_csv_path, index=False)

print(f'\n📁 Đã lưu kết quả chi tiết:  {detail_csv}')
print(f'📁 Đã lưu bảng tổng hợp:     {summary_csv_path}')
display(FileLink(str(detail_csv)))
display(FileLink(str(summary_csv_path)))
