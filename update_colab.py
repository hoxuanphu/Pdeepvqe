import json

with open("train_base_deepvqe_colab.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        
        # Update CONFIG
        if "CONFIG = {" in source and "scheduler_patience" in source:
            new_source = """CONFIG = {
    'seed': 1234,
    'sample_rate': 16000,
    'n_fft': 512,
    'hop_length': 256,
    'win_length': 512,
    
    'batch_size': 4,
    'num_workers': 2,
    'epochs': 100,
    'lr': 1e-3,          # Cập nhật theo repo gốc
    'grad_clip': 5.0,
    
    # Loss weights
    'lamda_ri': 30,
    'lamda_mag': 70,
    'compress_factor': 0.3,
    
    'train_csv': f'{data_dir}/train.csv',
    'valid_csv': f'{data_dir}/valid.csv',
    'test_csv': f'{data_dir}/test.csv',
    
    'output_dir': '/content/drive/MyDrive/DeepVQE_Workspace/checkpoints/deepvqe_vctk',
    
    # Scheduler
    'scheduler_factor': 0.5,
    'scheduler_patience': 5,
    'scheduler_min_lr': 1e-6,
    
    # Early stopping
    'early_stopping_patience': 15,
}

print('Config:', CONFIG)
"""
            cell["source"] = [line + "\n" for line in new_source.split("\n")[:-1]]
            
        # Update compute_loss
        if "def compute_loss(" in source and "loss_wav_weight" in source:
            new_source = """import csv

def make_stft(wav, n_fft, hop_length, win_length, win):
    \"\"\"wav [B, T] -> spec [B, F, T_frames, 2]\"\"\"
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                      window=win, return_complex=True)
    return torch.view_as_real(spec)

def make_istft(spec, n_fft, hop_length, win_length, win, length=None):
    \"\"\"spec [B, F, T_frames, 2] -> wav [B, T]\"\"\"
    complex_spec = torch.complex(spec[..., 0], spec[..., 1])
    return torch.istft(complex_spec, n_fft=n_fft, hop_length=hop_length,
                       win_length=win_length, window=win, length=length)

def compute_loss(model, noisy_wav, clean_wav, cfg, win):
    \"\"\"Tính HybridLoss giống repo SEtrain: Compressed RI + Compressed Mag + negative SI-SNR.\"\"\"
    n_fft = cfg['n_fft']
    hop = cfg['hop_length']
    win_len = cfg['win_length']
    c = cfg['compress_factor']
    
    device = noisy_wav.device
    
    # Forward pass
    noisy_spec = make_stft(noisy_wav, n_fft, hop, win_len, win)
    pred_stft = model(noisy_spec)
    
    # Cắt padding cho độ dài phù hợp
    clean_spec = make_stft(clean_wav, n_fft, hop, win_len, win)
    min_t = min(pred_stft.shape[2], clean_spec.shape[2])
    pred_stft = pred_stft[:, :, :min_t, :]
    true_stft = clean_spec[:, :, :min_t, :]
    
    # Tách Real / Imaginary
    pred_stft_real, pred_stft_imag = pred_stft[:,:,:,0], pred_stft[:,:,:,1]
    true_stft_real, true_stft_imag = true_stft[:,:,:,0], true_stft[:,:,:,1]
    
    # Tính Magnitude
    pred_mag = torch.sqrt(pred_stft_real**2 + pred_stft_imag**2 + 1e-12)
    true_mag = torch.sqrt(true_stft_real**2 + true_stft_imag**2 + 1e-12)
    
    # Phổ nén (Compressed Spectrum)
    pred_real_c = pred_stft_real / (pred_mag**(1 - c))
    pred_imag_c = pred_stft_imag / (pred_mag**(1 - c))
    true_real_c = true_stft_real / (true_mag**(1 - c))
    true_imag_c = true_stft_imag / (true_mag**(1 - c))
    
    # RI Loss & Mag Loss
    real_loss = torch.mean((pred_real_c - true_real_c)**2)
    imag_loss = torch.mean((pred_imag_c - true_imag_c)**2)
    mag_loss = torch.mean((pred_mag**c - true_mag**c)**2)
    
    # ISTFT để tính SI-SNR
    # Cửa sổ lúc này có lũy thừa 0.5 theo repo gốc, nhưng ta có thể dùng cửa sổ hann bình thường để đồng bộ
    y_pred = torch.istft(pred_stft_real + 1j*pred_stft_imag, n_fft=n_fft, hop_length=hop, win_length=win_len, window=win)
    y_true = torch.istft(true_stft_real + 1j*true_stft_imag, n_fft=n_fft, hop_length=hop, win_length=win_len, window=win)
    
    # Đảm bảo cùng độ dài
    min_wav_len = min(y_pred.shape[-1], y_true.shape[-1])
    y_pred = y_pred[..., :min_wav_len]
    y_true = y_true[..., :min_wav_len]
    
    y_target = torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true / (torch.sum(torch.square(y_true), dim=-1, keepdim=True) + 1e-8)
    sisnr = -torch.log10(torch.sum(torch.square(y_target), dim=-1, keepdim=True) / torch.sum(torch.square(y_pred - y_target), dim=-1, keepdim=True) + 1e-8).mean()
    
    # Tổng Loss = 30 * RI + 70 * Mag + SISNR
    loss = cfg['lamda_ri'] * (real_loss + imag_loss) + cfg['lamda_mag'] * mag_loss + sisnr
    
    return loss, {'loss': loss.item(), 'ri_loss': (real_loss + imag_loss).item(), 'mag_loss': mag_loss.item(), 'sisnr': sisnr.item()}

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_loss, bad_epochs):
    # Xử lý DataParallel: lấy state_dict từ module gốc
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save({
        'model': model_state,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'best_loss': best_loss,
        'bad_epochs': bad_epochs,
    }, str(path))

def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    target = model.module if hasattr(model, 'module') else model
    target.load_state_dict(ckpt['model'])
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler and ckpt.get('scheduler'):
        scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt.get('epoch', 0), ckpt.get('best_loss'), ckpt.get('bad_epochs', 0)

def append_log(log_path, row_dict):
    log_path = Path(log_path)
    file_exists = log_path.exists() and log_path.stat().st_size > 0
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

print('Hàm tiện ích đã sẵn sàng.')
"""
            cell["source"] = [line + "\n" for line in new_source.split("\n")[:-1]]

        # Update train loop print logging
        if "train_l1_wav" in source and "avg_train['loss']" in source:
            new_source = source.replace("wav={avg_train['l1_wav']:.4f}, spec={avg_train['l1_spec']:.4f}", 
                                      "ri={avg_train['ri_loss']:.4f}, mag={avg_train['mag_loss']:.4f}, sisnr={avg_train['sisnr']:.4f}")
            new_source = new_source.replace("wav={avg_valid['l1_wav']:.4f}, spec={avg_valid['l1_spec']:.4f}", 
                                      "ri={avg_valid['ri_loss']:.4f}, mag={avg_valid['mag_loss']:.4f}, sisnr={avg_valid['sisnr']:.4f}")
            
            # Update CSV log fields
            new_source = new_source.replace("'train_l1_wav': f\"{avg_train['l1_wav']:.6f}\",", "'train_ri_loss': f\"{avg_train['ri_loss']:.6f}\",")
            new_source = new_source.replace("'train_l1_spec': f\"{avg_train['l1_spec']:.6f}\",", "'train_mag_loss': f\"{avg_train['mag_loss']:.6f}\",")
            new_source = new_source.replace("'valid_l1_wav': f\"{avg_valid['l1_wav']:.6f}\",", "'valid_ri_loss': f\"{avg_valid['ri_loss']:.6f}\",")
            new_source = new_source.replace("'valid_l1_spec': f\"{avg_valid['l1_spec']:.6f}\",", "'valid_mag_loss': f\"{avg_valid['mag_loss']:.6f}\",")
            
            # Add SISNR to log
            new_source = new_source.replace("'train_mag_loss': f\"{avg_train['mag_loss']:.6f}\",", "'train_mag_loss': f\"{avg_train['mag_loss']:.6f}\",\n        'train_sisnr': f\"{avg_train['sisnr']:.6f}\",")
            new_source = new_source.replace("'valid_mag_loss': f\"{avg_valid['mag_loss']:.6f}\",", "'valid_mag_loss': f\"{avg_valid['mag_loss']:.6f}\",\n        'valid_sisnr': f\"{avg_valid['sisnr']:.6f}\",")
            
            cell["source"] = [line + "\n" for line in new_source.split("\n")[:-1]]

with open("train_base_deepvqe_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated loss logic inside train_base_deepvqe_colab.ipynb")
