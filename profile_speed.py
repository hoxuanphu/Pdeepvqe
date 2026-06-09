import torch
import torchaudio
import time
from deepvqe import DeepVQE
from pathlib import Path

# Mock config
CONFIG = {
    'sample_rate': 16000,
    'n_fft': 512,
    'hop_length': 256,
    'win_length': 512,
    'compress_factor': 0.3,
    'lamda_ri': 30,
    'lamda_mag': 70,
    'use_amp': True,
    'batch_size': 8
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

model = DeepVQE().to(device)
model.train()
window = torch.hann_window(CONFIG['win_length']).to(device)

def make_stft(wav, n_fft, hop_length, win_length, win):
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                      window=win, return_complex=True)
    return torch.view_as_real(spec)

def compute_loss(model, noisy_wav, clean_wav, cfg, win):
    n_fft = cfg['n_fft']
    hop = cfg['hop_length']
    win_len = cfg['win_length']
    c = cfg['compress_factor']
    
    noisy_spec = make_stft(noisy_wav, n_fft, hop, win_len, win)
    pred_stft = model(noisy_spec)
    
    clean_spec = make_stft(clean_wav, n_fft, hop, win_len, win)
    min_t = min(pred_stft.shape[2], clean_spec.shape[2])
    pred_stft = pred_stft[:, :, :min_t, :]
    true_stft = clean_spec[:, :, :min_t, :]
    
    pred_stft_real, pred_stft_imag = pred_stft[:,:,:,0], pred_stft[:,:,:,1]
    true_stft_real, true_stft_imag = true_stft[:,:,:,0], true_stft[:,:,:,1]
    
    pred_mag = torch.sqrt(pred_stft_real**2 + pred_stft_imag**2 + 1e-12)
    true_mag = torch.sqrt(true_stft_real**2 + true_stft_imag**2 + 1e-12)
    
    pred_real_c = pred_stft_real / (pred_mag**(1 - c))
    pred_imag_c = pred_stft_imag / (pred_mag**(1 - c))
    true_real_c = true_stft_real / (true_mag**(1 - c))
    true_imag_c = true_stft_imag / (true_mag**(1 - c))
    
    real_loss = torch.mean((pred_real_c - true_real_c)**2)
    imag_loss = torch.mean((pred_imag_c - true_imag_c)**2)
    mag_loss = torch.mean((pred_mag**c - true_mag**c)**2)
    
    y_pred = torch.istft(pred_stft_real + 1j*pred_stft_imag, n_fft=n_fft, hop_length=hop, win_length=win_len, window=win)
    y_true = clean_wav
    
    min_wav_len = min(y_pred.shape[-1], y_true.shape[-1])
    y_pred = y_pred[..., :min_wav_len]
    y_true = y_true[..., :min_wav_len]
    
    y_target = torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true / (torch.sum(torch.square(y_true), dim=-1, keepdim=True) + 1e-8)
    sisnr = -torch.log10(torch.sum(torch.square(y_target), dim=-1, keepdim=True) / torch.sum(torch.square(y_pred - y_target), dim=-1, keepdim=True) + 1e-8).mean()
    
    loss = cfg['lamda_ri'] * (real_loss + imag_loss) + cfg['lamda_mag'] * mag_loss + sisnr
    return loss, {'loss': loss.item()}

# Dummy data: 8 samples of 3 seconds at 16kHz
noisy = torch.randn(CONFIG['batch_size'], 48000).to(device)
clean = torch.randn(CONFIG['batch_size'], 48000).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler('cuda', enabled=True)

# Warmup
with torch.amp.autocast('cuda', enabled=CONFIG['use_amp']):
    loss, _ = compute_loss(model, noisy, clean, CONFIG, window)
scaler.scale(loss).backward()
optimizer.step()
optimizer.zero_grad()

print("Testing V2 Speed (AMP ON, batch=8)...")
t0 = time.time()
for i in range(10):
    with torch.amp.autocast('cuda', enabled=CONFIG['use_amp']):
        loss, _ = compute_loss(model, noisy, clean, CONFIG, window)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    torch.cuda.synchronize() if device.type == 'cuda' else None
t1 = time.time()
print(f"Time V2 (10 iters): {t1-t0:.4f}s -> {(t1-t0)/10:.4f}s/it")

print("Testing V1 Speed (AMP OFF, batch=4)...")
noisy_v1 = torch.randn(4, 48000).to(device)
clean_v1 = torch.randn(4, 48000).to(device)

t0 = time.time()
for i in range(10):
    loss, _ = compute_loss(model, noisy_v1, clean_v1, CONFIG, window)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize() if device.type == 'cuda' else None
t1 = time.time()
print(f"Time V1 (10 iters): {t1-t0:.4f}s -> {(t1-t0)/10:.4f}s/it")

