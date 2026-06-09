import time
import torch
import torchaudio

wav = torch.randn(1, 48000)

# Test 1: 48k to 16k (ratio 3:1)
t0 = time.time()
res1 = torchaudio.functional.resample(wav, 48000, 16000)
t1 = time.time()
print(f"48k -> 16k: {t1-t0:.4f}s")

# Test 2: 15200 to 16000 (ratio 19:20)
t0 = time.time()
res2 = torchaudio.functional.resample(wav, 15200, 16000)
t1 = time.time()
print(f"15200 -> 16k: {t1-t0:.4f}s")

# Test 3: 15333 to 16000 (weird ratio)
t0 = time.time()
res3 = torchaudio.functional.resample(wav, 15333, 16000)
t1 = time.time()
print(f"15333 -> 16k: {t1-t0:.4f}s")
