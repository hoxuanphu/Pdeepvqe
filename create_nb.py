import json

mixing_code = """
import os
import random
import torchaudio
import torch
import math
import glob
import json
from tqdm import tqdm

# --- CONFIGURATION ---
SAMPLE_RATE = 16000
DATA_DIR = "/content/drive/MyDrive/TSE_Dataset"
OUTPUT_DIR = os.path.join(DATA_DIR, "mixed_dataset")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Quét file (Lấy danh sách wav/flac)
print("Scanning files...")
libri_files = glob.glob(os.path.join(DATA_DIR, "LibriSpeech", "*", "*", "*", "*.flac"))
musan_noise = glob.glob(os.path.join(DATA_DIR, "musan", "noise", "*", "*.wav")) + \
              glob.glob(os.path.join(DATA_DIR, "musan", "music", "*", "*.wav")) + \
              glob.glob(os.path.join(DATA_DIR, "musan", "speech", "*", "*.wav"))
wham_noise = glob.glob(os.path.join(DATA_DIR, "cv", "noise", "*.wav")) # Tùy cấu trúc wham
rir_files = glob.glob(os.path.join(DATA_DIR, "RIRS_NOISES", "simulated_rirs", "*", "*", "*.wav")) + \
            glob.glob(os.path.join(DATA_DIR, "RIRS_NOISES", "real_rirs_isotropic_noises", "*.wav"))

all_noise = musan_noise + wham_noise

# Nhóm LibriSpeech theo speaker
speakers = {}
for f in libri_files:
    parts = f.replace("\\\\", "/").split('/')
    spk_id = parts[-3]
    if spk_id not in speakers:
        speakers[spk_id] = []
    speakers[spk_id].append(f)

spk_list = list(speakers.keys())
print(f"Found {len(spk_list)} speakers, {len(all_noise)} noises, {len(rir_files)} RIRs")

def get_audio(path, max_len=64000):
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(wav)
    if wav.shape[1] > max_len:
        start = random.randint(0, wav.shape[1] - max_len)
        wav = wav[:, start:start+max_len]
    elif wav.shape[1] < max_len:
        repeats = math.ceil(max_len / wav.shape[1])
        wav = wav.repeat(1, repeats)[:, :max_len]
    return wav

def convolve_rir(audio, rir_path):
    rir_wav, sr = torchaudio.load(rir_path)
    if sr != SAMPLE_RATE:
        rir_wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(rir_wav)
    rir_wav = rir_wav[random.randint(0, rir_wav.shape[0]-1):].unsqueeze(0)
    audio = torchaudio.functional.fftconvolve(audio, rir_wav)
    return audio[:, :64000]

def calc_energy(wav):
    return torch.sum(wav ** 2) / wav.shape[1]

# --- MIXING PIPELINE ---
NUM_SAMPLES = 5000 # Chạy thử 5000 samples
metadata_list = []

print("Mixing dataset...")
for i in tqdm(range(NUM_SAMPLES)):
    try:
        # Chọn Target
        target_spk = random.choice(spk_list)
        target_files = speakers[target_spk]
        if len(target_files) < 2: continue
        target_path = random.choice(target_files)
        enroll_path = random.choice([f for f in target_files if f != target_path])
        
        target_wav = get_audio(target_path)
        enroll_wav = get_audio(enroll_path)
        
        # RIR
        rir_path = random.choice(rir_files)
        target_reverb = convolve_rir(target_wav, rir_path)
        target_energy = calc_energy(target_reverb)
        
        # PHÂN LOẠI CASE (10% cơ hội cho Negative Case)
        rand_case = random.random()
        is_negative = rand_case < 0.10
        
        if is_negative:
            # --- NEGATIVE CASE ---
            # Mixture không có Target
            mixture = torch.zeros_like(target_reverb)
            # Target = 0 để model học cách dập âm
            target_reverb = torch.zeros_like(target_reverb)
            case_name = "absent_speaker"
        else:
            # --- POSITIVE CASE ---
            mixture = target_reverb.clone()
            case_name = "full_tse" if rand_case < 0.7 else ("target_noise" if rand_case > 0.85 else "target_interferer")
        
        # Thêm Interferer
        # Negative Case -> 100% có Interferer. Positive Case -> 85% có Interferer.
        if is_negative or rand_case < 0.85:
            interf_spk = random.choice([s for s in spk_list if s != target_spk])
            interf_path = random.choice(speakers[interf_spk])
            interf_wav = get_audio(interf_path)
            interf_reverb = convolve_rir(interf_wav, random.choice(rir_files))
            
            sir_db = random.uniform(-5, 10)
            interf_energy = calc_energy(interf_reverb)
            scale_sir = math.sqrt(target_energy / (interf_energy * (10 ** (sir_db / 10)) + 1e-8))
            mixture += interf_reverb * scale_sir
            
        # Thêm Noise
        # Negative Case -> 100% có Noise. Positive Case -> 85% có Noise.
        if is_negative or rand_case < 0.70 or rand_case > 0.85:
            noise_path = random.choice(all_noise)
            noise_wav = get_audio(noise_path)
            
            snr_db = random.uniform(-5, 15)
            noise_energy = calc_energy(noise_wav)
            scale_snr = math.sqrt(target_energy / (noise_energy * (10 ** (snr_db / 10)) + 1e-8))
            mixture += noise_wav * scale_snr
            
        # Chống Clip
        max_amp = torch.max(torch.abs(mixture))
        if max_amp > 0.9:
            mixture = mixture * (0.9 / max_amp)
            if not is_negative:
                target_reverb = target_reverb * (0.9 / max_amp)
            enroll_wav = enroll_wav * (0.9 / max_amp)
            
        # Lưu file
        mix_name = f"mix_{i:05d}.wav"
        tgt_name = f"tgt_{i:05d}.wav"
        enl_name = f"enl_{i:05d}.wav"
        
        torchaudio.save(os.path.join(OUTPUT_DIR, mix_name), mixture, SAMPLE_RATE)
        torchaudio.save(os.path.join(OUTPUT_DIR, tgt_name), target_reverb, SAMPLE_RATE)
        torchaudio.save(os.path.join(OUTPUT_DIR, enl_name), enroll_wav, SAMPLE_RATE)
        
        # Caching metadata (Thêm trường is_negative để map với train_personalized.py)
        metadata_list.append({
            "mixture": mix_name,
            "target": tgt_name,
            "enrollment": enl_name,
            "target_speaker": target_spk,
            "target_utterance": target_path.split("/")[-1],
            "enroll_utterance": enroll_path.split("/")[-1],
            "case": case_name,
            "is_negative": is_negative
        })
    except Exception as e:
        continue

with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata_list, f, indent=4)
print("Hoàn tất tạo Dataset!")
"""

nb = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Bước 1: Setup Google Drive & Thư mục"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "!pip install torchaudio speechbrain tqdm\\n",
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "import os\n",
    "data_dir = '/content/drive/MyDrive/TSE_Dataset'\n",
    "os.makedirs(data_dir, exist_ok=True)\n",
    "os.chdir(data_dir)\n",
    "print(f'Working directory: {os.getcwd()}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Bước 2: Tải & Giải nén các Dataset (Chỉ chạy 1 lần)"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "import urllib.request\n",
    "import tarfile\n",
    "import zipfile\n",
    "import os\n",
    "from tqdm import tqdm\n",
    "\n",
    "class DownloadProgressBar(tqdm):\n",
    "    def update_to(self, b=1, bsize=1, tsize=None):\n",
    "        if tsize is not None:\n",
    "            self.total = tsize\n",
    "        self.update(b * bsize - self.n)\n",
    "\n",
    "datasets = {\n",
    "    'train-clean-100': 'https://openslr.trmal.net/resources/12/train-clean-100.tar.gz',\n",
    "    'train-clean-360': 'https://openslr.trmal.net/resources/12/train-clean-360.tar.gz',\n",
    "    'musan': 'https://openslr.trmal.net/resources/17/musan.tar.gz',\n",
    "    'wham_noise': 'https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip',\n",
    "    'rirs_noises': 'https://openslr.trmal.net/resources/28/rirs_noises.zip'\n",
    "}\n",
    "for name, url in datasets.items():\n",
    "    file_name = url.split('/')[-1]\n",
    "    if not os.path.exists(file_name):\n",
    "        print(f'\\nĐang tải {name}...')\n",
    "        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=name) as t:\n",
    "            urllib.request.urlretrieve(url, filename=file_name, reporthook=t.update_to)\n",
    "    else:\n",
    "        print(f'\\n{name} đã được tải.')\n",
    "    \n",
    "    print(f'Đang giải nén {name}... (vui lòng đợi)')\n",
    "    if file_name.endswith('.tar.gz'):\n",
    "        with tarfile.open(file_name, 'r:gz') as tar:\n",
    "            tar.extractall()\n",
    "    elif file_name.endswith('.zip'):\n",
    "        with zipfile.ZipFile(file_name, 'r') as zip_ref:\n",
    "            zip_ref.extractall()\n",
    "print('\\nHoàn tất tải và giải nén dữ liệu gốc!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
       "# Bước 3: Mix Audio động và Lưu vào Drive\n", 
       "Đoạn code này áp dụng công thức 70% TSE, 15% Noise, 15% Interferer, và 10% Negative Case (Absent Speaker)."
    ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [line + "\n" for line in mixing_code.split("\n")[:-1]]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("d:/AI20K/deepvqe/data_prep_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
