import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "train_phase2_mamba_deepvqe_kaggle_v1.ipynb"

PHASE2_FILES = [
    "ablation/modules/__init__.py",
    "ablation/modules/mamba.py",
    "ablation/deepvqe_ablation.py",
    "ablation/ablation_config.py",
    "ablation/train_ablation.py",
    "ablation/configs/Mamba_b2_h384.yaml",
    "ablation/configs/GAN_Mamba_b2_h384.yaml",
    "ablation/test_phase2_mamba.py",
    "ablation/scripts/benchmark_rtf.py",
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip() + "\n",
    }


def phase2_patch_cell():
    payload = {}
    for rel in PHASE2_FILES:
        path = ROOT / rel
        payload[rel] = path.read_text(encoding="utf-8")

    files_literal = json.dumps(payload, ensure_ascii=False, indent=2)
    return code(
        f"""
from pathlib import Path
import importlib

PHASE2_FILES = {files_literal}

for rel_path, content in PHASE2_FILES.items():
    target = Path(rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding='utf-8')
    print(f'Wrote {{target}}')

importlib.invalidate_caches()
print('Phase 2 Mamba runtime patch applied.')
"""
    )


cells = [
    md(
        """
# Huấn luyện DeepVQE Phase 2 Mamba trên Kaggle

Notebook này dùng cấu hình `Mamba_b2_h384` để thay bottleneck GRU bằng Native PyTorch Mamba/SSM có cache streaming. Notebook cũng có sẵn cell smoke test, RTF benchmark, quality eval và ONNX export.
"""
    ),
    md("## 1. Cài đặt môi trường & chuẩn bị Kaggle Working Dir"),
    code(
        """
!pip install -q wandb gdown matplotlib soundfile pandas tqdm einops pesq pystoi pyyaml torchmetrics psutil onnx onnxruntime onnxsim

import os
import sys
import shutil
from pathlib import Path

WORK_DIR = Path('/kaggle/working/DeepVQE_Workspace')
WORK_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_DIR)
print(f'Working dir: {Path.cwd()}')

SHARED_CHECKPOINTS_DIR = Path('/kaggle/working/checkpoint_phase2_mamba')
SHARED_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
print(f'Checkpoint dir: {SHARED_CHECKPOINTS_DIR}')
"""
    ),
    md("## 2. Clone mã nguồn DeepVQE"),
    code(
        """
import subprocess

GIT_REPO = 'https://github.com/hoxuanphu/Pdeepvqe.git'
GIT_BRANCH = None  # Ví dụ: 'phase2-mamba' nếu bạn đã push branch riêng.
REPO_DIR = WORK_DIR / 'deepvqe'

if not REPO_DIR.exists():
    cmd = ['git', 'clone']
    if GIT_BRANCH:
        cmd += ['--branch', GIT_BRANCH]
    cmd += [GIT_REPO, str(REPO_DIR)]
    subprocess.run(cmd, check=True)
else:
    print(f'Thư mục {REPO_DIR} đã tồn tại. Đang cập nhật code mới nhất...')
    subprocess.run(['git', '-C', str(REPO_DIR), 'pull', '--ff-only'], check=False)

os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
print(f'Repo dir: {Path.cwd()}')
"""
    ),
    md("## 2.5 Áp dụng code Phase 2 Mamba vào runtime"),
    phase2_patch_cell(),
    md("## 2.6 Kiểm tra cấu hình Mamba"),
    code(
        """
from ablation.deepvqe_ablation import get_ablation_config, DeepVQE_Ablation, StreamDeepVQE_Ablation, count_parameters
from ablation.ablation_config import get_train_config, get_model_config_id

print('Model config id for GAN preset:', get_model_config_id('GAN_Mamba_b2_h384'))
cfg = get_ablation_config('Mamba_b2_h384')
print(cfg)

model = DeepVQE_Ablation.from_config_id('Mamba_b2_h384')
stream_model = StreamDeepVQE_Ablation.from_config_id('Mamba_b2_h384')
print(f'Mamba params: {count_parameters(model):,}')
print('Stream cache names:')
for name in stream_model.get_cache_names():
    print(' ', name)
"""
    ),
    md("## 3. Tải bộ dữ liệu VoiceBank-DEMAND"),
    code(
        """
import os
import zipfile
import subprocess
from pathlib import Path

data_dir = WORK_DIR / 'data' / 'voicebank-demand'
data_dir.mkdir(parents=True, exist_ok=True)

datasets = {
    'clean_trainset_28spk_wav': 'https://drive.google.com/file/d/1NJr2O4Ik6ueSFlIGSvub8dnFXGTHJ2PG/view?usp=sharing',
    'noisy_trainset_28spk_wav': 'https://drive.google.com/file/d/1OqpDIvpVyaTnMbwY1Qt__hfX3X4siMtU/view?usp=sharing',
    'clean_testset_wav': 'https://drive.google.com/file/d/1GQc-T1R4FNrhRjTn7AAvAenZTIQEazeH/view?usp=sharing',
    'noisy_testset_wav': 'https://drive.google.com/file/d/1rimmCqxXRYRIXZcPkGjQiacr6j1QsMAH/view?usp=sharing',
}

for folder_name, url in datasets.items():
    extract_path = data_dir / folder_name
    zip_path = data_dir / f'{folder_name}.zip'
    if extract_path.exists() and any(extract_path.iterdir()):
        print(f'{folder_name}: đã có dữ liệu, bỏ qua.')
        continue
    if not zip_path.exists():
        print(f'Tải {folder_name}.zip...')
        subprocess.run(['gdown', '--fuzzy', url, '-O', str(zip_path)], check=True)
    print(f'Giải nén {zip_path.name}...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

print(f'Dataset dir: {data_dir}')
"""
    ),
    md("## 4. Tạo CSV train/valid/test"),
    code(
        """
import glob
import pandas as pd

def create_csv(clean_dir, noisy_dir, output_csv):
    clean_files = sorted(glob.glob(str(Path(clean_dir) / '*.wav')))
    noisy_files = sorted(glob.glob(str(Path(noisy_dir) / '*.wav')))
    if len(clean_files) != len(noisy_files):
        raise ValueError(f'Số lượng file không khớp: {len(clean_files)} clean vs {len(noisy_files)} noisy')
    rows = []
    for clean_path, noisy_path in zip(clean_files, noisy_files):
        name = Path(clean_path).name
        rows.append({
            'ID': Path(name).stem,
            'clean_wav': str(Path(clean_path).resolve()),
            'noisy_wav': str(Path(noisy_path).resolve()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f'Wrote {output_csv}: {len(df)} samples')

create_csv(data_dir / 'clean_trainset_28spk_wav', data_dir / 'noisy_trainset_28spk_wav', data_dir / 'train_full.csv')
create_csv(data_dir / 'clean_testset_wav', data_dir / 'noisy_testset_wav', data_dir / 'test.csv')

df_train_full = pd.read_csv(data_dir / 'train_full.csv').sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(len(df_train_full) * 0.90)
df_train_full.iloc[:split_idx].to_csv(data_dir / 'train.csv', index=False)
df_train_full.iloc[split_idx:].to_csv(data_dir / 'valid.csv', index=False)

print('train/valid/test ready:')
for name in ['train.csv', 'valid.csv', 'test.csv']:
    p = data_dir / name
    print(f'  {p}: {len(pd.read_csv(p))} rows')
"""
    ),
    md("## 5. Cấu hình chạy Phase 2"),
    code(
        """
import json
from pathlib import Path

CONFIG = {
    # Đổi thành 'GAN_Mamba_b2_h384' nếu muốn train Mamba cùng Phase 1 GAN loss.
    'config_id': 'Mamba_b2_h384',
    'run_name': 'phase2_mamba_b2_h384_v1',
    'seed': 1234,

    'sample_rate': 16000,
    'n_fft': 512,
    'hop_length': 256,
    'win_length': 512,
    'stft_window': 'sqrt_hann',

    'batch_size': 8,
    'valid_batch_size': 8,
    'grad_accum_steps': 1,
    'epochs': 80,
    'num_workers': 2,
    'persistent_workers': False,
    'prefetch_factor': 2,
    'progress_update_every': 10,
    'max_consecutive_nonfinite_batches': 25,
    'lr': 3e-4,
    'weight_decay': 0.0,
    'grad_clip_norm': 1.0,
    'scheduler_factor': 0.5,
    'scheduler_patience': 5,
    'scheduler_min_lr': 1e-6,
    'early_stopping_patience': 15,

    'lamda_ri': 30.0,
    'lamda_mag': 70.0,
    'compress_factor': 0.3,
    'lamda_adv': 0.05,
    'lambda_fm': 2.0,

    'train_csv': str(data_dir / 'train.csv'),
    'valid_csv': str(data_dir / 'valid.csv'),
    'test_csv': str(data_dir / 'test.csv'),

    'output_dir': str(SHARED_CHECKPOINTS_DIR / 'phase2_mamba_b2_h384_v1'),
    'results_dir': str(WORK_DIR / 'results' / 'phase2_mamba_b2_h384_v1'),
    'onnx_dir': str(WORK_DIR / 'onnx_models' / 'phase2_mamba_b2_h384_v1'),
    'resume_existing': False,
    'resume_checkpoint': 'last.pt',

    'use_amp': False,
    'augment': True,
    'aug_gain_range_db': [-6.0, 6.0],
    'aug_snr_remix_range': [0.0, 20.0],
    'aug_prob': 0.5,

    'use_wandb': True,
    'wandb_project': 'DeepVQE-Phase2-Mamba',
    'wandb_tags': ['phase2', 'mamba', 'kaggle'],
    'wandb_notes': 'Phase 2 Mamba_b2_h384 sequence-modeling run on Kaggle.',
    'wandb_watch': True,
    'wandb_watch_log_freq': 100,
    'eval_pesq_every': 5,
    'eval_pesq_samples': 50,
    'log_audio_every': 5,
    'log_audio_samples': 1,

    'run_training': True,
    'run_smoke_tests': True,
    'run_rtf_benchmark': True,
    'run_quality_eval': True,
    'run_onnx_export': False,
}

Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
Path(CONFIG['results_dir']).mkdir(parents=True, exist_ok=True)
Path(CONFIG['onnx_dir']).mkdir(parents=True, exist_ok=True)

print(json.dumps(CONFIG, indent=2))
"""
    ),
    md("## 5.5 Đăng nhập WandB"),
    code(
        """
if CONFIG.get('use_wandb', True):
    import os
    import wandb

    wandb_key = os.environ.get('WANDB_API_KEY')
    if not wandb_key:
        try:
            from kaggle_secrets import UserSecretsClient
            wandb_key = UserSecretsClient().get_secret('WANDB_API_KEY')
        except Exception as exc:
            print(f'Không lấy được WANDB_API_KEY từ Kaggle Secrets/env: {exc}')

    if wandb_key:
        wandb.login(key=wandb_key, relogin=True)
        print('Đã đăng nhập WandB bằng WANDB_API_KEY.')
    else:
        print('Chưa có WANDB_API_KEY; train_ablation.py vẫn chạy, nhưng wandb có thể chuyển sang anonymous/offline tùy cấu hình môi trường.')
else:
    print('WandB đang tắt trong CONFIG.')
"""
    ),
    md("## 6. Ghi config override cho `train_ablation.py`"),
    code(
        """
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
override = {
    'experiment': {
        'name': CONFIG['run_name'],
        'config_id': CONFIG['config_id'],
        'seed': CONFIG['seed'],
        'output_dir': CONFIG['output_dir'],
        'resume_from': str(Path(CONFIG['output_dir']) / CONFIG['resume_checkpoint']) if CONFIG['resume_existing'] else None,
    },
    'stft': {
        'n_fft': CONFIG['n_fft'],
        'hop_length': CONFIG['hop_length'],
        'win_length': CONFIG['win_length'],
        'window': CONFIG['stft_window'],
    },
    'data': {
        'sample_rate': CONFIG['sample_rate'],
        'clip_seconds': 3.0,
        'num_workers': CONFIG['num_workers'],
        'persistent_workers': CONFIG['persistent_workers'],
        'prefetch_factor': CONFIG['prefetch_factor'],
        'pin_memory': torch.cuda.is_available(),
        'train_manifest': CONFIG['train_csv'],
        'valid_manifest': CONFIG['valid_csv'],
        'test_manifest': CONFIG['test_csv'],
        'augment': CONFIG['augment'],
        'aug_gain_range_db': CONFIG['aug_gain_range_db'],
        'aug_snr_remix_range': CONFIG['aug_snr_remix_range'],
        'aug_prob': CONFIG['aug_prob'],
    },
    'optimizer': {
        'lr': CONFIG['lr'],
        'weight_decay': CONFIG['weight_decay'],
        'grad_clip_norm': CONFIG['grad_clip_norm'],
    },
    'scheduler': {
        'factor': CONFIG['scheduler_factor'],
        'patience': CONFIG['scheduler_patience'],
        'min_lr': CONFIG['scheduler_min_lr'],
    },
    'training': {
        'device': device,
        'batch_size': CONFIG['batch_size'],
        'valid_batch_size': CONFIG.get('valid_batch_size', CONFIG['batch_size']),
        'grad_accum_steps': CONFIG.get('grad_accum_steps', 1),
        'progress_update_every': CONFIG.get('progress_update_every', 10),
        'max_consecutive_nonfinite_batches': CONFIG.get('max_consecutive_nonfinite_batches', 25),
        'epochs': CONFIG['epochs'],
        'use_amp': CONFIG['use_amp'],
        'disable_tqdm': False,
    },
    'loss': {
        'lamda_ri': CONFIG['lamda_ri'],
        'lamda_mag': CONFIG['lamda_mag'],
        'compress_factor': CONFIG['compress_factor'],
        'lamda_adv': CONFIG['lamda_adv'],
        'lambda_fm': CONFIG['lambda_fm'],
    },
    'logging': {
        'use_wandb': CONFIG.get('use_wandb', True),
        'wandb_project': CONFIG.get('wandb_project', 'DeepVQE-Phase2-Mamba'),
        'wandb_tags': CONFIG.get('wandb_tags', ['phase2', 'mamba', 'kaggle']),
        'wandb_notes': CONFIG.get('wandb_notes', 'Phase 2 Mamba_b2_h384 sequence-modeling run on Kaggle.'),
        'wandb_watch': CONFIG.get('wandb_watch', True),
        'wandb_watch_log_freq': CONFIG.get('wandb_watch_log_freq', 100),
        'eval_pesq_every': CONFIG.get('eval_pesq_every', 5),
        'eval_pesq_samples': CONFIG.get('eval_pesq_samples', 50),
        'log_audio_every': CONFIG.get('log_audio_every', 5),
        'log_audio_samples': CONFIG.get('log_audio_samples', 1),
    },
}

override_path = Path(CONFIG['output_dir']) / 'phase2_mamba_train_override.json'
override_path.write_text(json.dumps(override, indent=2), encoding='utf-8')
print(f'Override config: {override_path}')
print(f'Device: {device}')
"""
    ),
    md("## 7. Smoke test Phase 2 trước khi train"),
    code(
        """
import subprocess
import sys
import pandas as pd
from IPython.display import display, FileLink

def run_py(args, check=True):
    cmd = [sys.executable, *[str(arg) for arg in args]]
    print('\\n$ ' + ' '.join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(REPO_DIR), text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f'Lệnh lỗi với exit code {result.returncode}: {cmd}')
    return result

if CONFIG['run_smoke_tests']:
    run_py(['-m', 'py_compile', 'ablation/modules/mamba.py', 'ablation/deepvqe_ablation.py', 'ablation/test_phase2_mamba.py'])
    run_py(['ablation/test_phase2_mamba.py', '--frames', '3'])
    run_py(['ablation/test_ablation_streaming.py', '--configs', 'Mamba_b2_h384', '--frames', '3', '--atol', '1e-5'])
else:
    print('run_smoke_tests=False, bỏ qua smoke test.')
"""
    ),
    md("## 8. Train Mamba"),
    code(
        """
train_cmd = [
    'ablation/train_ablation.py',
    '--config-id', CONFIG['config_id'],
    '--config-json', override_path,
    '--train-manifest', CONFIG['train_csv'],
    '--valid-manifest', CONFIG['valid_csv'],
    '--output-dir', CONFIG['output_dir'],
    '--device', device,
    '--epochs', str(CONFIG['epochs']),
    '--batch-size', str(CONFIG['batch_size']),
    '--num-workers', str(CONFIG['num_workers']),
    '--early-stop-patience', str(CONFIG['early_stopping_patience']),
]
if CONFIG['resume_existing']:
    resume_path = Path(CONFIG['output_dir']) / CONFIG['resume_checkpoint']
    if resume_path.exists():
        train_cmd += ['--resume', resume_path]
if not torch.cuda.is_available():
    train_cmd += ['--no-pin-memory']

if CONFIG['run_training']:
    run_py(train_cmd)
else:
    print('run_training=False, bỏ qua training.')
"""
    ),
    md("## 9. Kiểm tra checkpoint & log"),
    code(
        """
output_dir = Path(CONFIG['output_dir'])
for name in ['best.pt', 'last.pt', 'config.json', 'train_log.txt']:
    p = output_dir / name
    if p.exists():
        print(f'{name}: {p} ({p.stat().st_size / 1024:.1f} KB)')
    else:
        print(f'MISSING: {p}')

log_path = output_dir / 'train_log.txt'
if log_path.exists():
    print('\\nLast log lines:')
    print('\\n'.join(log_path.read_text(encoding='utf-8', errors='ignore').splitlines()[-10:]))
"""
    ),
    md("## 10. RTF benchmark CPU/GPU"),
    code(
        """
rtf_csv = Path(CONFIG['results_dir']) / 'phase2_rtf_benchmark.csv'

if CONFIG['run_rtf_benchmark']:
    run_py([
        'ablation/scripts/benchmark_rtf.py',
        '--output', rtf_csv,
        '--configs', 'Baseline', 'D1b_gru768', 'Mamba_b2_h384',
        '--devices', 'cpu', 'cuda',
        '--frames', '63',
        '--warmup', '1',
        '--repeats', '3',
    ])
else:
    print('run_rtf_benchmark=False, bỏ qua RTF benchmark.')

if rtf_csv.exists():
    display(pd.read_csv(rtf_csv))
"""
    ),
    md("## 11. Quality eval trên test set"),
    code(
        """
quality_csv = Path(CONFIG['results_dir']) / 'ablation_quality.csv'
best_ckpt = Path(CONFIG['output_dir']) / 'best.pt'

if CONFIG['run_quality_eval']:
    if not best_ckpt.exists():
        raise FileNotFoundError(f'Không tìm thấy best checkpoint: {best_ckpt}')
    run_py([
        'ablation/eval_ablation_quality.py',
        '--config-id', CONFIG['config_id'],
        '--checkpoint', best_ckpt,
        '--manifest', CONFIG['test_csv'],
        '--output', quality_csv,
        '--device', device,
    ])
else:
    print('run_quality_eval=False, bỏ qua quality eval.')

if quality_csv.exists():
    display(pd.read_csv(quality_csv))
"""
    ),
    md("## 12. ONNX export streaming"),
    code(
        """
onnx_csv = Path(CONFIG['results_dir']) / 'ablation_onnx.csv'

if CONFIG['run_onnx_export']:
    if not best_ckpt.exists():
        raise FileNotFoundError(f'Không tìm thấy best checkpoint: {best_ckpt}')
    run_py([
        'ablation/export_ablation_onnx.py',
        '--config-id', CONFIG['config_id'],
        '--checkpoint', best_ckpt,
        '--output-dir', CONFIG['onnx_dir'],
        '--results', onnx_csv,
        '--device', 'cpu',
    ])
else:
    print('run_onnx_export=False, bỏ qua ONNX export.')

if onnx_csv.exists():
    display(pd.read_csv(onnx_csv))
"""
    ),
    md("## 13. Nén kết quả để tải về"),
    code(
        """
archive_base = Path('/kaggle/working/phase2_mamba_training_output')
archive_path = archive_base.with_suffix('.zip')
if archive_path.exists():
    archive_path.unlink()

shutil.make_archive(str(archive_base), 'zip', root_dir=str(WORK_DIR))
print(f'Đã nén kết quả: {archive_path}')
display(FileLink(str(archive_path)))
"""
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Created {OUTPUT}")
