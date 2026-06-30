# DeepVQE Ablation Workspace

This repository is a DeepVQE research workspace focused on speech enhancement, architecture ablation, streaming deployment checks, and Phase 1 adversarial training.

The original DeepVQE implementation is based on the paper [DeepVQE: Real Time Deep Voice Quality Enhancement for Joint Acoustic Echo Cancellation, Noise Suppression and Dereverberation](https://arxiv.org/pdf/2306.03177.pdf). In this project, the active scope is noise suppression / speech enhancement with the NS-only ablation model family.

## Current Status

The latest local evaluation summary is in [result/bao_cao_danh_gia_model.md](result/bao_cao_danh_gia_model.md).

Best current generator:

| Category | Selected model | Reason |
| --- | --- | --- |
| Best overall | `D1b_gru768` | Highest PESQ, SI-SDR/STOI in the leading group, and best RTF in the current summary. |
| Best PESQ | `D1b_gru768` | PESQ enhanced `2.8736`. |
| Best STOI / SI-SDR | `B4` | Highest STOI and SI-SDR, but lower PESQ than `D1b_gru768`. |
| Do not use | `B1a` | All main quality metrics degrade strongly. |

The recommended next experiment is `GAN_D1b_gru768`: train-only GAN loss on top of the `D1b_gru768` generator. The discriminator is not used during inference, so deploy-time architecture and RTF remain those of `D1b_gru768`.

## Repository Layout

| Path | Purpose |
| --- | --- |
| [deepvqe.py](deepvqe.py) | Original/offline DeepVQE model implementation. |
| [stream/](stream) | Stateful streaming model and streaming modules. |
| [ablation/deepvqe_ablation.py](ablation/deepvqe_ablation.py) | Parameterized ablation model variants. |
| [ablation/ablation_config.py](ablation/ablation_config.py) | Training defaults and train presets such as `GAN_D1b_gru768`. |
| [ablation/discriminator.py](ablation/discriminator.py) | PatchGAN / multi-scale discriminator and GAN losses. |
| [ablation/train_ablation.py](ablation/train_ablation.py) | CLI training script for ablation variants. |
| [ablation/eval_ablation_quality.py](ablation/eval_ablation_quality.py) | PESQ/STOI/SI-SDR evaluation script. |
| [ablation/run_ablation_benchmark.py](ablation/run_ablation_benchmark.py) | Params, MACs/FLOPs, causality, streaming parity, and RTF benchmark. |
| [ablation/export_ablation_onnx.py](ablation/export_ablation_onnx.py) | Streaming ONNX export and parity check. |
| [ablation/docs/](ablation/docs) | Roadmaps, training plans, and review notes. |
| [result/](result) | Local evaluation artifacts and model comparison report. |

## Requirements

The pinned dependencies are listed in [requirements.txt](requirements.txt).

```bash
pip install -r requirements.txt
```

For quality evaluation, install the optional metric packages used by notebooks/scripts when needed:

```bash
pip install pesq pystoi torchmetrics pandas tqdm pyyaml
```

Note: this workspace is often trained on Kaggle/Colab. The local machine used for repository edits may not have PyTorch installed.

## Main Config IDs

Architecture configs are defined in [ablation/deepvqe_ablation.py](ablation/deepvqe_ablation.py).

| Config | Meaning |
| --- | --- |
| `Baseline` | Original ablation-compatible DeepVQE baseline. |
| `D1b_gru768` | Baseline with bottleneck GRU hidden size increased from 576 to 768. Current best overall generator. |
| `B1b` | Per-channel PReLU activation variant. |
| `B2` | ECA-F attention in residual blocks. |
| `B3b` | Frequency SE skip gating. |
| `B4` | Loss-focused run that leads STOI/SI-SDR in the current report. |
| `C1a-g2` | Grouped residual convolution legacy variant. |

Training presets are defined in [ablation/ablation_config.py](ablation/ablation_config.py).

| Preset | Base architecture | Training change |
| --- | --- | --- |
| `GAN_Baseline` | `Baseline` | Single-scale GAN loss. |
| `GAN_MSD3` | `Baseline` | Multi-scale discriminator, 3 scales, feature matching. |
| `GAN_D1b_gru768` | `D1b_gru768` | Recommended Phase 1 GAN run: 3-scale discriminator and feature matching. |

## Kaggle Notebook

Use this notebook for the current recommended run:

[train_phase1_gan_d1b_gru768_deepvqe_kaggle_v1.ipynb](train_phase1_gan_d1b_gru768_deepvqe_kaggle_v1.ipynb)

Key settings inside the notebook:

```python
CONFIG = {
    "config_id": "D1b_gru768",
    "result_config_id": "GAN_D1b_gru768",
    "run_name": "phase1_gan_d1b_gru768_v1",
    "num_d_scales": 3,
    "lambda_fm": 2.0,
}
```

The notebook currently uses one GPU by default. WandB run name is controlled by `CONFIG["run_name"]`, and WandB only starts when `CONFIG["use_wandb"] = True`.

## Training

Train the current recommended preset:

```bash
python ablation/train_ablation.py --config-id GAN_D1b_gru768
```

Train a plain architecture config:

```bash
python ablation/train_ablation.py --config-id D1b_gru768
```

Override manifests and device:

```bash
python ablation/train_ablation.py \
  --config-id GAN_D1b_gru768 \
  --train-manifest data/manifests/train.jsonl \
  --valid-manifest data/manifests/valid.jsonl \
  --device cuda \
  --epochs 80 \
  --batch-size 8
```

Use a YAML override:

```bash
python ablation/train_ablation.py \
  --config-id D1b_gru768 \
  --config-yaml ablation/configs/GAN_D1b_gru768.yaml
```

## Evaluation

Evaluate a trained checkpoint:

```bash
python ablation/eval_ablation_quality.py \
  --config-id GAN_D1b_gru768 \
  --checkpoint runs/ablation/GAN_D1b_gru768/best.pt \
  --manifest data/manifests/test.jsonl \
  --output results/ablation_quality.csv \
  --device cuda
```

Run architecture and streaming benchmark:

```bash
python ablation/run_ablation_benchmark.py \
  --configs GAN_D1b_gru768 \
  --device cuda \
  --output results/ablation_arch_benchmark.csv
```

Verify streaming parity:

```bash
python ablation/test_ablation_streaming.py --configs GAN_D1b_gru768
```

Export streaming ONNX:

```bash
python ablation/export_ablation_onnx.py \
  --config-id GAN_D1b_gru768 \
  --checkpoint runs/ablation/GAN_D1b_gru768/best.pt \
  --output-dir onnx_models/ablation \
  --results results/ablation_onnx.csv \
  --device cpu
```

Collect result CSVs:

```bash
python ablation/collect_ablation_results.py
```

## Inference

Run WAV inference with a checkpoint:

```bash
python infer.py \
  --input input.wav \
  --output enhanced.wav \
  --ckpt deepvqe_trained_on_DNS3.tar \
  --device cuda
```

The ablation scripts operate on STFT tensors internally with:

| STFT setting | Value |
| --- | --- |
| Sample rate | 16 kHz |
| `n_fft` | 512 |
| `win_length` | 512 |
| `hop_length` | 256 |
| Window | `sqrt_hann` for ablation training |

## Notes For Future Experiments

- Keep architecture ablations and training-objective ablations separate when comparing results.
- `GAN_D1b_gru768` changes training only; inference should be benchmarked as `D1b_gru768`.
- Compare RTF only on the same hardware/device type.
- Keep a fixed test manifest for all reported PESQ/STOI/SI-SDR comparisons.
- Do not promote a variant based on one metric alone; use PESQ, STOI, SI-SDR, listening checks, and RTF together.

