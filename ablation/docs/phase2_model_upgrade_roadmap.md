# DeepVQE Phase 2 Model Upgrade Roadmap

## Objective

Find a model-level upgrade that beats Baseline V4 on perceptual quality without repeating failed Phase 1 probes.

Current winner:

```text
Baseline V4
```

Phase 1 rejected directions:

```text
PReLU activation: B1a/B1b
ECA-F attention: B2
Depthwise residual blocks: C1
Heavy loss-only B4: lower PESQ
```

Detailed design notes are in `ablation/docs/model_upgrade_phase2_plan.md`.

## Success Definition

A Phase 2 candidate is promoted only if it passes all core gates:

```text
PESQ enhanced >= Baseline + 0.03
STOI enhanced >= Baseline - 0.001
SI-SDR enhanced >= Baseline - 0.10 dB
RTF measured on same device/session as Baseline
No obvious audible artifacts on fixed listening samples
```

For a research-only candidate, allow higher RTF if:

```text
PESQ enhanced >= Baseline + 0.05
```

but do not mark it deployable until latency is recovered through smaller variants, distillation, or compression.

## Roadmap Overview

| Stage | Candidate | Goal | Decision Gate |
| --- | --- | --- | --- |
| 0 | Baseline lock | Reconfirm reference metrics and eval protocol | Baseline numbers reproduced on current Kaggle/runtime |
| 1 | D1b GRU 768 | Test whether bottleneck capacity is limiting quality | Promote if quality gain clears gates |
| 2 | D1a GRU 704 | Recover latency if D1b improves quality but is too slow | Promote if close to D1b quality with better RTF |
| 3 | D3 Hybrid Head | Test whether CCM output head is limiting quality | Promote only if quality improves and listening samples are clean |
| 4 | D2 Temporal Refine | Test structured temporal refinement only if D1 gives signal | Promote if it beats D1 tradeoff |
| 5 | Stop/Distill | If best Phase 2 model is too slow, distill/compress; if none pass, stop architecture changes | Decide final winner |

## Stage 0: Baseline Lock

Before training Phase 2, freeze the reference protocol.

Required actions:

```text
1. Re-evaluate Baseline V4 on the same test.csv used for all ablations.
2. Record CUDA device, batch/eval mode, total audio seconds, processing seconds, RTF.
3. Save baseline eval CSV and summary.
4. Confirm no CPU-vs-CUDA RTF comparison is used.
```

Required artifacts:

```text
baseline_v4_quality.csv
baseline_v4_eval_summary.md
baseline_v4_config.json
```

Gate:

```text
Do not start Phase 2 if baseline eval is not reproducible.
```

## Stage 1: D1b GRU 768

Priority: highest.

Change:

```text
Baseline bottleneck GRU hidden: 576 -> 768
Encoder/decoder/residual blocks unchanged
CCM unchanged
Loss unchanged from baseline
```

Why:

The first failed probes changed local feature transforms. D1b tests a stronger temporal-context hypothesis.

Training:

```text
config_id: D1b_gru768
notebook: train_d1b_gru768_deepvqe_kaggle_v1.ipynb
resume_existing: False for official run
epochs: 80
batch_size: 8
seed: 1234
```

Gate:

```text
If PESQ >= baseline + 0.03 and guardrails pass:
  promote D1b or continue to D1a if RTF is too high.

If PESQ >= baseline + 0.05 but RTF is high:
  keep D1b as teacher/research candidate.

If PESQ < baseline + 0.02:
  reject D1 branch and move to D3.
```

## Stage 2: D1a GRU 704

Priority: conditional.

Train D1a only if D1b improves quality but costs too much latency/params.

Change:

```text
Baseline bottleneck GRU hidden: 576 -> 704
```

Training:

```text
config_id: D1a_gru704
notebook: train_d1a_gru704_deepvqe_kaggle_v1.ipynb
resume_existing: False for official run
epochs: 80
batch_size: 8
seed: 1234
```

Gate:

```text
If D1a keeps at least 70% of D1b PESQ gain with better RTF:
  prefer D1a for deploy candidate.

If D1a loses most of D1b gain:
  keep D1b as research/teacher candidate and move to D3.
```

## Stage 3: D3 Hybrid Head

Priority: second independent architecture idea.

Change:

```text
Final decoder emits 29 channels instead of 27.
27 channels -> original CCM.
2 channels -> complex residual correction.
enhanced = CCM_output + beta * correction
```

Start conservative:

```text
beta initialized small or fixed <= 0.1 at first.
No attention.
No PReLU.
No depthwise residual.
Baseline loss first.
```

Training:

```text
config_id: D3_hybrid_head
notebook: train_d3_hybrid_head_deepvqe_kaggle_v1.ipynb
resume_existing: False for official run
epochs: 80
batch_size: 8
seed: 1234
```

Extra validation:

```text
Listen to at least 20 fixed samples.
Check for musical noise, speech buzz, high-frequency artifacts, or over-suppression.
```

Gate:

```text
If PESQ >= baseline + 0.03, guardrails pass, and listening check passes:
  promote D3.

If PESQ improves but audible artifacts appear:
  reject or constrain correction branch.

If PESQ < baseline + 0.02:
  reject D3.
```

## Stage 4: D2 Temporal Refinement

Priority: conditional and last.

Train D2 only if D1 suggests bottleneck/context matters.

Change:

```text
Keep GRU hidden 576.
Add a small causal temporal refinement residual block after bottleneck FC.
```

Candidate form:

```text
y = GRU_FC(x)
y = y + CausalTemporalRefine(y)
```

Training:

```text
config_id: D2_temporal_refine
notebook: train_d2_temporal_refine_deepvqe_kaggle_v1.ipynb
resume_existing: False for official run
epochs: 80
batch_size: 8
seed: 1234
```

Gate:

```text
If D2 beats D1a/D1b quality-latency tradeoff:
  promote D2.

If D2 is worse than D1 or adds streaming complexity without quality gain:
  reject D2.
```

## Stage 5: Final Decision

After candidates are evaluated, choose exactly one state:

```text
1. Deploy Baseline V4 unchanged.
2. Promote best Phase 2 candidate.
3. Keep best Phase 2 candidate as teacher only, then distill into Baseline.
4. Stop architecture changes and move to data-centric training.
```

Winner selection order:

```text
1. Highest PESQ gain above threshold.
2. Guardrails pass.
3. Lower RTF.
4. Fewer params.
5. Simpler streaming/ONNX export.
```

## Fixed Training Protocol

Use this unless a stage explicitly overrides it:

```text
seed = 1234
sample_rate = 16000
n_fft = 512
hop_length = 256
win_length = 512
stft_window = sqrt_hann
batch_size = 8
valid_batch_size = 4
optimizer = Adam
lr = 1e-3
weight_decay = 0.0
grad_clip = 5.0
use_amp = True
epochs = 80
resume_existing = False
```

Keep:

```text
same train.csv
same valid.csv
same test.csv
same eval code
same checkpoint selection rule
same CUDA/CPU comparison discipline
```

## Required Artifacts Per Candidate

Save these before making a decision:

```text
config.json
notebook_config.json
best.pt
last.pt
train_log.csv
eval_metrics_per_sample.csv
eval_metrics_summary.csv
ablation_quality.csv
ablation_arch_benchmark.csv
ablation_summary.csv
```

If ONNX/streaming export is attempted:

```text
ablation_onnx.csv
streaming_parity_report.txt
streaming_latency_report.txt
```

## Stop Rules

Stop Phase 2 early if:

```text
D1b fails clearly and D3 fails clearly.
```

Do not spend compute on D1a if D1b has no quality signal.

Do not spend compute on D2 if D1 has no quality signal.

Do not revive ECA-F/PReLU/depthwise combinations unless Phase 2 fully fails and there is a new implementation reason to revisit them.

## Immediate Next Action

Implement and train:

```text
D1b_gru768
```

This is the first Phase 2 gate and should decide whether stronger temporal capacity is a useful direction.
