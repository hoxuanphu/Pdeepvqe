# DeepVQE Model Upgrade Phase 2 Plan

## 1. Current Decision State

Baseline V4 is the current winner. Recent ablation results do not justify continuing the first set of architecture/loss probes:

| Variant | Change | Decision |
| --- | --- | --- |
| B1a | Shared PReLU | Reject: quality dropped clearly. |
| B1b | Per-channel PReLU | Reject: not better than baseline. |
| B2 | ECA-F in residual blocks | Reject: tiny PESQ gain, about 5x slower RTF. |
| C1 | Depthwise residual blocks | Reject: result not good enough. |
| B4 | Asymmetric mag + MR-STFT loss | Reject as winner: PESQ dropped despite slight STOI/SI-SDR gains. |

Do not prioritize `B3a`, `B3b`, `C2`, `C3`, or `C4` now because they combine failed signals: PReLU, ECA-F, and/or depthwise residual blocks.

## 2. Phase 2 Principle

Small local tweaks have not beaten Baseline V4. Phase 2 should modify only the parts most likely to increase quality:

1. Bottleneck temporal context.
2. Output mask/filter head.

Keep the baseline encoder/decoder/residual stack unchanged unless a Phase 2 probe gives a strong signal.

Primary metrics:

- PESQ enhanced.
- STOI enhanced as guardrail.
- SI-SDR enhanced as guardrail.
- RTF overall on the same hardware/session as baseline.
- Params and checkpoint compatibility notes.

## 3. Option D1: Stronger Bottleneck

### Idea

Keep the baseline architecture, but increase the GRU hidden size in the bottleneck:

```text
Baseline: gru_hidden = 576
D1a:      gru_hidden = 704
D1b:      gru_hidden = 768
```

The bottleneck receives flattened encoder features `(128 * 9 = 1152)` per frame and maps:

```text
1152 -> GRU hidden -> 1152
```

Increasing `gru_hidden` adds temporal modeling capacity without changing the encoder, decoder, skip connections, or CCM output mechanism.

### Why This Is First

The failed probes mostly changed local feature transforms. D1 targets temporal context, which is more likely to affect difficult non-stationary noise and speech preservation.

### Expected Tradeoff

| Aspect | Expected |
| --- | --- |
| Quality | Highest chance of real gain among Phase 2 options. |
| Params | Increases moderately. |
| RTF | Increases; must be measured. |
| Implementation risk | Low. |
| Streaming parity | Straightforward if stream bottleneck hidden cache shape is updated. |

### Train Order

Train only one first:

```text
D1b: gru_hidden = 768
```

If D1b improves quality but RTF is too high, train:

```text
D1a: gru_hidden = 704
```

### Pass Criteria

D1 passes if:

```text
PESQ >= baseline + 0.03
STOI >= baseline - 0.001
SI-SDR >= baseline - 0.10 dB
RTF overall <= 1.5x baseline
```

If PESQ improves by at least `+0.05`, allow RTF up to `2.0x baseline` as a research candidate, then consider distillation/compression.

### Reject Criteria

Reject D1 if:

```text
PESQ < baseline + 0.02
```

or if STOI/SI-SDR drops beyond guardrails.

## 4. Option D2: Bottleneck Temporal Refinement

### Idea

Keep `gru_hidden = 576`, then add a small causal temporal refinement block after the bottleneck projection.

Concept:

```text
x_flat = rearrange(x, "b c t f -> b t (c f)")
y = GRU(x_flat)
y = FC(y)
y = y + TemporalRefine(y)
y = rearrange(y, "b t (c f) -> b c t f")
```

Candidate refinement block:

```text
LayerNorm/BatchNorm
Causal depthwise Conv1d over time
Pointwise Linear/Conv1d
ELU
Residual add
```

### Why This Exists

If D1b improves quality, that suggests the bottleneck is the right place to invest. D2 tests whether a structured temporal residual block can improve quality with less parameter growth than a larger GRU.

### Expected Tradeoff

| Aspect | Expected |
| --- | --- |
| Quality | Medium upside. |
| Params | Small to moderate increase. |
| RTF | Depends on Conv1d implementation; benchmark required. |
| Implementation risk | Medium. |
| Streaming parity | Requires a causal temporal cache for the refinement block. |

### When To Train

Train D2 only if one of these is true:

```text
D1 improves quality but RTF/params are too high.
D1 is close to passing and suggests bottleneck context matters.
```

Do not train D2 before D1.

## 5. Option D3: Hybrid Output Head

### Idea

Keep the baseline CCM output, but add a small direct complex residual/correction branch at the output head.

Baseline final decoder emits `27` channels used by CCM:

```text
deblock1 -> 27 channels -> CCM -> enhanced complex spectrogram
```

D3 emits extra correction channels:

```text
deblock1 -> 29 channels
27 channels -> CCM
2 channels  -> complex residual correction
enhanced = CCM(mask27, noisy) + beta * correction2
```

Alternative blend:

```text
enhanced = CCM(mask27, noisy) + alpha * CRM(noisy, correction2)
```

Start with the residual correction version because it is easier to constrain and debug.

### Why This Is Worth Testing

The current CCM head applies a 3x3 complex convolving mask. It may leave residual artifacts that a small correction branch can fix. This option changes the final synthesis behavior directly rather than changing local residual blocks.

### Expected Tradeoff

| Aspect | Expected |
| --- | --- |
| Quality | Good upside if CCM is the limiting factor. |
| Params | Small increase. |
| RTF | Small to moderate increase. |
| Implementation risk | Medium-high. |
| Streaming parity | Must update stream CCM/head output contract. |

### Pass Criteria

D3 passes if:

```text
PESQ >= baseline + 0.03
STOI >= baseline - 0.001
SI-SDR >= baseline - 0.10 dB
RTF overall <= 1.25x baseline
```

Because D3 changes the output head, require listening checks on at least 20 fixed samples before calling it a winner.

### Reject Criteria

Reject D3 if it increases PESQ but creates audible musical noise, speech buzz, or high-frequency artifacts.

## 6. Recommended Execution Order

Run in this exact order:

```text
1. D1b: Baseline + GRU hidden 768
2. D1a: Baseline + GRU hidden 704, only if D1b quality improves but RTF is too high
3. D3: Hybrid CCM + residual complex correction head
4. D2: Bottleneck temporal refinement, only if D1 suggests bottleneck context matters
```

Stop early if a candidate passes strongly.

## 7. Training Protocol

Use the same locked protocol as Baseline V4:

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
resume_existing = False for first official run
```

Keep train/valid/test CSVs identical to baseline.

Checkpoint selection:

1. Save `last.pt` every epoch.
2. Save `best.pt` by valid loss for continuity.
3. For final decision, evaluate a fixed valid subset by PESQ/STOI/SI-SDR and record which checkpoint wins perceptual metrics.

Do not compare B4-style modified loss values against baseline loss values. For Phase 2 architecture probes, keep the baseline loss first unless explicitly testing a separate loss variant.

## 8. Evaluation Protocol

For every candidate, save:

```text
ablation_arch_benchmark.csv
ablation_quality.csv
ablation_onnx.csv, if exported
ablation_summary.csv
best.pt
last.pt
config.json
notebook_config.json
```

Quality eval must include:

```text
Test samples
Model params
Device
Total audio seconds
Processing seconds
PESQ enhanced/noisy/delta
STOI enhanced/noisy/delta
SI-SDR enhanced/noisy/delta
RTF mean
RTF overall
```

RTF rules:

- Compare RTF only on the same device type and preferably the same Kaggle session.
- Do not compare CPU RTF with CUDA RTF.
- Warm up before timed benchmark if using a microbenchmark.

## 9. Decision Table

| Result Pattern | Decision |
| --- | --- |
| PESQ +0.03 or more, guardrails pass, RTF acceptable | Promote candidate. |
| PESQ +0.02 or less, even if STOI/SI-SDR improves | Reject or mark inconclusive. |
| PESQ improves but STOI/SI-SDR drops beyond guardrail | Reject unless listening test proves improvement. |
| Quality improves but RTF is too high | Keep as teacher/research candidate; do not deploy. |
| D1 improves but too slow | Train D1a or distill D1b into baseline. |
| All D options fail | Stop architecture changes; move to data-centric tuning or teacher-student distillation. |

## 10. Naming

Suggested config IDs:

```text
D1a_gru704
D1b_gru768
D2_temporal_refine
D3_hybrid_head
```

Suggested notebook names:

```text
train_d1b_gru768_deepvqe_kaggle_v1.ipynb
train_d1a_gru704_deepvqe_kaggle_v1.ipynb
train_d3_hybrid_head_deepvqe_kaggle_v1.ipynb
train_d2_temporal_refine_deepvqe_kaggle_v1.ipynb
```

## 11. Immediate Next Step

Implement and train:

```text
D1b: Baseline + GRU hidden 768
```

This is the best first Phase 2 probe because it is simple, targets temporal context directly, keeps the rest of DeepVQE intact, and gives a clear answer about whether the baseline is capacity-limited.
