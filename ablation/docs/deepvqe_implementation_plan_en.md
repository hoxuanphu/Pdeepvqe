# Detailed Implementation Plan: DeepVQE Optimization (Ablation Study)

This plan outlines in high detail (code & specs level) how to program and deploy architectural improvements for the DeepVQE model based on the [Roadmap](deepvqe_optimization_roadmap.md).

> [!IMPORTANT]
> **Critical Constraint:** Do not interfere with or modify the original `deepvqe.py` file. All experimental source code will be separated into a new file `deepvqe_ablation.py`.
> Weight loading (`state_dict`) with `strict=True` is only valid for the Baseline. Other variants must use `strict=False`.

> [!WARNING]
> **Crucial Implementation Note:** The absolute highest priority is `StreamDeepVQE_Ablation` (the stateful streaming version). Without it, RTF/ONNX evaluation is meaningless. Every Offline architectural change must have a direct Stateful Streaming counterpart.

---

## 1. Architecture Design Specs

Using **Parameterized Design**, we build a single network that adapts via a flexible configuration dictionary.

### 1.1. ActivationFactory
Centralized management of the activation function to allow easy switching.
- **Input:** `prelu_type` (None, 'shared', 'per_channel'), `channels`
- **Output:**
  - `None` $\rightarrow$ `nn.ELU()`
  - `shared` $\rightarrow$ `nn.PReLU(1)`
  - `per_channel` $\rightarrow$ `nn.PReLU(channels)`

### 1.2. CausalECA_F (Frequency-pooling ECA)
Designed to maintain causality and minimize parameter cost.
- **Mechanism:** 
  1. Input shape: `(B, C, T, F)`
  2. Pool over Frequency: `y = x.mean(dim=3, keepdim=True)` $\rightarrow$ shape `(B, C, T, 1)`
  3. Conv1D over Channels: Transpose to `(B*T, 1, C)`, pass through `nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)` (Conv1D slides across the C axis).
  4. Sigmoid and expand-multiply back into `x`.
- **Streaming:** The `forward(x)` function processes per frame (`T=1`), **requires NO temporal cache** because pooling happens purely on the `F` axis.

### 1.3. ResidualBlock_Ablation
- Original main `conv`: `nn.Conv2d(C, C, kernel_size=(4,3))` + `nn.ZeroPad2d([1,1,3,0])`.
- If `dw_residual=True` (DW-Separable Conv):
  - Replaced by 2 layers:
    1. Depthwise: `nn.Conv2d(C, C, kernel_size=(4,3), groups=C)` + `nn.ZeroPad2d([1,1,3,0])`
    2. Pointwise: `nn.Conv2d(C, C, kernel_size=1)`
- **ECA_F injection:** If `use_eca_f=True`, apply `CausalECA_F` to the conv branch before the residual addition: `return ECA_F(y) + x` (Keeps the identity path clean).

### 1.4. EncoderBlock_Ablation & DecoderBlock_Ablation
- If `main_block_eca_f=True` (only for Stage 1 - B3):
  - EncoderBlock: Apply `CausalECA_F` immediately after `self.elu(self.bn(self.conv(self.pad(x))))`, before the `resblock`.
  - DecoderBlock: Apply `CausalECA_F` immediately after `self.elu(self.bn(self.deconv(self.resblock(y))))`. (Note: The module's skip connection `y = x + skip` must remain untouched). **Important Note:** Do not apply `main_block_eca_f` to the last decoder block (`is_last=True`) to avoid destroying the output mask.

---

## 2. Configuration & Ablation Matrix

### Ablation Matrix - Phase 1

| ID       | prelu_type    | dw_residual | use_eca_f | main_block_eca_f | Note |
| -------- | ------------- | ----------- | --------- | ---------------- | ---- |
| Baseline | None          | False       | False     | False            | Sanity & Scratch |
| B1a      | shared        | False       | False     | False            | Probe Pass |
| B1b      | per_channel   | False       | False     | False            | Probe Pass |
| B2       | None          | False       | True      | False            | Probe Pass |
| B3 (Opt) | None          | False       | True      | True             | Probe Pass |
| C1       | None          | True        | False     | False            | Deploy Pass |
| C2       | None          | True        | True      | False            | Deploy Pass |
| C3       | Best B1       | True        | False     | False            | Deploy Pass |
| C4       | Best B1       | True        | True      | False            | Deploy Pass |

---

## 3. Streaming Cache Contract

The `StreamDeepVQE_Ablation` will receive a `cache` parameter (typically a list of tensors) corresponding to each layer.

1. **Conv / Residual Conv / DW-Conv Cache:** 
   - With temporal kernel `K_t = 4`, the past padding requirement is `3`.
   - Cache Shape: `(B, C, 3, F)` (Stores the 3 most recent frames).
2. **GRU Hidden Cache:**
   - Shape: `(num_layers, B, hidden_size)` (DeepVQE default num_layers = 1, hidden_size=576).
3. **CCM Cache:**
   - Caches past frames required for the Complex Convolving Mask logic.
4. **CausalECA_F Cache:**
   - Absolutely NO cache required (0 bytes).

---

## 4. Training Budget & Eval Protocol (MANDATORY RULES)

- **Training Budget:** Freeze all hyperparams for all experiments to ensure fairness. All variants must share identical: `data split`, `seed policy`, `optimizer`, `LR schedule`, `max epochs/steps`, `early stopping patience`, `batch size`, `loss`, `augmentations`, and checkpoint selection rule.
- **Eval Set Protocol:** Eval split is rigidly fixed. No random crops during eval. Uses a shared manifest file.
- **Reproducibility Metadata:** When exporting CSV results, it is MANDATORY to log: `git_commit, config_hash, seed, hardware info, torch_version, onnxruntime_version, num_threads, checkpoint_id`.

---

## 5. Evaluation Plan (6 Rigorous Steps)

**Step 1: Parity & Causality Check**
- Streaming Parity between Offline and Stateful Streaming (error ~ 0).
- Causality check (future-past error = 0).

**Step 2: Untrained Benchmark (PyTorch Profiler)**
- Use Profiler primarily for *Measured latency / operator time* (Yields Stateful RTF).
- Use `ptflops` only as a secondary reference for MACs/FLOPs.
- **Pass Rules:** 
  - *Probe Pass (B1, B2, B3):* Allow slight compute increase ($\le +1\%$).
  - *Deploy Pass (C1, C2, C3, C4):* Params/FLOPs/Stateful RTF/ONNX latency $\le$ Baseline.

**Step 3: Training Runner (Dependency Graph Pruning)**
- Train Stage 1 (B-variants).
  - *Probe Tie-Breaker (B1a vs B1b):* 1) Higher metric. 2) Lower RTF/Latency. 3) Prefer simpler (B1a over B1b).
  - *Pruning Rules:* If B1 fails blatantly -> Prune C3. If B2 fails blatantly -> Prune C2. If they fail marginally (near noise threshold) -> Keep C4 to test interaction effects with DW-Conv.

**Step 4: Quality Evaluation**
- Evaluate PESQ, STOI, DNSMOS (OVRL, SIG, BAK), SI-SDR. (ERLE excluded).

**Step 5: Stateful ONNX Export**
- Export streaming stateful ONNX and test PyTorch parity (`abs_error < 1e-4`).

**Step 6: Collection & Phase 2 Escalation**
- Aggregate data to CSV.
- **Escalation Rules:**
  - Quality passes, RTF/ONNX fails -> Trigger Stage 4 (Reduce GRU).
  - RTF passes, Quality margin is tiny -> Trigger Stage 3 (Increase DW-Conv for headroom).
  - Phase 1 fails completely -> Halt experiments, review protocol.
  - *Stage 4 Stop Condition:* If E1/E2/E3 all fail Deploy Quality or fail RTF/ONNX latency, halt Phase 2 and conclude that comprehensive re-architecture or review of data/loss/training protocol is required.
