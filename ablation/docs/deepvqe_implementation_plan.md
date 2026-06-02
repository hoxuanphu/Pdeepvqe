# Kế Hoạch Triển Khai Chi Tiết: Tối Ưu Hóa DeepVQE (Ablation Study)

Kế hoạch này phác thảo chi tiết (mức độ code & specs) cách lập trình và triển khai các cải tiến kiến trúc cho model DeepVQE dựa trên [Roadmap](deepvqe_optimization_roadmap.md). 

> [!IMPORTANT]
> **Ràng buộc quan trọng:** Không can thiệp hay sửa đổi file `deepvqe.py` gốc. Toàn bộ mã nguồn thử nghiệm sẽ được tách riêng vào file mới `deepvqe_ablation.py`.
> Load trọng số (`state_dict`) bằng `strict=True` chỉ được dùng cho Baseline. Các biến thể khác bắt buộc `strict=False`.

> [!WARNING]
> **Lưu ý tối quan trọng khi Code:** Ưu tiên hàng đầu là `StreamDeepVQE_Ablation` (bản stateful streaming). Không có module này, mọi đánh giá RTF/ONNX là vô nghĩa. Mọi thay đổi trong cấu trúc Offline đều phải có bản ánh xạ tương đương sang Streaming Stateful.

---

## 1. Đặc Tả Kiến Trúc (Architecture Design Specs)

Áp dụng **Parametrized Design**, xây dựng một mạng duy nhất nhận các tham số config linh hoạt.

### 1.1. ActivationFactory
Quản lý tập trung hàm activation để có thể chuyển đổi dễ dàng.
- **Input:** `prelu_type` (None, 'shared', 'per_channel'), `channels`
- **Output:**
  - `None` $\rightarrow$ `nn.ELU()`
  - `shared` $\rightarrow$ `nn.PReLU(1)`
  - `per_channel` $\rightarrow$ `nn.PReLU(channels)`

### 1.2. CausalECA_F (Frequency-pooling ECA)
Được thiết kế để không phá vỡ tính causal và tiết kiệm nhất có thể.
- **Cơ chế:** 
  1. Input shape: `(B, C, T, F)`
  2. Pool over Frequency: `y = x.mean(dim=3, keepdim=True)` $\rightarrow$ shape `(B, C, T, 1)`
  3. Conv1D over Channels: Transpose thành `(B*T, 1, C)`, đưa qua `nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)` (Conv1D chạy dọc trục C).
  4. Sigmoid và nhân lại vào `x`.
- **Streaming:** Hàm `forward(x)` chỉ xử lý theo frame hiện tại (`T=1`), **không cần duy trì cache** vì pooling diễn ra trên trục `F`.

### 1.3. ResidualBlock_Ablation
- Lớp `conv` chính gốc là: `nn.Conv2d(C, C, kernel_size=(4,3))` + `nn.ZeroPad2d([1,1,3,0])`.
- Nếu `dw_residual=True` (DW-Separable Conv):
  - Thay thế bằng 2 lớp:
    1. Depthwise: `nn.Conv2d(C, C, kernel_size=(4,3), groups=C)` + `nn.ZeroPad2d([1,1,3,0])`
    2. Pointwise: `nn.Conv2d(C, C, kernel_size=1)`
- **ECA_F injection:** Nếu `use_eca_f=True`, áp dụng `CausalECA_F` lên nhánh conv trước khi cộng residual: `return ECA_F(y) + x`. (Giữ cho identity path sạch).

### 1.4. EncoderBlock_Ablation & DecoderBlock_Ablation
- Nếu `main_block_eca_f=True` (chỉ áp dụng cho Stage 1 - B3):
  - EncoderBlock: Áp dụng `CausalECA_F` ngay sau đoạn `self.elu(self.bn(self.conv(self.pad(x))))`, trước khi đi vào `resblock`.
  - DecoderBlock: Áp dụng `CausalECA_F` ngay sau đoạn `self.elu(self.bn(self.deconv(self.resblock(y))))`. (Tuyệt đối không chèn vào nhánh `skip_conv`). **Lưu ý quan trọng:** Không áp dụng `main_block_eca_f` cho block cuối cùng (`is_last=True`) để tránh làm hỏng mask output.

---

## 2. Bảng Cấu Hình & Pruning (Ablation Matrix)

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

## 3. Streaming Cache Contract (Giao thức Cache)

Bản `StreamDeepVQE_Ablation` sẽ nhận parameter `cache` dưới dạng danh sách (list of tensors) tương ứng với từng layer.

1. **Conv / Residual Conv / DW-Conv Cache:** 
   - Với kernel time `K_t = 4`, lượng padding tương lai là `3`.
   - Shape Cache: `(B, C, 3, F)` (Lưu trữ 3 frame gần nhất).
2. **GRU Hidden Cache:**
   - Shape: `(num_layers, B, hidden_size)` (DeepVQE mặc định num_layers = 1, hidden_size=576).
3. **CCM Cache:**
   - Cần cache frame cho thuật toán Complex Convolving Mask.
4. **CausalECA_F Cache:**
   - Hoàn toàn KHÔNG cần (0 bytes).

---

## 4. Training Budget & Eval Protocol (QUY TẮC BẮT BUỘC)

- **Training Budget:** Đóng băng toàn bộ hyperparams cho mọi thí nghiệm để đảm bảo công bằng. Tất cả variants sử dụng chung: `data split`, `seed policy`, `optimizer`, `LR schedule`, `max epochs/steps`, `early stopping patience`, `batch size`, `loss`, `augmentations`, và quy tắc chọn checkpoint.
- **Eval Set Protocol:** Eval split được khóa cố định, không random crop khi test, dùng chung 1 manifest file.
- **Reproducibility Metadata:** Khi log kết quả ra CSV, BẮT BUỘC log: `git_commit, config_hash, seed, hardware info, torch_version, onnxruntime_version, num_threads, checkpoint_id`.

---

## 5. Kế Hoạch Đánh Giá (6 Bước Nghiêm Ngặt)

**Bước 1: Parity & Causality Check**
- Streaming Parity giữa bản Offline và Stateful Streaming (sai số ~ 0).
- Causality check (sai số tương lai-quá khứ = 0).

**Bước 2: Untrained Benchmark (PyTorch Profiler)**
- Dùng Profiler đo *Measured latency / operator time* (Lấy Stateful RTF).
- Dùng `ptflops` chỉ để tham chiếu phụ cho MACs/FLOPs.
- **Pass Rules:** 
  - *Probe Pass (B1, B2, B3):* Cho phép compute tăng nhẹ ($\le +1\%$).
  - *Deploy Pass (C1, C2, C3, C4):* Params/FLOPs/Stateful RTF/ONNX latency $\le$ Baseline.

**Bước 3: Training Runner (Dependency Graph Pruning)**
- Train Stage 1 (B-variants).
  - *Probe Tie-Breaker (B1a vs B1b):* 1) Điểm cao hơn. 2) RTF/Latency thấp hơn. 3) B1a ưu tiên hơn B1b.
  - *Quy tắc cắt tia (Pruning):* Nếu B1 fail sập sàn -> Cắt C3. Nếu B2 fail sập sàn -> Cắt C2. Nếu fail marginal (sát ranh giới nhiễu) -> Vẫn giữ lại C4 để check interaction effect với DW-Conv.

**Bước 4: Quality Evaluation**
- Đánh giá PESQ, STOI, DNSMOS (OVRL, SIG, BAK), SI-SDR. (Bỏ ERLE).

**Bước 5: Stateful ONNX Export**
- Export streaming stateful ONNX và test PyTorch parity (`abs_error < 1e-4`).

**Bước 6: Tổng Hợp & Phase 2 Escalation**
- Gộp data ra CSV.
- **Escalation Rules:**
  - Quality tốt, RTF/ONNX fail -> Trigger Stage 4 (Giảm GRU).
  - RTF tốt, Quality margin siêu nhỏ -> Trigger Stage 3 (Tăng DW-Conv lấy headroom).
  - Phase 1 fail toàn tập -> Dừng thực nghiệm.
  - *Stage 4 Stop Condition:* Nếu E1/E2/E3 đều fail Deploy Quality hoặc fail RTF/ONNX latency, thì dừng Phase 2, kết luận cần re-architecture toàn diện hoặc xem xét lại data/loss/training protocol.
