# DeepVQE Optimization Roadmap (Ablation-Driven)

Mục tiêu: Đưa ra lộ trình thử nghiệm (ablation study) nhằm tối ưu performance (accuracy) của model DeepVQE gốc bằng cách thay đổi/thêm/bớt các component, dưới ràng buộc khắt khe về computation (Params, FLOPs/MACs) và real-time factor (RTF).

*Lưu ý: Mọi đề xuất dưới đây đều là các **giả thuyết (hypothesis)** cần được kiểm chứng qua các bước ablation riêng lẻ, không gộp chung vào một mô hình duy nhất từ ban đầu để tránh nhiễu kết quả.*

---

## 1. Các Giả Thuyết Tối Ưu (Optimization Hypotheses)

### Depthwise Separable Convolutions (DW-Conv)
- **Giả thuyết:** Các lớp `Conv2d` tiêu chuẩn tốn rất nhiều FLOPs/Params. Thay thế bằng DW-Conv sẽ giảm compute đáng kể, tạo dư địa để thêm các module tăng cường chất lượng.
- **Giải pháp:** Bắt đầu áp dụng DW-Conv **chỉ tại các ResidualBlock**. Giữ nguyên các main conv trong Encoder/Decoder ở các phase đầu.

### Efficient Channel Attention (Causal ECA)
- **Giả thuyết:** Thêm ECA vào các Block giúp mạng tập trung vào các channel quan trọng với chi phí Params/FLOPs cực rẻ.
- **Constraint Causal (Quan trọng):** Không dùng ECA gốc. Cần định nghĩa rõ:
  - **ECA-F (Frequency-only):** Chỉ pooling trên trục Frequency (F). Ưu tiên thử nghiệm ở Phase 1.
  - **ECA-CT (Causal Time):** Chuyển sang Deferred (hoãn lại) cho đến khi có streaming cache contract.

### Activation Function (ELU -> PReLU)
- **Giả thuyết:** Dùng PReLU có thể học tốt hơn ở vùng âm. Dạng Shared (1 tham số chung) hoặc Per-Channel (1 tham số mỗi channel).

### Bottleneck Tuning (GRU)
- **Giả thuyết:** Giảm `hidden_size` GRU từ 576 xuống 544 hoặc 512 tiết kiệm nhiều Params/RTF. Chuyển sang Deferred.

---

## 2. Tiêu Chí Đánh Giá (Pass Criteria & Metrics)

### 2.1 Pass Criteria
- **Probe Pass (Stage 1 - Khám phá):** Cho phép Params/FLOPs tăng nhẹ ($\le +1\%$). Tiêu chí chất lượng: Primary metric đạt ngưỡng.
  - *Tie-Breaker cho Probe Pass:* Nếu B1a và B1b đều pass, ưu tiên chọn theo thứ tự: (1) Primary metric cao hơn $\rightarrow$ (2) Stateful RTF / ONNX latency thấp hơn $\rightarrow$ (3) Đơn giản hơn (B1a shared > B1b per_channel).
- **Deploy Pass (Stage 2 - Mô hình cuối):** Bắt buộc Params/FLOPs/Stateful RTF/ONNX latency $\le$ Baseline.
  - Primary quality metric vượt ngưỡng (ví dụ: $\Delta$ PESQ $\ge +0.03$ HOẶC $\Delta$ DNSMOS OVRL $\ge +0.03$ so với Baseline).
  - Các metric khác (STOI, SI-SDR) không được giảm vượt quá sai số ngẫu nhiên.

### 2.2 Metrics & Causal Checks
- Sử dụng PESQ, STOI, DNSMOS, SI-SDR. *(Bỏ qua ERLE).*
- **Đo lường Compute:** Dùng **PyTorch Profiler** hoặc log latency thực tế, không tin hoàn toàn `ptflops`.
- Đo **Stateful Streaming RTF** thông qua module stream. Xuất **streaming stateful ONNX** và kiểm tra parity.

---

## 3. Training Budget & Eval Protocol (BẮT BUỘC ĐỒNG NHẤT)

Để đảm bảo tính công bằng khoa học, mọi ablation variant phải tuân thủ nghiêm ngặt hợp đồng huấn luyện và đánh giá sau:

- **Training Budget Khóa Chặt:** Tất cả các biến thể phải chia sẻ chung: `data split`, `seed policy`, `optimizer`, `LR schedule`, `max epochs/steps`, `early stopping patience`, `batch size`, `loss function`, `augmentations` và `checkpoint selection rule`.
- **Eval Set Protocol:** Eval split được khóa cố định, không random crop khi eval, dùng chung một danh sách manifest cho mọi biến thể. Bắt buộc log số lượng utterance/duration.
- **Train from Scratch:** Mặc định train lại từ đầu cho các biến thể Stage 1 & 2. (Có chạy Sanity Check đo pre-trained baseline để làm reference).
- **Reproducibility Metadata:** Bắt buộc log các thông số sau cho mỗi model checkpoint: `git_commit`, `config_hash` hoặc `full_config`, `seed`, `hardware info`, `torch_version`, `onnxruntime_version`, `num_threads`, `checkpoint_id`.

---

## 4. Ma Trận Triển Khai (Ablation Matrix) & Dependency Graph

### Phase 1: Scope Cốt Lõi

**Stage 0: Đo đạc Baseline gốc**
- Train Baseline model (từ scratch). Đo Params, MACs/FLOPs, Stateful RTF, Eval Benchmark.

**Stage 1: Low-risk Accuracy Probe**
- `B1a:` Baseline + PReLU (Shared).
- `B1b:` Baseline + PReLU (Per-channel).
- `B2:` Baseline + ECA-F ở ResidualBlock.
- `B3 (Optional):` Baseline + ECA-F ở Encoder/Decoder Block.

**Stage 2: Compute Reallocation**
- `C1:` Thay Conv2d trong ResidualBlock bằng DW-Conv.
- `C2:` Mô hình C1 + ECA-F (từ B2).
- `C3:` Mô hình C1 + PReLU (chọn Best B1 theo Tie-breaker).
- `C4:` Mô hình C1 + ECA-F + PReLU.

> **Variant Dependency Graph (Quy tắc Pruning mềm):**
> - Nếu `B1a/B1b` fail Probe Pass một cách rõ rệt (hụt xa mốc) $\rightarrow$ Bỏ `C3`.
> - Nếu `B2` fail rõ rệt $\rightarrow$ Bỏ `C2`.
> - Nếu `B1` hoặc `B2` chỉ fail marginal (mấp mé ranh giới nhiễu), vẫn có thể giữ lại biến thể kết hợp `C4` để kiểm tra tác động tương hỗ (interaction effect) khi đi kèm với DW-Conv (do capacity bị thay đổi).

---

### Phase 2: Deferred Scope & Risk Escalation Rules

Việc kích hoạt Phase 2 không còn cảm tính, mà phụ thuộc hoàn toàn vào kết quả của Phase 1:

**Quy tắc kích hoạt Escalation (Trigger Rules):**
1. **Quality tốt nhưng RTF Fail:** Nếu model tốt nhất Phase 1 vượt ngưỡng accuracy nhưng vi phạm rào cản RTF $\rightarrow$ Kích hoạt **Stage 4** (Giảm GRU hidden size).
2. **RTF tốt nhưng Quality Margin hẹp:** Nếu model qua bài test RTF nhưng điểm số chỉ nhích siêu nhẹ $\rightarrow$ Kích hoạt **Stage 3** (Ép DW-Conv sâu hơn) để ép giải phóng thêm compute headroom, từ đó có thể nhồi thêm Attention.
3. **Phase 1 Fail Toàn Bộ:** Dừng dự án ablation, quay lại kiểm tra data/loss/training protocol. Không kích hoạt Phase 2.

**Stage 3: Aggressive Lightweight**
- `D1:` DW-Conv ở ResidualBlock + một phần EncoderBlock.
- `D2:` DW-Conv trên toàn bộ Conv chính.
- `D3:` D2 + ECA-F.

**Stage 4: Bottleneck Tuning**
- `E1:` Giảm `hidden_size` GRU 576 -> 544.
- `E2:` Giảm 576 -> 512.
- `E3:` E2 + Baseline Knowledge Distillation.

> **Stage 4 Stop Condition:** Nếu E1/E2/E3 đều fail Deploy Quality hoặc fail RTF/ONNX latency, nên dừng hoàn toàn Phase 2 và kết luận dự án cần re-architecture toàn diện hoặc xem xét lại data/loss/training protocol.
