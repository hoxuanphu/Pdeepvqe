# DeepVQE Optimization Roadmap (Ablation-Driven)

Mục tiêu: Đưa ra lộ trình thử nghiệm (ablation study) nhằm tối ưu performance (accuracy) của model DeepVQE gốc (phiên bản **NS-only**, đã loại bỏ nhánh Far-end và khối Cross-Attention) bằng cách thay đổi/thêm/bớt các component, dưới ràng buộc khắt khe về computation (Params, FLOPs/MACs) và real-time factor (RTF).

**Phạm vi bài toán:** Dự án này chỉ tối ưu **khử nhiễu nền (noise suppression / speech enhancement)**. Không tối ưu khử tiếng vang far-end (AEC), không tối ưu khử âm vang phòng (dereverberation), và không dùng các metric/benchmark dành riêng cho echo hoặc reverb để ra quyết định ablation.

*Lưu ý: Mọi đề xuất dưới đây đều là các **giả thuyết (hypothesis)** cần được kiểm chứng qua các bước ablation riêng lẻ, không gộp chung vào một mô hình duy nhất từ ban đầu để tránh nhiễu kết quả.*

---

## 1. Các Giả Thuyết Tối Ưu (Optimization Hypotheses)

### Depthwise Separable Convolutions (DW-Conv)
- **Giả thuyết:** Các lớp `Conv2d` tiêu chuẩn tốn rất nhiều FLOPs/Params. Thay thế bằng DW-Conv sẽ giảm compute đáng kể, tạo dư địa để thêm các module tăng cường chất lượng.
- **Giải pháp:** Bắt đầu áp dụng DW-Conv **chỉ tại các ResidualBlock**. Giữ nguyên các main conv trong Encoder/Decoder ở các phase đầu.
- **Lưu ý quan trọng:** Mọi biến thể DW-Conv chỉ được xem là "compute win" nếu giảm được **Stateful Streaming RTF hoặc ONNX latency thực tế**, không chỉ giảm MACs/FLOPs lý thuyết.

### Efficient Channel Attention (Causal ECA)
- **Giả thuyết:** Thêm ECA vào các Block giúp mạng tập trung vào các channel quan trọng với chi phí Params/FLOPs cực rẻ.
- **Constraint Causal (Quan trọng):** Không dùng ECA gốc. Cần định nghĩa rõ:
  - **ECA-F (Frequency-only):** Đầu vào `[B, C, T, F]`. ECA-F chỉ được pooling theo trục tần số `F`, tuyệt đối không pooling theo trục thời gian `T`. Phải pass bài test Causal (thay đổi các frame tương lai không làm đổi output hiện tại). Ưu tiên thử nghiệm ở Phase 1.
  - **ECA-CT (Causal Time):** Chuyển sang Deferred (hoãn lại) cho đến khi có streaming cache contract.

### Activation Function (ELU -> PReLU)
- **Giả thuyết:** Dùng PReLU có thể học tốt hơn ở vùng âm. Dạng Shared (1 tham số chung) hoặc Per-Channel (1 tham số mỗi channel).

### Bottleneck Tuning (GRU)
- **Giả thuyết:** Giảm `hidden_size` GRU từ 576 xuống 544 hoặc 512 tiết kiệm nhiều Params/RTF. Chuyển sang Deferred.
- **Nâng cấp (Sub-band RNN):** Thay vì dùng 1 GRU lớn, chia dải tần thành 2-4 sub-bands và dùng các Grouped GRU nhỏ chạy song song để tiết kiệm Params/FLOPs mà không mất ngữ cảnh thời gian.

### Complex Convolving Mask (CCM) Optimization (Deferred)
- **Giả thuyết:** Áp dụng mask 2D $(m+1) \times (2n+1)$ lân cận trên mặt phẳng phức là quá nặng.
- **Giải pháp:** Sử dụng Sparse/Cross-shaped CCM (chỉ lấy lân cận theo hình chữ thập) hoặc Decoupled Magnitude-Phase CCM để giảm lượng lớn FLOPs ở đầu ra. (Lưu ý: Hạng mục này tạm thời chuyển sang Phase 3 / Deferred vì chưa được đưa vào ma trận Phase 1 & 2).

### Skip-Connection Gating
- **Giả thuyết:** Lớp 1x1 Conv ở skip-connection truyền cả nhiễu từ Encoder sang Decoder do thiếu tính chọn lọc.
- **Giải pháp:** Thêm Squeeze-and-Excitation (SE) hoặc Causal ECA vào skip-block để mô hình tự động "khóa" các channel nhiễu và "mở" channel tiếng nói.

### Decoder Sub-pixel Convolution Optimization
- **Giả thuyết:** Quá trình PixelShuffle tăng số kênh lên gấp 4 lần trước khi reshape, gây thắt cổ chai về compute.
- **Giải pháp:** Sử dụng DW-Conv ngay trước bước Sub-pixel reshape để giảm lượng MACs sinh ra.

### Hàm Loss chuyên biệt cho Khử nhiễu
- **Giả thuyết:** Model dễ bị over-suppression (cắt lẹm giọng) và nghe thiếu tự nhiên khi dùng các hàm Loss cũ.
- **Giải pháp:** Thêm Asymmetric Loss (phạt nặng quá trình cắt lẹm) và Multi-Resolution STFT Loss (học đặc trưng ở nhiều độ phân giải thời gian/tần số) mà không làm tăng inference cost.
- **Giới hạn:** Loss chỉ nên phục vụ mục tiêu giữ giọng sạch và giảm nhiễu nền. Không thêm thành phần loss/target dành cho echo cancellation hoặc dereverberation.

---

## 2. Tiêu Chí Đánh Giá (Pass Criteria & Metrics)

### 2.1 Pass Criteria
- **Primary Metric:** Phải được cố định trong config trước Stage 0. Mặc định dùng **DNSMOS OVRL** cho perceptual quality; nếu eval set có clean reference chất lượng cao thì dùng **PESQ**. Không được đổi primary metric sau khi đã nhìn kết quả ablation.
- **Guardrail Metrics (Ngưỡng an toàn):** Các metric STOI và SI-SDR không được giảm quá ngưỡng định sẵn so với Baseline:
  - `STOI >= baseline - 0.002`
  - `SI-SDR >= baseline - 0.1 dB`
  - Nếu dùng PESQ làm guardrail: `PESQ >= baseline - 0.01`
- **Probe Pass (Stage 1 - Khám phá):** Cho phép Params/FLOPs tăng nhẹ ($\le +1\%$). Tiêu chí chất lượng: Primary metric đạt ngưỡng.
  - *Tie-Breaker cho Probe Pass:* Nếu B1a và B1b đều pass, ưu tiên chọn theo thứ tự: (1) Primary metric cao hơn $\rightarrow$ (2) Stateful RTF / ONNX latency thấp hơn $\rightarrow$ (3) Đơn giản hơn (B1a shared > B1b per_channel).
- **Deploy Pass (Stage 2 - Mô hình cuối):** Bắt buộc Params/FLOPs/Stateful RTF/ONNX latency $\le$ Baseline.
  - Primary metric vượt ngưỡng (ví dụ: $\Delta$ DNSMOS OVRL $\ge +0.03$ so với Baseline).
  - Guardrail metrics phải thỏa mãn ngưỡng an toàn ở trên.
- **Đánh giá nhiễu thống kê (Variance):** Cần xác định vùng "marginal pass/fail" (ví dụ: $|\Delta| \le 0.01$ là vùng nhiễu). Các biến thể tốt nhất nên được chạy với 2-3 seeds ngẫu nhiên hoặc dùng paired bootstrap trên eval set để xác nhận không phải do may mắn thống kê.

### 2.2 Metrics & Causal Checks
- Sử dụng PESQ, STOI, DNSMOS, SI-SDR cho bài toán NS-only.
- Bỏ qua ERLE và các metric chuyên cho AEC/dereverberation; không dùng mức giảm echo/reverb làm tiêu chí pass/fail.
- **Đo lường Compute:** Dùng **PyTorch Profiler** hoặc log latency thực tế, không tin hoàn toàn `ptflops`.
- Đo **Stateful Streaming RTF** thông qua module stream. Xuất **streaming stateful ONNX** và kiểm tra parity.

---

## 3. Training Budget & Eval Protocol (BẮT BUỘC ĐỒNG NHẤT)

Để đảm bảo tính công bằng khoa học, mọi ablation variant phải tuân thủ nghiêm ngặt hợp đồng huấn luyện và đánh giá sau:

- **Training Budget Khóa Chặt (Architecture Ablation):** Các biến thể kiến trúc phải chia sẻ chung: `data split`, `seed policy`, `optimizer`, `LR schedule`, `max epochs/steps`, `early stopping patience`, `batch size`, `loss function` (baseline), `augmentations`.
- **Loss Ablation Độc Lập:** Riêng các thử nghiệm về Loss (`B4`) là một nhóm độc lập, chỉ thay đổi hàm loss và giữ nguyên cấu trúc mạng Baseline.
- **Tích hợp Loss (Stage 2):** Nếu `B4` vượt qua vòng kiểm tra, nó sẽ được áp dụng lại cho các ứng viên kiến trúc tốt nhất ở Stage 2. Lưu ý: Các combined run này phải được **train lại từ scratch** theo cùng training budget, không được lấy checkpoint cũ để fine-tune ngắn. Kết quả báo cáo phải phân tách rõ: Architecture-only gain, Loss-only gain, và Combined gain.
- **Dataset Scope NS-only:** Train/valid/eval phải được hiểu là bài toán clean speech + additive/background noise. Không đưa far-end echo, echo reference, hoặc reverb-specific target vào pipeline nếu mục tiêu không phải AEC/dereverberation.
- **Data Splits (Tránh Leakage):**
  - **Validation set:** Dùng để chọn checkpoint và early stopping (dựa trên Validation Loss hoặc Composite Score). Tuyệt đối không dùng tập Eval/Test để chọn checkpoint.
  - **Ablation/Dev Eval set:** Được khóa cố định, không random crop, dùng chung một danh sách manifest cho mọi biến thể. Tập này dùng để so sánh các biến thể thử nghiệm trong suốt quá trình ablation.
  - **Final Test set:** Chỉ được dùng đúng một lần duy nhất để báo cáo kết quả mô hình cuối cùng sau khi đã chọn xong.
- **Train from Scratch:** Mặc định train lại từ đầu cho các biến thể Stage 1 & 2. (Có chạy Sanity Check đo pre-trained baseline để làm reference).
- **Reproducibility Metadata:** Bắt buộc log các thông số sau cho mỗi model checkpoint: `git_commit`, `config_hash` hoặc `full_config`, `seed`, `hardware info`, `torch_version`, `onnxruntime_version`, `num_threads`, `checkpoint_id`.

---

## 4. Ma Trận Triển Khai (Ablation Matrix) & Dependency Graph

### Phase 1: Scope Cốt Lõi

**Stage 0: Đo đạc Baseline gốc**
- Train Baseline model (từ scratch). Đo Params, MACs/FLOPs, Stateful RTF, Eval Benchmark.

**Stage 1: Low-risk Accuracy Probe & Training Strategy**
- `B1a:` Baseline + PReLU (Shared).
- `B1b:` Baseline + PReLU (Per-channel).
- `B2:` Baseline + ECA-F ở ResidualBlock.
- `B3a:` Baseline + ECA-F ở Skip-connection (Skip-Gating).
- `B3b:` Baseline + SE ở Skip-connection.
- `B4:` Baseline + Asymmetric Loss & Multi-Resolution STFT Loss (Zero Inference Cost).

**Stage 2: Compute Reallocation & Deep Tuning**
- `C1:` Thay Conv2d trong ResidualBlock bằng DW-Conv.
- `C2a:` Mô hình C1 + DW-Subpixel ở Decoder.
- `C2b:` Mô hình C1 + ECA-F (từ B2).
- `C3:` Mô hình C1 + PReLU (từ B1a/B1b).
- `C4:` Mô hình C3 + ECA-F (từ B2).

> **Variant Dependency Graph (Quy tắc Pruning mềm):**
> - Nếu `B1a/B1b` fail Probe Pass một cách rõ rệt (hụt xa mốc) $\rightarrow$ Bỏ tích hợp PReLU vào biến thể tổng hợp (ví dụ `C4`).
> - Nếu `B2` fail rõ rệt $\rightarrow$ Bỏ `C2b` và các nhánh dùng ECA-F như `C4`, trừ khi muốn giữ một run kiểm tra interaction effect khi đi kèm DW-Conv.
> - Nếu `B1a/B1b` hoặc `B2` chỉ fail marginal (mấp mé ranh giới nhiễu), vẫn có thể giữ lại biến thể kết hợp `C4` để kiểm tra tác động tương hỗ.

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
- `E3:` Thay thế GRU bằng Grouped GRU (Sub-band RNN) chia 2 dải tần. (Lưu ý: Đây là thay đổi re-architecture rủi ro trung bình-cao, dễ làm mất context chéo giữa các dải tần và gây artifact).
- `E4:` Cấu hình Best(E1,E2,E3) + Baseline Knowledge Distillation (Teacher-Student).

> **Stage 4 Stop Condition:** Nếu E1/E2/E3 đều fail Deploy Quality hoặc fail RTF/ONNX latency, nên dừng hoàn toàn Phase 2 và kết luận dự án cần re-architecture toàn diện hoặc xem xét lại data/loss/training protocol.
