# Lộ Trình Nâng Cấp Kiến Trúc DeepVQE (Advanced Architectures Roadmap)

Lộ trình này chia các đề xuất lớn thành từng Phase cụ thể, ưu tiên dựa trên nguyên tắc: **Hiệu quả cao nhất (High ROI) và Rủi ro phá vỡ RTF thấp nhất (Low Real-Time RTF Risk)**.

Mỗi Phase là một dự án nghiên cứu nhỏ lẻ, cần được cô lập trong quá trình thử nghiệm (Ablation).

---

## Phase 1: MetricGAN & Adversarial Training (Zero Inference Cost)
**Mục tiêu:** Cải thiện đáng kể độ tự nhiên của giọng nói (Naturalness/Perceptual Quality), giảm hiện tượng méo tiếng (artifacts) khi xử lý nhiễu nặng, mà **không làm tăng bất kỳ phép tính (FLOPs/Params) nào khi Deploy**.

**Ứng viên Generator ưu tiên hiện tại:** Dựa trên `result/bao_cao_danh_gia_model.md`, `D1b_gru768` là lựa chọn cân bằng tốt nhất: PESQ cao nhất, SI-SDR/STOI nằm trong nhóm đầu và RTF tốt nhất trong bảng summary. Vì vậy Phase 1 nên chạy trước với preset `GAN_D1b_gru768`: kiến trúc inference giữ nguyên `D1b_gru768`, Discriminator chỉ dùng khi train.

**Các bước triển khai:**
1. **Thiết kế Discriminator:**
   - Xây dựng một module `Discriminator` dùng mạng CNN 2D (tương tự PatchGAN hoặc kiến trúc discriminator của CMGAN).
   - *Đầu vào:* Phổ âm thanh sạch (Clean) / Phổ do DeepVQE tạo ra (Enhanced).
   - *Đầu ra:* Phân biệt Thật/Giả (Adversarial) HOẶC dự đoán trực tiếp điểm PESQ (MetricGAN).
2. **Tích hợp Loss Function:**
   - Viết lại quy trình train trong `train_ablation.py`.
   - Loss tổng = `α * STFT_Loss + β * Asymmetric_Loss + γ * GAN_Loss`.
   - Update luân phiên Generator (DeepVQE) và Discriminator.
3. **Tiêu chí Đánh giá (Pass/Fail):**
   - Inference RTF không đổi (đảm bảo 100%).
   - Điểm DNSMOS OVRL tăng ít nhất `+0.05` đến `+0.1`.
   - Điểm STOI/PESQ có thể đi ngang hoặc nhích nhẹ, nhưng cảm nhận nghe thực tế phải trong trẻo hơn.

---

## Phase 2: Nâng cấp Lõi Sequence Modeling (Thay thế GRU)
**Mục tiêu:** Giải quyết điểm yếu "quên" ngữ cảnh và giới hạn receptive field của GRU bằng các mạng State-of-the-Art cho Sequence Modeling, tập trung vào **Mamba (State Space Models)** và **Causal Conformer**.

**Các bước triển khai:**
1. **Khám phá Mamba Block (Ưu tiên số 1):**
   - Nhập khẩu (Import) hoặc xây dựng `MambaBlock` thuần PyTorch có hỗ trợ Streaming (Stateful).
   - Thay thế `Bottleneck` (hiện là GRU 576-hidden) bằng 2 block Mamba với hidden size tương đương (hoặc nhỏ hơn để ép Params).
   - **Benchmarking khắt khe:** Chạy profile để đảm bảo Stateful RTF và RAM/VRAM inference không vượt quá Baseline.
2. **Khám phá Squeezeformer/Conformer (Dự phòng):**
   - Viết block Causal Squeezeformer (Bản thu gọn của Conformer).
   - So sánh trực tiếp với Mamba về điểm số vs. RTF. (Lưu ý: Mamba thường chiếm ưu thế tuyệt đối về RTF so với dòng họ Attention).
3. **Tiêu chí Đánh giá:**
   - **RTF Check:** Mamba bắt buộc phải có Stateful RTF tương đương hoặc nhanh hơn GRU.
   - **Quality:** Phải thấy sự cải thiện rõ rệt ở SI-SDR (thể hiện việc mô hình hiểu cấu trúc ngữ âm tốt hơn ở khoảng thời gian dài).

---

## Phase 3: Tối Ưu Hóa Phân Giải Phổ Phức (Complex Masking)
**Mục tiêu:** Thay thế lớp `CCM` hiện tại bằng các cơ chế giải quyết Magnitude (biên độ) và Phase (Pha) hiệu quả và nhẹ nhàng hơn. Tín hiệu âm thanh trong trẻo phụ thuộc rất lớn vào Phase.

**Các bước triển khai:**
1. **Thử nghiệm Decoupled Masking:**
   - Tách layer cuối cùng của Decoder thành 2 nhánh:
     - Nhánh 1: Tạọ `Magnitude Mask` (Real, kích hoạt bằng ReLU hoặc Sigmoid).
     - Nhánh 2: Tạo `Phase Residual` (dự đoán góc lệch pha $\Delta \theta$).
   - Kết hợp lại: $\hat{X} = (|X| \odot M_{mag}) \cdot e^{j(\theta_X + \Delta \theta)}$.
2. **Thử nghiệm Taylor-SENet Mask:**
   - Áp dụng cơ chế Taylor Expansion Block nhỏ nhắn gọn gọn để thay thế $3 \times 3$ CCM (do 3x3 filter theo mặt phẳng phức rất tốn compute ở lớp cuối).
3. **Tiêu chí Đánh giá:**
   - Số MACs/FLOPs giảm so với CCM gốc.
   - Chỉ số PESQ (vốn rất nhạy với sai lệch pha) phải tăng.

---

## Tóm tắt Timeline Đề xuất (Execution Plan)

> [!CAUTION]
> **Quy tắc Bất di bất dịch:** Chỉ thực hiện 1 Phase/Thay đổi tại một thời điểm. Việc gộp chung (ví dụ vừa thêm Mamba vừa train GAN) sẽ làm nhiễu kết quả nghiên cứu.

* **Tuần 1:** Triển khai **Phase 1 (MetricGAN/Adversarial Loss)** trên file code hiện tại. Đây là "Low hanging fruit" mang lại kết quả cảm nhận tức thì. (Bạn chỉ cần duyệt để tôi bắt đầu code Discriminator).
  - Lệnh ưu tiên sau khi có dữ liệu manifest: `python ablation/train_ablation.py --config-id GAN_D1b_gru768`.
* **Tuần 2:** Thiết lập môi trường và module **Mamba (Phase 2)**. Viết script đo RTF riêng cho Mamba để đảm bảo tính khả thi trên Edge Devices trước khi train thực tế.
* **Tuần 3:** Thay thế và so sánh **Decoupled Mask / Taylor Mask (Phase 3)** với CCM. So đo từng FLOPs một theo đúng tinh thần của roadmap Ablation cũ.

## Các Câu Hỏi Mở Cần Bạn Quyết Định (Open Questions)

1. Bạn có đồng ý với thứ tự ưu tiên: **GAN Loss -> Mamba Bottleneck -> Complex Mask** không?
2. Trong Phase 1, bạn muốn tôi thiết kế một **Discriminator phân biệt Real/Fake chuẩn WGAN** (dễ train, ổn định), hay một **MetricGAN Discriminator dự đoán trực tiếp điểm PESQ** (Khó train hơn nhưng tối ưu PESQ chính xác nhất)?
3. Đối với Phase 2, việc cài đặt Mamba (từ package `mamba-ssm`) có yêu cầu biên dịch C++/CUDA. Môi trường triển khai cuối cùng của DeepVQE có cho phép dùng các custom CUDA kernel này không, hay bắt buộc phải là thuần PyTorch Native?
