# Lộ Trình Triển Khai: Phase 2 - Nâng cấp Lõi Sequence Modeling (Thay thế GRU)

Mục tiêu của Phase 2 là giải quyết điểm yếu "quên" ngữ cảnh (forgetting) và giới hạn receptive field của kiến trúc GRU hiện tại trong `DeepVQE`. Chúng ta sẽ thay thế GRU (hiện đang nằm ở module `Bottleneck_Ablation` và `StreamBottleneck_Ablation`) bằng các kiến trúc State-of-the-Art cho Sequence Modeling. Ứng cử viên ưu tiên số 1 là **Mamba (State Space Models)**, và dự phòng là **Causal Squeezeformer**.

## Các Câu Hỏi Mở Cần Thống Nhất

> [!WARNING]
> 1. **Về môi trường triển khai Mamba:** DeepVQE khi deploy (trên Edge devices/Mobile/Server) có hỗ trợ sử dụng custom CUDA kernel (như của `mamba-ssm`) không? Nếu không, chúng ta có nên ưu tiên triển khai Mamba thuần PyTorch (Native PyTorch) ngay từ đầu để đảm bảo tính tương thích, dù tốc độ training có thể chậm hơn?
> 2. **Cấu hình Mamba:** Thay vì GRU (Baseline mặc định dùng 576, còn `D1b_gru768` là config riêng trong `ABLATION_CONFIGS`), chúng ta sẽ bắt đầu thử nghiệm Mamba với cấu hình nào? Đề xuất: 2 blocks Mamba với hidden size = 256 hoặc 384 để giữ số lượng tham số (Params) tương đương hoặc nhỏ hơn Baseline.
> 3. **Về Causal Squeezeformer (Dự phòng):** Có thiết kế sẵn block Squeezeformer để chạy song song so sánh với Mamba không, hay dồn 100% tài nguyên vào thử nghiệm Mamba trước, nếu Mamba fail (về RTF) mới chuyển sang Squeezeformer?

## Các Bước Triển Khai Đề Xuất

---

### Bước 1: Mamba Module

Tạo mới các module Mamba phục vụ cho quá trình thay thế GRU.

- **File mới:** `ablation/modules/mamba.py` (hoặc tương đương)
- **Nhiệm vụ:**
  - Xây dựng hoặc wrap lớp `MambaBlock`.
  - Đảm bảo hỗ trợ Streaming (Stateful) bằng cách truyền và nhận `cache` (hidden state) giữa các frame, tương tự như cách `GRU` hoạt động trong `StreamBottleneck_Ablation`.

---

### Bước 2: Tích hợp vào DeepVQE Ablation Core

Tích hợp Mamba vào kiến trúc Ablation hiện tại.

- **File chỉnh sửa:** `ablation/deepvqe_ablation.py`
- **Nhiệm vụ:**
  - Bổ sung cấu hình mới vào `ABLATION_CONFIGS`, ví dụ: `Mamba_b2_h384` (2 blocks, hidden 384).
  - Tạo class `MambaBottleneck_Ablation(nn.Module)` thay thế cho `Bottleneck_Ablation`. Thay vì dùng `nn.GRU`, module này sẽ khởi tạo N lớp `MambaBlock`.
  - Tạo class `StreamMambaBottleneck_Ablation(nn.Module)` thay thế cho `StreamBottleneck_Ablation`. Module này phải xử lý việc update và truyền `cache` (state) của Mamba qua từng time-step.
  - **Lưu ý quan trọng:** Cập nhật hàm `StreamDeepVQE_Ablation.init_cache` vì cấu trúc Mamba cache sẽ khác biệt hoàn toàn với GRU cache `(1, B, hidden_size)`. Đây là điểm rất dễ bị thiếu sót khi implement streaming.
  - Chỉnh sửa `DeepVQE_Ablation` và các logic liên quan để khởi tạo `MambaBottleneck_Ablation` dựa trên config.

---

### Bước 3: Training & Benchmarking Scripts

Cập nhật các script để hỗ trợ train và đo lường RTF cho kiến trúc mới.

- **File chỉnh sửa:** 
  - `ablation/train_ablation.py`
  - `ablation/ablation_config.py`: Training config thực tế đi qua file này. Cần cập nhật để parse và nhận diện các preset mới như `Mamba_b2_h384` hoặc `GAN_Mamba_b2_h384`.
- **File mới / File cập nhật:** Cần viết script chuyên dụng để đo Stateful RTF (Real-Time Factor). Có thể tạo thư mục `ablation/scripts/benchmark_rtf.py` hoặc tái sử dụng / cập nhật pattern từ script hiện có `ablation/run_ablation_benchmark.py`.
  - Phải đo được RTF trên CPU và GPU cho cả bản Offline và Stream.
  - Cần profile chi tiết RAM/VRAM usage để so sánh trực tiếp với Baseline GRU.

---

### Bước 4: Khả Năng Deploy (ONNX Export)

Bảo đảm mô hình sau Phase 2 vẫn tuân thủ tiêu chuẩn deploy của dự án.

- **File chỉnh sửa:** `ablation/export_ablation_onnx.py` và wrapper `StreamDeepVQE_AblationONNXWrapper` trong `ablation/deepvqe_ablation.py`.
- **Nhiệm vụ:**
  - Kiểm tra và thêm code hỗ trợ export Mamba (hoặc Squeezeformer) sang định dạng ONNX.
  - Xử lý các node/operation đặc thù để đảm bảo ONNX runtime hỗ trợ inference ổn định.

## Kế Hoạch Kiểm Chứng (Verification)

### 1. Automated Tests
- Chạy thử nghiệm trên `MambaBottleneck_Ablation` và `StreamMambaBottleneck_Ablation` để đảm bảo:
  - Shape đầu vào/đầu ra khớp với GRU Bottleneck hiện tại.
  - Chế độ Stream (truyền từng frame) cho kết quả **khớp trong sai số số học cho cùng causal path** (dùng `torch.allclose` với tolerance rõ ràng) so với chế độ Offline. Chú ý với Mamba hoặc stateful normalization có thể sẽ có chênh lệch nhỏ về mặt số học.

### 2. Benchmarking (Pre-training)
- Khởi tạo model Mamba với trọng số ngẫu nhiên.
- Chạy script đo RTF và so sánh với `D1b_gru768`. Mamba bắt buộc phải có Stateful RTF tương đương hoặc nhanh hơn GRU.
- Báo cáo số liệu Params / MACs.

### 3. Quality Verification (Training)
- Chạy train thử nghiệm Mamba trong 5-10 epoch để kiểm tra tính ổn định (không bị NaN/Exploding gradient).
- Train hội tụ và so sánh chỉ số: Kỳ vọng thấy sự cải thiện rõ rệt ở SI-SDR (do khả năng hiểu ngữ cảnh tốt hơn) so với Baseline GRU. RTF không được vượt ngưỡng cho phép.
