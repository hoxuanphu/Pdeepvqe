# Tổng Quan Phương Pháp Tối Ưu Performance DeepVQE

Dựa trên lộ trình thử nghiệm `deepvqe_optimization_roadmap.md`, dưới đây là tổng hợp các phương pháp tối ưu performance (độ chính xác/chất lượng âm thanh) của model DeepVQE với ràng buộc nghiêm ngặt là **không làm tăng Real-Time Factor (RTF) và tính toán (Params, FLOPs/MACs) so với Baseline**.

Lộ trình sử dụng phương pháp **Ablation Study** (thử nghiệm riêng lẻ từng giả thuyết) để kiểm chứng và đánh đổi (trade-off) computation.

## 1. Các Phương Pháp Tối Ưu Component (Giả Thuyết)

Chiến lược chính là **giảm tải ở một nơi và đắp vào nơi khác (Compute Reallocation)**:

*   **Giảm tải tính toán bằng Depthwise Separable Convolutions (DW-Conv):**
    *   *Mục đích:* Các lớp `Conv2d` tiêu chuẩn tốn rất nhiều FLOPs/Params. Việc thay thế bằng DW-Conv sẽ giảm đáng kể lượng tính toán.
    *   *Cách làm:* Bắt đầu áp dụng DW-Conv một cách an toàn **chỉ tại các ResidualBlock**, giữ nguyên các Conv chính ở Encoder/Decoder để tránh làm hỏng luồng trích xuất đặc trưng cốt lõi.
*   **Tăng cường chất lượng bằng Efficient Channel Attention (Causal ECA):**
    *   *Mục đích:* Giúp mạng tập trung vào các channel quan trọng nhưng tốn cực kỳ ít Params/FLOPs.
    *   *Cách làm (Ràng buộc Causal):* Dùng **ECA-F** (chỉ pooling trên trục Frequency) ở các block để đảm bảo tính Causal (không vi phạm streaming).
*   **Cải thiện Activation Function (PReLU thay vì ELU):**
    *   *Mục đích:* Giúp model học biểu diễn tốt hơn ở vùng giá trị âm.
    *   *Cách làm:* Thử nghiệm PReLU dạng Shared (1 tham số chung) hoặc Per-Channel (1 tham số cho mỗi channel).
*   **Bottleneck Tuning (Giảm kích thước GRU):**
    *   *Mục đích:* Layer GRU chiếm tỷ trọng tính toán và Params rất lớn. Giảm kích thước sẽ tiết kiệm nhiều RTF.
    *   *Cách làm:* Giảm `hidden_size` từ 576 xuống 544 hoặc 512 (chỉ áp dụng khi cần thiết).

## 2. Chiến Lược Triển Khai (Ablation Matrix)

Lộ trình được chia thành các Phase với các bộ lọc tiêu chí khắt khe để đảm bảo model cuối cùng thỏa mãn ràng buộc:

### Phase 1: Tập trung vào thay đổi cốt lõi (Core Scope)
*   **Stage 1 - Thăm dò (Low-risk Accuracy Probe):** Thử cấy riêng lẻ PReLU hoặc ECA-F vào Baseline. Ở bước này, cho phép Params/FLOPs tăng rất nhẹ ($\le +1\%$) để xem component nào thực sự giúp tăng Quality (PESQ, DNSMOS).
*   **Stage 2 - Tái phân bổ tính toán (Compute Reallocation):** Đây là bước chốt hạ cho Phase 1. Bắt đầu đưa **DW-Conv** vào ResidualBlock để ép giảm Params/FLOPs/RTF. Sau đó, kết hợp các thành phần "chiến thắng" ở Stage 1 (ECA-F, PReLU) vào cùng.
    *   *Tiêu chí Pass (Deploy Pass):* Mô hình lúc này **bắt buộc** phải có Params/FLOPs/Stateful RTF/ONNX latency $\le$ Baseline, đồng thời Quality metric phải tăng đáng kể (ví dụ: $\Delta$ PESQ $\ge +0.03$).

### Phase 2: Mở rộng rủi ro (Risk Escalation - Chỉ kích hoạt khi cần)
Phase 2 không chạy mặc định mà phụ thuộc vào việc Phase 1 có đạt mục tiêu hay không:
*   **Stage 3 - Cắt giảm mạnh tay (Aggressive Lightweight):** Nếu model Stage 2 đạt RTF nhưng Quality tăng quá ít, lộ trình sẽ ép dùng DW-Conv sâu hơn nữa (vào cả Encoder/Decoder) để giải phóng thêm compute, lấy chỗ nhồi thêm Attention.
*   **Stage 4 - Ép Bottleneck:** Nếu model Stage 2 có Quality rất tốt nhưng lại vi phạm (vượt quá) RTF, lộ trình sẽ buộc phải cắt giảm kích thước GRU (từ 576 xuống 544/512) để hạ RTF xuống dưới mức Baseline.

## 3. Đánh Giá Khách Quan

Để kết quả ablation đáng tin cậy, roadmap quy định chặt chẽ:
*   **Đo Real-time Factor (RTF):** Không tin hoàn toàn vào profiler đếm lý thuyết. Bắt buộc đo **Stateful Streaming RTF** và ONNX latency thực tế.
*   **Khóa Budget:** Mọi model thử nghiệm phải chạy cùng một setting data, optimizer, LR schedule,... và train lại từ đầu (train from scratch) để tránh bias.

> **Tóm lại:** Hướng đi chủ đạo của roadmap này là dùng **DW-Conv để "mua" lại ngân sách tính toán**, sau đó dùng ngân sách dư ra đó để "đầu tư" vào **PReLU và ECA-F** nhằm tăng cường độ chính xác, trong khi tổng chi phí (RTF, FLOPs) luôn được giữ ở mức bằng hoặc thấp hơn Baseline.
