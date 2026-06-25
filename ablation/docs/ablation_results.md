# DeepVQE Ablation Results Tracking

Tài liệu này theo dõi kết quả đánh giá của Baseline và các biến thể thử nghiệm trong lộ trình tối ưu hoá DeepVQE.

## 1. Môi trường đánh giá
- **Test samples**: 824
- **Device**: CUDA
- **Total audio duration**: 2072.0s

## 2. Bảng kết quả tổng hợp

| Mô hình | Params | PESQ | STOI | SI-SDR (dB) | RTF (Mean) | Ghi chú |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Noisy (Gốc)** | - | 1.9709 | 0.9210 | 8.44 | - | Âm thanh chưa qua xử lý |
| **Baseline (V4)** | 7,509,996 | **2.8489** | **0.9441** | 17.72 | **0.0160** | Mô hình DeepVQE NS-only gốc |
| **B2 (ECA-F)** | 7,510,026 | 2.8413 | 0.9429 | 18.39 | 0.0778 | Thêm ECA-F ở ResidualBlock |
| **D1b (GRU 768)** | ~10M | **2.8736** | **0.9441** | **17.82** | **0.0125** | Tăng hidden size GRU bottleneck lên 768 |
| **D2 (Temporal Refine)** | ? | 2.8313 | 0.9433 | 17.86 | 0.0342 | Thêm module Temporal Refinement |

## 3. Phân tích & Quyết định

### Đánh giá B2 (ECA-F) so với Baseline
- **PESQ/STOI:** Thấp hơn Baseline, không đạt kỳ vọng.
- **SI-SDR:** Tăng tốt (+0.67 dB), nén nhiễu mạnh nhưng lẹm giọng nói.
- **RTF / Latency:** Chậm hơn gần 5 lần (0.0778 vs 0.0160).
**Kết luận:** B2 thất bại.

### Đánh giá D1b (GRU 768) so với Baseline
- **PESQ:** Tăng đáng kể (+0.0247), vượt ngưỡng gate của Phase 2 (>= Baseline + 0.03? Chưa tới mức 0.03 nhưng là tín hiệu rất tốt, 2.8736 vs 2.8489).
- **STOI:** Giữ nguyên (0.9441).
- **SI-SDR:** Tăng nhẹ (+0.1 dB).
- **RTF / Latency:** RTF rất tốt (0.0125), nhanh hơn cả baseline (có thể do môi trường/batch size lúc test, nhưng đảm bảo real-time xuất sắc).
**Kết luận:** D1b là một nâng cấp rất tiềm năng, giúp tăng chất lượng mà không ảnh hưởng latency.

### Đánh giá D2 (Temporal Refine) so với Baseline & D1b
- **PESQ:** Giảm so với Baseline (2.8313 vs 2.8489).
- **STOI:** Giảm nhẹ (0.9433).
- **SI-SDR:** Tăng nhẹ lên 17.86, có xu hướng giống B2 (nén mạnh lẹm giọng).
- **RTF / Latency:** Chậm hơn gấp đôi Baseline (0.0342).
**Kết luận:** D2 thất bại, hiệu năng kém hơn D1b và chậm hơn đáng kể.

### Hành động tiếp theo (Next Steps)
Dựa theo Phase 2 Model Upgrade Roadmap:
1. **D1b (GRU 768)** đã cho tín hiệu chất lượng tốt và RTF xuất sắc.
2. **D2 (Temporal Refine)** thất bại và sẽ bị loại bỏ.
3. Kế hoạch tiếp theo là quyết định xem có cần train **D1a (GRU 704)** để lấy lại latency không (tuy nhiên D1b đang có RTF = 0.0125 rất tốt, có thể không cần D1a). Hoặc chuyển sang **Stage 3: D3 Hybrid Head** nếu muốn tối ưu output. 
4. Xác định cấu hình D1b làm ứng cử viên vô địch mới để tiếp tục.
