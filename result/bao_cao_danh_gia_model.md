# Báo cáo đánh giá các model DeepVQE

## 1. Dữ liệu đánh giá

Báo cáo này tổng hợp kết quả từ các file `eval_metrics_summary.csv` trong thư mục `result/`. Mỗi model được đánh giá trên 824 mẫu trong file `eval_metrics_per_sample.csv`.

Các chỉ số noisy tham chiếu trong summary là giống nhau giữa các model:

- PESQ noisy: 1.9709
- STOI noisy: 0.9210
- SI-SDR noisy: 8.44 dB

Quy ước đọc kết quả:

- PESQ, STOI, SI-SDR càng cao càng tốt.
- Delta improvement càng cao càng tốt; giá trị dương nghĩa là model cải thiện so với noisy input.
- RTF càng thấp càng tốt; RTF < 1 nghĩa là có khả năng chạy real-time.
- RTF chỉ nên so sánh trực tiếp khi được đo trên cùng phần cứng và cùng kiểu thiết bị; nếu có khác biệt giữa CPU và GPU thì số này có thể lệch đáng kể.

## 2. Diễn giải ký hiệu model

| Ký hiệu | Diễn giải ngắn | Khác gì so với baseline |
|---|---|---|
| Baseline | Mô hình DeepVQE gốc | Mốc so sánh chuẩn, không thay đổi kiến trúc. |
| D1b_gru768 | Baseline + GRU hidden 768 | Tăng sức chứa bottleneck GRU từ 576 lên 768, giữ nguyên các khối còn lại. |
| B1a | PReLU dùng chung tham số | Thay ELU bằng PReLU shared ở các khối liên quan. |
| B1b | PReLU theo từng kênh | Cũng thay ELU bằng PReLU, nhưng mỗi kênh có tham số riêng. |
| B2 | ECA-F trong residual blocks | Thêm attention ECA-F, tăng nhẹ năng lực nhưng làm chậm đáng kể. |
| B3b | Skip gate kiểu `se_f` | Thêm cơ chế gating theo tần số ở nhánh skip; là biến thể kết hợp của nhánh B3. |
| B4 | Asymmetric mag + MR-STFT loss | Không đổi mạnh kiến trúc, nhưng đổi hàm loss sang biến thể thiên về chất lượng phổ. |
| C1a-G2 | Residual block grouped conv với `res_groups=2` | Thay residual conv sang grouped conv, giảm độ dày tính toán so với baseline. |

Ghi chú:

- `B1a/B1b` là nhóm thay đổi activation.
- `B2/B3b` là nhóm thêm cơ chế chú ý/gating.
- `B4` chủ yếu đổi objective huấn luyện.
- `C1a-G2` là biến thể nhóm `C1a`, trong đó phần residual convolution dùng `groups=2`.

## 3. Bảng kết quả trung bình từ summary

Lưu ý: RTF ở các file summary không hoàn toàn đồng nhất, phép đo được chạy trên CPU và GPU khác nhau, nên phần so sánh RTF nên được hiểu là tương đối và chỉ thật sự công bằng khi cùng môi trường đo.
| Model | PESQ enhanced | PESQ Δ | STOI enhanced | STOI Δ | SI-SDR enhanced (dB) | SI-SDR Δ (dB) | RTF mean | RTF overall | Real-time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | 2.8490 | 0.8781 | 0.9441 | 0.0231 | 17.72 | 9.28 | 0.375979 | 0.374990 | Có |
| D1b_gru768 | 2.8736 | 0.9027 | 0.9441 | 0.0231 | 17.82 | 9.37 | 0.012577 | 0.012381 | Có |
| B1a | 1.1139 | -0.8570 | 0.6913 | -0.2297 | -14.31 | -22.76 | 0.388954 | 0.388672 | Có |
| B1b | 2.8623 | 0.8914 | 0.9435 | 0.0225 | 17.49 | 9.05 | 0.013425 | 0.013212 | Có |
| B2 | 2.8626 | 0.8918 | 0.9440 | 0.0230 | 17.67 | 9.22 | 0.076433 | 0.077359 | Có |
| B3b | 2.3404 | 0.3695 | 0.9337 | 0.0127 | 17.07 | 8.62 | 0.406263 | 0.406251 | Có |
| B4 | 2.7901 | 0.8192 | 0.9453 | 0.0243 | 17.87 | 9.43 | 0.408504 | 0.406850 | Có |
| C1a-G2 | 2.3077 | 0.3368 | 0.9333 | 0.0122 | 17.15 | 8.71 | 0.351012 | 0.349205 | Có |

## 4. Nhận xét theo từng nhóm chỉ số

### 4.1. Chất lượng PESQ

`D1b_gru768` đạt PESQ enhanced cao nhất với 2.8736, nhỉnh hơn `B2` 0.0110 điểm và `B1b` 0.0113 điểm. Đây là model tốt nhất hiện có về PESQ và cũng cải thiện 0.0246 điểm so với baseline.

`B2` và `B1b` vẫn nằm rất sát nhóm đầu với PESQ lần lượt là 2.8626 và 2.8623. `Baseline` cũng rất mạnh với PESQ 2.8490, nhưng vẫn thấp hơn `D1b_gru768` 0.0246 điểm. `B4` đạt 2.7901, vẫn tốt nhưng kém hơn nhóm `D1b/B2/B1b/Baseline` về PESQ.

`B1a` là trường hợp bất thường: PESQ enhanced chỉ đạt 1.1139 và PESQ Δ = -0.8570, nghĩa là chất lượng PESQ giảm mạnh so với noisy input.

### 4.2. Độ rõ và độ tương đồng nội dung theo STOI

`B4` có STOI enhanced cao nhất với 0.9453 và STOI Δ = 0.0243. `D1b_gru768` và `Baseline` cùng đạt 0.9441, nên nằm trong nhóm đầu ngay sát `B4`. Nhóm tiếp theo gồm `B2` 0.9440 và `B1b` 0.9435; các chênh lệch này rất nhỏ, nên có thể xem là tương đối sát nhau.

`B3b` và `C1a-G2` vẫn cải thiện STOI so với noisy input nhưng mức cải thiện thấp hơn, lần lượt là 0.0127 và 0.0122.

`B1a` làm giảm STOI từ 0.9210 xuống 0.6913, không phù hợp để sử dụng.

### 4.3. SI-SDR và khả năng khử nhiễu

`B4` đạt SI-SDR enhanced cao nhất với 17.87 dB và SI-SDR Δ = 9.43 dB. `D1b_gru768` đứng ngay sau với 17.82 dB và SI-SDR Δ = 9.37 dB, cao hơn baseline 0.10 dB. Đây là model tốt nhất nếu ưu tiên cải thiện tín hiệu theo SI-SDR.

`Baseline` đứng tiếp theo với 17.72 dB, sau đó là `B2` với 17.67 dB và `B1b` với 17.49 dB. Nhóm này đều cho mức cải thiện SI-SDR lớn, trên 9 dB hoặc xấp xỉ 9 dB.

`B1a` tiếp tục là outlier xấu với SI-SDR enhanced -14.31 dB và SI-SDR Δ = -22.76 dB.

### 4.4. Tốc độ suy luận

Tất cả các model đều có RTF < 1, nghĩa là đều đạt điều kiện real-time theo summary.

`D1b_gru768` nhanh nhất rõ rệt với RTF mean 0.012577. `B1b` đứng thứ hai với RTF mean 0.013425, vẫn rất nhanh. `B2` đứng thứ ba với RTF mean 0.076433, vẫn nhanh hơn nhiều so với baseline.

`C1a-G2`, `Baseline`, `B1a`, `B3b` và `B4` có RTF mean trong khoảng 0.35-0.41. Trong nhóm này, `B4` chậm nhất với RTF mean 0.408504, nhưng vẫn đủ real-time.

Lưu ý: RTF ở các file summary không hoàn toàn đồng nhất, phép đo được chạy trên CPU và GPU khác nhau, nên phần so sánh RTF nên được hiểu là tương đối và chỉ thật sự công bằng khi cùng môi trường đo.

## 5. Xếp hạng tổng quan

| Hạng mục | Model nổi bật | Lý do |
|---|---|---|
| PESQ tốt nhất | D1b_gru768 | PESQ cao nhất 2.8736, PESQ Δ 0.9027 |
| STOI tốt nhất | B4 | STOI cao nhất 0.9453, STOI Δ 0.0243 |
| SI-SDR tốt nhất | B4 | SI-SDR enhanced 17.87 dB, SI-SDR Δ 9.43 dB |
| Tốc độ tốt nhất | D1b_gru768 | RTF mean thấp nhất 0.012577 |
| Cân bằng chất lượng/tốc độ | D1b_gru768 | PESQ cao nhất, STOI/SI-SDR nằm trong nhóm đầu và RTF rất thấp |
| Không nên dùng | B1a | Tất cả chỉ số chính đều giảm mạnh so với noisy input |

## 6. Kết luận và đề xuất
Lần đánh giá này đã có một ứng viên vượt baseline khá rõ là `D1b_gru768`: PESQ tăng 0.0246, SI-SDR tăng 0.10 dB, STOI giữ nguyên ở mức 0.9441, và RTF mean còn thấp nhất trong nhóm summary hiện tại.

`D1b_gru768` là lựa chọn cân bằng nhất và là ứng viên ưu tiên nếu muốn giữ lại một model duy nhất cho giai đoạn tiếp theo.

`B4` vẫn phù hợp nếu ưu tiên chất lượng tín hiệu theo STOI và SI-SDR. Model này cho STOI và SI-SDR tốt nhất, nhưng PESQ thấp hơn `D1b_gru768`.

`B1b` là lựa chọn tốc độ tốt, tuy nhiên `D1b_gru768` đã vượt nhẹ cả về tốc độ lẫn chất lượng.

`B2` vẫn tốt nhưng không còn dẫn đầu về PESQ. `Baseline` vẫn là mốc so sánh chuẩn, nhưng đã bị `D1b_gru768` vượt trên PESQ và SI-SDR, còn STOI thì ngang nhau.

`B3b` và `C1a-G2` có cải thiện so với noisy input nhưng kém hơn các model tốt nhất trên hầu hết chỉ số, nên không phải ứng viên ưu tiên.

`B1a` không nên sử dụng trong trạng thái hiện tại vì làm giảm chất lượng rõ rệt ở PESQ, STOI và SI-SDR.
