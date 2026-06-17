# Kế hoạch triển khai B4 (Loss Ablation) trên Notebook riêng

Mục tiêu của phương án `B4` là cải thiện chất lượng khử nhiễu (tránh bị lẹm giọng) thông qua việc sử dụng **Asymmetric Loss** và **Multi-Resolution STFT Loss** mà không làm tăng chi phí tính toán khi inference. B4 là một hướng đi rất hứa hẹn và đúng đắn cho bài toán NS-only.

Để đảm bảo tính an toàn và minh bạch (không làm hỏng mốc baseline), chúng ta sẽ nhân bản file notebook thành một phiên bản riêng biệt cho B4 và huấn luyện từ đầu (train from scratch).

## Cấu hình B4 Đề Xuất (Đã được làm nhẹ)

> [!TIP]
> - **Hyperparameters an toàn ban đầu:**
>   - `asym_alpha = 2.0`: Phạt nhẹ over-suppression. (Nếu để `5.0` như lúc đầu cộng với `lamda_mag = 70` sẽ quá rủi ro, dễ giữ lại nhiều noise).
>   - `use_multi_res = True`: Bật tính năng Multi-Res STFT Loss.
>   - `multi_res_ffts = [256, 1024]`: Chỉ dùng 2 độ phân giải bổ sung để tránh rủi ro vọt GPU Memory/chậm Colab.
>   - `multi_res_weight = 0.25`: Khởi điểm an toàn.
> - Nếu PESQ/STOI chưa có cải thiện, chúng ta sẽ scale các thông số này lên sau.

## Proposed Changes

Chúng ta sẽ tạo một file notebook mới bằng cách copy từ file baseline.

### [NEW] [train_b4_deepvqe_colab_v4a.ipynb](file:///d:/AI20K/deepvqe/train_b4_deepvqe_colab_v4a.ipynb)

*(Được copy từ `train_base_deepvqe_colab_v4a.ipynb`)*

#### 1. Cell "5. Cấu hình Hyperparameters"
- **Đổi thư mục Checkpoint để không resume nhầm:**
  ```python
  'checkpoint_dir': f'{SHARED_CHECKPOINTS_DIR}/deepvqe_vctk_B4_scratch_v1',
  'output_dir': f'{SHARED_CHECKPOINTS_DIR}/deepvqe_vctk_B4_scratch_v1',
  'resume_checkpoint': 'last.pt',
  'resume_existing': False,
  ```
  Lần chạy chính thức đầu tiên phải giữ `resume_existing=False` để train từ scratch. Chỉ đổi sang `True` nếu Colab bị ngắt sau khi chính B4 đã tạo `last.pt` hợp lệ trong thư mục trên.
- **Thêm cấu hình Loss B4:**
  ```python
  'config_id': 'B4',
  'loss_variant': 'asym_mrstft_v1',
  'asym_alpha': 2.0,
  'use_multi_res': True,
  'multi_res_ffts': [256, 1024],
  'multi_res_hops': [128, 512],
  'multi_res_wins': [256, 1024],
  'multi_res_weight': 0.25,
  ```

#### 2. Cell "8. Hàm tiện ích: STFT, Loss, Checkpoint, Log, Metrics"
- **Tính Asymmetric Loss**: 
  ```python
  mag_diff = pred_mag**c - true_mag**c
  asym_alpha = cfg.get('asym_alpha', 1.0)
  if asym_alpha > 1.0:
      mag_loss = torch.mean(torch.where(mag_diff < 0, (mag_diff)**2 * asym_alpha, (mag_diff)**2))
  else:
      mag_loss = torch.mean(mag_diff**2)
  ```
- **Thêm Multi-Resolution STFT Loss**:
  Dùng vòng lặp quét qua `multi_res_ffts`, tính STFT cho `y_pred` và `y_true`. Sau đó tính `Spectral Convergence Loss` và `L1 Log-Magnitude Loss`.
  ```python
  loss = cfg['lamda_ri'] * (real_loss + imag_loss) + cfg['lamda_mag'] * mag_loss + sisnr
  if cfg.get('use_multi_res'):
      loss = loss + cfg['multi_res_weight'] * multi_res_loss
  ```
- Trả về biến `multi_res_loss` trong dict để log ra màn hình.

#### 3. Cell "9. Vòng lặp huấn luyện (Training Loop)"
- Điều chỉnh phần `print(...)` để log thêm giá trị `multi_res_loss`.

## Verification Plan

> [!WARNING]
> Lưu ý quan trọng: **KHÔNG** so sánh trực tiếp giá trị Validation Loss của B4 với Baseline, vì thang đo của hàm loss đã bị thay đổi (do Asym và Multi-Res). Mọi sự so sánh "hơn thua" phải dựa vào full perceptual eval: `PESQ, STOI, SI-SDR, DNSMOS`.

### Sanity Checks (Trước khi train dài)
Cần thực hiện các bài kiểm tra nhanh (chạy vài batch đầu) để đảm bảo an toàn:
1. **Fallback Test:** Thử set `use_multi_res=False` và `asym_alpha=1.0` $\rightarrow$ giá trị loss in ra phải gần bằng baseline cũ.
2. **Zero Loss Test:** Truyền `y_true` thay cho `y_pred` vào hàm tính multi_res $\rightarrow$ `multi_res_loss` phải $\approx 0$.
3. **Stability Test:** Đảm bảo hàm loss không trả về `NaN` hay `Inf`, tiến trình backward (tính gradient) diễn ra trơn tru.
4. **Memory Smoke Test:** Theo dõi Peak GPU Memory trên Colab xem bộ phân giải `[256, 1024]` có gây tràn RAM hay không.
5. **Config Integrity:** Mở file `config.json` lưu trong checkpoint dir để kiểm tra xem config B4 đã được save đúng chuẩn chưa.
