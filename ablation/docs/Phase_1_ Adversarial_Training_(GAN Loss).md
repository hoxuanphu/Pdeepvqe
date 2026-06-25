# Kế hoạch Triển khai Phase 1: Adversarial Training (GAN Loss) cho DeepVQE

Trong Phase 1, mục tiêu của chúng ta là thêm một mạng Discriminator để ép mô hình DeepVQE (Generator) tạo ra âm thanh tự nhiên hơn, giảm hiện tượng "robotic" hay cắt lẹm giọng. Quan trọng nhất, **Discriminator sẽ bị vứt bỏ khi Inference**, nên RTF (Real-Time Factor) không đổi.

Dựa trên việc đọc mã nguồn `train_ablation.py`, dưới đây là phương án triển khai chi tiết:

---

## 1. Quyết định Thiết kế cốt lõi (Design Choices)

Thay vì dùng **MetricGAN** (đòi hỏi phải chạy hàm tính PESQ cho từng batch trong lúc train, gây thắt cổ chai nghẽn cổ chai CPU và cực kỳ chậm), tôi đề xuất sử dụng **Least Squares GAN (LSGAN) / PatchGAN** trên Spectrogram.
* **Mạng D (Discriminator):** Mạng CNN 2D nhỏ gọn (khoảng 3-4 lớp Conv2d), nhận đầu vào là Magnitude Spectrogram.
* **Cơ chế:** 
  * D sẽ học cách chấm điểm `1.0` cho âm thanh sạch (Clean/Target) và `0.0` cho âm thanh do DeepVQE tạo ra.
  * G (DeepVQE) sẽ học cách "lừa" D để D chấm output của nó là `1.0`.

---

## 2. Các File Cần Chỉnh Sửa & Viết Mới

### A. Viết mới file `ablation/discriminator.py`
* Tạo một class `Discriminator(nn.Module)`:
  * Đầu vào: Magnitude Spectrogram `[B, 1, F, T]` (F=257).
  * Các lớp: `Conv2d` kết hợp `LeakyReLU` và `BatchNorm2d` (hoặc `InstanceNorm2d`).
  * Đầu ra: Trả về một ma trận điểm số PatchGAN (đánh giá từng vùng thời gian-tần số) hoặc 1 giá trị vô hướng.

### B. Sửa file `ablation/train_ablation.py`

#### 1. Khởi tạo Mô hình & Optimizer
* Trong `make_model()`, ngoài việc khởi tạo `DeepVQE_Ablation`, ta sẽ khởi tạo thêm mạng `Discriminator` và đẩy lên `device`.
* Trong `make_optimizer_scheduler()`, tách ra trả về 2 bộ Optimizer & Scheduler:
  * `opt_G` (cho DeepVQE)
  * `opt_D` (cho Discriminator, thường learning rate của D sẽ nhỏ hơn G, ví dụ: $1 \times 10^{-4}$).

#### 2. Cấu trúc lại vòng lặp Train trong `run_epoch()`
Mã nguồn hiện tại chỉ có một bước update gradient. Cần phải chia làm 2 bước luân phiên:

**Bước 1: Train Discriminator (Update `opt_D`)**
* `fake_spec_wav = model(mixture_spec).detach()` (Không tính gradient cho G ở bước này).
* Tính Magnitude của `fake_spec_wav` và `target_spec`.
* Gọi mạng D: 
  * `pred_real = D(mag(target_spec))`
  * `pred_fake = D(mag(fake_spec_wav))`
* `loss_D = 0.5 * MSE(pred_real, 1.0) + 0.5 * MSE(pred_fake, 0.0)`
* `loss_D.backward()` $\rightarrow$ `opt_D.step()`

**Bước 2: Train Generator (Update `opt_G`)**
* Tính STFT Loss và SISNR Loss như cũ (`real_loss + imag_loss + mag_loss + sisnr`).
* Lấy phổ `fake_spec` vừa tạo tính Magnitude, cho chạy qua D:
  * `pred_fake_for_G = D(mag(fake_spec))`
* Tính Adv Loss (Lừa Discriminator):
  * `loss_adv = MSE(pred_fake_for_G, 1.0)`
* Tổng hợp Loss G: 
  * `loss_G = STFT_SISNR_Loss + lambda_adv * loss_adv` (Với `lambda_adv` có thể là $0.01$ đến $0.1$ để không phá vỡ STFT loss gốc).
* `loss_G.backward()` $\rightarrow$ `opt_G.step()`

#### 3. Xử lý Checkpoint & Resume
* Sửa các hàm `save_checkpoint` và `load_checkpoint` để lưu trữ và tải lên trạng thái của cả `model_D`, `opt_D`, và `scaler_D` (nếu dùng AMP).

---

## 3. Quản lý Config (`ablation/ablation_config.py`)
Cần thêm các tham số vào Config để kiểm soát GAN:
* `training.use_gan: True/False` (Để bật tắt chức năng này dễ dàng).
* `loss.lamda_adv: 0.05` (Trọng số của GAN Loss so với STFT Loss).

---

## 4. Xác nhận / Câu hỏi mở (User Feedback Required)

> [!IMPORTANT]
> **Vui lòng duyệt các quyết định sau trước khi tôi bắt đầu code:**
> 1. **Kiến trúc Discriminator:** Bạn đồng ý dùng LSGAN/PatchGAN trên phổ Amplitude (Magnitude Spectrogram) chứ? Đây là cách cân bằng tốt nhất giữa tốc độ train và chất lượng âm thanh, tránh được overhead do tính PESQ liên tục.
> 2. **Trọng số Loss:** Tôi dự định set `lambda_adv = 0.05` làm mặc định ban đầu để GAN Loss không "áp đảo" SISNR/STFT Loss. Bạn có gợi ý khác không?
> 3. **Tách Biến / Tương thích ngược:** Để an toàn cho các Phase cũ, tôi sẽ bọc code GAN trong lệnh if: `if cfg["training"].get("use_gan", False):`. Như vậy các file config không có dòng này vẫn sẽ train bình thường như cũ. Bạn đồng ý chứ?
