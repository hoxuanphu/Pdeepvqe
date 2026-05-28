# Lộ trình Nâng cấp DeepVQE với ECAPA-TDNN (Cá nhân hóa)

Tài liệu này cung cấp lộ trình chi tiết để nâng cấp mô hình **DeepVQE** (Non-personalized) thành **Personalized DeepVQE (pDeepVQE)**. Tài liệu đã được tổng hợp và tinh chỉnh bởi toàn bộ các phản biện kỹ thuật chuyên sâu (Reviewer 1 & 2) để đảm bảo mô hình có thể hoạt động ổn định, toán học chuẩn xác và chống chịu tốt nhất trong thực tế.

## Mục tiêu (Goal)
Chuyển đổi bài toán từ Khử nhiễu (Speech Enhancement) thành Trích xuất giọng nói mục tiêu (Target Speech Extraction - TSE).

---

## Giai đoạn 1: Chuẩn bị Môi trường và Phụ thuộc
- **Cài đặt:** `pip install speechbrain torchaudio`
- **Mô hình Pretrained:** Sử dụng `speechbrain/spkrec-ecapa-voxceleb` (192-dim embedding).

---

## Giai đoạn 2: Pipeline Dữ liệu (Data Preparation)

### 2.1. Cấu trúc Mixture và Tỷ lệ trộn
Tập dữ liệu huấn luyện nên bao phủ đa dạng các trường hợp. Tỷ lệ trộn khuyến nghị:
- **70% TSE hoàn chỉnh:** Target + Interfering Speech (1-2 người) + Noise.
- **15% Chỉ khử nhiễu:** Target + Noise (Không có người nói thứ hai). *Lưu ý: Ở case này VẪN PHẢI truyền đúng Enrollment của Target để ép mô hình hình thành thói quen luôn luôn nhìn vào Embedding.*
- **15% Chỉ tách giọng:** Target + Interfering Speech (Không có tiếng ồn).

### 2.2. Room Impulse Response (RIR) và Ground Truth Target
- **RIR (Mô phỏng vang phòng):** Bắt buộc dùng RIR để chập (convolve) vào âm thanh khô. Nguồn dữ liệu: `RIRS_NOISES` (OpenSLR 28) hoặc `MIT RIR Survey`.
- **Định nghĩa Ground Truth để tính Loss:** Giai đoạn đầu nên dùng **Reverberant Target** (Âm thanh đích đã có vang phòng). Không nên ép mô hình vừa tách giọng vừa khử vang ngay từ đầu.

### 2.3. Định nghĩa SNR/SIR và Curriculum Learning
- Áp dụng **Curriculum Learning**: Ở những epoch đầu, mix SNR và SIR cao (Dễ: 10dB ~ 20dB). Sau đó giảm dần tỷ lệ xuống (Khó: -5dB ~ 5dB).

### 2.4. Tiền xử lý Enrollment Audio và Metadata Caching
- **VAD (Voice Activity Detection):** Cắt bỏ khoảng lặng (silence trimming).
- **Trường hợp âm thanh ngắn:** Nếu enrollment gốc < 3s, tiến hành lặp lại (repeat) tín hiệu.
- **Cache Embeddings:** Chạy ECAPA sinh vector lưu sẵn vào đĩa (`.npy`) để tăng tốc train. 
  - **[BẮT BUỘC 1]:** Cache Embedding phải lưu theo **từng đoạn enrollment segment riêng biệt**, KHÔNG lưu gộp theo `speaker_id`. Một speaker cần có đa dạng các segment khác nhau để giúp mô hình robust (chống chịu tốt) hơn với các chất lượng giọng khác nhau.
  - **[BẮT BUỘC 2]:** Phải lưu kèm Metadata (speaker_id, utterance_id, sample_rate, duration) dưới dạng `.json` để tránh hiểm họa Data Leakage. Khẳng định lại, cached `spk_emb` truyền vào Mixture phải luôn luôn thuộc về một utterance_id khác với Target.

---

## Giai đoạn 3: Nâng cấp Kiến trúc Code (`deepvqe.py`)

### Hỗ trợ Đa đầu vào, Xử lý Sample Rate & Khởi tạo FiLM
*Lưu ý kỹ thuật: Thiết kế này đóng vai trò là **Baseline vững chắc**. Sau khi Phase 1 chạy ổn, nếu khả năng tách speaker vẫn yếu thì mới nên mở rộng FiLM sang các nhánh Decoder/Skip-connections.*

```python
from speechbrain.inference.speaker import EncoderClassifier
import torchaudio.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PersonalizedBottleneck(nn.Module):
    def __init__(self, input_size, hidden_size, emb_dim=192):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        self.film_gamma = nn.Linear(emb_dim, input_size)
        self.film_beta = nn.Linear(emb_dim, input_size)
        
        # [QUAN TRỌNG]: Khởi tạo FiLM làm Identity Mapping ban đầu
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.ones_(self.film_gamma.bias) # Gamma ~ 1
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias) # Beta ~ 0
        
    def forward(self, x, spk_emb):
        y = rearrange(x, 'b c t f -> b t (c f)')
        y, _ = self.gru(y)
        y = self.fc(y)
        
        # [QUAN TRỌNG]: L2-Normalize Embedding (Set rõ eps=1e-8 để an toàn toán học)
        spk_emb = F.normalize(spk_emb, p=2, dim=-1, eps=1e-8)
        
        gamma = self.film_gamma(spk_emb).unsqueeze(1)
        beta = self.film_beta(spk_emb).unsqueeze(1)
        
        y_film = gamma * y + beta
        return rearrange(y_film, 'b t (c f) -> b c t f', c=x.shape[1])

class PersonalizedDeepVQE(nn.Module):
    def __init__(self, deepvqe_sr=16000, device="cpu"):
        super().__init__()
        # Tránh hardcode "cuda", linh hoạt theo thiết bị
        self.ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            run_opts={"device": device}
        )
        for param in self.ecapa.parameters():
            param.requires_grad = False
            
        self.resample = T.Resample(orig_freq=deepvqe_sr, new_freq=16000) if deepvqe_sr != 16000 else nn.Identity()
        # ... (các khối khác)
        self.bottle = PersonalizedBottleneck(128*9, 64*9, emb_dim=192)

    def forward(self, x_mixture_stft, enrollment_wav=None, spk_emb=None):
        if spk_emb is None:
            if enrollment_wav is None:
                raise ValueError("Must provide either enrollment_wav or spk_emb")
            with torch.no_grad():
                enroll_16k = self.resample(enrollment_wav)
                spk_emb = self.ecapa.encode_batch(enroll_16k)
                
                # Kiểm tra shape trước khi squeeze phòng khi SpeechBrain cập nhật API
                if spk_emb.dim() == 3:
                    spk_emb = spk_emb.squeeze(1) # (B, 192)
                
        # Forward DeepVQE
        # ...
        en_xr = self.bottle(en_x5, spk_emb)
        # ...
        # return x_enh
```

---

## Giai đoạn 4: Hàm Mất Mát (Loss Function) và Chỉ số Đánh giá

### 4.1. Công thức Mất mát và Phase Training
Không sử dụng tất cả Loss cùng lúc. Hãy đi theo 3 Phase huấn luyện:

- **Phase 1 (Baseline Training - Tái tạo Tín hiệu):** Chỉ tập trung vào SI-SDR và L1/MSE phổ STFT. 
  $$ \text{Loss}_{P1} = \lambda_1 \times (-\text{SI-SDR}(x_{enh}, x_{target})) + \lambda_2 \times \text{L1}(|X_{enh}|, |X_{target}|) $$

- **Phase 2 (Speaker Consistency Loss):** Thêm auxiliary loss (với $\alpha$ cực nhỏ: 0.01-0.1).
  $$ \text{Loss}_{total} = \text{Loss}_{P1} + \alpha \times (1 - \text{CosineSimilarity}(emb_{target\_waveform}, emb_{enh})) $$
  > [!WARNING]
  > **Differentiability Cảnh báo Đỏ:** Khởi tạo một `eval_ecapa` hoàn toàn tách biệt và luôn gọi `eval_ecapa.eval()`. 
  > 1. Set `requires_grad=False` cho mọi trọng số của `eval_ecapa` để KHÔNG update trọng số. Tuy nhiên, việc này **sẽ tiêu tốn thêm rất nhiều vRAM** vì đồ thị đạo hàm phải luân chuyển qua ECAPA ngược về $x_{enh}$.
  > 2. Đảm bảo luồng $x_{enh}$ (STFT) -> ISTFT -> Waveform -> `eval_ecapa` phải giữ nguyên khả năng vi phân (differentiable). 
  > 3. TUYỆT ĐỐI KHÔNG bọc hàm này bằng `with torch.no_grad():` nếu muốn gradient backprop.
  > *(Mẹo tối ưu: Chỉ bật Speaker Loss sau khi đã tune ổn định batch size / chiều dài clip, hoặc tính Speaker Loss ngắt quãng mỗi N batch để tiết kiệm bộ nhớ).*

- **Phase 3 (Negative Case Training):** Đưa thêm các mẫu "Absent speaker" (Cung cấp Target Embedding của một người không hề có mặt trong âm thanh).
  > [!CAUTION]
  > **Tuyệt đối không dùng SI-SDR cho target chuỗi Zeros** vì phép chia cho năng lượng bằng 0 sẽ khiến đạo hàm nổ (NaN). Đối với Negative Sample, hãy dùng Loss để triệt tiêu năng lượng: **Kết hợp Waveform Energy Loss (L1/MSE) + Magnitude Suppression Loss** để mạng ổn định nhất.

### 4.2. Metric Đánh giá 
- **Đo hiệu năng tách lọc:** SI-SDRi, SDRi.
- **Đo chất lượng âm học:** PESQ (Wideband), STOI.
- **Đo nhận diện người đích:** **Speaker Similarity (Cosine) / EER** để đảm bảo mô hình giữ đúng giọng.
- **Metric chuyên dụng cho Negative Case:** Không dùng SI-SDR, sử dụng **Output Energy Ratio (Attenuation)**:
  $$ \text{Attenuation} = 10 \times \log_{10} \left( \frac{\text{Energy}(x_{enh})}{\text{Energy}(x_{mixture})} \right) $$
  *(Đo lường đầu ra bị đè nén năng lượng tốt đến mức nào khi cung cấp sai ID. **Chỉ số này càng Âm càng tốt**).*

---

## Giai đoạn 5: Chiến lược Huấn luyện (Training Strategy)

- **Sanity Check (Overfit test):** Trích 1-2 batch để train overfit. Đảm bảo loss tụt tiệm cận 0, gradient sạch.
- **Đóng băng ECAPA (Freeze):** Luôn đóng băng ECAPA gốc (ở nhánh Embedding) trừ khi dataset của bạn khổng lồ.
- **Hyperparameters Baseline:** AdamW, Learning Rate $1e-4$. Giảm LR khi plateau. Early Stopping dựa trên SI-SDRi tập validation.
