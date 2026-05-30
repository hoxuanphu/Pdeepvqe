# Lộ trình Nâng cấp DeepVQE với Internal Embedding (Cá nhân hóa)

Tài liệu này cung cấp một **Option thay thế** cho lộ trình dùng ECAPA-TDNN, dựa trên nghiên cứu *"Personalized Speech Enhancement Without a Separate Speaker Embedding Model"* (Microsoft, 2024).

Phương pháp này **KHÔNG CẦN** bất kỳ mô hình nhận diện giọng nói riêng biệt nào. Thay vào đó, nó tận dụng chính trạng thái nội bộ của DeepVQE để đại diện cho danh tính người nói. Đây là một lựa chọn cực kỳ tốt cho các ứng dụng real-time và on-device.

## Mục tiêu (Goal)
Chuyển đổi bài toán từ Khử nhiễu thành Trích xuất giọng nói mục tiêu (TSE) bằng một mô hình DeepVQE duy nhất. Mục tiêu là đạt tốc độ xử lý rất cao (do loại bỏ được mô hình ECAPA-TDNN) và giảm thiểu hiện tượng xóa nhầm giọng.

> [!NOTE]
> *Lưu ý về tốc độ (RTF):* Bài báo gốc đạt RTF 0.0135 trên Intel i7-10700K cho mô hình nhỏ (PVQE-S). Tuy nhiên, kiến trúc `deepvqe.py` hiện tại của repo (GRU 576, 5 khối encoder) khác với bài báo. Do đó, tốc độ thực tế cần được benchmark lại chứ không nên mặc định < 0.02.

---

## Giai đoạn 1: Pipeline Dữ liệu (Data Preparation)

Dữ liệu huấn luyện kế thừa cấu trúc trộn (Mix) của TSE, giúp tiết kiệm bước cache embedding offline bằng ECAPA khi train.

### 1.1. Cấu trúc Mixture
- **60% TSE hoàn chỉnh:** Target + Interfering Speech + Noise.
- **15% Chỉ khử nhiễu:** Target + Noise.
- **15% Chỉ tách giọng:** Target + Interfering Speech.
- **10% Tình huống vắng mặt (Negative Case / Absent Speaker):** Mixture chỉ chứa Interferer + Noise, Target bị ép thành Silence.

> [!TIP]
> Tỷ lệ 60/15/15/10 ở trên là cấu hình **mở rộng thực dụng** cho bài toán TSE thực tế của repo. Theo chuẩn nguyên bản của bài báo: Clip train dài 40s, trong đó 30% có background speech (SIR [0, 20]).

> [!WARNING]
> **Khuyến nghị Clip Length khi Train:**
> Phương pháp Internal Embedding phụ thuộc vào GRU tích lũy context theo thời gian để xây dựng speaker representation. Clip quá ngắn (ví dụ 4s như config ECAPA) sẽ khiến average pooling cho embedding rất noisy.
> - **Nếu VRAM cho phép:** Dùng clip **10s–20s** trở lên (paper dùng 40s).
> - **Nếu VRAM hạn chế:** Giảm batch size xuống 2-4, kết hợp **gradient accumulation** (accumulate 4-8 steps) để bù lại effective batch size.
> - Config `clip_seconds` trong `personalized_train_config.py` cần được tăng tương ứng khi chuyển sang mode Internal Embedding.

### 1.2. Enrollment Audio
- Trong mỗi batch huấn luyện, lấy ngẫu nhiên một đoạn âm thanh sạch (hoặc có nhiễu nhẹ) của Target Speaker làm `enrollment_wav`.
- **Độ dài Enrollment:** Khuyến nghị dùng **5s đến 10s** khi train (bài báo dùng 10s). Nếu dữ liệu ngắn hơn 3s thì lặp lại (repeat). Khi chạy đánh giá (eval), nên cố định 5s hoặc 10s để đảm bảo công bằng.
- **Nhiễu trong Enrollment:** Theo bài báo, 50% enrollment clip nên bị thêm nhiễu (SNR [0, 40]) để mô hình robust hơn.

---

## Giai đoạn 2: Nâng cấp Kiến trúc Model (`deepvqe_internal.py`)

Việc "không đổi kiến trúc" mà bài báo nhắc tới là so sánh với mô hình Personalized DeepVQE (đã có khối kết hợp Fusion), chứ không phải DeepVQE nguyên bản. So với `deepvqe.py` gốc, chúng ta **vẫn phải sửa đổi Bottleneck khá nhiều**: thêm khối Projection, 2 lớp GRU, các lớp LayerNorm (LN) và cơ chế chạy Enrollment pass.

### Thiết kế Code Đề xuất:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PersonalizedBottleneck(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.emb_dim = hidden_size
        
        # Kích thước internal embedding lấy từ GRU LN là hidden_size.
        # CHÚ Ý: Paper chiếu embedding để match với size của flattened features, sau đó concat.
        # Để sát chuẩn paper nhất (với repo này input_size = 128*9 = 1152):
        self.spk_proj1 = nn.Linear(self.emb_dim, input_size)
        self.spk_ln1 = nn.LayerNorm(input_size)
        
        # Mạng giảm chiều sau khi Concat
        self.fusion_proj = nn.Linear(input_size * 2, input_size)
        
        # (Nếu muốn giảm nhẹ tham số, có thể dùng fusion_dim=240 độc lập như một biến thể)
        self.fusion_ln = nn.LayerNorm(input_size)
        
        # Flatten LN trước khi concat theo Figure 1 của paper
        self.feature_ln = nn.LayerNorm(input_size)

        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
        # Lớp LayerNorm sau GRU (Nơi lấy Internal Embedding)
        self.gru_ln = nn.LayerNorm(hidden_size)
        
        self.fc = nn.Linear(hidden_size, input_size)

    def extract_internal_embedding(self, enrollment_features):
        """Pass 1: Rút trích Vector K từ âm thanh mẫu"""
        B = enrollment_features.size(0)
        # CHUẨN THEO PAPER: Khởi tạo speaker embedding bằng vector 0 và cho đi qua chung 1 đường chiếu (proj)
        zero_spk = torch.zeros(B, self.emb_dim, device=enrollment_features.device, dtype=enrollment_features.dtype)
        
        k = self.spk_ln1(F.elu(self.spk_proj1(zero_spk)))
        k = k.unsqueeze(1).expand(-1, enrollment_features.shape[1], -1)
        
        enrollment_features = self.feature_ln(enrollment_features)
        x = torch.cat([enrollment_features, k], dim=-1)
        x = self.fusion_ln(F.elu(self.fusion_proj(x)))
        
        y, _ = self.gru1(x)
        y, _ = self.gru2(y)
        y = self.gru_ln(y)
        
        # Average pooling theo trục thời gian (T)
        spk_emb = torch.mean(y, dim=1) # (B, hidden_size)
        return spk_emb

    def forward(self, mixture_features, spk_emb=None):
        """Pass 2: Xử lý âm thanh nhiễu dựa vào Vector K"""
        if spk_emb is None:
            B = mixture_features.size(0)
            spk_emb = torch.zeros(B, self.emb_dim, device=mixture_features.device, dtype=mixture_features.dtype)
            
        k = self.spk_ln1(F.elu(self.spk_proj1(spk_emb)))
        k = k.unsqueeze(1).expand(-1, mixture_features.shape[1], -1) 
        
        mixture_features = self.feature_ln(mixture_features)
        x = torch.cat([mixture_features, k], dim=-1)
        x = self.fusion_ln(F.elu(self.fusion_proj(x)))

        y, _ = self.gru1(x)
        y, _ = self.gru2(y)
        y = self.gru_ln(y)
        
        out = self.fc(y)
        return out

class PersonalizedDeepVQE_Internal(nn.Module):
    def __init__(self):
        super().__init__()
        # Các khối Encoder, Decoder...
        # Lưu ý: GRU hidden size 256 theo PVQE-S trong paper, hoặc 576 theo repo gốc (64*9).
        # Nếu giữ hidden_size=576 thì vẫn map được old.bottle.gru -> new.gru1 và old.bottle.fc -> new.fc; nếu đổi 256 thì không map trực tiếp được.
        self.bottle = PersonalizedBottleneck(input_size=128*9, hidden_size=576)

    # Cần tạo hàm helper encode vì decoder yêu cầu skip connections (en_x0..en_x5)
    def encode(self, stft):
        # Chạy qua 5 block encoder...
        # return en_x5, [en_x0, en_x1, ...]
        pass

    def forward(self, mixture_stft, enrollment_stft=None, spk_emb=None):
        # 1. Đi qua Encoder cho Mixture
        mix_feat, mix_skips = self.encode(mixture_stft)
        mix_feat_flat = rearrange(mix_feat, 'b c t f -> b t (c f)')
        
        # 2. Rút trích Vector K (hoặc dùng cache lúc Inference)
        if spk_emb is None:
            if enrollment_stft is None:
                raise ValueError("Cần truyền enrollment_stft nếu chưa có spk_emb cache")
            
            # GHI CHÚ: Enrollment chỉ cần output cuối (en_x5) để rút trích embedding.
            # Bỏ qua skip connections của enrollment.
            enroll_feat, _ = self.encode(enrollment_stft)
            enroll_feat_flat = rearrange(enroll_feat, 'b c t f -> b t (c f)')
            
            spk_emb = self.bottle.extract_internal_embedding(enroll_feat_flat)
        
        # 3. Lọc tiếng 
        enh_feat = self.bottle(mix_feat_flat, spk_emb)
        
        # 4. Reshape lại cho Decoder sử dụng kích thước gốc của mix_feat
        B, C, T, Freq = mix_feat.shape
        enh_feat = rearrange(enh_feat, 'b t (c f) -> b c t f', c=C, f=Freq)
        
        # 5. Đi qua Decoder kèm skip connections (mix_skips)...
        # return x_enh
```

> [!IMPORTANT]
> **Ghi chú Triển khai (Implementation Note):** 
> - **Thiết lập Config Train riêng:** Trong `personalized_train_config.py` hiện đang chạy theo ECAPA (clip_seconds=4.0, module=deepvqe_personalized). Cần tạo config riêng cho Internal mode: `module=deepvqe_internal`, sử dụng luồng nạp `enrollment STFT` thay vì load spk_emb, tăng clip length lên 10-20s, giảm batch size và kết hợp gradient accumulation.
> - **Kiến trúc Flatten LN:** Paper (Figure 1) mô tả có một LayerNorm ngay sau Flatten trước khi concat. Đoạn code mẫu ở trên đã tích hợp sẵn `feature_ln` để bám sát chuẩn thiết kế này.
> - **Lúc Train:** KHÔNG bọc pass xử lý Enrollment bằng `torch.no_grad()`. Khối Encoder và Bottleneck cần được cập nhật gradient end-to-end.
> - **Lúc Inference:** Nên trích xuất `spk_emb` một lần duy nhất cho mỗi user và cache lại trong RAM/Session. Ở các bước xử lý audio stream tiếp theo, chỉ cần tái sử dụng `spk_emb` đó để không phải chạy lại hàm enrollment.

---

## Giai đoạn 2.5: Chiến lược Warm-start và Phase Training

Bài báo gốc train **from scratch end-to-end** với dữ liệu rất lớn (DNS Challenge). Nếu dataset của bạn nhỏ hơn đáng kể, nên áp dụng chiến lược warm-start:

1. **Load Pretrained Encoder/Decoder:** Import trọng số từ checkpoint DeepVQE gốc (`deepvqe.py`). Với khối Bottleneck mới, mặc dù thêm fusion path, nhưng nếu giữ `hidden_size=576`, ta có thể thử **warm-start map trọng số cũ**: load `old.bottle.gru -> new.bottle.gru1` và `old.bottle.fc -> new.bottle.fc`, đồng thời khởi tạo mạng fusion gần với no-op (nhân dạng gần identity) để giảm "sốc" khi fine-tune.
2. **Freeze Encoder giai đoạn đầu (5–10 epoch):** Train tập trung khối `PersonalizedBottleneck` mới để hội tụ projection/fusion mà không phá encoder đã pretrained.
3. **Unfreeze dần:** Sau khi loss ổn định, mở dần encoder (từ block 5 ngược lên block 1) với learning rate nhỏ hơn (ví dụ: LR encoder = 0.1 × LR bottleneck).
4. **Nếu train from scratch:** Vẫn nên dùng Curriculum Learning — epoch đầu mix SNR/SIR cao (10–20dB), sau đó giảm dần.

> [!TIP]
> Nếu repo đã có checkpoint DeepVQE non-personalized chất lượng tốt, đây là cách **tiết kiệm thời gian nhất** để bootstrap mô hình Internal Embedding.

---

## Giai đoạn 3: Hàm Mất Mát (Loss Function) và Chỉ số Đánh giá

### 3.1. Hàm Mất Mát
> [!NOTE]
> **Khác biệt với Paper:** Bài báo gốc dùng `Complex Compressed MSE loss` (với exponent 0.3 và beta 0.7).
> Tuy nhiên, với implementation thực dụng của repo hiện tại, bạn hoàn toàn có thể tiếp tục dùng Loss End-to-End thay thế:
> $$ \text{Loss} = \lambda_1 \times (-\text{SI-SDR}(x_{enh}, x_{target})) + \lambda_2 \times \text{L1}(|X_{enh}|, |X_{target}|) $$

- **Trường hợp Negative Case (Absent Speaker):** Vẫn áp dụng Waveform Energy Loss (L1/MSE) để ép mạng triệt tiêu toàn bộ âm thanh nếu danh tính không khớp. Lưu ý tỷ lệ Negative Case 10% là **extension thực dụng của dự án**, không phải recipe gốc của paper. Giai đoạn đầu nên train positive/background-speech trước cho hội tụ, sau đó mới bật absent-speaker loss để tránh model học triệt tiêu quá mạnh.

### 3.2. Metrics Đánh Giá
*Lưu ý Scope:* Scope hiện tại của repo chủ yếu nhận `mixture_stft` dạng (B,F,T,2) cho bài toán NS/TSE. Các metrics đánh giá Echo (AECMOS/ERLE) chỉ phù hợp nếu sau này dự án bổ sung thêm far-end path.
Để đánh giá mô hình trong scope NS/TSE, cần chú ý các metrics:
- **TSOS (Target Speaker Over-Suppression):** Đo lường mức độ xóa nhầm giọng người nói đích (rất quan trọng).
- **BAK SUPPR (Background Suppression) / Energy Reduction:** Dùng để đo lường trong tình huống chỉ có background speech hoặc wrong-speaker.
- **DNSMOS P.835:** Đo chất lượng tổng thể (OVRL, SIG, BAK) để so với paper.
- **PESQ / SI-SDR:** Dành cho các bộ test synthetic/reference-based (bộ test có ground truth).

---

## Giai đoạn 4: Chiến lược Triển khai và Tránh Xung Đột

Hiện tại repo đang có sẵn `deepvqe_personalized.py` chạy theo thiết kế ECAPA-TDNN (sử dụng mạng FiLM và yêu cầu truyền `spk_emb` tĩnh). Để tránh đụng độ tên class `PersonalizedBottleneck` và tránh làm rối luồng xử lý, **Cách ít rủi ro và dễ bảo trì nhất là tạo một file mã nguồn mới hoàn toàn:**

1. **Tạo file `deepvqe_internal.py`**: Chứa toàn bộ code của khối `PersonalizedBottleneck` và `PersonalizedDeepVQE_Internal` được đề xuất ở trên.
2. **Tái sử dụng các khối nguyên bản**: Bạn có thể import lại các khối `FE`, `EncoderBlock`, `DecoderBlock`, `CCM` từ `deepvqe.py` gốc vào file mới này.
3. **Quản lý cấu hình (Config)**: Để dự án linh hoạt chuyển đổi giữa 2 phương pháp, hãy thiết lập tham số trong file config (ví dụ `personalized_train_config.py`) để hệ thống biết nên nạp model từ file nào:
```python
# personalized_train_config.py
"model": {
    "module": "deepvqe_internal",                  # Hoặc "deepvqe_personalized" cho ECAPA
    "class_name": "PersonalizedDeepVQE_Internal", 
    "internal_hidden_size": 576,                   # Hoặc 256 để gần với bài báo PVQE-S hơn
}
```

### So sánh Tối ưu (Internal vs ECAPA)

| Tiêu chí | Option 1: ECAPA-TDNN | Option 2: Internal Embedding |
| :--- | :--- | :--- |
| **Trạng thái Code** | Đã implement (`deepvqe_personalized.py`) | Prototype implemented (`deepvqe_internal.py`), chưa tích hợp pipeline train |
| **Phụ thuộc ngoài** | SpeechBrain, torchaudio | Không cần thêm dependency |
| **Dung lượng Model** | Thêm ~15MB-30MB cho ECAPA | Tăng vừa phải, cần benchmark params/RTF |
| **Bộ nhớ lúc Train** | Thấp nếu cache embedding; cao nếu bật Speaker Consistency Loss | Cao hơn do chạy Encoder 2 lần (enrollment + mixture), cần clip dài (≥10s) |
| **Triển khai (Deployment)** | Cần ship 2 mô hình, luồng dữ liệu riêng | Đơn giản hơn, 1 mô hình duy nhất |
| **Chất lượng Embedding** | Pretrained trên VoxCeleb, đã chứng minh robust | Phụ thuộc chất lượng training data của bạn |
| **Ứng dụng phù hợp** | Khi cần embedding mạnh, dataset nhỏ | On-device, real-time, khi không muốn dependency ngoài |
