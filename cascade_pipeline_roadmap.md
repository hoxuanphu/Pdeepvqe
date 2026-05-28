# Lộ trình Phương án Ghép nối (Cascade Pipeline) cho Target Speech Extraction

Tài liệu này cung cấp lộ trình chi tiết để xây dựng hệ thống **Target Speech Extraction (TSE)** theo phương pháp **Ghép nối (Cascade)**. Phương pháp này hoàn toàn **KHÔNG CẦN TRAINING**, độc lập 100% với phương án `pDeepVQE` (trong file `deepvqe_personalized.py`), và giữ nguyên vẹn mô hình khử nhiễu `DeepVQE` gốc (trong file `deepvqe.py`).

---

## 1. Mục tiêu (Goal)
- Đạt được khả năng tách giọng người nói mục tiêu (Target Speaker) từ môi trường có nhiều người nói và tiếng ồn.
- Triển khai tức thì (Zero-shot / No-training).
- Tận dụng sức mạnh của các mô hình SOTA (State-of-the-Art) mã nguồn mở.
- Tái sử dụng `DeepVQE` gốc làm màng lọc nhiễu môi trường (Denoise) cực mạnh ở chốt chặn cuối cùng.

---

## 2. Kiến trúc Hệ thống (3 Bước Tuần tự)

Hệ thống sẽ hoạt động như một băng chuyền nhà máy đi qua 3 trạm kiểm định:

### Bước 1: Tách nguồn âm Mù (Blind Speech Separation)
- **Mục đích:** Tách đoạn âm thanh hỗn hợp (Mixture) thành số lượng nguồn âm cố định mà mô hình hỗ trợ (thường là 2) đoạn âm thanh riêng biệt. Hệ thống lúc này chưa cần biết ai là ai, chỉ cần tách sạch các giọng nói đè lên nhau.
- **Công cụ khuyến nghị:** Mô hình **SepFormer** (Ví dụ: `speechbrain/sepformer-whamr16k` - hỗ trợ cả tiếng ồn và vang phòng).
- **Input:** `Mixture_Audio` (Bắt buộc: Mono, 16 kHz - Áp dụng chung cho cả SepFormer, ECAPA và DeepVQE).
- **Output:** `S1_Audio`, `S2_Audio` (Giả sử phân tách ra 2 nguồn, không nhất thiết là 2 người).

### Bước 2: Nhận diện & Khớp giọng nói (Speaker Matching)
- **Mục đích:** Tìm xem trong `S1` và `S2`, đâu mới là giọng nói của người mà chúng ta cần tìm.
- **Công cụ khuyến nghị:** Mô hình **ECAPA-TDNN** (`speechbrain/spkrec-ecapa-voxceleb`).
- **Input:**
  1. `Enrollment_Audio` (Đoạn ghi âm mẫu 3-5 giây của mục tiêu - Bắt buộc: Mono, 16 kHz).
  2. `S1_Audio`, `S2_Audio` (Từ Bước 1).
- **Thuật toán:**
  1. Trích xuất Vector đặc trưng: $Emb_{target}$, $Emb_{S1}$, $Emb_{S2}$.
  2. Tính độ tương đồng Cosine (Cosine Similarity):
     - $Score_1 = \cos(Emb_{target}, Emb_{S1})$
     - $Score_2 = \cos(Emb_{target}, Emb_{S2})$
  3. Chọn luồng chiến thắng: $S_{selected} = S_1$ nếu $Score_1 > Score_2$ ngược lại là $S_2$.

### Bước 3: Đánh bóng & Khử nhiễu cuối (Final Denoising)
- **Mục đích:** Mô hình tách nguồn (SepFormer) thường chỉ giỏi tách giọng người với người, nhưng hay bỏ sót tiếng ồn nền (quạt, gió, đường phố) hoặc sinh ra nhiễu cơ học (artifacts). Trạm cuối này dùng để "giặt sạch" file audio.
- **Công cụ:** Mô hình **DeepVQE** gốc của bạn (import trực tiếp từ `deepvqe.py`).
- **Input:** `S_{selected}` (chuyển sang miền STFT như cách DeepVQE yêu cầu).
- **Output:** `Final_Target_Audio`.

---

## 3. Triển khai Code Độc Lập

Bạn không cần chạm vào `deepvqe.py` hay `deepvqe_personalized.py`. Thay vào đó, tạo một script Inference hoàn toàn mới (ví dụ: `infer_cascade.py`).

Dưới đây là mã giả (Pseudo-code) minh họa luồng đi:

```python
import torch
import torchaudio
import torch.nn.functional as F

# Xử lý tương thích ngược phiên bản SpeechBrain
try:
    from speechbrain.inference.separation import SepformerSeparation as separator
    from speechbrain.inference.speaker import EncoderClassifier as speaker_recognizer
except ImportError:
    from speechbrain.pretrained import SepformerSeparation as separator
    from speechbrain.pretrained import EncoderClassifier as speaker_recognizer

# 1. TÁI SỬ DỤNG DEEPVQE GỐC (Hoàn toàn không sửa code deepvqe.py)
from deepvqe import DeepVQE

class CascadeTSEPipeline:
    def __init__(self, device=None, threshold=0.35, margin_threshold=0.03):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.threshold = threshold
        self.margin_threshold = margin_threshold
        
        # Load Bước 1: SepFormer
        self.sep = separator.from_hparams(
            source="speechbrain/sepformer-whamr16k", 
            run_opts={"device": device}
        )
        
        # Load Bước 2: ECAPA-TDNN
        self.ecapa = speaker_recognizer.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            run_opts={"device": device}
        )
        
        # Load Bước 3: DeepVQE nguyên bản
        self.deepvqe = DeepVQE().to(device)
        ckpt = torch.load("path_to_pretrained_deepvqe.pth", map_location=device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.deepvqe.load_state_dict(state_dict)
        self.deepvqe.eval()

    def process(self, mixture_wav, enrollment_wav):
        with torch.no_grad():
            # ---- BƯỚC 1: TÁCH NGUỒN ----
            # est_sources: (Batch, Thời gian, Số nguồn) -> ví dụ (1, T, 2)
            est_sources = self.sep.separate_batch(mixture_wav)
            s1 = est_sources[:, :, 0]
            s2 = est_sources[:, :, 1]
            
            # ---- BƯỚC 2: KHỚP GIỌNG ----
            # [!] LƯU Ý: Đây là mã giả tối giản. Trong thực tế, KHÔNG nên encode toàn bộ s1/s2.
            # Cần chạy hàm vad_and_window_score() để cắt bỏ silence và lấy mean/max của các cửa sổ.
            emb_target = self.ecapa.encode_batch(enrollment_wav).squeeze(1)
            emb_s1 = self.ecapa.encode_batch(s1).squeeze(1)
            emb_s2 = self.ecapa.encode_batch(s2).squeeze(1)
            
            # Normalize và tính Cosine Similarity
            emb_target = F.normalize(emb_target, dim=-1)
            score_s1 = F.cosine_similarity(emb_target, F.normalize(emb_s1, dim=-1))
            score_s2 = F.cosine_similarity(emb_target, F.normalize(emb_s2, dim=-1))
            
            # [!] LƯU Ý BATCH SIZE: Lệnh mean().item() dưới đây chỉ an toàn khi Batch Size = 1.
            # Nếu Inference Batch > 1, cần xử lý logic so sánh cho từng phần tử trong batch.
            s1_val = score_s1.mean().item()
            s2_val = score_s2.mean().item()
            
            top_score = max(s1_val, s2_val)
            margin = abs(s1_val - s2_val)
            
            # Xử lý Negative/Uncertain Case
            if top_score < self.threshold or margin < self.margin_threshold:
                # Hệ thống không chắc chắn hoặc người cần tìm không có mặt
                # Trả về khoảng lặng. [!] LƯU Ý: code thực tế cần đảm bảo 
                # length/sample rate của tensor zeros này khớp với output pipeline cuối cùng
                return torch.zeros_like(mixture_wav)
                
            # Lấy luồng chiến thắng
            if s1_val > s2_val:
                s_selected = s1
            else:
                s_selected = s2
                
            # ---- BƯỚC 3: DỌN NHIỄU BẰNG DEEPVQE GỐC ----
            # Chuyển đổi wav -> STFT format mà DeepVQE yêu cầu (như trong infer.py)
            # pseudo function: compute_stft_for_deepvqe()
            s_selected_stft = compute_stft_for_deepvqe(s_selected) 
            
            final_stft = self.deepvqe(s_selected_stft)
            
            # pseudo function: compute_istft_from_deepvqe()
            final_wav = compute_istft_from_deepvqe(final_stft)
            
            return final_wav
```

---

## 4. Quản trị Rủi ro & Tối ưu (Đặc biệt quan trọng)

> [!WARNING]
> **Giới hạn bộ nhớ (VRAM OOM):**
> Pipeline này nhồi cả 3 mô hình khổng lồ (SepFormer, ECAPA, DeepVQE) vào chung một GPU.
> - **Giải pháp:** Load tuần tự hoặc đẩy SepFormer lên CPU nếu chạy off-line.

> [!CAUTION]
> **Lỗi dây chuyền (Cascading Errors) - Nhược điểm cốt lõi của non-end-to-end:**
> Lỗi ở SepFormer sẽ truyền thẳng xuống ECAPA và DeepVQE. Nếu SepFormer tách sai (làm méo tiếng, vỡ tần số), DeepVQE tuyệt đối **không thể cứu được** (speaker identity). DeepVQE chỉ khử nhiễu, nó không có khả năng khôi phục lại giọng nói đã bị phá hủy từ bước trước.

> [!IMPORTANT]
> **Vấn đề số lượng người nói (Fixed Speaker Count):**
> Các mô hình như `sepformer-whamr16k` thường bị "hardcode" để tách đúng 2 nguồn. 
> - **Nếu có 1 người:** Nó có thể chia đôi giọng của người đó ra 2 luồng, hoặc 1 luồng xả rác.
> - **Nếu có 3 người trở lên:** Ít nhất 2 giọng sẽ bị nhồi chung vào 1 luồng, gây nhiễu loạn cho ECAPA ở bước đối chiếu. 
> Đây là lý do chính khiến phương án huấn luyện pDeepVQE (End-to-End) vượt trội hơn trong thực tế.

> [!TIP]
> **Tối ưu So khớp Giọng nói (Speaker Matching):**
> KHÔNG nên đưa toàn bộ đoạn audio `S1` và `S2` vào ECAPA để encode, đặc biệt nếu có nhiều khoảng lặng (silence) hoặc artifacts.
> - **Kỹ thuật tối ưu:** Bắt buộc dùng **VAD** để cắt bỏ silence. Sau đó chia đoạn có speech thành các cửa sổ nhỏ (windows). Tính embedding cho từng cửa sổ rồi lấy `mean` hoặc `max score`. Điều này giúp vector ổn định hơn rất nhiều.

> [!TIP]
> **Thiết lập Ngưỡng (Threshold & Margin):**
> Ngưỡng 0.35 chỉ là con số ước lượng. Cần **Calibrate (hiệu chuẩn)** lại bằng Validation set nội bộ.
> Thay vì chỉ dùng Absolute Threshold (Ngưỡng tuyệt đối), hãy dùng thêm **Margin**: `margin = score_top_1 - score_top_2`.
> - Nếu `score_top_1` quá thấp, HOẶC `margin` quá nhỏ (ví dụ: `0.45` và `0.42`) $\rightarrow$ Cảnh báo "Không chắc chắn" (Uncertainty) hoặc loại bỏ, vì rất có thể hệ thống đang phân vân giữa 2 giọng bị mix lẫn.

> [!WARNING]
> **Phản ứng phụ của DeepVQE (Tác dụng ngược):**
> DeepVQE đặt sau SepFormer có thể "vừa giúp vừa hại". DeepVQE vốn được train để khử nhiễu tự nhiên (môi trường). Các "nhiễu" do SepFormer sinh ra (separation artifacts) là nhiễu cơ học phi tuyến tính. DeepVQE có thể hiểu lầm các artifacts này và vô tình bóp méo âm sắc (timbre) của người nói.
> - **Giải pháp:** Luôn **Benchmark 2 modes** độc lập để so sánh: `SepFormer + ECAPA` (Không dùng DeepVQE) và `SepFormer + ECAPA + DeepVQE`. Đánh giá xem có thực sự cần DeepVQE ở chốt cuối không.

---

## 5. Kế hoạch song hành

- Phương án này có thể code và **chạy thử ngay hôm nay** (chỉ tốn khoảng 1-2 tiếng viết file `infer_cascade.py`). Nó đóng vai trò là một **Baseline mạnh** để đo lường.
- Phương án `pDeepVQE` (trong `deepvqe_personalized.py`) cứ tiến hành training song song. Sau khi train xong, bạn mang kết quả của `pDeepVQE` ra so sánh với phương án Cascade này. Mô hình nào chạy nhanh hơn, tốn ít VRAM hơn và tách sạch hơn thì chọn triển khai thương mại. Thường thì pDeepVQE sẽ thắng ở tốc độ và độ nhẹ.

---

## 6. Tiêu chí Đánh giá (Evaluation Metrics)

Để biết Cascade thực sự tốt đến đâu, cần thiết lập quy trình benchmark với 2 mode:
- **Mode 1:** SepFormer + ECAPA (Chỉ tách giọng)
- **Mode 2:** SepFormer + ECAPA + DeepVQE (Tách giọng + Khử nhiễu)

**Tiêu chí bắt buộc:**
- **Chất lượng Tách / Khử nhiễu:** `SI-SDRi` (đo sự cải thiện), `PESQ` (chất lượng giọng cảm nhận), `STOI` (độ dễ hiểu).
- **Speaker Identity:** `Cosine Similarity` (đo xem giọng xuất ra có bị méo tiếng/sai người so với mẫu hay không).
- **Hiệu năng:** `Latency` (độ trễ miligiây), `VRAM footprint` (lượng bộ nhớ chiếm dụng).
