# Đánh Giá Chuyên Gia: DeepVQE Optimization Roadmap (Ablation-Driven)

**Tài liệu gốc:** [deepvqe_optimization_roadmap.md](file:///d:/AI20K/deepvqe/ablation/docs/deepvqe_optimization_roadmap.md)  
**Ngày đánh giá:** 2026-06-14  
**Vai trò:** Chuyên gia trung lập — ML Systems & Speech Enhancement  

---

## 1. Đánh Giá Tổng Quan

Roadmap này là một tài liệu ablation study **trên mức trung bình** cho một dự án cá nhân/nhóm nhỏ. Nó thể hiện tư duy thực nghiệm có kỷ luật: tách biến thể, định nghĩa tiêu chí pass/fail, phân pha rủi ro, và có dependency graph giữa các thí nghiệm. Đây là những yếu tố mà nhiều dự án tối ưu model trong thực tế thường bỏ qua.

Tuy nhiên, khi xét dưới tiêu chuẩn của một experimental protocol có thể publish hoặc đưa vào production pipeline, roadmap vẫn có **những lỗ hổng phương pháp luận đáng kể** cần được xử lý trước khi kết quả ablation có giá trị khoa học đầy đủ.

### Điểm mạnh cốt lõi

| Khía cạnh | Đánh giá |
|---|---|
| Tách architecture ablation và loss ablation | ✅ Đúng chuẩn — tránh confounding |
| Yêu cầu train from scratch cho combined runs | ✅ Nghiêm ngặt — tránh fine-tune bias |
| Phân loại rủi ro theo Phase/Stage | ✅ Thực tế — không ôm đồm |
| Causal constraint cho ECA | ✅ Quan trọng cho streaming — ít roadmap nào đề cập |
| Reproducibility metadata bắt buộc | ✅ Chuyên nghiệp |
| Dependency graph với pruning rules | ✅ Tiết kiệm compute khi hypothesis fail |

### Điểm yếu cốt lõi

| Khía cạnh | Mức độ |
|---|---|
| Thiếu power analysis / sample size justification | 🔴 Nghiêm trọng |
| Không có learning rate warmup / schedule chuẩn hóa | 🟡 Trung bình |
| Eval protocol thiếu chi tiết về noise conditions | 🟡 Trung bình |
| Không đề cập data augmentation strategy | 🟡 Trung bình |
| Thiếu perceptual listening test hoặc proxy | 🟠 Đáng lưu ý |

---

## 2. Phân Tích Chất Lượng Từng Giả Thuyết

### 2.1 PReLU (B1a/B1b) — 🟡 Giả thuyết cần kiểm chứng, đang chờ kết quả mới

**Giả thuyết gốc:** *"PReLU có thể học tốt hơn ở vùng âm"*

**Trạng thái:** B1a đang được train lại. Chưa có đánh giá hợp lệ.

**Nhận xét:**
- Giả thuyết này hơi chung chung. ELU đã xử lý vùng âm tốt (smooth saturation), và lợi thế lý thuyết của PReLU chủ yếu nằm ở việc tránh "dying neurons" — vốn là vấn đề của ReLU, **không phải ELU**.
- Tuy nhiên, PReLU cho phép model tự học slope ở vùng âm, có thể hữu ích nếu ELU saturation đang giới hạn gradient flow ở một số layer. Đây là giả thuyết cần dữ liệu để confirm hoặc reject.
- **Rủi ro tiềm ẩn cần theo dõi khi có kết quả:** BatchNorm + PReLU interaction có thể gây training instability. Nên log loss curve, gradient norm của PReLU parameter, và learned α value.

> [!NOTE]
> Khi B1a train xong, cần đánh giá cẩn thận trước khi kết luận. Nếu PReLU shared cho kết quả marginal, nên kiểm tra:
> 1. Loss curve có smooth hay oscillate?
> 2. Learned α value so với α = 1 (identity) — nếu α ≈ 1 thì PReLU không học được gì hữu ích
> 3. Gradient norm qua training để phát hiện instability

---

### 2.2 ECA-F (B2, B3) — ✅ Giả thuyết hợp lý, nhưng RTF đã cảnh báo

- Ý tưởng frequency-only pooling để giữ causality là **chính xác về mặt lý thuyết**.
- Tuy nhiên, [phase1_training_plan.md](file:///d:/AI20K/deepvqe/ablation/docs/phase1_training_plan.md) ghi nhận RTF tăng mạnh trên Tesla T4 → tạm hoãn. Đây là quyết định đúng.
- **Vấn đề chưa nêu:** ECA-F pooling trên trục F sẽ collapse thông tin spectral resolution. Với speech enhancement, thông tin tần số chính là phân biệt speech/noise. Nếu pooling quá mạnh, model có thể mất discrimination ở dải tần hẹp.

---

### 2.3 DW-Conv (C1) — ✅ Giả thuyết mạnh, deploy candidate hàng đầu

- Giảm **20.2% params** và **43.6% MACs** là con số rất tốt.
- Roadmap đã cảnh báo đúng: chỉ tính là "win" khi giảm RTF thực tế, không chỉ lý thuyết.
- **Rủi ro chưa đề cập:** DW-Conv giảm cross-channel interaction trong ResidualBlock. Nếu ResidualBlock đang đóng vai trò refine cross-channel features (ví dụ: harmonics structure), DW-Conv có thể làm mất chất lượng ở vùng tần số tương tác.

---

### 2.4 Skip-Connection Gating (B3a/B3b) — 🟡 Giả thuyết thú vị nhưng chưa đủ cơ sở

- Giả thuyết "skip truyền cả nhiễu" là hợp lý về mặt trực giác, nhưng:
  - Chưa có diagnostic nào chứng minh skip connection thực sự là bottleneck chất lượng
  - Không rõ liệu 1×1 Conv hiện tại đã đủ capacity để filter hay chưa
  - SE block thêm parameters và latency — cần đo trước khi commit

---

### 2.5 Loss Function (B4) — ✅ Giả thuyết mạnh, zero inference cost

- Asymmetric Loss để chống over-suppression và Multi-Resolution STFT Loss là **state-of-the-art practice** trong speech enhancement (đã được validate trong nhiều paper: HiFi-GAN, Demucs, CMGAN).
- Zero inference cost → không có lý do gì để không thử.
- **Thiếu sót:** Roadmap không nêu rõ trọng số giữa các thành phần loss sẽ được tune như thế nào. Nếu tune trọng số loss, cần coi đây là thêm một chiều hyperparameter search và tính vào training budget.

---

### 2.6 GRU Bottleneck Tuning (E1-E4) — ⚠️ Hợp lý nhưng rủi ro cao

- GRU hidden size 576 → 512 giảm ~20% params ở bottleneck, nhưng bottleneck là nơi model encode temporal context. Giảm quá mạnh sẽ ảnh hưởng đến khả năng track noise statistics qua thời gian.
- **Grouped GRU (E3)** là re-architecture rủi ro cao. Roadmap đã đánh giá đúng mức rủi ro.
- **Knowledge Distillation (E4)** là ý tưởng tốt nhưng cần teacher model đủ mạnh. Nếu baseline chính là teacher, KD gain sẽ rất hạn chế vì teacher và student cùng capacity class.

---

## 3. Đánh Giá Phương Pháp Luận Thực Nghiệm

### 3.1 Điểm mạnh đã có

1. **Train from scratch cho mọi variant** — Tránh confounding từ pretrained weights. Đây là best practice.
2. **Tách validation / dev_eval / final_test** — Đã sửa từ review trước. Đúng chuẩn.
3. **Guardrail metrics có ngưỡng số** (STOI ≥ baseline - 0.002, SI-SDR ≥ baseline - 0.1 dB) — Cho phép automated pass/fail.
4. **Reproducibility metadata bắt buộc** — `git_commit`, `config_hash`, `seed`, `hardware_info` — code implementation trong [ablation_config.py](file:///d:/AI20K/deepvqe/ablation/ablation_config.py) đã khớp với roadmap.

### 3.2 Lỗ hổng phương pháp luận

#### 🔴 A. Thiếu Statistical Power Analysis

Roadmap đề cập "chạy 2-3 seeds" cho best variants, nhưng:
- **Không định nghĩa** bao nhiêu seed là đủ cho mỗi stage
- **Không nêu** paired test nào sẽ dùng (paired t-test, Wilcoxon, bootstrap CI?)
- **Không nêu** significance level (α = 0.05? 0.01?)
- Với eval set nhỏ (VCTK-DEMAND ~800 utterances), variance giữa các seed có thể rất lớn. **Δ PESQ = 0.01 hoàn toàn có thể nằm trong noise**.

> [!IMPORTANT]
> Với ngưỡng "marginal" |Δ| ≤ 0.01, bạn cần ít nhất **3 seeds** cho mỗi variant ở Stage 1 để có paired comparison có ý nghĩa. Lý tưởng là 5 seeds nếu compute cho phép. Nếu không, cần dùng paired bootstrap trên eval set (≥1000 resamples) và báo cáo 95% CI.

#### 🟡 B. Training Budget Chưa Được Justify

- 80 epochs (theo phase1_training_plan) hoặc 100 epochs (theo config) — nhưng không có evidence nào cho thấy đây là đủ epochs cho convergence.
- Baseline đạt PESQ 2.856 tại epoch nào? B1a fail tại epoch nào? Nếu B1a chưa converge, kết luận "B1a fail" có thể premature.
- **Đề xuất:** Nên ghi rõ convergence criterion (ví dụ: validation loss không giảm trong 10 epochs liên tiếp) thay vì cố định epoch count.

#### 🟡 C. Eval Conditions Không Đủ Chi Tiết

Roadmap dùng VCTK-DEMAND nhưng không nêu:
- SNR levels nào trong test set? (0dB, 5dB, 10dB, 15dB, 20dB?)
- Noise types nào? Nếu test set chỉ có stationary noise, model tốt ở PESQ nhưng có thể fail ở real-world non-stationary noise.
- Phân tách kết quả theo SNR level và noise type sẽ cho insight sâu hơn nhiều so với chỉ báo cáo overall average.

#### 🟠 D. Thiếu Perceptual Evaluation

- PESQ và STOI là proxy metrics, không phải ground truth perceptual quality.
- DNSMOS là machine-learned metric, tốt hơn nhưng vẫn có bias.
- Không có bất kỳ mention nào về subjective listening test, MUSHRA, hoặc AB preference test.
- **Với dự án cá nhân/nhóm nhỏ, điều này chấp nhận được**, nhưng nên acknowledge rõ ràng đây là limitation.

---

## 4. Đánh Giá Code Infrastructure

Dựa trên review code thực tế trong [deepvqe_ablation.py](file:///d:/AI20K/deepvqe/ablation/deepvqe_ablation.py) và [ablation_config.py](file:///d:/AI20K/deepvqe/ablation/ablation_config.py):

### Điểm mạnh
- **Config-driven design:** `ABLATION_CONFIGS` dict + `get_ablation_config()` cho phép tạo variant bằng config, không cần sửa model code → đúng chuẩn ablation.
- **Backward compatibility:** `_normalize_model_config()` xử lý legacy keys → robust khi config evolve.
- **Streaming parity:** Mỗi module có cả offline (`DeepVQE_Ablation`) và streaming variant (`StreamDeepVQE_Ablation`) → cho phép kiểm tra streaming RTF ngay.
- **`from_offline()` class method:** Convert offline → streaming weights → tiện cho deploy pipeline.

### Điểm cần cải thiện

| Vấn đề | Chi tiết |
|---|---|
| B3a/B3b chưa có trong code | Roadmap liệt kê Skip-Gating (SE/ECA ở skip-connection), nhưng `ABLATION_CONFIGS` không có `B3a`, `B3b`. Chỉ có `B3` (ECA-F ở main block). Roadmap và code **không đồng bộ**. |
| C2a chưa có trong code | `C2a` (DW-Subpixel Decoder) được nêu trong roadmap nhưng `ABLATION_CONFIGS` chỉ có `C2` generic. |
| B4 (Loss ablation) chưa có config | Loss config trong `BASE_TRAIN_CONFIG` chỉ có `lamda_ri`, `lamda_mag`, `compress_factor`. Chưa có Asymmetric Loss hay Multi-Res STFT Loss. |
| `augment: False` | Data augmentation bị tắt trong base config. Nếu tất cả variants đều train không augment, kết quả sẽ đồng nhất nhưng có thể không phản ánh performance trên data thực tế. |

> [!NOTE]
> Sự không đồng bộ giữa roadmap doc và actual code config là rủi ro thường gặp trong dự án ablation. Nên tạo một script tự động kiểm tra mọi variant ID trong roadmap đều có config tương ứng trong code.

---

## 5. So Sánh Với Industry Practice

| Tiêu chuẩn | Roadmap hiện tại | Best practice (Meta Denoiser / Google SRNN / MS DCCRN) |
|---|---|---|
| Ablation isolation | ✅ Một biến một lúc | ✅ Tương đương |
| Training budget lock | ✅ Có | ✅ Có, thường 200-500k steps |
| Multi-seed evaluation | ⚠️ "2-3 seeds cho best" | ✅ 3-5 seeds cho mọi variant |
| Statistical test | ⚠️ Chỉ mention, chưa chọn test | ✅ Paired t-test hoặc bootstrap |
| SNR-stratified evaluation | ❌ Không đề cập | ✅ Luôn báo cáo per-SNR |
| Noise-type breakdown | ❌ Không đề cập | ✅ Babble, Factory, Engine, etc. |
| Subjective evaluation | ❌ Không | ✅ MUSHRA hoặc AB test cho final model |
| Compute efficiency metric | ✅ RTF + ONNX latency | ✅ Tương đương, thường thêm memory footprint |
| Automated CI/CD ablation | ❌ Manual | ✅ Automated pipeline |
| Effect size reporting | ❌ Chỉ raw Δ | ✅ Cohen's d hoặc % relative |

---

## 6. Trạng Thái Thực Nghiệm Hiện Tại

**B1a (PReLU shared)** đang được train lại. Chưa có đánh giá chất lượng hợp lệ.

> [!IMPORTANT]
> File `phase1_training_plan.md` có ghi kết quả B1a cũ, nhưng đó là từ lần train trước và **không nên được dùng để ra quyết định**. Cần chờ kết quả từ lần train mới hoàn tất.

### Checklist khi B1a train xong

Khi B1a hoàn tất, cần đánh giá theo đúng protocol đã định trong roadmap:

1. **Quality metrics:** So sánh PESQ, STOI, SI-SDR với baseline đã khóa
2. **Guardrail check:** STOI ≥ baseline - 0.002, SI-SDR ≥ baseline - 0.1 dB
3. **Compute check:** Params/MACs gần như không đổi (chỉ +20 params)
4. **RTF check:** Đo trên cùng hardware với baseline
5. **Convergence check:** Ghi lại best epoch, final loss, early stopping trigger
6. **PReLU diagnostic:** Log learned α value, kiểm tra có hội tụ ổn định không

### Tác động lên Roadmap (tùy kết quả)

- **Nếu B1a pass:** Tiến hành B1b để so sánh shared vs per-channel, rồi chọn winner cho C3/C4
- **Nếu B1a fail marginal** (|Δ| nhỏ, trong vùng nhiễu): Vẫn nên train B1b, và giữ C3/C4 trong roadmap
- **Nếu B1a fail rõ rệt:** Cần root cause analysis trước khi quyết định có train B1b không — tránh lãng phí compute nếu bản chất PReLU không phù hợp với architecture này

---

## 7. Các Rủi Ro Chưa Được Đề Cập Trong Roadmap

### R1: Overfitting trên Eval Set nhỏ
Nếu dev_eval set chỉ có vài trăm utterances, việc so sánh 10+ variants trên cùng một set sẽ dẫn đến **multiple comparison problem**. Variant "tốt nhất" có thể chỉ là may mắn thống kê.

**Mitigation:** Bonferroni correction hoặc Holm-Bonferroni cho số lượng variants được so sánh.

### R2: Confirmation Bias trong Staged Design
Staged design (Phase 1 → Phase 2) có ưu điểm tiết kiệm compute, nhưng cũng tạo confirmation bias: nếu Phase 1 cho kết quả marginal, có xu hướng "kích hoạt Phase 2" để justify thêm thử nghiệm thay vì dừng lại.

**Mitigation:** Roadmap đã có trigger rules — đây là điểm tốt. Nên thêm rule: *"Nếu best Phase 1 variant chỉ hơn baseline < 0.02 PESQ, declare Phase 1 inconclusive và review toàn bộ approach trước khi Phase 2."*

### R3: Training trên Colab/Kaggle
Dựa trên các notebook file names (`train_base_deepvqe_colab.ipynb`, `train_ablation_B1_kaggle.ipynb`), training đang chạy trên free/shared GPU platforms. Điều này tạo ra:
- **Variance do hardware:** GPU khác nhau giữa các session
- **Session timeout:** Training có thể bị gián đoạn
- **Non-reproducibility:** Khó đảm bảo cùng hardware cho mọi variant

> [!CAUTION]
> Nếu các variant được train trên các GPU khác nhau (T4 vs P100 vs A100), sự khác biệt RTF giữa các variant sẽ **vô nghĩa**. RTF phải được đo trên cùng một hardware, cùng một session, cùng workload.

### R4: Thiếu Ablation cho Data Pipeline
Roadmap focus 100% vào model architecture và loss, nhưng không có variant nào ablate:
- Data augmentation on vs off
- Different SNR mixing strategies
- Clip length (3s — có đủ?)
- Sample rate impact

Trong nhiều bài báo speech enhancement, data pipeline changes tạo ra impact lớn hơn architecture changes.

---

## 8. Đề Xuất Ưu Tiên (Xếp Theo Mức Độ Quan Trọng)

### 🔴 Phải làm trước khi tiếp tục ablation

1. **Lock evaluation protocol chi tiết:** Định nghĩa rõ SNR conditions, noise types, và reporting format (per-SNR breakdown, overall average). Ghi vào config, không phải doc.

2. **Thêm multi-seed requirement cho Stage 1:** Tối thiểu 3 seeds. Nếu compute hạn chế, ít nhất phải có paired bootstrap trên eval set.

3. **Đánh giá B1a đang train:** Khi B1a train xong, đánh giá theo đúng protocol (quality + guardrail + compute + convergence). Nếu fail, cần root cause analysis trước khi train B1b.

4. **Đồng bộ roadmap và code:** B3a, B3b, C2a chưa có trong `ABLATION_CONFIGS`. Hoặc thêm vào code, hoặc sửa roadmap.

### 🟡 Nên làm để tăng giá trị khoa học

5. **Thêm effect size reporting:** Báo cáo Cohen's d hoặc relative % thay vì chỉ raw Δ.

6. **Thêm convergence analysis:** Ghi best epoch, final epoch loss, và early stopping trigger cho mỗi variant.

7. **Tạo bảng cost/benefit tracker:** Mỗi variant cần ghi: expected compute change, actual compute change, quality change, implementation effort. Roadmap review trước đã đề xuất — nên thực hiện.

8. **RTF benchmark chuẩn hóa:** Tất cả RTF phải được đo trên cùng GPU, cùng batch size, cùng input length, warm-up 50 iterations trước khi đo.

### 🟢 Nice-to-have

9. **Ablation dashboard:** Tự động hóa so sánh variants bằng script (đã có `collect_ablation_results.py` — nên extend thêm visualization).

10. **Perceptual spot-check:** Nghe thử 10-20 samples từ mỗi variant trên các noise conditions khó (low SNR, non-stationary). Không cần formal MUSHRA, nhưng sanity check bằng tai sẽ phát hiện artifacts mà metrics bỏ sót.

---

## 9. Kết Luận

| Tiêu chí | Điểm (1-5) | Ghi chú |
|---|:---:|---|
| Tính hệ thống | **4/5** | Phân pha rõ, dependency graph tốt |
| Tính khoa học | **3/5** | Thiếu power analysis, chưa có statistical test |
| Tính thực tiễn | **4/5** | Code infrastructure khớp roadmap (phần lớn), có streaming |
| Tính đầy đủ | **3/5** | Thiếu data ablation, eval detail, perceptual test |
| Quản lý rủi ro | **4/5** | Trigger rules tốt, nhưng thiếu rủi ro platform/hardware |
| **Tổng thể** | **3.5/5** | Trên trung bình, cần bổ sung statistical rigor |

**Verdict:** Roadmap đủ tốt để chạy thực nghiệm và thu thập tín hiệu sớm, nhưng **chưa đủ nghiêm ngặt để claim kết quả là conclusive**. Các bổ sung về statistical testing, multi-seed, và per-condition evaluation sẽ nâng giá trị khoa học lên đáng kể mà không tốn thêm nhiều compute.

> [!TIP]
> Roadmap nên được coi là **living document**. Sau mỗi round ablation, cập nhật lại giả thuyết, ghi rõ kết quả pass/fail, và adjust ưu tiên. Đừng treat roadmap như kế hoạch cố định — treat nó như hypothesis tracker.
