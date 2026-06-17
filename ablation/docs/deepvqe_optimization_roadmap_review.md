# Nhan xet cap nhat ve DeepVQE Optimization Roadmap

Tai lieu goc: `ablation/docs/deepvqe_optimization_roadmap.md`

## Tong quan

Ban roadmap da duoc sua theo huong chat hon ro ret. Cac gop y quan trong o vong truoc da duoc dua vao dung huong:

- Lam ro phien ban DeepVQE dang xet la **NS-only**, da loai bo Far-end va Cross-Attention.
- Bo sung canh bao DW-Conv chi duoc xem la "compute win" khi giam **Stateful Streaming RTF** hoac **ONNX latency thuc te**, khong chi giam MACs/FLOPs ly thuyet.
- Mo ta ECA-F theo tensor `[B, C, T, F]` va nhan manh khong pooling theo truc thoi gian `T`.
- Tach architecture ablation va loss ablation.
- Bo sung checkpoint selection rule de tranh data leakage.
- Bo sung danh gia variance/statistical noise.
- Tach `C2` thanh `C2a` va `C2b`, giup ablation sach hon.
- Dua Grouped GRU/Sub-band RNN vao nhom re-architecture rui ro trung binh-cao.

Nhin chung, tai lieu hien da gan voi mot experimental protocol thuc te hon, khong chi la danh sach y tuong.

## Cac diem con nen chinh

### 1. Primary Metric van chua that su khoa

Roadmap hien ghi:

> PESQ neu co clean reference tot hoac DNSMOS OVRL lam Primary Metric.

Cach viet nay dung ve mat y tuong, nhung khi thuc thi ablation can khoa metric truoc khi train/eval. Neu de "PESQ hoac DNSMOS" qua lau, co nguy co chon metric sau khi da nhin ket qua.

De xuat sua thanh mot trong hai cach:

- Chon cu the ngay trong roadmap, vi du `Primary Metric: DNSMOS OVRL`.
- Hoac ghi ro primary metric nam trong experiment config va phai duoc co dinh truoc khi bat dau Stage 0.

Vi du:

> Primary metric phai duoc co dinh trong config truoc Stage 0. Mac dinh dung DNSMOS OVRL cho perceptual quality; neu eval set co clean reference chat luong cao thi dung PESQ. Khong duoc doi primary metric sau khi da nhin ket qua ablation.

### 2. Guardrail metrics can co nguong so

Hien tai roadmap noi STOI va SI-SDR "khong duoc giam vuot qua gioi han an toan", nhung chua dinh nghia gioi han an toan la bao nhieu.

Nen dat nguong tam thoi, sau do dieu chinh dua tren variance baseline:

- `STOI >= baseline - 0.002`
- `SI-SDR >= baseline - 0.1 dB`
- Neu dung PESQ lam guardrail: `PESQ >= baseline - 0.01`

Nguong cu the co the thay doi sau khi chay baseline nhieu seed, nhung nen co con so mac dinh de automation biet pass/fail.

### 3. Dependency Graph con sot ten C2

Stage 2 da tach:

- `C2a`: C1 + DW-Subpixel Decoder
- `C2b`: C1 + ECA-F

Nhung dependency graph van ghi:

> Neu B2 fail ro ret -> Bo C2.

Nen sua thanh:

> Neu `B2` fail ro ret -> Bo `C2b` va cac nhanh dung ECA-F nhu `C4`, tru khi muon giu mot run kiem tra interaction effect khi di kem DW-Conv.

Dong nay quan trong vi `C2a` la DW-Subpixel, khong phu thuoc truc tiep vao ket qua `B2`.

### 4. Eval set "chi dung duy nhat mot lan" co the gay mau thuan

Roadmap hien ghi eval set chi duoc dung duy nhat mot lan de bao cao ket qua sau cung. Neu hieu theo nghia nghiem ngat, dieu nay mau thuan voi ablation study, vi moi variant deu can duoc danh gia tren cung mot split de so sanh.

Nen tach ro ba loai split:

- **Validation set:** dung de chon checkpoint va early stopping.
- **Ablation/dev eval set:** dung de so sanh cac bien the trong roadmap.
- **Final test set:** chi dung mot lan cho model cuoi cung sau khi da chon xong.

De xuat phrasing:

> Validation set dung cho checkpoint selection. Ablation/dev eval set duoc khoa co dinh va dung de so sanh moi variant. Final test set chi duoc dung mot lan de bao cao model cuoi cung.

Neu hien tai chi co mot eval split, nen doi ten no thanh `dev_eval` va can tao them `final_test` neu muon bao cao ket qua nghiem tuc.

### 5. B4/BestLoss can ghi ro phai train lai

Phan tich hop loss vao Stage 2 da dung huong, nhung nen ghi ro combined run phai duoc train lai theo protocol da khoa.

Can tranh hieu nham rang co the lay checkpoint cua architecture candidate roi chi doi loss de fine-tune ngan, vi nhu vay budget se khac va ket qua kho so sanh.

De xuat them:

> Cac combined run dung `BestLoss` phai train lai tu scratch theo cung training budget, tru khi co mot nhom fine-tuning ablation rieng duoc dinh nghia doc lap.

### 6. Nen them bang cost/benefit neu roadmap dung de quan ly thuc nghiem

Day khong bat buoc, nhung se giup bien roadmap thanh tracker de chay experiment.

Moi variant nen co them cac cot:

- Expected Params change.
- Expected MACs change.
- Expected RTF/ONNX latency risk.
- Implementation risk.
- Quality hypothesis.
- Pass/fail rule.

Vi du:

| Variant | Expected compute | Latency risk | Impl risk | Quality hypothesis |
| --- | --- | --- | --- | --- |
| B1a PReLU shared | Gan nhu khong doi | Thap | Thap | Cai thien activation vung am |
| B2 ECA-F | Tang nhe | Thap-vua | Vua | Tang channel selectivity |
| C1 DW-Conv ResidualBlock | Giam MACs | Vua | Vua | Giai phong compute headroom |
| E3 Grouped GRU | Co the giam lon | Vua-cao | Cao | Giam bottleneck cost nhung rui ro mat cross-band context |

## Ket luan

Ban roadmap sau khi sua da kha chat. Cac van de lon cua ban truoc da duoc xu ly. Nhung truoc khi dung de chay experiment that, nen chinh them cac diem sau:

1. Khoa primary metric truoc Stage 0.
2. Dat nguong so cho STOI/SI-SDR guardrail.
3. Sua dependency graph tu `C2` thanh `C2b` va cac nhanh ECA-F.
4. Tach ro `validation`, `dev_eval`, va `final_test`.
5. Ghi ro combined architecture + BestLoss phai train lai tu scratch theo protocol.

Neu sua them cac diem nay, roadmap se du manh de lam checklist ablation va cung de automation hoa viec pass/fail cho tung variant.
