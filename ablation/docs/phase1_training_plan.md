Epoch  44/100 | Train Loss: 1.274235 (ri=0.0769, mag=0.0089, sisnr=-1.6569) | Valid Loss: 1.516308 (ri=0.0786, mag=0.0108, sisnr=-1.6005) | LR: 2.50e-04 | Time: 275s# Ke Hoach Train Phase 1 Sau Baseline Benchmark

Muc tieu: train truoc cac bien the it rui ro, danh gia ket qua that tren VCTK-DEMAND, sau do moi quyet dinh co quay lai train cac nhanh ECA-F hay khong.

## 1. Moc Baseline Da Khoa

Baseline checkpoint goc da duoc eval tren cung test split:

| Metric | Gia tri |
| --- | ---: |
| Params | 7,509,996 |
| MACs | 8.046G |
| FLOPs | 16.092G |
| Stateful RTF | 0.0145 |
| PESQ | 2.8560 |
| STOI | 0.9448 |
| SI-SDR | 18.57 |

Day la moc so sanh bat buoc cho tat ca bien the.

## 2. Ket Luan Tu Architecture Benchmark

- `B1a` va `B1b` gan nhu khong tang MACs/FLOPs, chi tang params rat nho. Day la nhom probe activation dang train truoc.
- `C1` giam params khoang 20.2% va MACs khoang 43.6%, la deploy candidate quan trong.
- `B2`, `B3`, `C2`, `C4` co ECA-F. Du MACs tang rat it, RTF tang manh tren Tesla T4, nen tam hoan train full.
- `C1b` hien gan nhu trung `C1` trong code/config, nen khong can train o vong dau tru khi muon do training variance.

## 3. Vong 1: Train Cac Candidate It Rui Ro

Chay truoc:

```python
AB_CONFIGS = ['B1a', 'B1b', 'C1']
RUN_TRAINING = True
RUN_QUALITY_EVAL = True
RUN_ONNX_EXPORT = False
AB_EPOCHS = 80
AB_BATCH_SIZE = 8
```

Y nghia:

- `B1a`: PReLU shared.
- `B1b`: PReLU per-channel.
- `C1`: DW-Conv trong ResidualBlock.

Tat ca phai train tu scratch voi cung split, seed, optimizer, LR schedule, loss, batch size va checkpoint selection rule.

## 4. Doc Ket Qua Sau Vong 1

So sanh tung bien the voi baseline:

- Quality:
  - PESQ phai cao hon baseline hoac it nhat khong giam dang ke.
  - STOI khong duoc giam ro.
  - SI-SDR khong duoc giam ro.
- Compute/deploy:
  - Params/MACs/FLOPs khong duoc vuot baseline neu la deploy candidate.
  - Stateful RTF ly tuong la <= baseline; neu tang nhe thi can can nhac bang quality gain.

Quy tac chon PReLU winner:

1. Quality cao hon.
2. Neu gan bang nhau, chon RTF thap hon.
3. Neu van gan bang, chon `B1a` vi don gian va nhe hon.

## 5. Vong 2: Train Bien The Ket Hop

Chi train `C3` neu ca hai dieu kien sau dung:

- `B1a` hoac `B1b` cho thay PReLU co loi ve quality.
- `C1` khong pha quality qua manh so voi baseline.

Neu `B1a` thang, nen train `C3` voi `prelu_type='shared'` thay vi mac dinh per-channel. Neu `B1b` thang, co the dung `C3` mac dinh.

Thiet lap:

```python
AB_CONFIGS = ['C3']
RUN_TRAINING = True
RUN_QUALITY_EVAL = True
RUN_ONNX_EXPORT = True
AB_EPOCHS = 80
AB_BATCH_SIZE = 8
```

Sau khi co `C3`, chay ONNX export/parity de kiem tra kha nang deploy streaming.

## 6. Khi Nao Quay Lai Train ECA-F

Tam hoan:

```python
['B2', 'B3', 'C2', 'C4']
```

Chi quay lai train cac nhanh nay neu:

- `B1a/B1b/C1/C3` khong tang quality du tot.
- Can kiem chung ECA-F du RTF benchmark ban dau xau.
- Co ke hoach toi uu lai implementation ECA-F de giam runtime overhead.

Neu quay lai ECA-F, thu tu nen la:

1. `B2`: ECA-F trong ResidualBlock.
2. `B3`: ECA-F them o main Encoder/Decoder, optional.
3. `C2`: DW-Conv + ECA-F, neu `B2` co loi.
4. `C4`: DW-Conv + ECA-F + PReLU, chi khi ca PReLU va ECA-F deu co tin hieu tot.

## 7. Output Can Luu Sau Moi Vong

Sau moi lan train/eval, can luu:

- `ablation_arch_benchmark.csv`
- `ablation_quality.csv`
- `ablation_onnx.csv` neu da export ONNX
- `ablation_summary.csv`
- Checkpoint `best.pt` va `last.pt` cua tung config

Ket qua cuoi cung phai duoc tong hop bang:

```bash
python ablation/collect_ablation_results.py
```

## 8. Trang Thai Hien Tai

- Baseline da duoc khoa va co quality benchmark day du PESQ/STOI/SI-SDR.
- Cac ablation variants moi co architecture benchmark, chua co quality benchmark vi chua train.
- Buoc tiep theo: train `B1a`, `B1b`, `C1`.
