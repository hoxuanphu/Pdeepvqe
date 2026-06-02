import json

with open('train_base_deepvqe_kaggle.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and update git clone cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if line.startswith('!git clone '):
                new_source.append('# TODO: Thay <your-username> bang ten tai khoan GitHub cua ban\n')
                new_source.append('!git clone -b ablation-study https://github.com/<your-username>/deepvqe.git\n')
            else:
                new_source.append(line)
        cell['source'] = new_source

# Append cells for Evaluation of Baseline
baseline_eval_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Đánh giá Baseline (Tuỳ chọn)\\n",
        "Chạy các ô dưới đây nếu bạn muốn đo Params, FLOPs, RTF và PESQ/STOI/SI-SDR của mô hình gốc để làm mốc so sánh."
    ]
}

baseline_benchmark = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "%cd /kaggle/working/DeepVQE_Workspace/deepvqe\\n",
        "!pip install ptflops\\n",
        "!python ablation/run_ablation_benchmark.py --configs Baseline"
    ]
}

baseline_quality = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "%cd /kaggle/working/DeepVQE_Workspace/deepvqe\\n",
        "!pip install pesq pystoi\\n",
        "!python ablation/eval_ablation_quality.py \\\\\\n",
        "    --config-id Baseline \\\\\\n",
        "    --checkpoint /kaggle/working/DeepVQE_Workspace/checkpoints/best.pt \\\\\\n",
        "    --manifest /kaggle/working/DeepVQE_Workspace/metadata/test.csv"
    ]
}

ablation_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Train Ablation Model (Stage 1)\\n",
        "Sử dụng các config_id như B1a, B1b, B2... đã được định nghĩa trong thư mục ablation/configs/"
    ]
}

ablation_train = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "%cd /kaggle/working/DeepVQE_Workspace/deepvqe\\n",
        "!python ablation/train_ablation.py \\\\\\n",
        "    --config-id B1a \\\\\\n",
        "    --train-manifest /kaggle/working/DeepVQE_Workspace/metadata/train.csv \\\\\\n",
        "    --valid-manifest /kaggle/working/DeepVQE_Workspace/metadata/valid.csv \\\\\\n",
        "    --output-dir /kaggle/working/DeepVQE_Workspace/checkpoints/B1a \\\\\\n",
        "    --epochs 100 \\\\\\n",
        "    --batch-size 8"
    ]
}

ablation_eval = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "%cd /kaggle/working/DeepVQE_Workspace/deepvqe\\n",
        "!python ablation/eval_ablation_quality.py \\\\\\n",
        "    --config-id B1a \\\\\\n",
        "    --checkpoint /kaggle/working/DeepVQE_Workspace/checkpoints/B1a/best.pt \\\\\\n",
        "    --manifest /kaggle/working/DeepVQE_Workspace/metadata/test.csv"
    ]
}

nb['cells'].extend([baseline_eval_md, baseline_benchmark, baseline_quality, ablation_md, ablation_train, ablation_eval])

with open('train_ablation_kaggle.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
