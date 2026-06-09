import json

with open(r'd:\AI20K\deepvqe\train_base_deepvqe_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cell_6 = """import os
import zipfile
import shutil
from pathlib import Path

# Thư mục chứa file ZIP trên Google Drive
drive_data_dir = "/content/drive/MyDrive/DeepVQE_Workspace/data/voicebank-demand"
os.makedirs(drive_data_dir, exist_ok=True)

# Thư mục trên máy ảo Colab (Local SSD) để giải nén và train cho nhanh
data_dir = "/content/data/voicebank-demand"
os.makedirs(data_dir, exist_ok=True)

datasets = [
    ("clean_trainset_28spk_wav.zip", "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip"),
    ("noisy_trainset_28spk_wav.zip", "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip"),
    ("clean_testset_wav.zip", "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip"),
    ("noisy_testset_wav.zip", "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip")
]

for filename, url in datasets:
    drive_zip = os.path.join(drive_data_dir, filename)
    local_zip = os.path.join(data_dir, filename)
    
    # 1. Tải về Google Drive nếu chưa có
    if not os.path.exists(drive_zip):
        print(f"Đang tải {filename} về Google Drive...")
        os.system(f"wget -q --show-progress {url} -O {drive_zip}")
    
    # 2. Copy từ Google Drive sang Local SSD
    if not os.path.exists(local_zip):
        print(f"Đang copy {filename} từ Drive sang Local SSD...")
        shutil.copy2(drive_zip, local_zip)
    
    # 3. Giải nén trên Local SSD
    extract_folder = local_zip.replace(".zip", "")
    if not os.path.exists(extract_folder):
        print(f"Đang giải nén {filename}...")
        with zipfile.ZipFile(local_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"Hoàn tất giải nén {filename}.")

print("Dữ liệu đã sẵn sàng trên Local SSD của Colab!")
"""

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        src = "".join(c.get('source', []))
        if 'zipfile' in src and 'datasets =' in src:
            c['source'] = [line + "\n" for line in new_cell_6.split("\n")[:-1]]
            break

with open(r'd:\AI20K\deepvqe\train_base_deepvqe_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated Colab dataset loading logic.")
