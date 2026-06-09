import json
import re

with open(r'd:\AI20K\deepvqe\train_base_deepvqe_kaggle.ipynb', 'r', encoding='utf-8') as f:
    k_nb = json.load(f)

with open(r'd:\AI20K\deepvqe\train_base_deepvqe_colab.ipynb', 'r', encoding='utf-8') as f:
    c_nb = json.load(f)

# Find specific cells in Colab
def find_cell_by_content(nb, keyword):
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            src = "".join(c.get('source', []))
            if keyword in src:
                return src
    return None

c_config_src = find_cell_by_content(c_nb, 'CONFIG = {')
c_utils_src = find_cell_by_content(c_nb, 'def compute_loss(')
c_train_src = find_cell_by_content(c_nb, 'best_loss = float')

# Modify Colab's CONFIG to match Kaggle environment
# Replace output_dir
c_config_src = re.sub(r"'output_dir': '.*?'", "'output_dir': '/kaggle/working/DeepVQE_Workspace/checkpoints/deepvqe_vctk'", c_config_src)
# Replace batch_size to 8 for Kaggle
c_config_src = re.sub(r"'batch_size': \d+,", "'batch_size': 8,", c_config_src)
# Replace csv paths
c_config_src = c_config_src.replace("f'{data_dir}/train.csv'", "f'{metadata_dir}/train.csv'")
c_config_src = c_config_src.replace("f'{data_dir}/valid.csv'", "f'{metadata_dir}/valid.csv'")
c_config_src = c_config_src.replace("f'{data_dir}/test.csv'", "f'{metadata_dir}/test.csv'")

# Now find and replace in Kaggle
for c in k_nb['cells']:
    if c['cell_type'] == 'code':
        src = "".join(c.get('source', []))
        if 'CONFIG = {' in src:
            c['source'] = [line + "\n" for line in c_config_src.split("\n")[:-1]]
        elif 'def compute_loss(' in src:
            c['source'] = [line + "\n" for line in c_utils_src.split("\n")[:-1]]
        elif 'best_loss = float' in src:
            c['source'] = [line + "\n" for line in c_train_src.split("\n")[:-1]]

with open(r'd:\AI20K\deepvqe\train_base_deepvqe_kaggle.ipynb', 'w', encoding='utf-8') as f:
    json.dump(k_nb, f, indent=1)

print("Kaggle notebook updated successfully.")
