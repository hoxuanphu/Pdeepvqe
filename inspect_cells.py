import json

with open(r'd:\AI20K\deepvqe\train_base_deepvqe_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('out.txt', 'w', encoding='utf-8') as out:
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            src = "".join(c.get('source', []))
            if 'CONFIG = {' in src or 'def create_csv' in src:
                out.write(src)
                out.write('\n\n---CELL---\n\n')
