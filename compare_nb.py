import json

with open(r'd:\AI20K\deepvqe\train_base_deepvqe_kaggle.ipynb', 'r', encoding='utf-8') as f:
    k_nb = json.load(f)
with open(r'd:\AI20K\deepvqe\train_base_deepvqe_colab.ipynb', 'r', encoding='utf-8') as f:
    c_nb = json.load(f)

with open('out.txt', 'w', encoding='utf-8') as out:
    out.write('Kaggle Cells:\n')
    for i, c in enumerate(k_nb['cells']):
        src = ''.join(c.get('source', []))
        out.write(f"[{i}] {c.get('cell_type')}: {src[:150].replace(chr(10), ' ')}...\n")

    out.write('\nColab Cells:\n')
    for i, c in enumerate(c_nb['cells']):
        src = ''.join(c.get('source', []))
        out.write(f"[{i}] {c.get('cell_type')}: {src[:150].replace(chr(10), ' ')}...\n")
