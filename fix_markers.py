with open('train_base_deepvqe_kaggle.ipynb', 'r', encoding='utf-8') as f:
    lines = f.readlines()

clean_lines = []
in_conflict = False
keep = True

for line in lines:
    if line.startswith('<<<<<<< HEAD'):
        in_conflict = True
        keep = False # discard the HEAD version (which might be the old one)
        continue
    elif line.startswith('======='):
        keep = True # keep the incoming version
        continue
    elif line.startswith('>>>>>>>'):
        in_conflict = False
        continue
    
    if keep:
        clean_lines.append(line)

with open('train_base_deepvqe_kaggle.ipynb', 'w', encoding='utf-8') as f:
    f.writelines(clean_lines)
