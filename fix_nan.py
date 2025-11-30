import json

# Read notebook
with open('notebooks/04_dl_model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell with run_baseline_evaluation
for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'source' in cell:
        src_list = cell['source']
        if any('def run_baseline_evaluation' in line for line in src_list):
            print(f"Found run_baseline_evaluation in cell {cell_idx}")
            # Find the index where we need to insert
            for i, line in enumerate(src_list):
                if 'Normalize features' in line or 'Normalize' in line:
                    print(f"Found 'Normalize' at index {i}: {line[:60]}")
                    # Check if previous line has y_reg_val
                    if i > 0:
                        prev_line = src_list[i-1]
                        print(f"Previous line: {prev_line[:60]}")
                        if 'y_reg_val' in prev_line or (i > 1 and 'y_reg_val' in src_list[i-2]):
                            # Insert the filtering code
                            new_lines = [
                                '        # For multitask models, filter out NaN values in y_reg\n',
                                '        if multitask and y_reg_train is not None:\n',
                                '            # Filter train set\n',
                                '            valid_train_mask = ~np.isnan(y_reg_train)\n',
                                '            apt_train = apt_train[valid_train_mask]\n',
                                '            mol_train = mol_train[valid_train_mask]\n',
                                '            y_class_train = y_class_train[valid_train_mask]\n',
                                '            y_reg_train = y_reg_train[valid_train_mask]\n',
                                '            \n',
                                '            # Filter val set\n',
                                '            valid_val_mask = ~np.isnan(y_reg_val)\n',
                                '            apt_val = apt_val[valid_val_mask]\n',
                                '            mol_val = mol_val[valid_val_mask]\n',
                                '            y_class_val = y_class_val[valid_val_mask]\n',
                                '            y_reg_val = y_reg_val[valid_val_mask]\n',
                                '        \n'
                            ]
                            # Insert before the "# Normalize" line
                            cell['source'] = src_list[:i] + new_lines + src_list[i:]
                            print(f"Inserted {len(new_lines)} lines before index {i}")
                            break
            break

# Write back
with open('notebooks/04_dl_model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done")
