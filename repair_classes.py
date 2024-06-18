import os
import glob

def adjust_class_indices(label_path):
    label_files = glob.glob(os.path.join(label_path, '*.txt'))
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            class_idx = int(parts[0]) - 1  # Adjust index to be 0-based
            new_line = ' '.join([str(class_idx)] + parts[1:])
            new_lines.append(new_line)
        with open(label_file, 'w') as f:
            f.write('\n'.join(new_lines))

# Adjust class indices in training and validation data
adjust_class_indices('datasets/train/labels')
adjust_class_indices('datasets/valid/labels')
