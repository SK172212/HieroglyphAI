# glyph_classification/scripts/list_train_val.py
import os
from pathlib import Path
import torch
from torchvision import datasets
from collections import Counter

# --- CONFIG: change these only if your training script used different values ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"            # same as your train script
MIN_IMAGES = 6
VAL_RATIO = 0.2
SEED = 42
OUT_DIR = PROJECT_ROOT / "models"                   # outputs (train/val lists placed here)
CKPT_PATH = OUT_DIR / "resnet18_v1.pth"
# ------------------------------------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

# load ImageFolder (this reproduces full_dataset in training)
full_dataset = datasets.ImageFolder(str(DATA_DIR))
print(f"Found {len(full_dataset.samples)} total samples under {DATA_DIR}")

# count targets and select classes with at least MIN_IMAGES
label_counts = Counter(full_dataset.targets)
kept_class_indices = sorted([idx for idx, c in label_counts.items() if c >= MIN_IMAGES])
print(f"Kept {len(kept_class_indices)} classes (min_images={MIN_IMAGES})")

old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_class_indices)}
kept_class_names = [full_dataset.classes[i] for i in kept_class_indices]

# filter samples same as training code
filtered_samples = []
for path, old_label in full_dataset.samples:
    if old_label in old_to_new:
        new_label = old_to_new[old_label]
        filtered_samples.append((path, new_label))

print(f"Filtered dataset size: {len(filtered_samples)} (after removing classes with <{MIN_IMAGES} images)")

# deterministic split (same seed and method as training script)
torch.manual_seed(SEED)
num_samples = len(filtered_samples)
num_val = int(VAL_RATIO * num_samples)
indices = torch.randperm(num_samples).tolist()
train_indices = indices[:(num_samples - num_val)]
val_indices = indices[(num_samples - num_val):]

train_samples = [filtered_samples[i] for i in train_indices]
val_samples   = [filtered_samples[i] for i in val_indices]

print(f"Recreated split -> Train: {len(train_samples)}  Val: {len(val_samples)}")

# Save file lists
train_list_path = OUT_DIR / "train_files.txt"
val_list_path = OUT_DIR / "val_files.txt"

with open(train_list_path, "w", encoding="utf-8") as f:
    for p, lbl in train_samples:
        f.write(f"{p}\t{lbl}\n")

with open(val_list_path, "w", encoding="utf-8") as f:
    for p, lbl in val_samples:
        f.write(f"{p}\t{lbl}\n")

print(f"Wrote train list: {train_list_path}")
print(f"Wrote val list:   {val_list_path}")

# Optional: print a few examples
print("\nSample train files:")
for p, lbl in train_samples[:10]:
    print(p, lbl)

print("\nSample val files:")
for p, lbl in val_samples[:10]:
    print(p, lbl)

# Print class names from the checkpoint (if checkpoint exists)
if CKPT_PATH.exists():
    try:
        ckpt = torch.load(str(CKPT_PATH), map_location="cpu")
        print("\nCheckpoint class names (from {}):".format(CKPT_PATH))
        for i, name in enumerate(ckpt["class_names"]):
            print(i, name)
    except Exception as e:
        print("Could not load checkpoint:", e)
else:
    print(f"\nNo checkpoint found at {CKPT_PATH} (skipping ckpt class name print)")