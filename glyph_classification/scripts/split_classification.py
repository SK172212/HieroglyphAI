import os
import random
import shutil
from pathlib import Path

# ===== EDIT THESE IF NEEDED =====
INPUT_ROOT = "dataset"
OUTPUT_ROOT = "classify_data"
TRAIN_RATIO = 0.9
SEED = 42
# ===============================

random.seed(SEED)

train_root = Path(OUTPUT_ROOT) / "train"
val_root = Path(OUTPUT_ROOT) / "val"

train_root.mkdir(parents=True, exist_ok=True)
val_root.mkdir(parents=True, exist_ok=True)

print("Splitting classification dataset (90/10)...")

for class_name in sorted(os.listdir(INPUT_ROOT)):
    class_dir = Path(INPUT_ROOT) / class_name
    if not class_dir.is_dir():
        continue

    images = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if len(images) < 2:
        print(f"[SKIP] {class_name}: not enough images ({len(images)})")
        continue

    random.shuffle(images)

    n_train = max(1, int(len(images) * TRAIN_RATIO))
    train_imgs = images[:n_train]
    val_imgs = images[n_train:]

    (train_root / class_name).mkdir(parents=True, exist_ok=True)
    (val_root / class_name).mkdir(parents=True, exist_ok=True)

    for img in train_imgs:
        shutil.copy2(class_dir / img, train_root / class_name / img)

    for img in val_imgs:
        shutil.copy2(class_dir / img, val_root / class_name / img)

    print(
        f"{class_name}: "
        f"{len(train_imgs)} train / {len(val_imgs)} val"
    )

print("Done.")
