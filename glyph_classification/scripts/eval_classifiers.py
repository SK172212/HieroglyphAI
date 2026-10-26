"""
eval_classifiers.py

Goal:
- Evaluate classifier v1 vs v2 on the SAME validation split:
    glyph_classification/data/val
- Only evaluate on classes that exist in BOTH models (shared classes).
- Save metrics + per-class accuracy + confusion matrices for both models.

How to run:
- Press Run in VS Code (no CLI required)
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # glyph_classification/
VAL_DIR = PROJECT_ROOT / "data" / "val"

V1_CKPT = PROJECT_ROOT / "models" / "v1" / "resnet18_v1_stratified_10ep_NOflip.pth"
V2_CKPT = PROJECT_ROOT / "models" / "v2" / "resnet18_v2_frozenSplit_min6_12ep_rot10.pth"  # change if your filename differs

OUT_DIR = PROJECT_ROOT / "output" / "eval_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
NUM_WORKERS = 0
IMG_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Helpers
# =========================
def load_ckpt(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_names = ckpt["class_names"]
    img_size = ckpt.get("img_size", IMG_SIZE)
    return ckpt, class_names, img_size


def build_model(num_classes: int, ckpt_state_dict):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt_state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def get_val_dataset(val_dir: Path, img_size: int):
    # IMPORTANT: must match training normalization
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(val_dir, transform=val_tf)
    return ds


def filter_to_shared_classes(ds, shared_class_names):
    """
    Keep only samples whose class folder name is in shared_class_names.
    Returns:
      - subset dataset
      - shared_class_names_sorted (consistent order)
    """
    # ds.classes is folder names, ds.class_to_idx maps name -> index
    shared = sorted([c for c in ds.classes if c in shared_class_names])

    if len(shared) == 0:
        raise RuntimeError("No shared classes found between dataset and checkpoints.")

    # Keep indices for samples whose label is in shared set
    keep_class_idxs = {ds.class_to_idx[c] for c in shared}

    keep_sample_indices = []
    for i, (_, y) in enumerate(ds.samples):
        if y in keep_class_idxs:
            keep_sample_indices.append(i)

    subset = Subset(ds, keep_sample_indices)
    return subset, shared


def evaluate(model, loader, ds_classes_shared, ds_class_to_idx):
    """
    Evaluates accuracy + confusion matrix (in dataset label space).
    Returns dict with:
      overall_acc, conf_mat, per_class_acc, counts
    """
    C = len(ds_classes_shared)
    # We build confusion matrix indexed by shared-class order (0..C-1)
    shared_name_to_newidx = {name: i for i, name in enumerate(ds_classes_shared)}

    conf = torch.zeros((C, C), dtype=torch.int64)

    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(imgs)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Update confusion (labels/preds are dataset indices, not 0..C-1 necessarily)
            for t, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                true_name = None
                pred_name = None

                # Convert dataset index -> folder name
                # ds_class_to_idx is name->idx; invert once outside would be faster, but this is fine for your sizes
                # We'll build inverse map quickly here:
                # (Better: pass idx_to_name dict in, but keeping simple.)
                # We'll do a tiny hack:
                # find name by scanning (small number of classes)
                for name in ds_class_to_idx:
                    if ds_class_to_idx[name] == t:
                        true_name = name
                    if ds_class_to_idx[name] == p:
                        pred_name = name
                    if true_name is not None and pred_name is not None:
                        break

                if true_name in shared_name_to_newidx and pred_name in shared_name_to_newidx:
                    conf[shared_name_to_newidx[true_name], shared_name_to_newidx[pred_name]] += 1

    overall_acc = correct / total if total > 0 else 0.0

    true_counts = conf.sum(dim=1)
    correct_counts = conf.diag()
    per_class_acc = (correct_counts.float() / true_counts.clamp(min=1).float()).tolist()

    return {
        "overall_acc": overall_acc,
        "total": total,
        "conf_mat": conf,
        "per_class_acc": per_class_acc,
        "true_counts": true_counts.tolist(),
        "correct_counts": correct_counts.tolist(),
    }


def save_confusion_csv(conf, class_names, out_path: Path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + class_names)
        for i, row in enumerate(conf.tolist()):
            w.writerow([class_names[i]] + row)


def save_per_class_csv(class_names, correct_counts, true_counts, per_class_acc, out_path: Path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "correct", "total", "acc"])
        for name, c, t, a in zip(class_names, correct_counts, true_counts, per_class_acc):
            w.writerow([name, c, t, f"{a:.6f}"])


# =========================
# Main
# =========================
def main():
    print("[INFO] Device:", DEVICE)
    print("[INFO] VAL_DIR:", VAL_DIR)

    ckpt1, v1_classes, v1_img_size = load_ckpt(V1_CKPT)
    ckpt2, v2_classes, v2_img_size = load_ckpt(V2_CKPT)

    print("[INFO] v1 classes:", len(v1_classes))
    print("[INFO] v2 classes:", len(v2_classes))

    shared_classes = set(v1_classes).intersection(set(v2_classes))
    print("[INFO] shared classes:", len(shared_classes))

    # Use a single image size (prefer v2 if different)
    img_size = v2_img_size

    ds = get_val_dataset(VAL_DIR, img_size=img_size)
    print("[INFO] val folder classes (from disk):", len(ds.classes))

    subset, shared_in_val = filter_to_shared_classes(ds, shared_classes)
    print("[INFO] shared classes present in VAL_DIR:", len(shared_in_val))
    print("[INFO] images evaluated (shared classes only):", len(subset))

    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Build models
    m1 = build_model(num_classes=len(v1_classes), ckpt_state_dict=ckpt1["model_state_dict"])
    m2 = build_model(num_classes=len(v2_classes), ckpt_state_dict=ckpt2["model_state_dict"])

    # IMPORTANT:
    # These models predict indices in THEIR OWN class ordering (v1_classes / v2_classes),
    # but the dataset labels are based on folder ordering in VAL_DIR.
    #
    # So we must remap dataset labels -> model labels for evaluation.
    #
    # Easiest fix: create a wrapper dataset that remaps labels for each model separately.
    #
    # We'll do that now.

    ds_class_to_idx = ds.class_to_idx  # folder name -> dataset label

    # Build remap: dataset_label -> model_label (only for shared classes)
    v1_name_to_idx = {name: i for i, name in enumerate(v1_classes)}
    v2_name_to_idx = {name: i for i, name in enumerate(v2_classes)}

    shared_dataset_label_to_v1 = {}
    shared_dataset_label_to_v2 = {}

    for cname in shared_in_val:
        dlab = ds_class_to_idx[cname]
        shared_dataset_label_to_v1[dlab] = v1_name_to_idx[cname]
        shared_dataset_label_to_v2[dlab] = v2_name_to_idx[cname]

    class RemapSubset(torch.utils.data.Dataset):
        def __init__(self, subset, remap_dict):
            self.subset = subset
            self.remap = remap_dict

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            x, y = self.subset[idx]
            return x, self.remap[y]

    subset_v1 = RemapSubset(subset, shared_dataset_label_to_v1)
    subset_v2 = RemapSubset(subset, shared_dataset_label_to_v2)

    loader_v1 = DataLoader(subset_v1, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    loader_v2 = DataLoader(subset_v2, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # For confusion matrices we want a consistent label set (shared_in_val order)
    # We'll compute confusion in shared class space by mapping model predictions back to class names.
    shared_order = shared_in_val  # list of class names

    # Evaluate v1
    # We'll make a mapping from v1 model idx -> class name for shared classes
    v1_idx_to_name = {i: name for i, name in enumerate(v1_classes)}
    v2_idx_to_name = {i: name for i, name in enumerate(v2_classes)}

    def evaluate_in_shared_space(model, loader, idx_to_name):
        C = len(shared_order)
        name_to_shared = {name: i for i, name in enumerate(shared_order)}
        conf = torch.zeros((C, C), dtype=torch.int64)
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(imgs)
                preds = logits.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                for t, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                    tn = idx_to_name[t]
                    pn = idx_to_name[p]
                    if tn in name_to_shared and pn in name_to_shared:
                        conf[name_to_shared[tn], name_to_shared[pn]] += 1

        overall_acc = correct / total if total else 0.0
        true_counts = conf.sum(dim=1)
        correct_counts = conf.diag()
        per_class_acc = (correct_counts.float() / true_counts.clamp(min=1).float()).tolist()

        return overall_acc, total, conf, correct_counts.tolist(), true_counts.tolist(), per_class_acc

    v1_acc, v1_total, v1_conf, v1_corr, v1_true, v1_pca = evaluate_in_shared_space(m1, loader_v1, v1_idx_to_name)
    v2_acc, v2_total, v2_conf, v2_corr, v2_true, v2_pca = evaluate_in_shared_space(m2, loader_v2, v2_idx_to_name)

    print("\n===== RESULTS (Shared Val Classes) =====")
    print("Images evaluated:", len(subset))
    print(f"v1 accuracy: {v1_acc:.4f}")
    print(f"v2 accuracy: {v2_acc:.4f}")
    print("=======================================\n")

    # Save outputs
    metrics = {
        "images_evaluated": len(subset),
        "shared_classes": shared_order,
        "v1": {"ckpt": str(V1_CKPT), "accuracy": v1_acc},
        "v2": {"ckpt": str(V2_CKPT), "accuracy": v2_acc},
    }

    metrics_path = OUT_DIR / "classifier_eval_sharedVal_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_confusion_csv(v1_conf, shared_order, OUT_DIR / "classifier_eval_sharedVal_confusion_v1.csv")
    save_confusion_csv(v2_conf, shared_order, OUT_DIR / "classifier_eval_sharedVal_confusion_v2.csv")

    save_per_class_csv(shared_order, v1_corr, v1_true, v1_pca, OUT_DIR / "classifier_eval_sharedVal_per_class_v1.csv")
    save_per_class_csv(shared_order, v2_corr, v2_true, v2_pca, OUT_DIR / "classifier_eval_sharedVal_per_class_v2.csv")

    print("[SAVED]")
    print("Metrics JSON:     ", metrics_path)
    print("Confusion v1 CSV: ", OUT_DIR / "classifier_eval_sharedVal_confusion_v1.csv")
    print("Confusion v2 CSV: ", OUT_DIR / "classifier_eval_sharedVal_confusion_v2.csv")
    print("Per-class v1 CSV: ", OUT_DIR / "classifier_eval_sharedVal_per_class_v1.csv")
    print("Per-class v2 CSV: ", OUT_DIR / "classifier_eval_sharedVal_per_class_v2.csv")


if __name__ == "__main__":
    main()