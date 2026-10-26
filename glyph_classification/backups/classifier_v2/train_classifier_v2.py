from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image

SEED = 42
MIN_IMAGES_PER_CLASS = 6

NUM_EPOCHS = 12
BATCH_SIZE = 16
LR = 1e-3

V1_CKPT_PATH = Path("glyph_classification/models/resnet18_v1.pth")

# Output
OUT_DIR = Path("glyph_classification/models")
RUN_NAME = "resnet18_v2_frozenSplit_min6_12ep_rot10"
SAVE_PATH = OUT_DIR / f"{RUN_NAME}.pth"
METRICS_PATH = OUT_DIR / f"{RUN_NAME}_metrics.jsonl"
CMAT_CSV_PATH = OUT_DIR / f"{RUN_NAME}_confusion_matrix.csv"
PERCLASS_CSV_PATH = OUT_DIR / f"{RUN_NAME}_per_class_acc.csv"

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def write_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def save_csv_matrix(path: Path, mat: torch.Tensor, row_names: list[str], col_names: list[str]):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    mat_cpu = mat.detach().cpu().tolist()
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + col_names)
        for rname, row in zip(row_names, mat_cpu):
            w.writerow([rname] + row)


def save_per_class_csv(path: Path, class_names: list[str], correct: torch.Tensor, total: torch.Tensor):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "correct", "total", "acc"])
        for cname, c, t in zip(class_names, correct.tolist(), total.tolist()):
            acc = (c / t) if t > 0 else 0.0
            w.writerow([cname, int(c), int(t), f"{acc:.6f}"])


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, samples: list[tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def filter_and_remap_split(imagefolder: datasets.ImageFolder, min_images: int):
    targets = [y for _, y in imagefolder.samples]
    counts = Counter(targets)

    kept_old_idxs = sorted([cls_idx for cls_idx, cnt in counts.items() if cnt >= min_images])
    kept_class_names = [imagefolder.classes[i] for i in kept_old_idxs]
    old_to_new = {old: new for new, old in enumerate(kept_old_idxs)}

    samples_remapped = []
    for path, old_label in imagefolder.samples:
        if old_label in old_to_new:
            samples_remapped.append((path, old_to_new[old_label]))

    return samples_remapped, kept_class_names


def intersect_classes(train_folder: datasets.ImageFolder, val_folder: datasets.ImageFolder, min_images: int):
    train_targets = [y for _, y in train_folder.samples]
    val_targets = [y for _, y in val_folder.samples]
    train_counts = Counter(train_targets)
    val_counts = Counter(val_targets)

    common_old_idxs = []
    for old_idx in range(len(train_folder.classes)):
        if train_counts.get(old_idx, 0) >= min_images and val_counts.get(old_idx, 0) >= 1:
            common_old_idxs.append(old_idx)

    common_old_idxs = sorted(common_old_idxs)   
    common_names = [train_folder.classes[i] for i in common_old_idxs]
    old_to_new = {old: new for new, old in enumerate(common_old_idxs)}

    def remap_samples(folder: datasets.ImageFolder):
        out = []
        for p, old in folder.samples:
            if old in old_to_new:
                out.append((p, old_to_new[old]))
        return out

    return remap_samples(train_folder), remap_samples(val_folder), common_names


def load_resnet18_classifier(num_classes: int, device: torch.device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    return model


def try_load_v1_weights(model: nn.Module, v1_path: Path, new_class_names: list[str], device: torch.device):
    if v1_path is None or not v1_path.exists():
        print("[INFO] No v1 checkpoint found/selected. Training from scratch.")
        return

    ckpt = torch.load(v1_path, map_location=device)
    v1_class_names = ckpt.get("class_names", None)
    state = ckpt.get("model_state_dict", None)

    if state is None:
        print(f"[WARN] v1 checkpoint missing model_state_dict: {v1_path}")
        return

    if v1_class_names == new_class_names:
        model.load_state_dict(state, strict=True)
        print(f"[INFO] Loaded FULL v1 weights (classes match): {v1_path}")
        return

    filtered = {k: v for k, v in state.items() if not k.startswith("fc.")}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[INFO] Loaded PARTIAL v1 weights (skipped fc due to class mismatch): {v1_path}")
    if missing:
        print("[INFO] Missing keys (expected):", missing)
    if unexpected:
        print("[INFO] Unexpected keys:", unexpected)

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    project_root = Path(__file__).resolve().parents[1] 
    train_dir = project_root / "data" / "train"
    val_dir = project_root / "data" / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
        "Train/Val directories not found.\n"
        f"Expected:\n {train_dir}\n {val_dir}\n"
        "Run split_classifier_dataset.py first."
        )

   
    train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]),
    ])


    train_folder = datasets.ImageFolder(train_dir)
    val_folder = datasets.ImageFolder(val_dir)


    train_samples, val_samples, class_names = intersect_classes(
    train_folder, val_folder, min_images=MIN_IMAGES_PER_CLASS
    )

    num_classes = len(class_names)
    if num_classes < 2:
        raise RuntimeError(
            f"Too few classes after filtering (num_classes={num_classes}). "
            "Add more images or lower MIN_IMAGES_PER_CLASS."
        )

    print("\n===== DATASET SUMMARY (FROZEN SPLIT) =====")
    print("Train dir:", train_dir)
    print("Val dir: ", val_dir)
    print(f"Classes kept (>= {MIN_IMAGES_PER_CLASS} imgs in train, >=1 in val): {num_classes}")
    print("Train samples:", len(train_samples))
    print("Val samples: ", len(val_samples))
    print("First 10 classes:", class_names[:10])
    print("=========================================\n")

    train_dataset = ImageDataset(train_samples, transform=train_transform)
    val_dataset = ImageDataset(val_samples, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if METRICS_PATH.exists():
        METRICS_PATH.unlink() 

    model = load_resnet18_classifier(num_classes=num_classes, device=device)
    try_load_v1_weights(model, V1_CKPT_PATH, class_names, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    # Train loop
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy_from_logits(logits, labels)

        train_loss /= max(1, len(train_loader))
        train_acc /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= max(1, len(val_loader))
        val_acc = (val_correct / val_total) if val_total else 0.0

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        write_jsonl(METRICS_PATH, {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "img_size": 224,
                "run_name": RUN_NAME,
                "min_images_per_class": MIN_IMAGES_PER_CLASS,
                "seed": SEED
            }, SAVE_PATH)
            print(f" Saved BEST -> {SAVE_PATH} (val acc {best_val_acc:.3f})")

    # Load best + evaluate confusion matrix
    ckpt = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("\nLoaded BEST checkpoint for evaluation:", SAVE_PATH)

    C = num_classes
    conf_mat = torch.zeros((C, C), dtype=torch.int64)

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            preds = logits.argmax(dim=1)

            for t, p in zip(labels.cpu(), preds.cpu()):
                conf_mat[int(t), int(p)] += 1

    true_counts = conf_mat.sum(dim=1)
    correct_counts = conf_mat.diag()
    per_class_acc = correct_counts.float() / true_counts.clamp(min=1).float()
    overall_acc = correct_counts.sum().float() / true_counts.sum().clamp(min=1).float()

    print("\nPer-class accuracy (val):")
    for i, acc in enumerate(per_class_acc):
        print(f"{class_names[i]}: {acc.item():.3f} ({int(correct_counts[i])}/{int(true_counts[i])})")
    print(f"\nOverall val accuracy (from confusion matrix): {overall_acc.item():.3f}")

    
    save_csv_matrix(CMAT_CSV_PATH, conf_mat, class_names, class_names)
    save_per_class_csv(PERCLASS_CSV_PATH, class_names, correct_counts, true_counts)
    print("\nSaved:")
    print(" Metrics JSONL:", METRICS_PATH)
    print(" Confusion matrix:", CMAT_CSV_PATH)
    print(" Per-class accuracy:", PERCLASS_CSV_PATH)
    print(" Best checkpoint:", SAVE_PATH)

    # Show top confusions
    print("\nTop Confusions (true -> predicted):")
    conf_no_diag = conf_mat.clone()
    conf_no_diag.fill_diagonal_(0)
    flat = conf_no_diag.flatten()
    topk = min(10, flat.numel())
    vals, idxs = torch.topk(flat, k=topk)

    shown = 0
    for v, idx in zip(vals.tolist(), idxs.tolist()):
        if v == 0:
            continue
        true_idx = idx // C
        pred_idx = idx % C
        print(f"{class_names[true_idx]} -> {class_names[pred_idx]}: {v}")
        shown += 1
    if shown == 0:
        print("(none)")


if __name__ == "__main__":
    main()