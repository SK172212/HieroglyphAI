from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageFilter


# ============================================================
# CONFIG
# ============================================================

SEED = 42
NUM_EPOCHS = 15
BATCH_SIZE = 16
LR = 3e-4
WEIGHT_DECAY = 1e-4
IMG_SIZE = 224

V2_CKPT_NAME = "resnet18_v2_frozenSplit_min6_12ep_rot10.pth"
RUN_NAME = "resnet18_v3_museum_aug_sqrtweighted"


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

TRAIN_DIR = SCRIPT_DIR / "data" / "train"
VAL_DIR = SCRIPT_DIR / "data" / "val"
LOCKED_TEST_DIR = SCRIPT_DIR / "data" / "locked_test"

V2_CKPT_PATH = PROJECT_ROOT / "models" / "v2" / V2_CKPT_NAME

OUT_DIR = SCRIPT_DIR / "output"
BEST_PATH = OUT_DIR / f"{RUN_NAME}_best_val_loss.pth"
FINAL_PATH = OUT_DIR / f"{RUN_NAME}_final_epoch.pth"
METRICS_PATH = OUT_DIR / f"{RUN_NAME}_metrics.jsonl"
CMAT_CSV_PATH = OUT_DIR / f"{RUN_NAME}_val_confusion_matrix.csv"
PERCLASS_CSV_PATH = OUT_DIR / f"{RUN_NAME}_val_per_class.csv"
CLASS_COUNTS_PATH = OUT_DIR / f"{RUN_NAME}_train_class_counts.csv"
SUMMARY_PATH = OUT_DIR / f"{RUN_NAME}_summary.json"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

class PadToSquare:
    """Pad to square without stretching the glyph."""
    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size
        if w == h:
            return img

        corners = [
            img.getpixel((0, 0)),
            img.getpixel((w - 1, 0)),
            img.getpixel((0, h - 1)),
            img.getpixel((w - 1, h - 1)),
        ]
        bg = tuple(
            int(sum(px[c] for px in corners) / len(corners))
            for c in range(3)
        )

        size = max(w, h)
        canvas = Image.new("RGB", (size, size), bg)
        canvas.paste(img, ((size - w) // 2, (size - h) // 2))
        return canvas


class RandomGaussianBlur:
    def __init__(self, probability=0.10, radius_min=0.1, radius_max=0.8):
        self.probability = probability
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() < self.probability:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius))
        return img


# ============================================================
# DATASET + CLASS MAPPING
# ============================================================

class ImageDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img, label


def build_train_samples(train_dir: Path):
    folder = datasets.ImageFolder(train_dir)
    class_names = list(folder.classes)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    samples = [
        (path, class_to_idx[folder.classes[old_idx]])
        for path, old_idx in folder.samples
    ]
    return samples, class_names, class_to_idx


def build_val_samples(val_dir: Path, class_to_idx: dict[str, int]):
    folder = datasets.ImageFolder(val_dir)
    samples = []
    ignored = []

    for path, folder_idx in folder.samples:
        name = folder.classes[folder_idx]
        if name not in class_to_idx:
            ignored.append((path, name))
        else:
            samples.append((path, class_to_idx[name]))

    return samples, ignored


# ============================================================
# TRANSFORMS
# ============================================================

def build_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        PadToSquare(),
        transforms.Resize(
            (IMG_SIZE, IMG_SIZE),
            interpolation=InterpolationMode.BILINEAR
        ),
        transforms.RandomRotation(
            10,
            interpolation=InterpolationMode.BILINEAR,
            fill=255
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.04, 0.04),
            scale=(0.95, 1.05),
            shear=3,
            interpolation=InterpolationMode.BILINEAR,
            fill=255
        ),
        transforms.RandomPerspective(
            distortion_scale=0.08,
            p=0.15,
            interpolation=InterpolationMode.BILINEAR,
            fill=255
        ),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.08,
            hue=0.02
        ),
        RandomGaussianBlur(probability=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = transforms.Compose([
        PadToSquare(),
        transforms.Resize(
            (IMG_SIZE, IMG_SIZE),
            interpolation=InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, val_transform


# ============================================================
# MODERATE IMBALANCE HANDLING
# ============================================================

def build_weighted_sampler(samples, num_classes):
    counts = Counter(label for _, label in samples)

    missing = [i for i in range(num_classes) if counts[i] == 0]
    if missing:
        raise RuntimeError(f"Training classes with zero samples: {missing}")

    # Moderate balancing: 1/sqrt(n), not full 1/n.
    # This helps rare classes without repeatedly oversampling a 5-image
    # class as aggressively as full inverse-frequency balancing would.
    class_weights = {
        class_idx: 1.0 / math.sqrt(count)
        for class_idx, count in counts.items()
    }
    sample_weights = [class_weights[label] for _, label in samples]

    generator = torch.Generator()
    generator.manual_seed(SEED)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    return sampler, counts


# ============================================================
# MODEL
# ============================================================

def create_model(num_classes, device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def load_v2_backbone(model, checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "V2 checkpoint not found. Expected:\n"
            f"{checkpoint_path}\n"
            "Do not train until this path is corrected."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict")
    if state is None:
        raise RuntimeError("V2 checkpoint has no model_state_dict.")

    backbone_state = {
        k: v for k, v in state.items()
        if not k.startswith("fc.")
    }

    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    unexpected_non_fc = [k for k in unexpected if not k.startswith("fc.")]

    if unexpected_non_fc:
        raise RuntimeError(f"Unexpected V2 keys: {unexpected_non_fc}")

    print("\nLoaded V2 backbone:", checkpoint_path)
    print("Old FC layer discarded; new 59-class FC layer initialized.")
    return True


# ============================================================
# OUTPUT HELPERS
# ============================================================

def write_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def save_checkpoint(path, model, class_names, class_to_idx,
                    epoch, val_loss, val_acc):
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "img_size": IMG_SIZE,
        "run_name": RUN_NAME,
        "seed": SEED,
        "num_epochs_planned": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate_initial": LR,
        "weight_decay": WEIGHT_DECAY,
        "epoch": epoch,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "preprocessing": "pad_to_square_then_resize",
        "sampler": "inverse_sqrt_class_frequency",
        "v2_backbone_loaded": True,
    }, path)


def save_confusion_matrix(path, matrix, class_names):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + class_names)
        for name, row in zip(class_names, matrix.tolist()):
            w.writerow([name] + row)


def calculate_per_class_metrics(conf_mat):
    true_total = conf_mat.sum(dim=1)
    pred_total = conf_mat.sum(dim=0)
    correct = conf_mat.diag()

    precision = correct.float() / pred_total.clamp(min=1).float()
    recall = correct.float() / true_total.clamp(min=1).float()
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)

    return {
        "correct": correct,
        "true_total": true_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_per_class_metrics(path, class_names, metrics):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "val_examples", "correct", "precision", "recall", "f1"])
        for i, name in enumerate(class_names):
            w.writerow([
                name,
                int(metrics["true_total"][i]),
                int(metrics["correct"][i]),
                f"{metrics['precision'][i].item():.6f}",
                f"{metrics['recall'][i].item():.6f}",
                f"{metrics['f1'][i].item():.6f}",
            ])


def save_train_counts(path, class_names, counts):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "train_images"])
        for i, name in enumerate(class_names):
            w.writerow([name, counts[i]])


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            n = labels.size(0)
            total_loss += loss.item() * n
            total_correct += (preds == labels).sum().item()
            total_examples += n

            for t, p in zip(labels.cpu(), preds.cpu()):
                conf_mat[int(t), int(p)] += 1

    return (
        total_loss / max(1, total_examples),
        total_correct / max(1, total_examples),
        conf_mat,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("\n" + "=" * 72)
    print("HIEROGLYPH CLASSIFIER V3 TRAINING")
    print("=" * 72)
    print("Device:", device)

    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        raise FileNotFoundError("V3 train/val directories are missing.")

    # Safety: protected classifier test directory must remain empty.
    locked_files = [
        p for p in LOCKED_TEST_DIR.rglob("*") if p.is_file()
    ] if LOCKED_TEST_DIR.exists() else []

    if locked_files:
        raise RuntimeError(
            "STOP: classifier_v3/data/locked_test contains files. "
            "The protected museum benchmark must remain untouched during training."
        )

    train_samples, class_names, class_to_idx = build_train_samples(TRAIN_DIR)
    val_samples, ignored_val = build_val_samples(VAL_DIR, class_to_idx)
    num_classes = len(class_names)

    if num_classes != 59:
        raise RuntimeError(f"Expected 59 classes, found {num_classes}.")
    if len(train_samples) != 1673:
        raise RuntimeError(f"Expected 1673 train images, found {len(train_samples)}.")
    if len(val_samples) != 62:
        raise RuntimeError(
            f"Expected 62 usable validation images, found {len(val_samples)}. "
            f"Ignored: {len(ignored_val)}"
        )
    if ignored_val:
        raise RuntimeError(
            "Validation contains classes outside the canonical V3 mapping: "
            f"{ignored_val[:5]}"
        )

    train_transform, val_transform = build_transforms()
    train_dataset = ImageDataset(train_samples, train_transform)
    val_dataset = ImageDataset(val_samples, val_transform)

    sampler, train_counts = build_weighted_sampler(train_samples, num_classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    val_counts = Counter(label for _, label in val_samples)
    represented_val = [i for i in range(num_classes) if val_counts[i] > 0]

    print("\n===== V3 DATASET SUMMARY =====")
    print(f"Classes:                   {num_classes}")
    print(f"Training images:           {len(train_samples)}")
    print(f"Validation images:         {len(val_samples)}")
    print(f"Val classes represented:   {len(represented_val)}/{num_classes}")
    print(f"Smallest train class:      {min(train_counts.values())}")
    print(f"Largest train class:       {max(train_counts.values())}")
    print("Sampler:                   inverse-sqrt class frequency")
    print("Locked museum benchmark:   UNTOUCHED")
    print("==============================")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if METRICS_PATH.exists():
        METRICS_PATH.unlink()

    save_train_counts(CLASS_COUNTS_PATH, class_names, train_counts)

    model = create_model(num_classes, device)
    load_v2_backbone(model, V2_CKPT_PATH, device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=LR / 20,
    )

    # IMPORTANT:
    # The validation set is only 62 clean images across 34/59 classes.
    # We therefore do NOT early-stop based on it.
    # We train the full predeclared 15 epochs.
    #
    # For the "best" checkpoint, use validation LOSS rather than raw
    # accuracy because loss is less coarse than accuracy on only 62 images.
    # We also save the final epoch separately.
    best_val_loss = float("inf")
    best_epoch = None
    best_val_acc_at_best_loss = None

    print("\nStarting full 15-epoch training...\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            n = labels.size(0)
            train_loss_sum += loss.item() * n
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += n

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        val_loss, val_acc, _ = evaluate(
            model, val_loader, criterion, device, num_classes
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"lr {current_lr:.6f} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        write_jsonl(METRICS_PATH, {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_acc_at_best_loss = val_acc
            save_checkpoint(
                BEST_PATH, model, class_names, class_to_idx,
                epoch, val_loss, val_acc
            )
            print(
                f"  Saved BEST-VAL-LOSS -> {BEST_PATH.name} "
                f"(epoch {epoch}, val loss {val_loss:.4f})"
            )

        scheduler.step()

    # Always preserve final epoch too, without using locked museum data.
    final_val_loss, final_val_acc, _ = evaluate(
        model, val_loader, criterion, device, num_classes
    )
    save_checkpoint(
        FINAL_PATH, model, class_names, class_to_idx,
        NUM_EPOCHS, final_val_loss, final_val_acc
    )

    # Our predeclared model-selection rule is lowest CLEAN validation loss.
    # The locked museum set remains untouched.
    ckpt = torch.load(BEST_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    selected_val_loss, selected_val_acc, conf_mat = evaluate(
        model, val_loader, criterion, device, num_classes
    )
    per_class = calculate_per_class_metrics(conf_mat)

    represented_mask = per_class["true_total"] > 0
    macro_recall = per_class["recall"][represented_mask].mean().item()
    macro_f1 = per_class["f1"][represented_mask].mean().item()

    print("\n" + "=" * 72)
    print("SELECTED V3 MODEL — CLEAN VALIDATION")
    print("=" * 72)
    print(f"Selection rule: lowest clean validation loss")
    print(f"Selected epoch: {best_epoch}/{NUM_EPOCHS}")
    print(f"Validation loss: {selected_val_loss:.4f}")
    print(f"Validation accuracy: {selected_val_acc:.4f}")
    print(f"Macro recall (34 represented classes): {macro_recall:.4f}")
    print(f"Macro F1 (34 represented classes): {macro_f1:.4f}")
    print("Protected museum benchmark used: NO")

    print("\nPer-class validation:")
    for i, name in enumerate(class_names):
        total = int(per_class["true_total"][i])
        if total == 0:
            continue
        correct = int(per_class["correct"][i])
        print(
            f"{name:<6} {correct}/{total} "
            f"recall={per_class['recall'][i].item():.3f} "
            f"f1={per_class['f1'][i].item():.3f}"
        )

    print("\nTop validation confusions:")
    conf_no_diag = conf_mat.clone()
    conf_no_diag.fill_diagonal_(0)
    flat = conf_no_diag.flatten()
    vals, idxs = torch.topk(flat, k=min(10, flat.numel()))
    shown = 0
    for value, index in zip(vals.tolist(), idxs.tolist()):
        if value == 0:
            continue
        t = index // num_classes
        p = index % num_classes
        print(f"{class_names[t]} -> {class_names[p]}: {value}")
        shown += 1
    if shown == 0:
        print("(none)")

    save_confusion_matrix(CMAT_CSV_PATH, conf_mat, class_names)
    save_per_class_metrics(PERCLASS_CSV_PATH, class_names, per_class)

    summary = {
        "run_name": RUN_NAME,
        "num_classes": num_classes,
        "train_images": len(train_samples),
        "val_images": len(val_samples),
        "val_classes_represented": len(represented_val),
        "epochs_completed": NUM_EPOCHS,
        "selection_rule": "lowest_clean_validation_loss",
        "selected_epoch": best_epoch,
        "selected_val_loss": selected_val_loss,
        "selected_val_accuracy": selected_val_acc,
        "selected_macro_recall_represented_classes": macro_recall,
        "selected_macro_f1_represented_classes": macro_f1,
        "final_epoch_val_loss": final_val_loss,
        "final_epoch_val_accuracy": final_val_acc,
        "optimizer": "AdamW",
        "learning_rate_initial": LR,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR",
        "sampler": "inverse_sqrt_class_frequency",
        "preprocessing": "pad_to_square_then_resize",
        "v2_backbone_loaded": True,
        "locked_test_used": False,
        "class_names": class_names,
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print(" Selected checkpoint:", BEST_PATH)
    print(" Final-epoch checkpoint:", FINAL_PATH)
    print(" Metrics:", METRICS_PATH)
    print(" Confusion matrix:", CMAT_CSV_PATH)
    print(" Per-class metrics:", PERCLASS_CSV_PATH)
    print(" Train counts:", CLASS_COUNTS_PATH)
    print(" Summary:", SUMMARY_PATH)

    print("\nIMPORTANT:")
    print("The six protected museum source groups were not used.")
    print("Do not use them to tune V3 after this run; they are the final held-out benchmark.")


if __name__ == "__main__":
    main()
