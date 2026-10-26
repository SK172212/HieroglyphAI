from pathlib import Path
import csv
import shutil

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    BASE_DIR
    / "data"
    / "museum_curated"
    / "train_unlabeled"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "v2"
    / "resnet18_v2_frozenSplit_min6_12ep_rot10.pth"
)

OUTPUT_CSV = (
    BASE_DIR
    / "analysis"
    / "v2_museum_predictions.csv"
)

SORTED_DIR = (
    BASE_DIR
    / "analysis"
    / "v2_prelabel_by_class"
)


def load_model(device):
    ckpt = torch.load(
        MODEL_PATH,
        map_location=device
    )

    class_names = ckpt["class_names"]
    img_size = ckpt.get("img_size", 224)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names)
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.to(device)
    model.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize(
                (img_size, img_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return model, class_names, preprocess


def get_images():
    valid_exts = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".tif",
        ".tiff",
    }

    return sorted(
        p for p in INPUT_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in valid_exts
    )


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    model, class_names, preprocess = (
        load_model(device)
    )

    images = get_images()

    print(
        "Museum crops found:",
        len(images)
    )

    if SORTED_DIR.exists():
        shutil.rmtree(SORTED_DIR)

    SORTED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    for i, image_path in enumerate(
        images,
        start=1
    ):
        img = Image.open(
            image_path
        ).convert("RGB")

        x = (
            preprocess(img)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(
                logits,
                dim=1
            ).squeeze(0)

        top_probs, top_indices = (
            torch.topk(probs, k=2)
        )

        pred1 = class_names[
            int(top_indices[0])
        ]

        pred2 = class_names[
            int(top_indices[1])
        ]

        conf1 = float(
            top_probs[0].item()
        )

        conf2 = float(
            top_probs[1].item()
        )

        margin = conf1 - conf2

        relative_path = (
            image_path.relative_to(
                BASE_DIR
            )
        )

        # Copy into predicted-class folder
        class_dir = (
            SORTED_DIR
            / pred1
        )

        class_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        destination = (
            class_dir
            / image_path.name
        )

        shutil.copy2(
            image_path,
            destination
        )

        rows.append(
            {
                "filename": image_path.name,
                "source_path": str(
                    relative_path
                ),
                "predicted_label": pred1,
                "confidence": conf1,
                "second_label": pred2,
                "second_confidence": conf2,
                "margin": margin,
            }
        )

        if (
            i % 100 == 0
            or i == len(images)
        ):
            print(
                f"{i}/{len(images)}"
            )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        fieldnames = [
            "filename",
            "source_path",
            "predicted_label",
            "confidence",
            "second_label",
            "second_confidence",
            "margin",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 60)
    print("V2 PRE-LABELING COMPLETE")
    print("=" * 60)
    print("Images:", len(rows))
    print("CSV:", OUTPUT_CSV)
    print(
        "Sorted copies:",
        SORTED_DIR
    )


if __name__ == "__main__":
    main()