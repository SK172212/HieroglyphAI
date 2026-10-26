from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw, ImageFont
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

# Detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo


# ============================================================
# CONFIG
# ============================================================

DETECTOR_SCORE_THRESH = 0.40
DETECTOR_NMS_THRESH = 0.40
CROP_PAD_FRACTION = 0.10
IMG_SIZE = 224

LOCKED_GROUPS = [
    ("HMANE", "HMANE_003"),
    ("HMANE", "HMANE_004"),
    ("MFA", "MFA_003"),
    ("MFA", "MFA_007"),
    ("MFA", "MFA_013"),
    ("MFA", "MFA_014"),
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}

V2_CKPT_NAME = "resnet18_v2_frozenSplit_min6_12ep_rot10.pth"
V3_CKPT_NAME = "resnet18_v3_museum_aug_sqrtweighted_best_val_loss.pth"


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent          # classifier_v3/
PROJECT_ROOT = SCRIPT_DIR.parent                     # glyph_classification/

DETECTOR_PATH = PROJECT_ROOT.parent / "glyph_det" / "experiments" / "v3A" / "model_final_v3A.pth"
V2_PATH = PROJECT_ROOT / "models" / "v2" / V2_CKPT_NAME
V3_PATH = SCRIPT_DIR / "output" / V3_CKPT_NAME

MUSEUM_SPLIT_ROOT = PROJECT_ROOT / "data" / "museum_split"

OUTPUT_ROOT = SCRIPT_DIR / "locked_comparison"
ANNOTATED_ROOT = OUTPUT_ROOT / "annotated"
CROPS_ROOT = OUTPUT_ROOT / "crops"
CSV_PATH = OUTPUT_ROOT / "v2_v3_predictions.csv"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"


# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# V3 PREPROCESSING
# ============================================================

class PadToSquare:
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


def build_v2_transform(img_size):
    # Matches V2 inference/training behavior: direct square resize.
    return transforms.Compose([
        transforms.Resize(
            (img_size, img_size),
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_v3_transform(img_size):
    # Matches V3 validation/inference preprocessing.
    return transforms.Compose([
        PadToSquare(),
        transforms.Resize(
            (img_size, img_size),
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ============================================================
# LOAD CLASSIFIERS
# ============================================================

def load_classifier(checkpoint_path: Path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_names = checkpoint.get("class_names")
    if not class_names:
        class_to_idx = checkpoint.get("class_to_idx")
        if not class_to_idx:
            raise RuntimeError(
                f"Checkpoint has neither class_names nor class_to_idx: {checkpoint_path}"
            )
        class_names = [
            name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])
        ]

    img_size = int(checkpoint.get("img_size", IMG_SIZE))

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))

    state = checkpoint.get("model_state_dict")
    if state is None:
        # Fallback in case an older checkpoint is a raw state_dict.
        state = checkpoint

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, class_names, img_size


@torch.no_grad()
def classify_crop(model, class_names, transform, crop: Image.Image, device):
    x = transform(crop).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    top_prob, top_idx = torch.max(probs, dim=0)

    k = min(2, len(class_names))
    top_probs, top_idxs = torch.topk(probs, k=k)

    result = {
        "label": class_names[int(top_idx)],
        "confidence": float(top_prob),
        "top2": [
            (class_names[int(i)], float(p))
            for p, i in zip(top_probs.cpu(), top_idxs.cpu())
        ],
    }

    if k >= 2:
        result["margin"] = float(top_probs[0] - top_probs[1])
    else:
        result["margin"] = float(top_probs[0])

    return result


# ============================================================
# LOAD DETECTOR
# ============================================================

def load_detector():
    if not DETECTOR_PATH.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {DETECTOR_PATH}")

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.WEIGHTS = str(DETECTOR_PATH)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DETECTOR_SCORE_THRESH
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = DETECTOR_NMS_THRESH

    # Keep detector device behavior robust on Mac.
    # Detectron2 generally supports CPU reliably; CUDA if available.
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
    else:
        cfg.MODEL.DEVICE = "cpu"

    return DefaultPredictor(cfg)


def detect_once(predictor, pil_image: Image.Image):
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    outputs = predictor(bgr)
    instances = outputs["instances"].to("cpu")

    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()

    detections = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [float(v) for v in box]
        detections.append({
            "box": (x1, y1, x2, y2),
            "detector_confidence": float(score),
        })

    # Stable visual order: roughly top-to-bottom, then left-to-right.
    detections.sort(
        key=lambda d: (
            round(d["box"][1] / 50.0),
            d["box"][0],
        )
    )
    return detections


# ============================================================
# CROP + DRAWING
# ============================================================

def padded_crop(img: Image.Image, box, pad_fraction=CROP_PAD_FRACTION):
    x1, y1, x2, y2 = box
    w, h = img.size

    bw = x2 - x1
    bh = y2 - y1
    px = bw * pad_fraction
    py = bh * pad_fraction

    left = max(0, int(np.floor(x1 - px)))
    top = max(0, int(np.floor(y1 - py)))
    right = min(w, int(np.ceil(x2 + px)))
    bottom = min(h, int(np.ceil(y2 + py)))

    if right <= left or bottom <= top:
        raise ValueError(f"Invalid padded crop: {(left, top, right, bottom)}")

    return img.crop((left, top, right, bottom)), (left, top, right, bottom)


def get_font(size=18):
    # Use a built-in/default font if a scalable system font is unavailable.
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        p = Path(candidate)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def text_box(draw, xy, text, font, fill, text_fill="white"):
    x, y = xy
    bbox = draw.multiline_textbbox((x, y), text, font=font, spacing=2)
    pad = 4
    rect = (
        bbox[0] - pad,
        bbox[1] - pad,
        bbox[2] + pad,
        bbox[3] + pad,
    )
    draw.rectangle(rect, fill=fill)
    draw.multiline_text(
        (x, y),
        text,
        font=font,
        fill=text_fill,
        spacing=2,
    )


def annotate_image(image, results):
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = get_font(18)

    for item in results:
        x1, y1, x2, y2 = item["box"]
        agree = item["v2_label"] == item["v3_label"]

        # Green-ish for agreement, red-ish for disagreement.
        # These colors are only for visual review, not model semantics.
        outline = (40, 150, 70) if agree else (210, 50, 50)
        label_bg = (30, 110, 50) if agree else (160, 35, 35)

        draw.rectangle(
            (int(x1), int(y1), int(x2), int(y2)),
            outline=outline,
            width=4,
        )

        status = "AGREE" if agree else "V2 != V3"
        text = (
            f"#{item['detection_id']} {status}\n"
            f"V2: {item['v2_label']} {item['v2_confidence']:.0%}\n"
            f"V3: {item['v3_label']} {item['v3_confidence']:.0%}"
        )

        # Prefer above box; if too close to top, put inside/below top edge.
        tx = max(0, int(x1))
        ty = max(0, int(y1) - 62)
        text_box(draw, (tx, ty), text, font, label_bg)

    return annotated


# ============================================================
# FIND LOCKED SOURCE GROUPS
# ============================================================

def find_group_dir(museum: str, group: str) -> Path:
    """
    Supports the likely museum_split layouts without silently substituting
    another source. We only accept a directory whose name exactly matches
    the frozen group name.
    """
    candidates = [
        MUSEUM_SPLIT_ROOT / "locked_test_sources" / museum / group,
        MUSEUM_SPLIT_ROOT / "locked_test" / museum / group,
        MUSEUM_SPLIT_ROOT / museum / group,
        MUSEUM_SPLIT_ROOT / group,
    ]

    for p in candidates:
        if p.is_dir():
            return p

    matches = [
        p for p in MUSEUM_SPLIT_ROOT.rglob(group)
        if p.is_dir() and p.name == group
    ]

    # Prefer paths that visibly contain "locked".
    locked_matches = [p for p in matches if "locked" in str(p).lower()]
    if len(locked_matches) == 1:
        return locked_matches[0]
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Could not uniquely locate frozen group {museum}/{group} under "
        f"{MUSEUM_SPLIT_ROOT}. Matches found: {matches}"
    )


def list_images(folder: Path):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


# ============================================================
# MAIN
# ============================================================

def main():
    device = get_device()

    print("\n" + "=" * 78)
    print("LOCKED MUSEUM VISUAL COMPARISON: V2 vs V3")
    print("=" * 78)
    print("Classifier device:", device)
    print("Detector checkpoint:", DETECTOR_PATH)
    print("V2 checkpoint:", V2_PATH)
    print("V3 checkpoint:", V3_PATH)
    print("Detector runs ONCE per image; exact same crops go to V2 and V3.")
    print("This script does NOT train or modify either model.\n")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ANNOTATED_ROOT.mkdir(parents=True, exist_ok=True)
    CROPS_ROOT.mkdir(parents=True, exist_ok=True)

    # Fail before processing if any required model is missing.
    for p in [DETECTOR_PATH, V2_PATH, V3_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Required checkpoint missing: {p}")

    # Resolve all six groups before touching any images.
    group_dirs = {}
    for museum, group in LOCKED_GROUPS:
        folder = find_group_dir(museum, group)
        images = list_images(folder)
        if not images:
            raise RuntimeError(f"No source images found in {folder}")
        group_dirs[(museum, group)] = (folder, images)
        print(f"{group}: {len(images)} source image(s) -> {folder}")

    print("\nLoading detector...")
    detector = load_detector()

    print("Loading V2...")
    v2_model, v2_classes, v2_img_size = load_classifier(V2_PATH, device)
    v2_transform = build_v2_transform(v2_img_size)

    print("Loading V3...")
    v3_model, v3_classes, v3_img_size = load_classifier(V3_PATH, device)
    v3_transform = build_v3_transform(v3_img_size)

    print(f"V2 classes: {len(v2_classes)}")
    print(f"V3 classes: {len(v3_classes)}")

    rows = []
    total_images = 0
    total_detections = 0
    total_agreements = 0
    total_disagreements = 0

    for museum, group in LOCKED_GROUPS:
        source_dir, image_paths = group_dirs[(museum, group)]

        annotated_dir = ANNOTATED_ROOT / group
        crops_dir = CROPS_ROOT / group
        annotated_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- {group}: {len(image_paths)} source image(s) ---")

        for image_index, image_path in enumerate(image_paths, start=1):
            total_images += 1

            # Respect EXIF orientation before detection and drawing.
            with Image.open(image_path) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")

            detections = detect_once(detector, image)
            comparison_results = []

            for detection_id, det in enumerate(detections, start=1):
                crop, padded_box = padded_crop(image, det["box"])

                v2 = classify_crop(
                    v2_model, v2_classes, v2_transform, crop, device
                )
                v3 = classify_crop(
                    v3_model, v3_classes, v3_transform, crop, device
                )

                agree = v2["label"] == v3["label"]
                total_detections += 1
                if agree:
                    total_agreements += 1
                else:
                    total_disagreements += 1

                crop_name = (
                    f"{image_path.stem}__det_{detection_id:03d}.jpg"
                )
                crop_path = crops_dir / crop_name
                crop.save(crop_path, quality=95)

                result = {
                    "detection_id": detection_id,
                    "box": det["box"],
                    "padded_box": padded_box,
                    "detector_confidence": det["detector_confidence"],
                    "v2_label": v2["label"],
                    "v2_confidence": v2["confidence"],
                    "v2_margin": v2["margin"],
                    "v3_label": v3["label"],
                    "v3_confidence": v3["confidence"],
                    "v3_margin": v3["margin"],
                    "agree": agree,
                }
                comparison_results.append(result)

                rows.append({
                    "museum": museum,
                    "group": group,
                    "source_image": image_path.name,
                    "source_path": str(image_path),
                    "detection_id": detection_id,
                    "detector_confidence": f"{det['detector_confidence']:.6f}",
                    "x1": f"{det['box'][0]:.2f}",
                    "y1": f"{det['box'][1]:.2f}",
                    "x2": f"{det['box'][2]:.2f}",
                    "y2": f"{det['box'][3]:.2f}",
                    "crop_path": str(crop_path),
                    "v2_label": v2["label"],
                    "v2_confidence": f"{v2['confidence']:.6f}",
                    "v2_margin": f"{v2['margin']:.6f}",
                    "v3_label": v3["label"],
                    "v3_confidence": f"{v3['confidence']:.6f}",
                    "v3_margin": f"{v3['margin']:.6f}",
                    "agree": agree,
                    # Blank column intentionally included so you can optionally
                    # add the true sign later without changing the raw outputs.
                    "true_label_optional": "",
                })

            annotated = annotate_image(image, comparison_results)
            out_name = f"{image_path.stem}__V2_vs_V3.jpg"
            out_path = annotated_dir / out_name
            annotated.save(out_path, quality=95)

            disagreements = sum(
                1 for r in comparison_results if not r["agree"]
            )
            print(
                f"[{image_index:02d}/{len(image_paths):02d}] "
                f"{image_path.name}: "
                f"{len(detections)} detections, "
                f"{disagreements} disagreement(s)"
            )

    fieldnames = [
        "museum",
        "group",
        "source_image",
        "source_path",
        "detection_id",
        "detector_confidence",
        "x1", "y1", "x2", "y2",
        "crop_path",
        "v2_label",
        "v2_confidence",
        "v2_margin",
        "v3_label",
        "v3_confidence",
        "v3_margin",
        "agree",
        "true_label_optional",
    ]

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    agreement_rate = (
        total_agreements / total_detections
        if total_detections else 0.0
    )

    summary = {
        "source_groups": [group for _, group in LOCKED_GROUPS],
        "source_images": total_images,
        "detected_crops": total_detections,
        "v2_v3_agreements": total_agreements,
        "v2_v3_disagreements": total_disagreements,
        "agreement_rate": agreement_rate,
        "detector_score_threshold": DETECTOR_SCORE_THRESH,
        "detector_nms_threshold": DETECTOR_NMS_THRESH,
        "crop_pad_fraction": CROP_PAD_FRACTION,
        "v2_checkpoint": str(V2_PATH),
        "v3_checkpoint": str(V3_PATH),
        "v2_preprocessing": "direct_resize_to_square",
        "v3_preprocessing": "pad_to_square_then_resize",
        "note": (
            "Agreement is not accuracy. True correctness requires human "
            "inspection or labels."
        ),
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("COMPARISON COMPLETE")
    print("=" * 78)
    print(f"Source images:       {total_images}")
    print(f"Detected crops:      {total_detections}")
    print(f"V2/V3 agreements:   {total_agreements}")
    print(f"V2/V3 disagreements:{total_disagreements}")
    print(f"Agreement rate:      {agreement_rate:.1%}")
    print("\nAnnotated images:")
    print(ANNOTATED_ROOT)
    print("\nIndividual detector crops:")
    print(CROPS_ROOT)
    print("\nPrediction CSV:")
    print(CSV_PATH)
    print("\nIMPORTANT: agreement rate is NOT accuracy.")
    print("Use your knowledge of the true glyphs to judge which prediction is correct.")


if __name__ == "__main__":
    main()