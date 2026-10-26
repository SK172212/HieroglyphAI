from pathlib import Path
import csv
import shutil

import cv2
import torch
from PIL import Image
import numpy as np

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

BASE_DIR = Path(__file__).resolve().parents[2]
CLASSIFIER_DIR = BASE_DIR / "glyph_classification"

DETECTOR_WEIGHTS = (
    BASE_DIR
    / "glyph_det"
    / "experiments"
    / "v3A"
    / "model_final_v3A.pth"
)

TRAIN_SOURCE_DIR = (
    CLASSIFIER_DIR
    / "data"
    / "museum_split"
    / "train_sources"
)

TEST_SOURCE_DIR = (
    CLASSIFIER_DIR
    / "data"
    / "museum_split"
    / "locked_test_sources"
)

OUTPUT_ROOT = (
    CLASSIFIER_DIR
    / "data"
    / "museum_crops"
)

TRAIN_OUTPUT_DIR = OUTPUT_ROOT / "train_unlabeled"
TEST_OUTPUT_DIR = OUTPUT_ROOT / "locked_test_unlabeled"

DETECTOR_SCORE_THRESH = 0.60
DETECTOR_NMS_THRESH = 0.40

CROP_PADDING_RATIO = 0.10

MIN_W = 20
MIN_H = 20
MAX_AREA_FRAC = 0.25
MAX_H_FRAC = 0.60

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

def get_device():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)
    return device


def load_detector(device):
    if not DETECTOR_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Detector weights not found:\n{DETECTOR_WEIGHTS}"
        )

    cfg = get_cfg()

    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.MODEL.DEVICE = (
        "cuda" if device.type == "cuda" else "cpu"
    )

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = str(DETECTOR_WEIGHTS)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        DETECTOR_SCORE_THRESH
    )

    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = (
        DETECTOR_NMS_THRESH
    )

    predictor = DefaultPredictor(cfg)

    return predictor

def is_reasonable_box(
    x1,
    y1,
    x2,
    y2,
    img_w,
    img_h,
):
    box_w = x2 - x1
    box_h = y2 - y1

    if box_w < MIN_W or box_h < MIN_H:
        return False

    box_area = box_w * box_h
    img_area = img_w * img_h

    if box_area / max(1.0, img_area) > MAX_AREA_FRAC:
        return False

    if box_h / max(1.0, img_h) > MAX_H_FRAC:
        return False

    return True


def clamp_box(
    x1,
    y1,
    x2,
    y2,
    w,
    h,
):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))

    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))

    return x1, y1, x2, y2


def crop_with_padding(
    img_rgb,
    box_xyxy,
    pad_ratio,
):
    h, w = img_rgb.shape[:2]

    x1, y1, x2, y2 = box_xyxy

    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = box_w * pad_ratio
    pad_y = box_h * pad_ratio

    x1p = x1 - pad_x
    y1p = y1 - pad_y
    x2p = x2 + pad_x
    y2p = y2 + pad_y

    x1i, y1i, x2i, y2i = clamp_box(
        x1p,
        y1p,
        x2p,
        y2p,
        w,
        h,
    )

    if x2i <= x1i or y2i <= y1i:
        return None

    crop = img_rgb[
        y1i:y2i,
        x1i:x2i,
    ]

    if crop.size == 0:
        return None

    return Image.fromarray(crop)

def get_source_images(source_root):
    images = []

    for path in source_root.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in VALID_EXTENSIONS
        ):
            images.append(path)

    return sorted(images)


def artifact_group_from_path(
    image_path,
    source_root,
):
    relative = image_path.relative_to(source_root)

    parts = relative.parts

    if len(parts) < 3:
        raise ValueError(
            f"Unexpected source path structure: {relative}"
        )

    museum = parts[0]
    group = parts[1]

    return museum, group

def extract_split(
    predictor,
    source_root,
    output_root,
    split_name,
):
    images = get_source_images(source_root)

    print()
    print("=" * 60)
    print(split_name)
    print("=" * 60)
    print("Source photos:", len(images))

    manifest_rows = []

    total_raw_detections = 0
    total_saved_crops = 0
    total_rejected_boxes = 0

    for image_index, image_path in enumerate(
        images,
        start=1,
    ):
        museum, group = artifact_group_from_path(
            image_path,
            source_root,
        )

        img_bgr = cv2.imread(str(image_path))

        if img_bgr is None:
            print(
                f"[WARN] Could not read: {image_path}"
            )
            continue

        h, w = img_bgr.shape[:2]
        img_rgb = img_bgr[:, :, ::-1]

        outputs = predictor(img_bgr)

        instances = outputs["instances"].to("cpu")

        boxes = (
            instances
            .pred_boxes
            .tensor
            .numpy()
        )

        scores = (
            instances
            .scores
            .numpy()
        )

        total_raw_detections += len(boxes)

        image_output_dir = (
            output_root
            / museum
            / group
        )

        image_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        saved_for_image = 0

        for detection_index, (box, score) in enumerate(
            zip(boxes, scores),
            start=1,
        ):
            x1, y1, x2, y2 = box.tolist()

            reasonable = is_reasonable_box(
                x1,
                y1,
                x2,
                y2,
                w,
                h,
            )

            if not reasonable:
                total_rejected_boxes += 1
                continue

            crop = crop_with_padding(
                img_rgb,
                (x1, y1, x2, y2),
                CROP_PADDING_RATIO,
            )

            if crop is None:
                total_rejected_boxes += 1
                continue

            saved_for_image += 1
            total_saved_crops += 1

            crop_filename = (
                f"{museum}_"
                f"{group}_"
                f"{image_path.stem}_"
                f"crop_{saved_for_image:03d}.png"
            )

            crop_path = (
                image_output_dir
                / crop_filename
            )

            crop.save(crop_path)

            manifest_rows.append(
                {
                    "split": split_name,
                    "museum": museum,
                    "artifact_group": group,
                    "source_image": str(
                        image_path.relative_to(
                            CLASSIFIER_DIR
                        )
                    ),
                    "crop_file": str(
                        crop_path.relative_to(
                            CLASSIFIER_DIR
                        )
                    ),
                    "detector_score": float(score),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "source_width": w,
                    "source_height": h,
                }
            )

        print(
            f"[{image_index:03d}/{len(images):03d}] "
            f"{museum}/{group}/{image_path.name} "
            f"-> {saved_for_image} crops"
        )

    manifest_path = (
        output_root
        / f"{split_name}_crop_manifest.csv"
    )

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = [
            "split",
            "museum",
            "artifact_group",
            "source_image",
            "crop_file",
            "detector_score",
            "x1",
            "y1",
            "x2",
            "y2",
            "source_width",
            "source_height",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print("Raw detections:   ", total_raw_detections)
    print("Rejected boxes:   ", total_rejected_boxes)
    print("Saved crops:      ", total_saved_crops)
    print("Manifest:         ", manifest_path)

    return total_saved_crops

def main():
    device = get_device()
    predictor = load_detector(device)

    if OUTPUT_ROOT.exists():
        print(
            f"Removing previous crop output:\n"
            f"{OUTPUT_ROOT}"
        )
        shutil.rmtree(OUTPUT_ROOT)

    TRAIN_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEST_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_count = extract_split(
        predictor,
        TRAIN_SOURCE_DIR,
        TRAIN_OUTPUT_DIR,
        "train",
    )

    test_count = extract_split(
        predictor,
        TEST_SOURCE_DIR,
        TEST_OUTPUT_DIR,
        "locked_test",
    )

    print()
    print("=" * 60)
    print("ALL EXTRACTION COMPLETE")
    print("=" * 60)
    print("Training crops:    ", train_count)
    print("Locked test crops: ", test_count)
    print("Total crops:       ", train_count + test_count)
    print()
    print("Output:")
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()