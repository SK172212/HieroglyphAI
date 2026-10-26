import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer


# ============================================================
# MODEL PATHS
# ============================================================

DEFAULT_DETECTOR_WEIGHTS = "models/model_final_v3A.pth"

DEFAULT_CLASSIFIER_WEIGHTS = (
    "models/resnet18_v3_museum_aug_sqrtweighted_best_val_loss.pth"
)


# ============================================================
# CONFIDENCE SETTINGS
# ============================================================

# Keep the app's detector threshold unchanged.
DETECTOR_SCORE_THRESH = 0.40

# A classification is displayed only if:
# 1. The top confidence is at least 0.75.
# 2. The top class beats the second class by at least 0.20.
CLASSIFIER_CONF_THRESH = 0.75
CLASSIFIER_MARGIN_THRESH = 0.20

# Lowering NMS from 0.40 to 0.30 suppresses more duplicate boxes.
DETECTOR_NMS_THRESH = 0.30


# ============================================================
# CROP AND BOX FILTER SETTINGS
# ============================================================

# Padding placed around detector boxes before classification.
CROP_PADDING_RATIO = 0.10

# Filters for extremely small noise boxes and unusually large boxes.
MIN_W = 20
MIN_H = 20
MAX_AREA_FRAC = 0.25
MAX_H_FRAC = 0.60


# ============================================================
# DRAWING SETTINGS
# ============================================================

# OpenCV uses BGR colors.
BOX_COLOR = (0, 255, 0)
LABEL_COLOR = (0, 0, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.60
FONT_THICKNESS = 2


def get_device() -> torch.device:
    """
    Select CUDA when available; otherwise use the CPU.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    return device


def load_detector(
    detector_weights_path: str,
    device: torch.device,
):
    """
    Load the Detectron2 glyph detector.
    """
    detector_path = Path(detector_weights_path)

    if not detector_path.exists():
        raise FileNotFoundError(
            f"Detector weights not found: {detector_weights_path}"
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
    cfg.MODEL.WEIGHTS = str(detector_path)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        DETECTOR_SCORE_THRESH
    )

    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = (
        DETECTOR_NMS_THRESH
    )

    predictor = DefaultPredictor(cfg)

    return predictor, cfg


class PadToSquare:
    """
    Match V3 inference preprocessing: preserve aspect ratio by padding
    the crop to a square using the mean color of its four corners.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size

        if w == h:
            return img

        corners = [
            img.getpixel((0, 0)),
            img.getpixel((w - 1, 0)),
            img.getpixel((0, h - 1)),
            img.getpixel((w - 1, h - 1)),
        ]

        background = tuple(
            int(sum(pixel[channel] for pixel in corners) / len(corners))
            for channel in range(3)
        )

        size = max(w, h)
        canvas = Image.new("RGB", (size, size), background)
        canvas.paste(
            img,
            ((size - w) // 2, (size - h) // 2),
        )

        return canvas


def load_classifier(
    classifier_weights_path: str,
    device: torch.device,
):
    """
    Load the ResNet-18 hieroglyph classifier.
    """
    classifier_path = Path(classifier_weights_path)

    if not classifier_path.exists():
        raise FileNotFoundError(
            f"Classifier weights not found: "
            f"{classifier_weights_path}"
        )

    checkpoint = torch.load(
        classifier_path,
        map_location=device,
    )

    class_names = checkpoint["class_names"]
    img_size = checkpoint.get("img_size", 224)

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    # V3 was trained/validated with aspect-ratio-preserving
    # pad-to-square preprocessing before the 224x224 resize.
    preprocess = transforms.Compose(
        [
            PadToSquare(),
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


@st.cache_resource(show_spinner="Loading AI models...")
def get_models():
    """
    Load and cache the detector and classifier for Streamlit.

    A Render redeploy or local Streamlit restart clears this cache.
    """
    device = get_device()

    detector, _ = load_detector(
        DEFAULT_DETECTOR_WEIGHTS,
        device,
    )

    classifier, class_names, preprocess = load_classifier(
        DEFAULT_CLASSIFIER_WEIGHTS,
        device,
    )

    return (
        device,
        detector,
        classifier,
        class_names,
        preprocess,
    )


def is_reasonable_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
) -> bool:
    """
    Return True if a detector box has a reasonable glyph-like size.

    This removes:
    - Invalid boxes
    - Very small noise boxes
    - Boxes covering an unusually large part of the image
    - Boxes spanning most of the image height
    """
    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return False

    if box_w < MIN_W or box_h < MIN_H:
        return False

    box_area = box_w * box_h
    img_area = img_w * img_h

    if (
        box_area / max(1.0, img_area)
        > MAX_AREA_FRAC
    ):
        return False

    if (
        box_h / max(1.0, img_h)
        > MAX_H_FRAC
    ):
        return False

    return True


def clamp_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    w: int,
    h: int,
):
    """
    Clamp bounding-box coordinates to the image boundaries.
    """
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))

    return x1, y1, x2, y2


def crop_with_padding(
    img_rgb: np.ndarray,
    box_xyxy,
    pad_ratio: float,
) -> Image.Image:
    """
    Crop one detected region and add proportional padding.
    """
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

    # Prevent empty crops.
    if x2i <= x1i or y2i <= y1i:
        x2i = min(w, x1i + 1)
        y2i = min(h, y1i + 1)

    crop = img_rgb[
        y1i:y2i,
        x1i:x2i,
    ]

    return Image.fromarray(crop)


def classify_crop(
    pil_crop: Image.Image,
    model,
    preprocess,
    class_names,
    device: torch.device,
):
    """
    Classify one detector crop.

    A known class is returned only when:
    - Top confidence is at least 0.75
    - Top confidence exceeds second-place confidence by at least 0.20

    Otherwise, the crop is marked as UNK.
    """
    x = (
        preprocess(pil_crop)
        .unsqueeze(0)
        .to(device)
    )

    with torch.inference_mode():
        logits = model(x)
        probabilities = torch.softmax(
            logits,
            dim=1,
        ).squeeze(0)

        number_of_results = min(
            2,
            len(class_names),
        )

        top_probs, top_indices = torch.topk(
            probabilities,
            k=number_of_results,
        )

    raw_label = class_names[
        int(top_indices[0].item())
    ]

    raw_conf = float(
        top_probs[0].item()
    )

    if number_of_results > 1:
        second_label = class_names[
            int(top_indices[1].item())
        ]

        second_conf = float(
            top_probs[1].item()
        )
    else:
        second_label = None
        second_conf = 0.0

    confidence_margin = (
        raw_conf - second_conf
    )

    passes_confidence = (
        raw_conf >= CLASSIFIER_CONF_THRESH
    )

    passes_margin = (
        confidence_margin
        >= CLASSIFIER_MARGIN_THRESH
    )

    if passes_confidence and passes_margin:
        shown_label = raw_label
    else:
        shown_label = "UNK"

    return {
        "raw_label": raw_label,
        "raw_conf": raw_conf,
        "second_label": second_label,
        "second_conf": second_conf,
        "confidence_margin": confidence_margin,
        "shown_label": shown_label,
    }


def draw_label_safe(
    img_bgr,
    label: str,
    x1i: int,
    y1i: int,
    x2i: int,
    y2i: int,
):
    """
    Draw a label near the box without allowing it to leave the canvas.
    """
    h, w = img_bgr.shape[:2]

    (text_w, text_h), baseline = cv2.getTextSize(
        label,
        FONT,
        FONT_SCALE,
        FONT_THICKNESS,
    )

    text_x = x1i
    text_y = y1i - 8

    # Move below the box if there is no space above.
    if text_y - text_h - baseline < 0:
        text_y = (
            y2i
            + text_h
            + baseline
            + 6
        )

    if text_y > h - 1:
        text_y = h - 1 - baseline

    if text_x + text_w > w - 1:
        text_x = max(
            0,
            (w - 1) - text_w,
        )

    cv2.putText(
        img_bgr,
        label,
        (text_x, text_y),
        FONT,
        FONT_SCALE,
        LABEL_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA,
    )


def analyze_image(image_path):
    """
    Detect glyphs, classify accepted detector crops, save outputs,
    and return results for the Streamlit app.
    """
    (
        device,
        detector,
        classifier,
        class_names,
        preprocess,
    ) = get_models()

    image_path = Path(image_path)
    image_stem = image_path.stem
    out_dir = image_path.parent

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    img_bgr = cv2.imread(
        str(image_path)
    )

    if img_bgr is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    h, w = img_bgr.shape[:2]

    print(
        f"Loaded image: {image_path} "
        f"({w}x{h})"
    )

    # Run detector.
    outputs = detector(img_bgr)
    all_instances = outputs["instances"].to("cpu")

    all_boxes = (
        all_instances
        .pred_boxes
        .tensor
        .numpy()
    )

    all_det_scores = (
        all_instances
        .scores
        .numpy()
    )

    print(
        f"Raw detections: {len(all_boxes)} "
        f"(score >= {DETECTOR_SCORE_THRESH})"
    )

    results = []

    annot_bgr = img_bgr.copy()
    img_rgb = img_bgr[:, :, ::-1]

    accepted_instance_indices = []
    accepted_count = 0
    rejected_count = 0

    for original_index, (box, det_score) in enumerate(
        zip(all_boxes, all_det_scores),
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
            rejected_count += 1

            print(
                f"  #{original_index:02d} "
                f"det={det_score:.3f} "
                f"REJECTED by box-size filter"
            )

            results.append(
                {
                    "id": original_index,
                    "bbox_xyxy": [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ],
                    "det_score": float(det_score),
                    "accepted_box": False,
                    "rejection_reason": (
                        "box_size_filter"
                    ),
                }
            )

            continue

        accepted_count += 1

        # Save the zero-based Detectron2 instance index.
        accepted_instance_indices.append(
            original_index - 1
        )

        pil_crop = crop_with_padding(
            img_rgb,
            (x1, y1, x2, y2),
            CROP_PADDING_RATIO,
        )

        classification = classify_crop(
            pil_crop,
            classifier,
            preprocess,
            class_names,
            device,
        )

        raw_label = classification[
            "raw_label"
        ]

        raw_conf = classification[
            "raw_conf"
        ]

        second_label = classification[
            "second_label"
        ]

        second_conf = classification[
            "second_conf"
        ]

        confidence_margin = classification[
            "confidence_margin"
        ]

        shown_label = classification[
            "shown_label"
        ]

        if shown_label == "UNK":
            print(
                f"  #{original_index:02d} "
                f"det={det_score:.3f} -> UNK "
                f"(top={raw_label} "
                f"{raw_conf:.3f}, "
                f"second={second_label} "
                f"{second_conf:.3f}, "
                f"margin={confidence_margin:.3f})"
            )
        else:
            print(
                f"  #{original_index:02d} "
                f"det={det_score:.3f} -> "
                f"{shown_label} "
                f"conf={raw_conf:.3f}, "
                f"margin={confidence_margin:.3f}"
            )

        x1i, y1i, x2i, y2i = clamp_box(
            x1,
            y1,
            x2,
            y2,
            w,
            h,
        )

        # Draw every accepted detector box, including unknown crops.
        cv2.rectangle(
            annot_bgr,
            (x1i, y1i),
            (x2i, y2i),
            BOX_COLOR,
            2,
        )

        # Only draw a Gardiner label when classification passes
        # both the confidence and margin tests.
        if shown_label != "UNK":
            label_text = (
                f"{shown_label} "
                f"{raw_conf:.2f}"
            )

            draw_label_safe(
                annot_bgr,
                label_text,
                x1i,
                y1i,
                x2i,
                y2i,
            )

        results.append(
            {
                "id": original_index,
                "bbox_xyxy": [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                ],
                "det_score": float(det_score),
                "accepted_box": True,
                "raw_label": raw_label,
                "raw_conf": raw_conf,
                "second_label": second_label,
                "second_conf": second_conf,
                "confidence_margin": (
                    confidence_margin
                ),
                "shown_label": shown_label,
                "cls_conf_thresh": (
                    CLASSIFIER_CONF_THRESH
                ),
                "cls_margin_thresh": (
                    CLASSIFIER_MARGIN_THRESH
                ),
            }
        )

    print(
        f"Accepted detections: "
        f"{accepted_count}"
    )

    print(
        f"Rejected by box-size filter: "
        f"{rejected_count}"
    )

    # Build a detection-only visualization containing only accepted boxes.
    if accepted_instance_indices:
        accepted_instances = all_instances[
            accepted_instance_indices
        ]
    else:
        accepted_instances = all_instances[
            []
        ]

    visualizer = Visualizer(
        img_rgb,
        metadata=MetadataCatalog.get(
            "__unused__"
        ),
        scale=1.0,
    )

    out_vis = visualizer.draw_instance_predictions(
        accepted_instances
    )

    det_vis_path = (
        out_dir
        / f"{image_stem}_det.png"
    )

    cv2.imwrite(
        str(det_vis_path),
        out_vis.get_image()[:, :, ::-1],
    )

    pipe_img_path = (
        out_dir
        / f"{image_stem}_detcls.png"
    )

    cv2.imwrite(
        str(pipe_img_path),
        annot_bgr,
    )

    json_path = (
        out_dir
        / f"{image_stem}_detcls.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    annot_rgb = annot_bgr[:, :, ::-1]

    return results, annot_rgb


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Detect hieroglyphs in an inscription image "
            "and classify each detected glyph."
        )
    )

    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input inscription image.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    analyze_image(args.image_path)


if __name__ == "__main__":
    main()