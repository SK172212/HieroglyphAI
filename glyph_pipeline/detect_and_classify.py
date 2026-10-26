# detect_and_classify.py
# DEBUG: print(sys.executable) if env issues ever return
#TO RUN THIS SCRIPT: python glyph_pipeline/detect_and_classify.py glyph_pipeline/test/ptest_(num).png

import json
import argparse
from pathlib import Path

import torch
from torchvision import transforms, models
from torch import nn

from PIL import Image
import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

DEFAULT_DETECTOR_WEIGHTS = "glyph_det/experiments/v2/model_final.pth"
DEFAULT_CLASSIFIER_WEIGHTS = "glyph_classification/models/v2/resnet18_v2_frozenSplit_min6_12ep_rot10.pth"

DETECTOR_SCORE_THRESH = 0.6
CLASSIFIER_CONF_THRESH = 0.40

# Crop padding around detector boxes before classification
CROP_PADDING_RATIO = 0.10

# Heuristic filters to reduce tiny noise boxes and huge region boxes
MIN_W = 20
MIN_H = 20
MAX_AREA_FRAC = 0.25
MAX_H_FRAC = 0.60

# Drawing settings (BGR colors)
BOX_COLOR = (0, 255, 0)
LABEL_COLOR = (0, 0, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2


def is_reasonable_box(x1, y1, x2, y2, img_w, img_h) -> bool:
    """
    Returns True if a detection box is likely to be a real glyph.
    Filters out tiny noise boxes and giant region boxes.
    """
    box_w = x2 - x1
    box_h = y2 - y1

    # Reject invalid or tiny boxes
    if box_w < MIN_W or box_h < MIN_H:
        return False

    box_area = box_w * box_h
    img_area = img_w * img_h

    # Reject boxes that cover too much of the image
    if (box_area / max(1.0, img_area)) > MAX_AREA_FRAC:
        return False

    # Reject boxes that are too tall (region-level detections)
    if (box_h / max(1.0, img_h)) > MAX_H_FRAC:
        return False

    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect hieroglyphs in an inscription image and classify each glyph."
    )

    parser.add_argument("image_path", type=str, help="Path to input inscription image")

    parser.add_argument(
        "--detector-weights",
        type=str,
        default=DEFAULT_DETECTOR_WEIGHTS,
        help="Path to trained Detectron2 detector weights",
    )

    parser.add_argument(
        "--classifier-weights",
        type=str,
        default=DEFAULT_CLASSIFIER_WEIGHTS,
        help="Path to trained classification model weights (.pth)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="glyph_pipeline/output",
        help="Directory to save outputs (a subfolder per image will be created).",
    )

    return parser.parse_args()


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_detector(detector_weights_path: str, device: torch.device):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.DEVICE = "cuda" if device.type == "cuda" else "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = detector_weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DETECTOR_SCORE_THRESH
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4

    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def load_classifier(classifier_weights_path: str, device: torch.device):
    ckpt = torch.load(classifier_weights_path, map_location=device)

    class_names = ckpt["class_names"]
    img_size = ckpt.get("img_size", 224)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return model, class_names, preprocess


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    return x1, y1, x2, y2


def crop_with_padding(img_rgb: np.ndarray, box_xyxy, pad_ratio: float) -> Image.Image:
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = box_xyxy

    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio

    x1p = x1 - pad_x
    y1p = y1 - pad_y
    x2p = x2 + pad_x
    y2p = y2 + pad_y

    x1i, y1i, x2i, y2i = clamp_box(x1p, y1p, x2p, y2p, w, h)

    # Guard: avoid empty crops
    if x2i <= x1i or y2i <= y1i:
        # fallback: minimal safe crop
        x2i = min(w, x1i + 1)
        y2i = min(h, y1i + 1)

    crop = img_rgb[y1i:y2i, x1i:x2i]
    return Image.fromarray(crop)


def classify_crop(pil_crop: Image.Image, model, preprocess, class_names, device):
    x = preprocess(pil_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        conf, idx = torch.max(probs, dim=0)

    raw_label = class_names[int(idx)]
    raw_conf = float(conf.item())

    shown_label = raw_label
    if raw_conf < CLASSIFIER_CONF_THRESH:
        shown_label = "UNK"
        
    return raw_label, raw_conf, shown_label


def draw_label_safe(img_bgr, label: str, x1i: int, y1i: int, x2i: int, y2i: int):
    """
    Place label near the box but keep it on-canvas (no cutoff).
    """
    h, w = img_bgr.shape[:2]

    (text_w, text_h), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)

    # Start above top-left
    text_x = x1i
    text_y = y1i - 8

    # If above goes off top, move below the box
    if text_y - text_h - baseline < 0:
        text_y = y2i + text_h + baseline + 6

    # Clamp vertically
    if text_y > h - 1:
        text_y = h - 1 - baseline

    # Clamp horizontally
    if text_x + text_w > w - 1:
        text_x = max(0, (w - 1) - text_w)

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


def main():
    args = parse_args()
    device = get_device()

    # Output subfolder named after the image
    base_out_dir = Path(args.output_dir)
    image_path = Path(args.image_path)
    image_stem = image_path.stem
    out_dir = base_out_dir / image_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise RuntimeError(f"Could not read image: {args.image_path}")

    h, w = img_bgr.shape[:2]
    print(f"Loaded image: {args.image_path} ({w}x{h})")

    # Load models
    detector, _ = load_detector(args.detector_weights, device)
    classifier, class_names, preprocess = load_classifier(args.classifier_weights, device)

    # Detect
    outputs = detector(img_bgr)
    instances = outputs["instances"].to("cpu")

    boxes = instances.pred_boxes.tensor.numpy()
    det_scores = instances.scores.numpy()

    print(f"Detections: {len(boxes)} (score >= {DETECTOR_SCORE_THRESH})")
    for j, (box, s) in enumerate(zip(boxes, det_scores), start=1):
        x1, y1, x2, y2 = box.tolist()
        print(f"  #{j:02d} det={s:.3f} box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

    # Classify each detection
    results = []
    annot_bgr = img_bgr.copy()
    img_rgb = img_bgr[:, :, ::-1]  # for PIL crops

    for j, (box, det_score) in enumerate(zip(boxes, det_scores), start=1):
        x1, y1, x2, y2 = box.tolist()

        reasonable = is_reasonable_box(x1, y1, x2, y2, w, h)

        pil_crop = crop_with_padding(img_rgb, (x1, y1, x2, y2), CROP_PADDING_RATIO)
        raw_label, raw_conf, shown_label = classify_crop(
            pil_crop, classifier, preprocess, class_names, device
        )

        if shown_label == "UNK":
            print(f"  #{j:02d} det={det_score:.3f} -> UNK (raw={raw_label} {raw_conf:.3f})")
        else:
            print(f"  #{j:02d} det={det_score:.3f} -> {shown_label} cls={raw_conf:.3f}")

        # Draw box + label
        x1i, y1i, x2i, y2i = clamp_box(x1, y1, x2, y2, w, h)
        cv2.rectangle(annot_bgr, (x1i, y1i), (x2i, y2i), BOX_COLOR, 2)

        if shown_label != "UNK":
            label_text = f"{shown_label} {raw_conf:.2f}"
            draw_label_safe(annot_bgr, label_text, x1i, y1i, x2i, y2i)

        results.append(
            {
                "id": j,
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "det_score": float(det_score),
                "raw_label": raw_label,
                "raw_conf": raw_conf,
                "shown_label": shown_label,
                "cls_conf_thresh": CLASSIFIER_CONF_THRESH,
            }
        )

    # Save detection-only viz (Detectron2 Visualizer)
    v = Visualizer(img_rgb, metadata=MetadataCatalog.get("__unused__"), scale=1.0)
    out_vis = v.draw_instance_predictions(instances)
    det_vis_path = out_dir / f"{image_stem}_det.png"
    cv2.imwrite(str(det_vis_path), out_vis.get_image()[:, :, ::-1])
    print("Wrote detection viz:", det_vis_path)

    # Save pipeline viz (our drawn boxes + labels)
    pipe_img_path = out_dir / f"{image_stem}_detcls.png"
    cv2.imwrite(str(pipe_img_path), annot_bgr)
    print("Wrote pipeline viz:", pipe_img_path)

    # Save JSON
    json_path = out_dir / f"{image_stem}_detcls.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Wrote JSON:", json_path)


if __name__ == "__main__":
    main()