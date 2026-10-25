from pathlib import Path

import cv2
import numpy as np

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer


PROJECT_ROOT = Path("/Users/sohankilaru/Desktop/hieroglyphProject")


V1_WEIGHTS = (
    PROJECT_ROOT / "glyph_det/experiments/v1/model_final.pth"
)
V3A_WEIGHTS = (
    PROJECT_ROOT / "glyph_det/experiments/v3A/model_final_v3A.pth"
)

IMAGE_PATH = PROJECT_ROOT / "glyph_det/test_images/test_01.jpeg"

OUTPUT_DIR = PROJECT_ROOT / "glyph_det/output/v1_vs_v3A"

SCORE_THRESHOLD = 0.50


def build_predictor(weights_path: Path) -> DefaultPredictor:
    cfg = get_cfg()

    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
    cfg.MODEL.DEVICE = "cpu"

    return DefaultPredictor(cfg)


def visualize_predictions(image_bgr, outputs, metadata):
    visualizer = Visualizer(
        image_bgr[:, :, ::-1],
        metadata=metadata,
        scale=1.0,
    )

    result = visualizer.draw_instance_predictions(
        outputs["instances"].to("cpu")
    )

    return result.get_image()[:, :, ::-1]


def add_title(image, title):
    title_height = 60

    titled = cv2.copyMakeBorder(
        image,
        title_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    cv2.putText(
        titled,
        title,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return titled


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in [V1_WEIGHTS, V3A_WEIGHTS, IMAGE_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise ValueError(f"OpenCV could not read: {IMAGE_PATH}")

    metadata = MetadataCatalog.get("glyph_visual_comparison")
    metadata.thing_classes = ["glyph"]

    print("Loading Version 1 detector...")
    v1_predictor = build_predictor(V1_WEIGHTS)

    print("Loading Version 3A detector...")
    v3a_predictor = build_predictor(V3A_WEIGHTS)

    print("Running inference on the same museum image...")
    v1_outputs = v1_predictor(image)
    v3a_outputs = v3a_predictor(image)

    v1_count = len(v1_outputs["instances"])
    v3a_count = len(v3a_outputs["instances"])

    v1_visual = visualize_predictions(
        image, v1_outputs, metadata
    )

    v3a_visual = visualize_predictions(
        image, v3a_outputs, metadata
    )

    v1_visual = add_title(
        v1_visual,
        f"V1: {v1_count} detection",
    )

    v3a_visual = add_title(
        v3a_visual,
        f"V3A: {v3a_count} detections",
    )

    v1_path = OUTPUT_DIR / "v1_prediction.jpg"
    v3a_path = OUTPUT_DIR / "v3A_prediction.jpg"
    comparison_path = OUTPUT_DIR / "v1_vs_v3A.jpg"

    cv2.imwrite(str(v1_path), v1_visual)
    cv2.imwrite(str(v3a_path), v3a_visual)

    comparison = np.hstack([
        v1_visual,
        v3a_visual,
    ])

    cv2.imwrite(str(comparison_path), comparison)

    print("\n--- DETECTOR COMPARISON ---")
    print(f"Test image: {IMAGE_PATH.name}")
    print(f"Confidence threshold: {SCORE_THRESHOLD}")
    print(f"V1 detections: {v1_count}")
    print(f"V3A detections: {v3a_count}")

    print("\nSaved files:")
    print(v1_path)
    print(v3a_path)
    print(comparison_path)


if __name__ == "__main__":
    main()
