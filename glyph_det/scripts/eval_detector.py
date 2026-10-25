import argparse
import json
from pathlib import Path

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


DATASET_NAME = "glyph_val_evaluation"


def setup_cfg(weights_path: str, score_thresh: float = 0.05):
    cfg = get_cfg()

    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = weights_path

    # Keep this low for COCO AP evaluation.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh

    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATASETS.TEST = (DATASET_NAME,)

    return cfg


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        required=True,
        help="Path to model_final.pth",
    )

    parser.add_argument(
        "--val-img-dir",
        default="glyph_det/splits/v3A/val/images",
    )

    parser.add_argument(
        "--val-json",
        default="glyph_det/splits/v3A/val/annotations.json",
    )

    parser.add_argument(
        "--out-dir",
        required=True,
    )

    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.05,
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    register_coco_instances(
        DATASET_NAME,
        {},
        args.val_json,
        args.val_img_dir,
    )

    cfg = setup_cfg(
        args.weights,
        args.score_thresh,
    )

    evaluator = COCOEvaluator(
        DATASET_NAME,
        output_dir=str(out_dir),
    )

    val_loader = build_detection_test_loader(
        cfg,
        DATASET_NAME,
    )

    model = DefaultTrainer.build_model(cfg)

    DetectionCheckpointer(model).load(
        cfg.MODEL.WEIGHTS
    )

    results = inference_on_dataset(
        model,
        val_loader,
        evaluator,
    )

    model_name = Path(args.weights).parent.name
    metrics_path = out_dir / f"{model_name}_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== COCO Evaluation Results ===")
    print(json.dumps(results, indent=2))
    print(f"\nSaved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()