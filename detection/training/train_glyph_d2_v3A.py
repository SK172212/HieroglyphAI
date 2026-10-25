import os

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo


def main():
    train_img_dir = "glyph_det/splits/v3A/train/images"
    train_json = "glyph_det/splits/v3A/train/annotations.json"

    val_img_dir = "glyph_det/splits/v3A/val/images"
    val_json = "glyph_det/splits/v3A/val/annotations.json"

    register_coco_instances(
        "glyph_train_v3A",
        {},
        train_json,
        train_img_dir,
    )

    register_coco_instances(
        "glyph_val_v3A",
        {},
        val_json,
        val_img_dir,
    )

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.MODEL.DEVICE = "cpu"

    cfg.DATASETS.TRAIN = ("glyph_train_v3A",)
    cfg.DATASETS.TEST = ("glyph_val_v3A",)

    cfg.DATALOADER.NUM_WORKERS = 0

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # Replace this path if your actual v2 model is elsewhere.
    cfg.MODEL.WEIGHTS = (
        "glyph_det/experiments/v2/model_final.pth"
    )

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00010
    cfg.SOLVER.MAX_ITER = 1500
    cfg.SOLVER.STEPS = (1000, 1300)
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = 250

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256

    cfg.TEST.EVAL_PERIOD = 250

    cfg.OUTPUT_DIR = "glyph_det/experiments/v3A"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)

    # Start a fresh v3A fine-tuning run from the v2 weights.
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
