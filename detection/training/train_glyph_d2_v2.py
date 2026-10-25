import os
from pathlib import Path

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo


def main():
    
    TRAIN_IMG_DIR = "glyph_det/splits/v2/train/images"
    TRAIN_JSON    = "glyph_det/splits/v2/train/annotations.json"

    VAL_IMG_DIR = "glyph_det/splits/v2/val/images"
    VAL_JSON    = "glyph_det/splits/v2/val/annotations.json"

    register_coco_instances("glyph_train_v2", {}, TRAIN_JSON, TRAIN_IMG_DIR)
    register_coco_instances("glyph_val_v2", {}, VAL_JSON, VAL_IMG_DIR)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
 
    cfg.MODEL.DEVICE = "cpu"

    cfg.DATASETS.TRAIN = ("glyph_train_v2",)
    cfg.DATASETS.TEST  = ("glyph_val_v2",)

    cfg.DATALOADER.NUM_WORKERS = 0

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.MODEL.WEIGHTS = "glyph_det/experiments/v1/model_final.pth"

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # ---- Training hyperparams ----
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00010          
    cfg.SOLVER.MAX_ITER = 1500            
    cfg.SOLVER.STEPS = (1000, 1300)       
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = 250

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256

    cfg.TEST.EVAL_PERIOD = 250

    cfg.OUTPUT_DIR = "glyph_det/output/output_glyph_v2"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=True)
    trainer.train()

if __name__ == "__main__":
    main()
