import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator


class TrainerWithEval(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
        os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=output_folder)


def main():
    TRAIN_IMG_DIR = "glyph_det/data/train/images"
    TRAIN_JSON    = "glyph_det/data/train/annotations.json"

    VAL_IMG_DIR = "glyph_det/data/val/images"
    VAL_JSON    = "glyph_det/data/val/annotations.json"

    register_coco_instances("glyph_train", {}, TRAIN_JSON, TRAIN_IMG_DIR)
    register_coco_instances("glyph_val", {}, VAL_JSON, VAL_IMG_DIR)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    cfg.MODEL.DEVICE = "cpu"

    cfg.DATASETS.TRAIN = ("glyph_train",)
    cfg.DATASETS.TEST = ("glyph_val",)

    cfg.DATALOADER.NUM_WORKERS = 0

    
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025

    
    cfg.SOLVER.MAX_ITER = 2000
    cfg.SOLVER.STEPS = (1500, 1800)   
    cfg.SOLVER.GAMMA = 0.1

    cfg.TEST.EVAL_PERIOD = 200

    cfg.SOLVER.CHECKPOINT_PERIOD = 500

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    cfg.OUTPUT_DIR = "./detectron2/output_glyph_v2"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    

    trainer = TrainerWithEval(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()