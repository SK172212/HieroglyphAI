import os
import cv2

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

def main():
    
    register_coco_instances(
        "glyph_train",
        {},
        "glyph_det/data/images/train/annotations.json",
        "glyph_det/data/images/train/images",
    )

    
    TEST_IMG = "glyph_det/data/images/val/images/det_03C.png" #change file path to selected test image

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = "detectron2/output_glyph_v1/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50  #change this to change confidence level
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3   # try 0.3 then 0.4


    predictor = DefaultPredictor(cfg)

    img = cv2.imread(TEST_IMG)
    if img is None:
        raise RuntimeError(f"Could not read image: {TEST_IMG}")

    outputs = predictor(img)

    meta = MetadataCatalog.get("glyph_train")
    v = Visualizer(img[:, :, ::-1], metadata=meta, scale=1.0)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    os.makedirs("output/predict", exist_ok=True)
    out_path = "output/predict/pred_03C_0.50_2.png"   #change this when testing on individual images
    cv2.imwrite(out_path, out.get_image()[:, :, ::-1])
    print("Wrote:", out_path)
    print("Detections:", len(outputs["instances"]))

if __name__ == "__main__":
    main()
