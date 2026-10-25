import os
import cv2

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo


def main():
    
    PROJECT_ROOT = "/Users/sohankilaru/Desktop/hieroglyphProject"
    WEIGHTS = os.path.join(PROJECT_ROOT, "detectron2", "output_glyph", "model_final.pth")
    IMAGE = os.path.join(PROJECT_ROOT, "detectron2", "glyph_coco_split", "val", "images", "PUT_AN_IMAGE_FILENAME_HERE.jpg")
    OUT = os.path.join(PROJECT_ROOT, "detectron2", "pred_vis.jpg")
    

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = WEIGHTS
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cpu"  

    predictor = DefaultPredictor(cfg)

    img = cv2.imread(IMAGE)
    if img is None:
        raise FileNotFoundError(
            f"Could not read image: {IMAGE}\n"
            f"Double-check the filename exists and is a .jpg/.png."
        )

    outputs = predictor(img)

    meta = MetadataCatalog.get("__unused__")
    meta.thing_classes = ["glyph"]

    vis = Visualizer(img[:, :, ::-1], metadata=meta, scale=1.0)
    out_img = vis.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1]

    cv2.imwrite(OUT, out_img)
    print("Wrote:", OUT)


if __name__ == "__main__":
    main()
