import cv2
import os

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

WEIGHTS_PATH = "glyph_det/experiments/v2/model_final.pth"
INPUT_DIR = "glyph_det/qualitative_test_images"
OUTPUT_DIR = "glyph_det/output/visualize/v2"
SCORE_THRESH = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = WEIGHTS_PATH
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH

predictor = DefaultPredictor(cfg)

metadata = MetadataCatalog.get("__unused")
metadata.thing_classes = ["glyph"]

for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith((".png")):
        continue

    img_path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(img_path)

    outputs = predictor(img)

    v = Visualizer(
        img[:, :, ::-1],
        metadata=metadata,
        scale=1.0
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    out_path = os.path.join(OUTPUT_DIR, fname)
    cv2.imwrite(out_path, v.get_image()[:, :, ::-1])

print("Done!")

