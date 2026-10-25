import os
import random
import cv2

from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer

def main():
    TRAIN_IMG_DIR = "glyph_det/data/images/train/images"
    TRAIN_JSON    = "glyph_det/data/images/train/annotations.json"
    VAL_IMG_DIR   = "glyph_det/data/images/val/images"
    VAL_JSON      = "glyph_det/data/images/val/annotations.json"

    # Register
    register_coco_instances("glyph_train_v1", {}, TRAIN_JSON, TRAIN_IMG_DIR)
    register_coco_instances("glyph_val_v1", {}, VAL_JSON, VAL_IMG_DIR)

    # Load dataset dicts
    dataset = DatasetCatalog.get("glyph_train_v1")
    meta = MetadataCatalog.get("glyph_train_v1")

    print("Loaded train samples:", len(dataset))
    if len(dataset) == 0:
        raise RuntimeError("Train dataset is empty. Paths/JSON mismatch.")

    os.makedirs("output/check_dataset", exist_ok=True)

    # Visualize a few random samples
    for i, d in enumerate(random.sample(dataset, k=min(3, len(dataset)))):
        img = cv2.imread(d["file_name"])
        if img is None:
            raise RuntimeError(f"Could not read image: {d['file_name']}")

        v = Visualizer(img[:, :, ::-1], metadata=meta, scale=1.0)
        out = v.draw_dataset_dict(d)
        out_path = f"output/check_dataset/sample_{i}.png"
        cv2.imwrite(out_path, out.get_image()[:, :, ::-1])
        print("Wrote:", out_path)

if __name__ == "__main__":
    main()
