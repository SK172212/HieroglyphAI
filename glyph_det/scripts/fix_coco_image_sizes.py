import json
import cv2
from pathlib import Path


COCO_JSON = "glyph_det/splits/v2/train/annotations.json"
IMAGES_DIR = "glyph_det/splits/v2/train/images"
OUT_JSON = "glyph_det/splits/v2/train/annotations.json"


with open(COCO_JSON, "r") as f:
    coco = json.load(f)

images = coco["images"]
updated = 0

for img in images:
    fn = img["file_name"]
    img_path = Path(IMAGES_DIR) / fn

    if not img_path.exists():
        print(f"[WARN] Missing image: {fn}")
        continue

    im = cv2.imread(str(img_path))
    if im is None:
        print(f"[WARN] Could not read image: {fn}")
        continue

    h, w = im.shape[:2]

    if img["width"] != w or img["height"] != h:
        img["width"] = w
        img["height"] = h
        updated += 1

print(f"[INFO] Updated {updated} image size entries")

with open(OUT_JSON, "w") as f:
    json.dump(coco, f)

print(f"[DONE] Wrote fixed annotations to {OUT_JSON}")