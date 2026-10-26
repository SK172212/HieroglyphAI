import json
from pathlib import Path
import cv2

ANN_PATH = Path("glyph_det/annotations/instances_default.json")
IMG_DIR = Path("glyph_det/data/images/default")

with open(ANN_PATH, "r") as f:
    coco = json.load(f)

images_by_id = {img["id"]: img for img in coco["images"]}

first_img = coco["images"][0]
img_id = first_img["id"]
file_name = first_img["file_name"]

img_path = IMG_DIR / file_name
print("Using image:", file_name)
print("Image path:", img_path)

img =  cv2.imread(str(img_path), cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"Could not read image at {img_path}. Check filename/path.")

anns = [a for a in coco["annotations"] if a["image_id"] == img_id]
print("Boxes in this image:", len(anns))

for a in anns:
    x, y, w, h = a["bbox"]
    x1, y1 = int(round(x)), int(round(y))
    x2, y2 = int(round(x + w)), int(round(y + h))
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

OUT_PATH = Path("glyph_det/preview_first_image_boxes.jpg")
cv2.imwrite(str(OUT_PATH), img)

