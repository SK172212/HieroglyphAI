import json
from pathlib import Path

ANNOTATION_PATH = Path("glyph_det/annotations/instances_default.json")

with open(ANNOTATION_PATH, "r") as f:
    coco = json.load(f)

print("Loaded COCO annotation file ")
print("Top-level keys:", list(coco.keys()))

print("\n=== COUNTS ===")
print("Number of images:", len(coco["images"]))
print("Number of annotations (boxes):", len(coco["annotations"]))
print("Number of categories (classes):", len(coco["categories"]))

print("\n=== CATEGORIES ===")
for cat in coco["categories"]:
    print(f'id={cat["id"]} name={cat["name"]}')
