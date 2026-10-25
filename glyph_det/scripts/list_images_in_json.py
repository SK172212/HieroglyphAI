import json

with open("glyph_det/annotations/instances_default_v2.json", "r") as f:
    coco = json.load(f)

image_names = [img["file_name"] for img in coco["images"]]

print(f"Found {len(image_names)} images:\n")
for name in image_names:
    print(name)
