import json
from pathlib import Path


INPUT_JSON = Path(
    "glyph_det/annotations/instances_master_v3A.json"
)

OUTPUT_JSON = Path(
    "glyph_det/annotations/instances_master_v3A_clean.json"
)

MISSING_FILENAME = "det_01A.png"


def main():
    with INPUT_JSON.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    removed_image_ids = {
        image["id"]
        for image in images
        if image.get("file_name") == MISSING_FILENAME
    }

    if not removed_image_ids:
        print(f"No image entry found for {MISSING_FILENAME}")
        return

    cleaned_images = [
        image
        for image in images
        if image["id"] not in removed_image_ids
    ]

    cleaned_annotations = [
        annotation
        for annotation in annotations
        if annotation.get("image_id") not in removed_image_ids
    ]

    coco["images"] = cleaned_images
    coco["annotations"] = cleaned_annotations

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    print("Removed image IDs:", removed_image_ids)
    print("Images before:", len(images))
    print("Images after:", len(cleaned_images))
    print("Annotations before:", len(annotations))
    print("Annotations after:", len(cleaned_annotations))
    print("Wrote:", OUTPUT_JSON)


if __name__ == "__main__":
    main()