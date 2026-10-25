import json
from pathlib import Path


OLD_JSON = Path(
    "glyph_det/annotations/instances_default_v2.json"
)

NEW_JSON = Path(
    "glyph_det/annotations/instances_default_v3A.json"
)

OUT_JSON = Path(
    "glyph_det/annotations/instances_master_v3A.json"
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def main() -> None:
    old = load_json(OLD_JSON)
    new = load_json(NEW_JSON)

    old_images = old.get("images", [])
    old_annotations = old.get("annotations", [])

    new_images = new.get("images", [])
    new_annotations = new.get("annotations", [])

    # Use one consistent category definition.
    old_categories = old.get("categories", [])
    new_categories = new.get("categories", [])

    if old_categories != new_categories:
        print("[WARN] Category definitions are not identical.")
        print("Old categories:", old_categories)
        print("New categories:", new_categories)

    # Start new IDs after the largest existing old IDs.
    max_image_id = max(
        (image["id"] for image in old_images),
        default=0,
    )

    max_annotation_id = max(
        (annotation["id"] for annotation in old_annotations),
        default=0,
    )

    image_id_map = {}

    renumbered_new_images = []

    for offset, image in enumerate(new_images, start=1):
        old_new_image_id = image["id"]
        safe_image_id = max_image_id + offset

        image_id_map[old_new_image_id] = safe_image_id

        updated_image = image.copy()
        updated_image["id"] = safe_image_id

        renumbered_new_images.append(updated_image)

    # Renumber annotations and update their image_id values.
    renumbered_new_annotations = []

    for offset, annotation in enumerate(
        new_annotations,
        start=1,
    ):
        updated_annotation = annotation.copy()

        updated_annotation["id"] = (
            max_annotation_id + offset
        )

        original_image_id = annotation["image_id"]

        if original_image_id not in image_id_map:
            raise KeyError(
                "Annotation refers to an image ID that "
                f"does not exist: {original_image_id}"
            )

        updated_annotation["image_id"] = (
            image_id_map[original_image_id]
        )

        renumbered_new_annotations.append(
            updated_annotation
        )

    merged = {}

    for key in ["info", "licenses"]:
        if key in old:
            merged[key] = old[key]

    merged["categories"] = (
        old_categories
        if old_categories
        else new_categories
    )

    merged["images"] = (
        old_images
        + renumbered_new_images
    )

    merged["annotations"] = (
        old_annotations
        + renumbered_new_annotations
    )

    save_json(merged, OUT_JSON)

    print("[DONE] COCO files merged")
    print(
        "Old:",
        len(old_images),
        "images,",
        len(old_annotations),
        "annotations",
    )
    print(
        "New:",
        len(new_images),
        "images,",
        len(new_annotations),
        "annotations",
    )
    print(
        "Merged:",
        len(merged["images"]),
        "images,",
        len(merged["annotations"]),
        "annotations",
    )
    print("Wrote:", OUT_JSON)


if __name__ == "__main__":
    main()