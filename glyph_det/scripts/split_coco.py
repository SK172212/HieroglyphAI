import json
import random
import shutil
from pathlib import Path

COCO_JSON = "glyph_det/annotations/instances_master_v3A.json"  
IMAGES_DIR = "glyph_det/images_all"                             
OUT_DIR = "glyph_det/splits/v3A"                                 

TRAIN_RATIO = 0.85
RANDOM_SEED = 42
ON_MISSING = "error"  


def load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, p: Path):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def split_coco(coco: dict, train_ratio: float, seed: int):
    images = coco.get("images", [])
    anns = coco.get("annotations", [])

    rng = random.Random(seed)
    images_shuffled = images[:]
    rng.shuffle(images_shuffled)

    n_train = int(len(images_shuffled) * train_ratio)
    train_images = images_shuffled[:n_train]
    val_images = images_shuffled[n_train:]

    train_image_ids = set(img["id"] for img in train_images)
    val_image_ids = set(img["id"] for img in val_images)

    train_anns = [a for a in anns if a.get("image_id") in train_image_ids]
    val_anns = [a for a in anns if a.get("image_id") in val_image_ids]

    def make_split(images_split, anns_split):
        out = {}
        # keep common COCO fields if present
        for k in ["info", "licenses", "categories"]:
            if k in coco:
                out[k] = coco[k]
        out["images"] = images_split
        out["annotations"] = anns_split
        return out

    return make_split(train_images, train_anns), make_split(val_images, val_anns)


def copy_images(images, src_dir: Path, dst_dir: Path, on_missing: str = "warn"):
    ensure_dir(dst_dir)
    missing = 0

    for img in images:
        fn = img.get("file_name")
        if not fn:
            continue

        src = src_dir / fn
        dst = dst_dir / fn
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            missing += 1
            if on_missing == "error":
                raise FileNotFoundError(f"Missing image: {src}")
            elif on_missing == "warn":
                print(f"[WARN] Missing image: {src}")
            elif on_missing == "ignore":
                pass
            continue

        shutil.copy2(src, dst)

    if missing and on_missing != "ignore":
        print(f"[INFO] Missing {missing} images (see warnings above).")


def main():
    coco_json = Path(COCO_JSON)
    images_dir = Path(IMAGES_DIR)
    out_dir = Path(OUT_DIR)

    if not coco_json.exists():
        raise FileNotFoundError(f"COCO_JSON not found: {coco_json.resolve()}")
    if not images_dir.exists():
        raise FileNotFoundError(f"IMAGES_DIR not found: {images_dir.resolve()}")

    coco = load_json(str(coco_json))

    # Basic sanity
    if "categories" not in coco or len(coco.get("categories", [])) == 0:
        print("[WARN] No categories found in COCO. Is this the right export?")

    print(f"[INFO] Master COCO: {coco_json}")
    print(f"[INFO] Images dir:  {images_dir}")
    print(f"[INFO] Out dir:     {out_dir}")
    print(f"[INFO] Images:      {len(coco.get('images', []))}")
    print(f"[INFO] Annotations: {len(coco.get('annotations', []))}")
    print(f"[INFO] Categories:  {len(coco.get('categories', []))}")
    print(f"[INFO] Split:       {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)} (train/val)")
    print(f"[INFO] Seed:        {RANDOM_SEED}")

    train_coco, val_coco = split_coco(coco, TRAIN_RATIO, RANDOM_SEED)

    train_dir = out_dir / "train"
    val_dir = out_dir / "val"

    ensure_dir(train_dir / "images")
    ensure_dir(val_dir / "images")

    train_json = train_dir / "annotations.json"
    val_json = val_dir / "annotations.json"

    save_json(train_coco, train_json)
    save_json(val_coco, val_json)

    copy_images(train_coco["images"], images_dir, train_dir / "images", on_missing=ON_MISSING)
    copy_images(val_coco["images"], images_dir, val_dir / "images", on_missing=ON_MISSING)

    print("\n[DONE]")
    print(f"Train: {len(train_coco['images'])} images, {len(train_coco['annotations'])} anns -> {train_json}")
    print(f"Val:   {len(val_coco['images'])} images, {len(val_coco['annotations'])} anns -> {val_json}")
    print(f"Images copied into:\n  {train_dir/'images'}\n  {val_dir/'images'}")


if __name__ == "__main__":
    main()