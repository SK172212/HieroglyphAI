import json
import random
import shutil
from pathlib import Path

SEED = 42
VAL_FRACTION = 0.20           
MIN_IMAGES_PER_CLASS = 6      
MIN_VAL_PER_CLASS = 1         
MIN_TRAIN_PER_CLASS = 5       
FREEZE_SPLIT = True           
RESET_SPLIT = False           

PROJECT_ROOT = Path(__file__).resolve().parents[1]  
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
VAL_DIR = PROJECT_ROOT / "data" / "val"
MANIFEST_PATH = PROJECT_ROOT / "data" / "split_manifest.json"

V1_TRAIN_LIST = PROJECT_ROOT / "models" / "train_files_v1.txt"

def load_v1_train_set():
    if not V1_TRAIN_LIST.exists():
        print("[WARN] train_files_v1.txt not found. No leakage filtering applied.")
        return set()
    
    with open(V1_TRAIN_LIST, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    v1_train_set = set()
    for line in lines:
        p = Path(line)

        try:
            rel = p.relative_to(RAW_DIR)
        except ValueError:
            rel = Path(line)
        
        v1_train_set.add(str(rel))
    
    print(f"[INFO] Loaded {len(v1_train_set)} v1 training images (will exclude from val).")
    return v1_train_set


def list_images_in_class(class_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    files = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files

def safe_clear_dir(d: Path):
    """Delete and recreate a directory."""
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

def copy_files(pairs):
    """pairs: list of (src_path, dst_path)"""
    for src, dst in pairs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build_split_from_scratch():
    """
    Returns:
      manifest dict with keys: seed, val_fraction, classes, train, val
      where train/val are lists of relative paths from RAW_DIR (class/file.ext)
    """
    v1_train_set = load_v1_train_set()

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW_DIR not found: {RAW_DIR}")

    rng = random.Random(SEED)

    class_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    class_dirs.sort(key=lambda p: p.name)

    train_rel = []
    val_rel = []
    classes_info = {}
    skipped_classes = {}

    for cdir in class_dirs:
        imgs = list_images_in_class(cdir)
        n = len(imgs)

        if n < MIN_IMAGES_PER_CLASS:
            if n > 0:
                skipped_classes[cdir.name] = n
            continue

        n_val = int(round(n * VAL_FRACTION))
        n_val = max(MIN_VAL_PER_CLASS, n_val)

        max_val_allowed = n - MIN_TRAIN_PER_CLASS
        if n_val > max_val_allowed:
            n_val = max_val_allowed

        if n_val < MIN_VAL_PER_CLASS:
            n_val = MIN_VAL_PER_CLASS
        if n_val >= n:
            n_val = n - 1

        imgs_shuf = imgs[:]
        rng.shuffle(imgs_shuf)

        imgs_rel = [str(p.relative_to(RAW_DIR)) for p in imgs_shuf]

        val_imgs = []
        train_imgs = []

        for p_obj, rel_path in zip(imgs_shuf, imgs_rel):

            if len(val_imgs) < n_val and rel_path not in v1_train_set:
                val_imgs.append(p_obj)
            else:
                train_imgs.append(p_obj)

        val_rel_c = [str(p.relative_to(RAW_DIR)) for p in val_imgs]
        train_rel_c = [str(p.relative_to(RAW_DIR)) for p in train_imgs]

        val_rel.extend(val_rel_c)
        train_rel.extend(train_rel_c)

        classes_info[cdir.name] = {
            "total": n,
            "val": len(val_rel_c),
            "train": len(train_rel_c),
        }

    manifest = {
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "min_images_per_class": MIN_IMAGES_PER_CLASS,
        "min_val_per_class": MIN_VAL_PER_CLASS,
        "min_train_per_class": MIN_TRAIN_PER_CLASS,
        "raw_dir": str(RAW_DIR),
        "train_dir": str(TRAIN_DIR),
        "val_dir": str(VAL_DIR),
        "classes": classes_info,
        "skipped_classes": skipped_classes,  
        "train": sorted(train_rel),
        "val": sorted(val_rel),
    }
    return manifest


def load_or_create_manifest():
    """
    If FREEZE_SPLIT and manifest exists, reuse it (unless RESET_SPLIT=True).
    Otherwise, create a new split and save it.
    """
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MANIFEST_PATH.exists() and FREEZE_SPLIT and not RESET_SPLIT:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        print(f"[INFO] Using existing frozen split manifest: {MANIFEST_PATH}")
        return manifest

    manifest = build_split_from_scratch()
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Wrote split manifest: {MANIFEST_PATH}")
    return manifest


def materialize_split(manifest):
    """
    Copies files from RAW_DIR into TRAIN_DIR and VAL_DIR preserving class folders.
    """
    safe_clear_dir(TRAIN_DIR)
    safe_clear_dir(VAL_DIR)

    train_pairs = [(RAW_DIR / rel, TRAIN_DIR / rel) for rel in manifest["train"]]
    val_pairs = [(RAW_DIR / rel, VAL_DIR / rel) for rel in manifest["val"]]

    missing = [str(src) for src, _ in (train_pairs + val_pairs) if not src.exists()]
    if missing:
        print("[WARN] Some files listed in manifest are missing from RAW_DIR.")
        print("[WARN] First few missing:")
        for m in missing[:10]:
            print("  ", m)
        raise FileNotFoundError("Missing files in RAW_DIR. Fix filenames and re-run.")

    copy_files(train_pairs)
    copy_files(val_pairs)


def print_summary(manifest):
    classes = manifest.get("classes", {})
    skipped = manifest.get("skipped_classes", {})
    num_classes = len(classes)
    total_train = len(manifest["train"])
    total_val = len(manifest["val"])
    total = total_train + total_val

    print("\n===== SPLIT SUMMARY =====")
    print(f"Classes kept (>= {manifest['min_images_per_class']} imgs): {num_classes}")
    print(f"Total images kept: {total}")
    print(f"Train: {total_train}")
    print(f"Val:   {total_val}")
    if total > 0:
        print(f"Val fraction overall: {total_val/total:.3f}")

    if skipped:
        print(f"\nClasses skipped (< {manifest['min_images_per_class']} imgs): {len(skipped)}")
        
        for cname, n in sorted(skipped.items(), key=lambda kv: kv[1])[:10]:
            print(f"  {cname}: {n} images")

    if classes:
        smallest = sorted(classes.items(), key=lambda kv: kv[1]["total"])[:10]
        print("\nSmallest kept classes (up to 10):")
        for cname, info in smallest:
            print(f"  {cname}: total={info['total']} train={info['train']} val={info['val']}")
    print("=========================\n")


def main():
    print(f"[INFO] RAW_DIR:    {RAW_DIR}")
    print(f"[INFO] TRAIN_DIR:  {TRAIN_DIR}")
    print(f"[INFO] VAL_DIR:    {VAL_DIR}")
    print(f"[INFO] MANIFEST:   {MANIFEST_PATH}")

    manifest = load_or_create_manifest()
    materialize_split(manifest)
    print_summary(manifest)

    print("[DONE] Split created.")

if __name__ == "__main__":
    main()