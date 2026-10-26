from pathlib import Path
import csv
import re
import shutil

BASE_DIR = Path(__file__).resolve().parents[2]

MAPPING_PATH = (
    BASE_DIR
    / "analysis"
    / "crop_contact_sheets"
    / "train"
    / "train_crop_mapping.csv"
)

REJECT_PATH = (
    BASE_DIR
    / "analysis"
    / "train_rejected_ids.txt"
)

NEEDS_CROP_PATH = (
    BASE_DIR
    / "analysis"
    / "train_needs_cropping_ids.txt"
)

SALVAGED_DIR = (
    BASE_DIR
    / "analysis"
    / "needs_cropping"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "museum_curated"
    / "train_unlabeled"
)

MANIFEST_PATH = (
    BASE_DIR
    / "data"
    / "museum_curated"
    / "train_curated_manifest.csv"
)


def load_id_file(path):
    ids = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if re.fullmatch(r"T\d{4}", line):
                ids.add(line)

    return ids


def load_mapping():
    mapping = {}

    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            mapping[row["crop_id"]] = row

    return mapping


def extract_crop_id(filename):
    match = re.match(r"(T\d{4})", filename)

    if not match:
        return None

    return match.group(1)


def extract_variant(filename):
    """
    Examples:

    T1788__MFA...png        -> A
    T1788_B_MFA...copy.png  -> B
    T1788_C_MFA...copy.png  -> C
    """

    match = re.match(
        r"T\d{4}_([A-Z])_",
        filename
    )

    if match:
        return match.group(1)

    return "A"


def main():
    mapping = load_mapping()

    rejected = load_id_file(REJECT_PATH)
    needs_crop = load_id_file(NEEDS_CROP_PATH)

    overlap = rejected & needs_crop

    if overlap:
        raise RuntimeError(
            f"IDs appear in both REJECT and NEEDS CROPPING: "
            f"{sorted(overlap)}"
        )

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest_rows = []

    keep_as_is_count = 0
    salvaged_count = 0

    # ========================================================
    # 1. COPY KEEP-AS-IS ORIGINAL CROPS
    # ========================================================

    for crop_id, row in sorted(mapping.items()):

        if crop_id in rejected:
            continue

        if crop_id in needs_crop:
            continue

        source_path = (
            BASE_DIR
            / row["crop_path"]
        )

        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing original crop: {source_path}"
            )

        museum = row["museum"]
        artifact_group = row["artifact_group"]

        dest_dir = (
            OUTPUT_DIR
            / museum
            / artifact_group
        )

        dest_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        destination = (
            dest_dir
            / f"{crop_id}.png"
        )

        shutil.copy2(
            source_path,
            destination
        )

        keep_as_is_count += 1

        manifest_rows.append(
            {
                "crop_id": crop_id,
                "variant": "A",
                "museum": museum,
                "artifact_group": artifact_group,
                "status": "keep_as_is",
                "original_crop_path": row["crop_path"],
                "curated_path": str(
                    destination.relative_to(BASE_DIR)
                ),
            }
        )

    # ========================================================
    # 2. COPY MANUALLY SALVAGED / RECROPPED IMAGES
    # ========================================================

    salvaged_files = sorted(
        p for p in SALVAGED_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
    )

    seen_names = set()

    for source_path in salvaged_files:

        crop_id = extract_crop_id(
            source_path.name
        )

        if crop_id is None:
            print(
                "[WARN] Could not determine T-ID:",
                source_path.name
            )
            continue

        if crop_id not in mapping:
            raise RuntimeError(
                f"{crop_id} not found in mapping."
            )

        row = mapping[crop_id]

        museum = row["museum"]
        artifact_group = row["artifact_group"]

        variant = extract_variant(
            source_path.name
        )

        clean_name = (
            f"{crop_id}_{variant}.png"
        )

        # Guard against accidental duplicate variant names.
        key = (
            museum,
            artifact_group,
            clean_name
        )

        if key in seen_names:
            raise RuntimeError(
                f"Duplicate curated name detected: {key}"
            )

        seen_names.add(key)

        dest_dir = (
            OUTPUT_DIR
            / museum
            / artifact_group
        )

        dest_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        destination = (
            dest_dir
            / clean_name
        )

        # Preserve edited image bytes.
        shutil.copy2(
            source_path,
            destination
        )

        salvaged_count += 1

        manifest_rows.append(
            {
                "crop_id": crop_id,
                "variant": variant,
                "museum": museum,
                "artifact_group": artifact_group,
                "status": "salvaged_recrop",
                "original_crop_path": row["crop_path"],
                "curated_path": str(
                    destination.relative_to(BASE_DIR)
                ),
            }
        )

    # ========================================================
    # 3. WRITE MANIFEST
    # ========================================================

    with open(
        MANIFEST_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        fieldnames = [
            "crop_id",
            "variant",
            "museum",
            "artifact_group",
            "status",
            "original_crop_path",
            "curated_path",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    total = (
        keep_as_is_count
        + salvaged_count
    )

    print()
    print("=" * 60)
    print("FINAL MUSEUM TRAINING POOL")
    print("=" * 60)
    print(
        "KEEP-AS-IS crops: ",
        keep_as_is_count
    )
    print(
        "Salvaged recrops: ",
        salvaged_count
    )
    print(
        "TOTAL:            ",
        total
    )
    print()
    print(
        "Output:",
        OUTPUT_DIR
    )
    print(
        "Manifest:",
        MANIFEST_PATH
    )


if __name__ == "__main__":
    main()