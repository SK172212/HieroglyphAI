from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import shutil

# ============================================================
# CONFIG
# ============================================================

MIN_TOTAL_IMAGES = 6
MIN_TRAIN_IMAGES = 5

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

# Old clean classifier dataset + frozen V2 split
OLD_RAW_DIR = BASE_DIR / "data" / "raw"
OLD_SPLIT_MANIFEST = BASE_DIR / "data" / "split_manifest.json"

# Reviewed museum classifier crops
MUSEUM_CURATED_DIR = (
    BASE_DIR / "data" / "museum_curated" / "train_unlabeled"
)

# Frozen museum source split
MUSEUM_SPLIT_DIR = BASE_DIR / "data" / "museum_split"
MUSEUM_TRAIN_SOURCES = MUSEUM_SPLIT_DIR / "train_sources"
MUSEUM_LOCKED_SOURCES = MUSEUM_SPLIT_DIR / "locked_test_sources"

# Museum labeling/review files
ANALYSIS_DIR = BASE_DIR / "analysis" / "v3"
PREDICTIONS_CSV = ANALYSIS_DIR / "v2_museum_predictions.csv"
CORRECTIONS_FILE = ANALYSIS_DIR / "museum_label_corrections.txt"

# Final V3 output
V3_DIR = BASE_DIR / "classifier_v3"
DATA_DIR = V3_DIR / "data"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
LOCKED_TEST_DIR = DATA_DIR / "locked_test"
ARCHIVED_DIR = DATA_DIR / "archived"
REJECTED_DIR = DATA_DIR / "rejected"

MANIFEST_DIR = V3_DIR / "manifests"


# ============================================================
# HELPERS
# ============================================================

def get_images(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def crop_id(value):
    return Path(str(value).strip()).stem


def unique_destination(folder, filename):
    destination = folder / filename

    if not destination.exists():
        return destination

    stem = Path(filename).stem
    suffix = Path(filename).suffix

    i = 2

    while True:
        candidate = folder / f"{stem}__{i}{suffix}"

        if not candidate.exists():
            return candidate

        i += 1


def copy_record(record, split, destination_root):

    if split == "rejected":
        destination_folder = destination_root
    else:
        destination_folder = destination_root / record["label"]

    destination_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    if record["source_type"] == "museum":

        filename = (
            f"{record['museum']}__"
            f"{record['source_group']}__"
            f"{record['path'].name}"
        )

    else:

        filename = (
            f"old__{record['label']}__"
            f"{record['path'].name}"
        )

    destination = unique_destination(
        destination_folder,
        filename
    )

    shutil.copy2(
        record["path"],
        destination
    )

    return {
        "split": split,
        "label": record["label"],
        "source_type": record["source_type"],
        "museum": record.get("museum", ""),
        "source_group": record.get("source_group", ""),
        "crop_id": record["crop_id"],
        "original_path": str(
            record["path"].relative_to(BASE_DIR)
        ),
        "v3_path": str(
            destination.relative_to(BASE_DIR)
        ),
    }


def write_manifest(path, rows):

    fields = [
        "split",
        "label",
        "source_type",
        "museum",
        "source_group",
        "crop_id",
        "original_path",
        "v3_path",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# START
# ============================================================

print("\n" + "=" * 72)
print("FINAL V3 CLASSIFIER DATASET BUILDER")
print("=" * 72)

print(f"Project root: {BASE_DIR}")
print(f"V3 output:    {V3_DIR}")


# ============================================================
# 1. REQUIRED INPUTS
# ============================================================

print("\n1. Checking required inputs...")

required = [
    OLD_RAW_DIR,
    OLD_SPLIT_MANIFEST,
    MUSEUM_CURATED_DIR,
    MUSEUM_TRAIN_SOURCES,
    MUSEUM_LOCKED_SOURCES,
    PREDICTIONS_CSV,
    CORRECTIONS_FILE,
]

for path in required:

    if not path.exists():
        raise SystemExit(
            f"ERROR: missing {path}"
        )

print("   Required inputs: PASS")


# ============================================================
# 2. OLD CLEAN DATA
# ============================================================

print("\n2. Loading old clean classifier data...")

old_records = []

for class_dir in sorted(
    OLD_RAW_DIR.iterdir()
):

    if not class_dir.is_dir():
        continue

    label = class_dir.name

    for p in get_images(class_dir):

        old_records.append({
            "path": p,
            "label": label,
            "source_type": "old_clean",
            "museum": "",
            "source_group": "",
            "crop_id": p.stem,
        })


print(
    f"   Old clean images: "
    f"{len(old_records)}"
)


# ============================================================
# 3. OLD FROZEN V2 SPLIT
# ============================================================

print("\n3. Loading frozen V2 clean split...")

with OLD_SPLIT_MANIFEST.open(
    encoding="utf-8"
) as f:

    old_split = json.load(f)


old_train_paths = set(
    old_split["train"]
)

old_val_paths = set(
    old_split["val"]
)


if old_train_paths & old_val_paths:

    raise SystemExit(
        "ERROR: overlap in old V2 train/val split."
    )


print(
    f"   Frozen V2 train: "
    f"{len(old_train_paths)}"
)

print(
    f"   Frozen V2 val:   "
    f"{len(old_val_paths)}"
)


# ============================================================
# 4. FROZEN MUSEUM GROUP ASSIGNMENTS
# ============================================================

print(
    "\n4. Loading frozen museum source assignments..."
)

museum_train_groups = set()
museum_locked_groups = set()


for museum_dir in sorted(
    MUSEUM_TRAIN_SOURCES.iterdir()
):

    if not museum_dir.is_dir():
        continue

    for group_dir in sorted(
        museum_dir.iterdir()
    ):

        if group_dir.is_dir():

            museum_train_groups.add(
                (
                    museum_dir.name,
                    group_dir.name
                )
            )


for museum_dir in sorted(
    MUSEUM_LOCKED_SOURCES.iterdir()
):

    if not museum_dir.is_dir():
        continue

    for group_dir in sorted(
        museum_dir.iterdir()
    ):

        if group_dir.is_dir():

            museum_locked_groups.add(
                (
                    museum_dir.name,
                    group_dir.name
                )
            )


if museum_train_groups & museum_locked_groups:

    raise SystemExit(
        "ERROR: museum group assigned to both "
        "train and locked test."
    )


print(
    f"   Museum training groups: "
    f"{len(museum_train_groups)}"
)

print(
    f"   Untouched locked groups: "
    f"{len(museum_locked_groups)}"
)


# ============================================================
# 5. CURATED MUSEUM CROPS
# ============================================================

print(
    "\n5. Loading reviewed museum classifier crops..."
)

museum_paths = get_images(
    MUSEUM_CURATED_DIR
)

museum_by_id = defaultdict(list)


for p in museum_paths:
    museum_by_id[p.stem].append(p)


duplicate_ids = {
    cid: paths
    for cid, paths in museum_by_id.items()
    if len(paths) > 1
}


if duplicate_ids:

    print(
        "\nERROR: duplicate museum crop IDs:"
    )

    for cid, paths in sorted(
        duplicate_ids.items()
    ):

        print(cid)

        for p in paths:
            print(
                "   ",
                p.relative_to(BASE_DIR)
            )

    raise SystemExit(
        "Resolve duplicate museum IDs first."
    )


print(
    f"   Curated museum crops: "
    f"{len(museum_paths)}"
)


# ============================================================
# 6. VERIFY CURATED CROPS ARE TRAIN-SOURCE ONLY
# ============================================================

print(
    "\n6. Verifying museum provenance..."
)

curated_groups = set()


for p in museum_paths:

    rel = p.relative_to(
        MUSEUM_CURATED_DIR
    )

    if len(rel.parts) < 3:

        raise SystemExit(
            f"ERROR: unexpected museum path: {rel}"
        )

    museum = rel.parts[0]
    group = rel.parts[1]

    curated_groups.add(
        (museum, group)
    )


locked_contamination = (
    curated_groups & museum_locked_groups
)


if locked_contamination:

    raise SystemExit(
        "ERROR: locked museum groups were found "
        "inside curated training crops: "
        f"{sorted(locked_contamination)}"
    )


unknown_groups = (
    curated_groups - museum_train_groups
)


if unknown_groups:

    raise SystemExit(
        "ERROR: curated crops contain groups not "
        "assigned to museum training: "
        f"{sorted(unknown_groups)}"
    )


print(
    f"   Curated source groups: "
    f"{len(curated_groups)}"
)

print(
    "   Locked-test contamination: 0"
)

print(
    "   Museum provenance: PASS"
)


# ============================================================
# 7. V2 MUSEUM PRELABELS
# ============================================================

print(
    "\n7. Loading V2 museum prelabels..."
)

predictions = {}


with PREDICTIONS_CSV.open(
    newline="",
    encoding="utf-8-sig"
) as f:

    reader = csv.DictReader(f)

    if not reader.fieldnames:

        raise SystemExit(
            "ERROR: prediction CSV has no header."
        )


    id_candidates = [
        "crop_id",
        "id",
        "filename",
        "file",
        "image",
        "image_name",
        "path",
        "image_path",
    ]


    label_candidates = [
        "predicted_label",
        "prediction",
        "predicted_class",
        "class",
        "label",
        "top1_label",
    ]


    id_col = next(
        (
            x
            for x in id_candidates
            if x in reader.fieldnames
        ),
        None
    )


    label_col = next(
        (
            x
            for x in label_candidates
            if x in reader.fieldnames
        ),
        None
    )


    if id_col is None or label_col is None:

        raise SystemExit(
            "ERROR: couldn't identify prediction "
            f"columns. Found: {reader.fieldnames}"
        )


    for row in reader:

        cid = crop_id(
            row[id_col]
        )

        label = (
            row[label_col]
            .strip()
        )


        if cid in predictions:

            raise SystemExit(
                f"ERROR: duplicate prediction "
                f"for {cid}"
            )


        predictions[cid] = label


print(
    f"   Predictions: "
    f"{len(predictions)}"
)


# ============================================================
# 8. MANUAL CORRECTIONS
# ============================================================

print(
    "\n8. Loading manual review decisions..."
)

corrections = {}


with CORRECTIONS_FILE.open(
    encoding="utf-8-sig"
) as f:

    for line_number, raw in enumerate(
        f,
        start=1
    ):

        line = raw.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 2:

            raise SystemExit(
                f"ERROR: malformed correction "
                f"line {line_number}: {raw!r}"
            )


        cid, decision = parts


        if cid in corrections:

            raise SystemExit(
                f"ERROR: duplicate correction "
                f"for {cid}"
            )


        corrections[cid] = decision


print(
    f"   Manual decisions: "
    f"{len(corrections)}"
)


# ============================================================
# 9. ID INTEGRITY
# ============================================================

print(
    "\n9. Verifying museum IDs..."
)

museum_ids = set(
    museum_by_id
)


missing_predictions = (
    museum_ids - set(predictions)
)


bad_corrections = (
    set(corrections) - museum_ids
)


if missing_predictions:

    raise SystemExit(
        f"ERROR: {len(missing_predictions)} "
        "museum crops have no prediction."
    )


if bad_corrections:

    raise SystemExit(
        "ERROR: correction IDs missing from "
        f"museum data: {sorted(bad_corrections)}"
    )


print(
    "   Museum ID integrity: PASS"
)


# ============================================================
# 10. APPLY VERIFIED MUSEUM LABELS
# ============================================================

print(
    "\n10. Applying verified museum labels..."
)

museum_records = []
rejected_records = []


for cid in sorted(museum_ids):

    p = museum_by_id[cid][0]

    rel = p.relative_to(
        MUSEUM_CURATED_DIR
    )

    museum = rel.parts[0]
    source_group = rel.parts[1]


    if (
        museum,
        source_group
    ) not in museum_train_groups:

        raise SystemExit(
            "ERROR: curated museum crop is "
            "not from a training source: "
            f"{rel}"
        )


    if cid in corrections:

        decision = corrections[cid]


        if decision.upper() == "REJECT":

            rejected_records.append({
                "path": p,
                "label": "REJECT",
                "source_type": "museum",
                "museum": museum,
                "source_group": source_group,
                "crop_id": cid,
            })

            continue


        label = decision


    else:

        label = predictions[cid]


    museum_records.append({
        "path": p,
        "label": label,
        "source_type": "museum",
        "museum": museum,
        "source_group": source_group,
        "crop_id": cid,
    })


print(
    f"   Museum accepted: "
    f"{len(museum_records)}"
)

print(
    f"   Museum rejected: "
    f"{len(rejected_records)}"
)


# ============================================================
# 11. INITIAL CLASS COUNTS
# ============================================================

combined_counts = Counter()


for r in old_records:
    combined_counts[
        r["label"]
    ] += 1


for r in museum_records:
    combined_counts[
        r["label"]
    ] += 1


total_eligible = {
    label
    for label, count
    in combined_counts.items()
    if count >= MIN_TOTAL_IMAGES
}


# ============================================================
# 12. PRELIMINARY OLD SPLIT
# ============================================================

prelim_train = []
prelim_val = []


for r in old_records:

    if r["label"] not in total_eligible:
        continue


    relative = str(
        r["path"].relative_to(
            OLD_RAW_DIR
        )
    )


    if relative in old_val_paths:

        prelim_val.append(r)


    elif relative in old_train_paths:

        prelim_train.append(r)


    else:

        # Class wasn't eligible for V2, but became
        # eligible for V3 after museum additions.
        prelim_train.append(r)


# All accepted museum crops come from frozen
# museum TRAIN sources.
prelim_train.extend(
    museum_records
)


prelim_train_counts = Counter(
    r["label"]
    for r in prelim_train
)


# ============================================================
# 13. FINAL CLASS ELIGIBILITY
# ============================================================

eligible_classes = {
    label
    for label in total_eligible
    if prelim_train_counts[label]
    >= MIN_TRAIN_IMAGES
}


archived_classes = (
    set(combined_counts)
    - eligible_classes
)


print(
    "\n11. Determining final V3 classes..."
)

print(
    f"   >= {MIN_TOTAL_IMAGES} total "
    f"and >= {MIN_TRAIN_IMAGES} train: "
    f"{len(eligible_classes)}"
)

print(
    f"   Archived classes: "
    f"{len(archived_classes)}"
)


# ============================================================
# 14. FINAL ASSIGNMENTS
# ============================================================

train_records = []
val_records = []
archived_records = []


for r in old_records:

    if r["label"] not in eligible_classes:

        archived_records.append(r)
        continue


    relative = str(
        r["path"].relative_to(
            OLD_RAW_DIR
        )
    )


    if relative in old_val_paths:

        val_records.append(r)


    else:

        train_records.append(r)


for r in museum_records:

    if r["label"] in eligible_classes:

        train_records.append(r)

    else:

        archived_records.append(r)


train_counts = Counter(
    r["label"]
    for r in train_records
)

val_counts = Counter(
    r["label"]
    for r in val_records
)


# ============================================================
# 15. FINAL CLASS AUDIT
# ============================================================

print(
    "\n" + "=" * 72
)

print(
    "FINAL V3 CLASS AUDIT"
)

print(
    "=" * 72
)

print(
    f"{'CLASS':<9}"
    f"{'TOTAL':>7}"
    f"{'TRAIN':>8}"
    f"{'VAL':>7}"
)

print(
    "-" * 31
)


for label in sorted(
    eligible_classes
):

    print(
        f"{label:<9}"
        f"{combined_counts[label]:>7}"
        f"{train_counts[label]:>8}"
        f"{val_counts[label]:>7}"
    )


# ============================================================
# 16. FINAL SAFETY CHECKS
# ============================================================

bad_train_classes = [
    label
    for label in eligible_classes
    if train_counts[label]
    < MIN_TRAIN_IMAGES
]


if bad_train_classes:

    raise SystemExit(
        "ERROR: classes below training minimum: "
        f"{bad_train_classes}"
    )


if (
    curated_groups
    & museum_locked_groups
):

    raise SystemExit(
        "ERROR: locked museum source leakage."
    )


# ============================================================
# 17. CLEAR ONLY GENERATED V3 OUTPUT
# ============================================================

print(
    "\n12. Rebuilding classifier_v3/data..."
)


for folder in [
    TRAIN_DIR,
    VAL_DIR,
    LOCKED_TEST_DIR,
    ARCHIVED_DIR,
    REJECTED_DIR,
]:

    if folder.exists():

        shutil.rmtree(
            folder
        )


if MANIFEST_DIR.exists():

    shutil.rmtree(
        MANIFEST_DIR
    )


MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Keep the directory present as a reminder that
# the true museum benchmark is intentionally untouched.
LOCKED_TEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 18. COPY DATA
# ============================================================

all_rows = []
train_rows = []
val_rows = []
archived_rows = []
rejected_rows = []


print(
    "   Copying training images..."
)


for r in sorted(
    train_records,
    key=lambda x: (
        x["label"],
        x["source_type"],
        str(x["path"])
    )
):

    row = copy_record(
        r,
        "train",
        TRAIN_DIR
    )

    train_rows.append(row)
    all_rows.append(row)


print(
    "   Copying clean validation images..."
)


for r in sorted(
    val_records,
    key=lambda x: (
        x["label"],
        str(x["path"])
    )
):

    row = copy_record(
        r,
        "val",
        VAL_DIR
    )

    val_rows.append(row)
    all_rows.append(row)


print(
    "   Copying archived images..."
)


for r in sorted(
    archived_records,
    key=lambda x: (
        x["label"],
        x["source_type"],
        str(x["path"])
    )
):

    row = copy_record(
        r,
        "archived",
        ARCHIVED_DIR
    )

    archived_rows.append(row)
    all_rows.append(row)


print(
    "   Copying rejected images..."
)


for r in sorted(
    rejected_records,
    key=lambda x: str(
        x["path"]
    )
):

    row = copy_record(
        r,
        "rejected",
        REJECTED_DIR
    )

    rejected_rows.append(row)
    all_rows.append(row)


# ============================================================
# 19. WRITE MANIFESTS
# ============================================================

print(
    "\n13. Writing manifests..."
)


write_manifest(
    MANIFEST_DIR
    / "all_v3_images.csv",
    all_rows
)


write_manifest(
    MANIFEST_DIR
    / "train.csv",
    train_rows
)


write_manifest(
    MANIFEST_DIR
    / "val.csv",
    val_rows
)


write_manifest(
    MANIFEST_DIR
    / "archived.csv",
    archived_rows
)


write_manifest(
    MANIFEST_DIR
    / "rejected.csv",
    rejected_rows
)


# Empty locked-test manifest on purpose.
write_manifest(
    MANIFEST_DIR
    / "locked_test.csv",
    []
)


# ============================================================
# 20. ACCOUNTING
# ============================================================

expected_total = (
    len(old_records)
    + len(museum_paths)
)


actual_total = (
    len(train_rows)
    + len(val_rows)
    + len(archived_rows)
    + len(rejected_rows)
)


if actual_total != expected_total:

    raise SystemExit(
        f"ERROR: expected {expected_total} "
        f"images but accounted for "
        f"{actual_total}."
    )


# Make sure every copied file exists.
for row in all_rows:

    p = BASE_DIR / row["v3_path"]

    if not p.exists():

        raise SystemExit(
            f"ERROR: copied file missing: {p}"
        )


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 72
)

print(
    "FINAL V3 BUILD COMPLETE"
)

print(
    "=" * 72
)


print(
    f"Trainable classes:        "
    f"{len(eligible_classes)}"
)

print(
    f"Archived classes:         "
    f"{len(archived_classes)}"
)

print()

print(
    f"Training images:          "
    f"{len(train_rows)}"
)

print(
    f"Clean validation images:  "
    f"{len(val_rows)}"
)

print(
    "Locked museum test:       "
    "0 crops (intentionally untouched)"
)

print(
    f"Archived images:          "
    f"{len(archived_rows)}"
)

print(
    f"Rejected images:          "
    f"{len(rejected_rows)}"
)

print()

print(
    f"Total accounted:          "
    f"{actual_total}"
)

print(
    f"Expected total:           "
    f"{expected_total}"
)

print()

print(
    f"Museum training groups:   "
    f"{len(curated_groups)}"
)

print(
    f"Protected museum groups:  "
    f"{len(museum_locked_groups)}"
)

print(
    "Locked-source leakage:    "
    "0 groups"
)

print()

print(
    "Original source data was NOT modified."
)

print(
    "Locked museum photographs remain untouched."
)