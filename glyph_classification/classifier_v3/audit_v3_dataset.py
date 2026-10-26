from pathlib import Path
from collections import Counter, defaultdict
import csv
import re

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OLD_RAW_DIR = BASE_DIR / "data" / "raw"
MUSEUM_DIR = BASE_DIR / "data" / "museum_curated" / "train_unlabeled"

ANALYSIS_DIR = BASE_DIR / "analysis" / "v3"
PREDICTIONS_CSV = ANALYSIS_DIR / "v2_museum_predictions.csv"
CORRECTIONS_FILE = ANALYSIS_DIR / "museum_label_corrections.txt"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# Gardiner-style labels:
# A1, D21, Aa1, N33a, etc.
LABEL_PATTERN = re.compile(r"^(?:[A-Z][a-z]?)[0-9]+[a-z]?$")


def crop_id(value):
    """Turn a filename/path into its crop ID."""
    return Path(str(value).strip()).stem


# ============================================================
# 1. AUDIT OLD / V2 DATA
# ============================================================

print("\n" + "=" * 70)
print("1. OLD CLEAN DATA")
print("=" * 70)

if not OLD_RAW_DIR.exists():
    raise SystemExit(f"ERROR: Missing {OLD_RAW_DIR}")

old_counts = Counter()
old_files = []

for class_dir in sorted(OLD_RAW_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    label = class_dir.name

    for p in class_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            old_counts[label] += 1
            old_files.append((p, label))

print(f"Old clean images: {len(old_files)}")
print(f"Old class folders: {len(old_counts)}")


# ============================================================
# 2. FIND ALL CURATED MUSEUM CROPS
# ============================================================

print("\n" + "=" * 70)
print("2. CURATED MUSEUM CROPS")
print("=" * 70)

museum_files = [
    p for p in MUSEUM_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
]

museum_by_id = defaultdict(list)

for p in museum_files:
    museum_by_id[p.stem].append(p)

print(f"Museum crops:       {len(museum_files)}")
print(f"Unique museum IDs:  {len(museum_by_id)}")

duplicate_museum_ids = {
    k: v for k, v in museum_by_id.items()
    if len(v) > 1
}

if duplicate_museum_ids:
    print("\nERROR: duplicate crop IDs in museum dataset:")

    for cid, paths in sorted(duplicate_museum_ids.items()):
        print(f"\n{cid}")
        for p in paths:
            print("   ", p.relative_to(BASE_DIR))

    raise SystemExit("\nSTOPPED: duplicate museum IDs must be resolved.")


# ============================================================
# 3. LOAD V2 MUSEUM PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("3. V2 MUSEUM PREDICTIONS")
print("=" * 70)

predictions = {}

with PREDICTIONS_CSV.open(newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    if not reader.fieldnames:
        raise SystemExit("ERROR: prediction CSV has no header.")

    print("CSV columns:", reader.fieldnames)

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
        (x for x in id_candidates if x in reader.fieldnames),
        None
    )

    label_col = next(
        (x for x in label_candidates if x in reader.fieldnames),
        None
    )

    if id_col is None or label_col is None:
        raise SystemExit(
            "\nERROR: Could not identify prediction columns.\n"
            f"Found: {reader.fieldnames}"
        )

    print(f"Using ID column:    {id_col}")
    print(f"Using label column: {label_col}")

    for row in reader:
        cid = crop_id(row[id_col])
        label = row[label_col].strip()

        if not cid or not label:
            continue

        if cid in predictions:
            raise SystemExit(
                f"ERROR: duplicate v2 prediction for {cid}"
            )

        predictions[cid] = label

print(f"Prediction entries: {len(predictions)}")


# ============================================================
# 4. LOAD MANUAL CORRECTIONS
# ============================================================

print("\n" + "=" * 70)
print("4. MANUAL REVIEW FILE")
print("=" * 70)

corrections = {}

with CORRECTIONS_FILE.open(encoding="utf-8-sig") as f:

    for line_number, raw in enumerate(f, start=1):

        line = raw.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 2:
            raise SystemExit(
                f"ERROR: malformed line {line_number}: {raw!r}"
            )

        cid, label = parts

        if cid in corrections:
            raise SystemExit(
                f"ERROR: duplicate correction for {cid}"
            )

        corrections[cid] = label

print(f"Manual decisions: {len(corrections)}")

reject_count = sum(
    1 for x in corrections.values()
    if x.upper() == "REJECT"
)

print(f"Manual REJECTs:   {reject_count}")


# ============================================================
# 5. VALIDATE MANUAL LABEL FORMAT
# ============================================================

invalid_manual_labels = []

for cid, label in corrections.items():

    if label.upper() == "REJECT":
        continue

    if not LABEL_PATTERN.match(label):
        invalid_manual_labels.append((cid, label))

if invalid_manual_labels:

    print("\nERROR: suspicious manual labels:")

    for cid, label in invalid_manual_labels:
        print(f"  {cid}: {label}")

    raise SystemExit(
        "\nSTOPPED: verify these labels before continuing."
    )

print("Manual label formatting: PASS")


# ============================================================
# 6. CROSS-CHECK ALL IDs
# ============================================================

print("\n" + "=" * 70)
print("5. ID INTEGRITY")
print("=" * 70)

museum_ids = set(museum_by_id)
prediction_ids = set(predictions)
correction_ids = set(corrections)

missing_predictions = sorted(museum_ids - prediction_ids)
extra_predictions = sorted(prediction_ids - museum_ids)
bad_corrections = sorted(correction_ids - museum_ids)

print(f"Museum crops without prediction: {len(missing_predictions)}")
print(f"Predictions without crop:        {len(extra_predictions)}")
print(f"Corrections without crop:        {len(bad_corrections)}")

if missing_predictions:
    print("\nMissing predictions:")
    for x in missing_predictions[:30]:
        print(" ", x)

if extra_predictions:
    print("\nExtra predictions:")
    for x in extra_predictions[:30]:
        print(" ", x)

if bad_corrections:
    print("\nCorrections with no corresponding crop:")
    for x in bad_corrections[:30]:
        print(" ", x)

if missing_predictions or bad_corrections:
    raise SystemExit(
        "\nSTOPPED: not every museum crop can be safely labeled."
    )


# ============================================================
# 7. CREATE FINAL MUSEUM LABELS IN MEMORY
# ============================================================

final_museum_labels = {}

manual_kept = 0
v2_unchanged = 0
rejected_ids = []

for cid in sorted(museum_ids):

    if cid in corrections:

        label = corrections[cid]

        if label.upper() == "REJECT":
            rejected_ids.append(cid)
            continue

        final_museum_labels[cid] = label
        manual_kept += 1

    else:

        label = predictions[cid]

        final_museum_labels[cid] = label
        v2_unchanged += 1


museum_counts = Counter(final_museum_labels.values())

print("\n" + "=" * 70)
print("6. FINAL VERIFIED MUSEUM DATA")
print("=" * 70)

print(f"Original museum crops:    {len(museum_files)}")
print(f"Rejected:                 {len(rejected_ids)}")
print(f"Final museum images:      {len(final_museum_labels)}")
print(f"Manually relabeled/kept:  {manual_kept}")
print(f"Unchanged v2 labels:      {v2_unchanged}")
print(f"Museum classes:           {len(museum_counts)}")


# ============================================================
# 8. COMBINE OLD + MUSEUM COUNTS
# ============================================================

all_classes = sorted(
    set(old_counts) | set(museum_counts)
)

combined_counts = Counter()

for label in all_classes:
    combined_counts[label] = (
        old_counts[label] + museum_counts[label]
    )


print("\n" + "=" * 70)
print("7. COMBINED V3 CLASS DISTRIBUTION")
print("=" * 70)

print(
    f"{'CLASS':<10}"
    f"{'OLD':>8}"
    f"{'MUSEUM':>10}"
    f"{'TOTAL':>9}"
)

print("-" * 37)

for label in sorted(
    all_classes,
    key=lambda x: (-combined_counts[x], x)
):
    print(
        f"{label:<10}"
        f"{old_counts[label]:>8}"
        f"{museum_counts[label]:>10}"
        f"{combined_counts[label]:>9}"
    )


# ============================================================
# 9. NEW CLASSES INTRODUCED BY MUSEUM DATA
# ============================================================

new_classes = sorted(
    set(museum_counts) - set(old_counts)
)

print("\n" + "=" * 70)
print("8. NEW CLASSES FROM MUSEUM DATA")
print("=" * 70)

if not new_classes:
    print("None")
else:
    for label in new_classes:
        print(
            f"{label:<8} "
            f"museum={museum_counts[label]:>4} "
            f"total={combined_counts[label]:>4}"
        )


# ============================================================
# 10. SMALL / POTENTIALLY PROBLEMATIC CLASSES
# ============================================================

print("\n" + "=" * 70)
print("9. CLASSES WITH FEWER THAN 6 TOTAL IMAGES")
print("=" * 70)

small_classes = [
    label for label in all_classes
    if combined_counts[label] < 6
]

if not small_classes:
    print("None")
else:
    for label in sorted(small_classes):
        print(
            f"{label:<8} "
            f"old={old_counts[label]:>3} "
            f"museum={museum_counts[label]:>3} "
            f"total={combined_counts[label]:>3}"
        )


# ============================================================
# 11. MUSEUM SOURCE-GROUP COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("10. MUSEUM SOURCE-GROUP COVERAGE")
print("=" * 70)

groups_by_class = defaultdict(set)

for cid, label in final_museum_labels.items():

    p = museum_by_id[cid][0]
    rel = p.relative_to(MUSEUM_DIR)

    parts = rel.parts

    museum = parts[0] if len(parts) >= 2 else "UNKNOWN"
    group = parts[1] if len(parts) >= 3 else "UNKNOWN"

    groups_by_class[label].add((museum, group))


print(
    f"{'CLASS':<10}"
    f"{'MUSEUM N':>10}"
    f"{'GROUPS':>10}"
)

print("-" * 30)

for label in sorted(
    museum_counts,
    key=lambda x: (-museum_counts[x], x)
):
    print(
        f"{label:<10}"
        f"{museum_counts[label]:>10}"
        f"{len(groups_by_class[label]):>10}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print(f"Old clean images:       {len(old_files)}")
print(f"Museum images kept:     {len(final_museum_labels)}")
print(f"Museum images rejected: {len(rejected_ids)}")
print(f"Combined images:        {len(old_files) + len(final_museum_labels)}")
print(f"Combined classes:       {len(all_classes)}")
print(f"New museum classes:     {len(new_classes)}")
print(f"Classes below 6 total:  {len(small_classes)}")

print(
    "\nNO FILES WERE COPIED, MOVED, DELETED, OR MODIFIED."
)