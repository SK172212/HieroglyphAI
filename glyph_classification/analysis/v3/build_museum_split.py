from pathlib import Path
import csv
import re
import shutil

BASE_DIR = Path(__file__).resolve().parents[2]

ANALYSIS_DIR = BASE_DIR / "analysis" / "v3"
SOURCE_DIR = BASE_DIR / "data" / "museum_sources"
OUTPUT_DIR = BASE_DIR / "data" / "museum_split"

GROUPS_FILE = ANALYSIS_DIR / "photo_groups.txt"
SPLIT_FILE = ANALYSIS_DIR / "locked_source_split.txt"

TRAIN_DIR = OUTPUT_DIR / "train_sources"
TEST_DIR = OUTPUT_DIR / "locked_test_sources"


def load_mapping(museum):
    mapping_file = (
        ANALYSIS_DIR
        / "contact_sheets"
        / museum
        / f"{museum}_photo_mapping.csv"
    )

    mapping = {}

    with open(mapping_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            number = int(row["number"])
            filename = row["filename"]
            mapping[number] = filename

    return mapping


def load_groups():
    groups = {}

    with open(GROUPS_FILE, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if ":" not in line:
                continue

            group_name, numbers_text = line.split(":", 1)
            group_name = group_name.strip()

            numbers = [
                int(x)
                for x in re.findall(r"\d+", numbers_text)
            ]

            groups[group_name] = numbers

    return groups


def load_split():
    train_groups = set()
    test_groups = set()

    current_section = None

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if line == "TRAIN:":
                current_section = "train"
                continue

            if line == "TEST:":
                current_section = "test"
                continue

            if current_section == "train":
                train_groups.add(line)

            elif current_section == "test":
                test_groups.add(line)

    return train_groups, test_groups


def copy_group(group_name, photo_numbers, destination_root, mappings):
    museum = group_name.split("_")[0]

    if museum not in mappings:
        raise ValueError(f"Unknown museum in group: {group_name}")

    museum_source_dir = SOURCE_DIR / museum
    museum_dest_dir = destination_root / museum / group_name
    museum_dest_dir.mkdir(parents=True, exist_ok=True)

    mapping = mappings[museum]

    copied = 0

    for photo_number in photo_numbers:
        if photo_number not in mapping:
            raise ValueError(
                f"{group_name}: photo number {photo_number} "
                f"not found in {museum} mapping CSV"
            )

        filename = mapping[photo_number]
        source_path = museum_source_dir / filename

        if not source_path.exists():
            raise FileNotFoundError(
                f"Source image not found: {source_path}"
            )

        destination_path = museum_dest_dir / filename
        shutil.copy2(source_path, destination_path)
        copied += 1

    return copied


def main():
    print("Loading group definitions...")
    groups = load_groups()

    print("Loading locked split...")
    train_groups, test_groups = load_split()

    all_split_groups = train_groups | test_groups
    all_defined_groups = set(groups.keys())

    missing_from_split = all_defined_groups - all_split_groups
    unknown_in_split = all_split_groups - all_defined_groups
    overlap = train_groups & test_groups

    if overlap:
        raise ValueError(
            f"Groups appear in BOTH train and test: {sorted(overlap)}"
        )

    if missing_from_split:
        raise ValueError(
            f"Groups defined in photo_groups.txt but not assigned "
            f"to train/test: {sorted(missing_from_split)}"
        )

    if unknown_in_split:
        raise ValueError(
            f"Groups listed in locked_source_split.txt but not found "
            f"in photo_groups.txt: {sorted(unknown_in_split)}"
        )

    mappings = {
        "MFA": load_mapping("MFA"),
        "HMANE": load_mapping("HMANE"),
    }

    # Rebuild output from scratch.
    if OUTPUT_DIR.exists():
        print(f"Removing old split folder: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    train_count = 0
    test_count = 0

    print("\nCopying TRAIN groups...")

    for group_name in sorted(train_groups):
        count = copy_group(
            group_name,
            groups[group_name],
            TRAIN_DIR,
            mappings,
        )
        train_count += count
        print(f"  {group_name}: {count} photos")

    print("\nCopying LOCKED TEST groups...")

    for group_name in sorted(test_groups):
        count = copy_group(
            group_name,
            groups[group_name],
            TEST_DIR,
            mappings,
        )
        test_count += count
        print(f"  {group_name}: {count} photos")

    total = train_count + test_count

    print("\n================================")
    print("Museum split complete")
    print("================================")
    print(f"Train photos:       {train_count}")
    print(f"Locked test photos: {test_count}")
    print(f"Total photos:       {total}")
    print(f"Output folder:      {OUTPUT_DIR}")


if __name__ == "__main__":
    main()