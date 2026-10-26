from pathlib import Path
import csv
import shutil

BASE_DIR = Path(__file__).resolve().parents[2]

MAPPING_PATH = (
    BASE_DIR
    / "analysis"
    / "crop_contact_sheets"
    / "train"
    / "train_crop_mapping.csv"
)

IDS_PATH = (
    BASE_DIR
    / "analysis"
    / "train_needs_cropping_ids.txt"
)

OUTPUT_DIR = (
    BASE_DIR
    / "analysis"
    / "needs_cropping"
)


def main():

    # Load IDs
    with open(IDS_PATH, "r", encoding="utf-8") as f:
        wanted_ids = {
            line.strip()
            for line in f
            if line.strip()
        }

    print("Needs cropping IDs:", len(wanted_ids))

    # Load crop mapping
    mapping = {}

    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            mapping[row["crop_id"]] = row

    missing = wanted_ids - set(mapping.keys())

    if missing:
        print("Missing IDs:")
        for crop_id in sorted(missing):
            print(" ", crop_id)

        raise RuntimeError(
            "Some crop IDs were not found in the mapping."
        )

    # Rebuild output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied = 0

    for crop_id in sorted(wanted_ids):

        row = mapping[crop_id]

        source_path = (
            BASE_DIR
            / row["crop_path"]
        )

        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing crop file: {source_path}"
            )

        # Prefix the stable T-ID so the image is easy to track.
        destination_name = (
            f"{crop_id}__{source_path.name}"
        )

        destination_path = (
            OUTPUT_DIR
            / destination_name
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

        copied += 1

    print()
    print("Copied:", copied)
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()