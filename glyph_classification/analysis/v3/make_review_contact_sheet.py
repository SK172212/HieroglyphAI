from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import csv

BASE_DIR = Path(__file__).resolve().parents[2]

MAPPING_PATH = (
    BASE_DIR
    / "analysis"
    / "crop_contact_sheets"
    / "train"
    / "train_crop_mapping.csv"
)

REVIEW_IDS_PATH = (
    BASE_DIR
    / "analysis"
    / "train_review_ids.txt"
)

OUTPUT_PATH = (
    BASE_DIR
    / "analysis"
    / "train_review_contact_sheet.jpg"
)

COLS = 4
THUMB_W = 300
THUMB_H = 300
LABEL_H = 40
MARGIN = 20
GAP = 15


def main():

    # -----------------------------
    # Load review IDs
    # -----------------------------

    review_ids = set()

    with open(REVIEW_IDS_PATH, "r") as f:
        for line in f:
            crop_id = line.strip()

            if crop_id:
                review_ids.add(crop_id)

    print(f"Review IDs: {len(review_ids)}")

    # -----------------------------
    # Load mapping
    # -----------------------------

    mapping = {}

    with open(
        MAPPING_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            mapping[row["crop_id"]] = row

    # -----------------------------
    # Verify IDs
    # -----------------------------

    missing = review_ids - set(mapping.keys())

    if missing:
        print("Missing IDs:")
        for crop_id in sorted(missing):
            print(crop_id)

        raise RuntimeError(
            "Some review IDs were not found in mapping."
        )

    review_ids = sorted(review_ids)

    # -----------------------------
    # Sheet dimensions
    # -----------------------------

    rows = (
        len(review_ids) + COLS - 1
    ) // COLS

    sheet_w = (
        MARGIN * 2
        + COLS * THUMB_W
        + (COLS - 1) * GAP
    )

    sheet_h = (
        MARGIN * 2
        + rows * (THUMB_H + LABEL_H)
        + (rows - 1) * GAP
    )

    sheet = Image.new(
        "RGB",
        (sheet_w, sheet_h),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    # -----------------------------
    # Add crops
    # -----------------------------

    for index, crop_id in enumerate(review_ids):

        row_data = mapping[crop_id]

        crop_path = (
            BASE_DIR
            / row_data["crop_path"]
        )

        with Image.open(crop_path) as img:

            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")

            img.thumbnail(
                (THUMB_W, THUMB_H)
            )

            thumb = Image.new(
                "RGB",
                (THUMB_W, THUMB_H),
                "white"
            )

            x_center = (
                THUMB_W - img.width
            ) // 2

            y_center = (
                THUMB_H - img.height
            ) // 2

            thumb.paste(
                img,
                (x_center, y_center)
            )

        row = index // COLS
        col = index % COLS

        x = (
            MARGIN
            + col * (THUMB_W + GAP)
        )

        y = (
            MARGIN
            + row
            * (THUMB_H + LABEL_H + GAP)
        )

        sheet.paste(
            thumb,
            (x, y)
        )

        label = (
            f"{crop_id}   "
            f"{row_data['museum']}   "
            f"{row_data['artifact_group']}"
        )

        draw.text(
            (x, y + THUMB_H + 8),
            label,
            fill="black"
        )

    # -----------------------------
    # Save
    # -----------------------------

    sheet.save(
        OUTPUT_PATH,
        quality=95
    )

    print()
    print("Review contact sheet created.")
    print("Images:", len(review_ids))
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()