from pathlib import Path
import csv
import math

from PIL import Image, ImageOps, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parents[2]

PREDICTIONS_CSV = (
    BASE_DIR
    / "analysis"
    / "v2_museum_predictions.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "analysis"
    / "prelabel_review_sheets"
)

COLS = 5
ROWS = 5
PER_SHEET = COLS * ROWS

CELL_W = 260
CELL_H = 300

IMAGE_W = 230
IMAGE_H = 220

LABEL_H = 65

MARGIN = 20
GAP = 10


def load_predictions():
    rows = []

    with open(
        PREDICTIONS_CSV,
        "r",
        encoding="utf-8"
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            row["confidence"] = float(
                row["confidence"]
            )
            row["second_confidence"] = float(
                row["second_confidence"]
            )
            rows.append(row)

    return rows


def main():

    rows = load_predictions()

    # Group by predicted label
    groups = {}

    for row in rows:
        label = row["predicted_label"]

        groups.setdefault(
            label,
            []
        ).append(row)

    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Sort each class by confidence ASCENDING
    # so the most suspicious predictions appear first.
    for label in groups:
        groups[label].sort(
            key=lambda r: r["confidence"]
        )

    summary_rows = []

    for label in sorted(groups):

        class_rows = groups[label]

        class_dir = (
            OUTPUT_DIR
            / label
        )

        class_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        num_sheets = math.ceil(
            len(class_rows)
            / PER_SHEET
        )

        for sheet_num in range(num_sheets):

            start = (
                sheet_num
                * PER_SHEET
            )

            end = min(
                start + PER_SHEET,
                len(class_rows)
            )

            subset = (
                class_rows[start:end]
            )

            sheet_w = (
                MARGIN * 2
                + COLS * CELL_W
                + (COLS - 1) * GAP
            )

            sheet_h = (
                MARGIN * 2
                + ROWS * CELL_H
                + (ROWS - 1) * GAP
            )

            sheet = Image.new(
                "RGB",
                (sheet_w, sheet_h),
                "white"
            )

            draw = ImageDraw.Draw(
                sheet
            )

            for idx, row in enumerate(subset):

                r = idx // COLS
                c = idx % COLS

                x = (
                    MARGIN
                    + c * (CELL_W + GAP)
                )

                y = (
                    MARGIN
                    + r * (CELL_H + GAP)
                )

                image_path = (
                    BASE_DIR
                    / row["source_path"]
                )

                with Image.open(
                    image_path
                ) as img:

                    img = ImageOps.exif_transpose(
                        img
                    )

                    img = img.convert("RGB")

                    img.thumbnail(
                        (IMAGE_W, IMAGE_H)
                    )

                    canvas = Image.new(
                        "RGB",
                        (IMAGE_W, IMAGE_H),
                        "white"
                    )

                    px = (
                        IMAGE_W - img.width
                    ) // 2

                    py = (
                        IMAGE_H - img.height
                    ) // 2

                    canvas.paste(
                        img,
                        (px, py)
                    )

                sheet.paste(
                    canvas,
                    (x, y)
                )

                line1 = (
                    f"{row['filename']}"
                )

                line2 = (
                    f"PRED: {row['predicted_label']} "
                    f"{row['confidence']:.2f}"
                )

                line3 = (
                    f"2nd: {row['second_label']} "
                    f"{row['second_confidence']:.2f}"
                )

                draw.text(
                    (
                        x,
                        y + IMAGE_H + 5
                    ),
                    line1,
                    fill="black"
                )

                draw.text(
                    (
                        x,
                        y + IMAGE_H + 23
                    ),
                    line2,
                    fill="black"
                )

                draw.text(
                    (
                        x,
                        y + IMAGE_H + 41
                    ),
                    line3,
                    fill="black"
                )

            output_path = (
                class_dir
                / (
                    f"{label}_review_"
                    f"{sheet_num + 1:03d}.jpg"
                )
            )

            sheet.save(
                output_path,
                quality=95
            )

        summary_rows.append(
            (
                label,
                len(class_rows),
                num_sheets
            )
        )

    print()
    print("=" * 60)
    print("PRELABEL REVIEW SHEETS COMPLETE")
    print("=" * 60)

    total_sheets = 0

    for label, count, sheets in summary_rows:
        total_sheets += sheets

        print(
            f"{label:8} "
            f"{count:4} images "
            f"{sheets:3} sheets"
        )

    print()
    print(
        "Total images:",
        len(rows)
    )

    print(
        "Total sheets:",
        total_sheets
    )

    print(
        "Output:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()