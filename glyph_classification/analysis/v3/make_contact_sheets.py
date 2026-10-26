from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont
import math
import csv

BASE_DIR = Path(__file__).resolve().parents[2]

SOURCE_ROOT = BASE_DIR / "data" / "museum_sources"
OUTPUT_DIR = BASE_DIR / "analysis" / "v3" / "contact_sheets"

PHOTOS_PER_SHEET = 30
COLS = 5

THUMB_W = 300
THUMB_H = 300
LABEL_H = 45

MARGIN = 20
GAP = 15

VALID_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".heic",
    ".webp",
    ".tif",
    ".tiff",
}


def get_images(folder):
    files = []

    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in VALID_EXTS:
            files.append(path)

    return sorted(files, key=lambda p: p.name.lower())


def make_thumbnail(image_path):
    with Image.open(image_path) as img:

        # Fix iPhone EXIF orientation automatically
        img = ImageOps.exif_transpose(img)

        img = img.convert("RGB")

        # Preserve aspect ratio
        img.thumbnail((THUMB_W, THUMB_H))

        canvas = Image.new(
            "RGB",
            (THUMB_W, THUMB_H),
            "white",
        )

        x = (THUMB_W - img.width) // 2
        y = (THUMB_H - img.height) // 2

        canvas.paste(img, (x, y))

        return canvas


def create_museum_sheets(museum_name):

    source_dir = SOURCE_ROOT / museum_name
    museum_out = OUTPUT_DIR / museum_name

    museum_out.mkdir(parents=True, exist_ok=True)

    images = get_images(source_dir)

    print(f"{museum_name}: {len(images)} images")

    mapping_rows = []

    for index, path in enumerate(images, start=1):
        mapping_rows.append(
            {
                "number": index,
                "filename": path.name,
            }
        )

    mapping_path = museum_out / f"{museum_name}_photo_mapping.csv"

    with open(mapping_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["number", "filename"],
        )
        writer.writeheader()
        writer.writerows(mapping_rows)

    total_sheets = math.ceil(
        len(images) / PHOTOS_PER_SHEET
    )

    rows_per_sheet = math.ceil(
        PHOTOS_PER_SHEET / COLS
    )

    sheet_w = (
        MARGIN * 2
        + COLS * THUMB_W
        + (COLS - 1) * GAP
    )

    sheet_h = (
        MARGIN * 2
        + rows_per_sheet * (THUMB_H + LABEL_H)
        + (rows_per_sheet - 1) * GAP
    )

    for sheet_index in range(total_sheets):

        start = sheet_index * PHOTOS_PER_SHEET
        end = min(
            start + PHOTOS_PER_SHEET,
            len(images),
        )

        sheet_images = images[start:end]

        sheet = Image.new(
            "RGB",
            (sheet_w, sheet_h),
            "white",
        )

        draw = ImageDraw.Draw(sheet)

        for local_index, image_path in enumerate(sheet_images):

            global_number = start + local_index + 1

            row = local_index // COLS
            col = local_index % COLS

            x = MARGIN + col * (THUMB_W + GAP)

            y = MARGIN + row * (
                THUMB_H
                + LABEL_H
                + GAP
            )

            try:
                thumb = make_thumbnail(image_path)

                sheet.paste(
                    thumb,
                    (x, y),
                )

                label = (
                    f"{museum_name} {global_number:03d}"
                )

                draw.text(
                    (x, y + THUMB_H + 8),
                    label,
                    fill="black",
                )

            except Exception as error:

                print(
                    f"Could not open {image_path.name}: "
                    f"{error}"
                )

        output_path = (
            museum_out
            / f"{museum_name}_contact_{sheet_index + 1:02d}.jpg"
        )

        sheet.save(
            output_path,
            quality=90,
        )

        print("Saved:", output_path)

    print("Mapping:", mapping_path)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    create_museum_sheets("MFA")
    create_museum_sheets("HMANE")

    print("\nDone.")


if __name__ == "__main__":
    main()