from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import csv
import math


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

CROP_ROOT = (
    BASE_DIR
    / "data"
    / "museum_crops"
    / "train_unlabeled"
)

OUTPUT_ROOT = (
    BASE_DIR
    / "analysis"
    / "crop_contact_sheets"
    / "train"
)

MAPPING_PATH = (
    OUTPUT_ROOT
    / "train_crop_mapping.csv"
)


# ============================================================
# CONTACT SHEET SETTINGS
# ============================================================

CROPS_PER_SHEET = 50

COLS = 5
ROWS = 10

THUMB_W = 220
THUMB_H = 220

LABEL_H = 35

MARGIN = 20
GAP = 12

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# HELPERS
# ============================================================

def get_crop_files():
    files = []

    for path in CROP_ROOT.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in VALID_EXTENSIONS
        ):
            files.append(path)

    return sorted(
        files,
        key=lambda p: str(p).lower()
    )


def make_thumbnail(image_path):
    with Image.open(image_path) as img:

        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")

        # Preserve aspect ratio.
        img.thumbnail(
            (THUMB_W, THUMB_H)
        )

        canvas = Image.new(
            "RGB",
            (THUMB_W, THUMB_H),
            "white",
        )

        x = (
            THUMB_W - img.width
        ) // 2

        y = (
            THUMB_H - img.height
        ) // 2

        canvas.paste(
            img,
            (x, y),
        )

        return canvas


def get_metadata_from_path(crop_path):
    relative = crop_path.relative_to(CROP_ROOT)

    # Expected structure:
    #
    # MFA/MFA_001/crop_file.png
    #
    # or
    #
    # HMANE/HMANE_001/crop_file.png

    parts = relative.parts

    if len(parts) < 3:
        raise ValueError(
            f"Unexpected crop path structure: {relative}"
        )

    museum = parts[0]
    artifact_group = parts[1]

    return museum, artifact_group


# ============================================================
# MAIN
# ============================================================

def main():

    crop_files = get_crop_files()

    print(
        f"Found {len(crop_files)} training crops."
    )

    if len(crop_files) == 0:
        raise RuntimeError(
            f"No crops found in:\n{CROP_ROOT}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Create stable crop mapping
    # --------------------------------------------------------

    mapping_rows = []

    for index, crop_path in enumerate(
        crop_files,
        start=1,
    ):
        museum, artifact_group = (
            get_metadata_from_path(crop_path)
        )

        crop_id = f"T{index:04d}"

        mapping_rows.append(
            {
                "crop_id": crop_id,
                "museum": museum,
                "artifact_group": artifact_group,
                "crop_filename": crop_path.name,
                "crop_path": str(
                    crop_path.relative_to(
                        BASE_DIR
                    )
                ),
            }
        )

    with open(
        MAPPING_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fieldnames = [
            "crop_id",
            "museum",
            "artifact_group",
            "crop_filename",
            "crop_path",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(mapping_rows)

    print(
        "Wrote mapping:",
        MAPPING_PATH,
    )

    # --------------------------------------------------------
    # Sheet dimensions
    # --------------------------------------------------------

    sheet_w = (
        MARGIN * 2
        + COLS * THUMB_W
        + (COLS - 1) * GAP
    )

    sheet_h = (
        MARGIN * 2
        + ROWS * (THUMB_H + LABEL_H)
        + (ROWS - 1) * GAP
    )

    total_sheets = math.ceil(
        len(crop_files)
        / CROPS_PER_SHEET
    )

    print(
        f"Creating {total_sheets} contact sheets..."
    )

    # --------------------------------------------------------
    # Generate sheets
    # --------------------------------------------------------

    for sheet_index in range(
        total_sheets
    ):

        start = (
            sheet_index
            * CROPS_PER_SHEET
        )

        end = min(
            start + CROPS_PER_SHEET,
            len(crop_files),
        )

        subset = crop_files[
            start:end
        ]

        sheet = Image.new(
            "RGB",
            (sheet_w, sheet_h),
            "white",
        )

        draw = ImageDraw.Draw(
            sheet
        )

        for local_index, crop_path in enumerate(
            subset
        ):

            global_index = (
                start
                + local_index
                + 1
            )

            crop_id = (
                f"T{global_index:04d}"
            )

            row = (
                local_index // COLS
            )

            col = (
                local_index % COLS
            )

            x = (
                MARGIN
                + col
                * (THUMB_W + GAP)
            )

            y = (
                MARGIN
                + row
                * (
                    THUMB_H
                    + LABEL_H
                    + GAP
                )
            )

            try:

                thumb = make_thumbnail(
                    crop_path
                )

                sheet.paste(
                    thumb,
                    (x, y),
                )

                museum, artifact_group = (
                    get_metadata_from_path(
                        crop_path
                    )
                )

                label = (
                    f"{crop_id} "
                    f"{museum} "
                    f"{artifact_group}"
                )

                draw.text(
                    (
                        x,
                        y
                        + THUMB_H
                        + 8,
                    ),
                    label,
                    fill="black",
                )

            except Exception as error:

                print(
                    f"[WARN] Could not open "
                    f"{crop_path}: "
                    f"{error}"
                )

        output_path = (
            OUTPUT_ROOT
            / (
                f"train_crops_"
                f"{sheet_index + 1:03d}.jpg"
            )
        )

        sheet.save(
            output_path,
            quality=92,
        )

        print(
            f"[{sheet_index + 1:03d}/"
            f"{total_sheets:03d}] "
            f"Saved {output_path.name}"
        )

    print()
    print("=" * 60)
    print("CONTACT SHEETS COMPLETE")
    print("=" * 60)

    print(
        "Crops:",
        len(crop_files),
    )

    print(
        "Sheets:",
        total_sheets,
    )

    print(
        "Mapping:",
        MAPPING_PATH,
    )

    print(
        "Output:",
        OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()