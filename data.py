"""Loading the metadata CSV and matching it to the image folders.
"""

from pathlib import Path

import pandas as pd

from . import config

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def load_metadata(path=config.METADATA_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found at {path}")

    df = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False, na_filter=False)

    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Metadata is missing required columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[config.REQUIRED_COLUMNS].copy()
    df["Code"] = df["Code"].astype(str).str.strip()
    return df


def collect_image_paths(image_dir=config.IMAGE_DIR):
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found at {image_dir}")

    paths_by_code = {}
    for folder in sorted(p for p in image_dir.iterdir() if p.is_dir()):
        images = sorted(
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        paths_by_code[folder.name] = images
    return paths_by_code


def build_image_dataframe(metadata_df, image_dir=config.IMAGE_DIR):
    paths_by_code = collect_image_paths(image_dir)

    rows = []
    for _, r in metadata_df.iterrows():
        code = r["Code"]
        for image_path in paths_by_code.get(code, []):
            rows.append(
                {
                    "Code": code,
                    "image_path": str(image_path),
                    "Material": r["Material"],
                    "Density": r["Density"],
                    "Percentage": r["Percentage"],
                    "Weight": r["Weight"],
                    "Total weight": r["Total weight"],
                }
            )

    columns = [
        "Code",
        "image_path",
        "Material",
        "Density",
        "Percentage",
        "Weight",
        "Total weight",
    ]
    return pd.DataFrame(rows, columns=columns)


def validate_dataset(
    metadata_df,
    image_dir=config.IMAGE_DIR,
    images_per_code=config.IMAGES_PER_CODE,
    strict=True,
):
    
    image_dir = Path(image_dir)

    codes = list(metadata_df["Code"])
    code_set = set(codes)
    duplicate_codes = sorted({c for c in codes if codes.count(c) > 1})

    folders = {p.name for p in image_dir.iterdir() if p.is_dir()}
    paths_by_code = collect_image_paths(image_dir)

    missing_folders = sorted(code_set - folders)  
    extra_folders = sorted(folders - code_set)  

    image_counts = {c: len(paths_by_code.get(c, [])) for c in codes}
    wrong_image_counts = {
        c: n for c, n in image_counts.items() if n != images_per_code
    }

    problems = []
    if duplicate_codes:
        problems.append(f"Duplicate Codes in metadata: {duplicate_codes}")
    if missing_folders:
        problems.append(f"Codes with no image folder: {missing_folders}")
    if extra_folders:
        problems.append(f"Image folders with no metadata row: {extra_folders}")
    if wrong_image_counts:
        problems.append(
            f"Codes without exactly {images_per_code} images: {wrong_image_counts}"
        )

    report = {
        "n_mixtures": len(codes),
        "n_unique_codes": len(code_set),
        "n_folders": len(folders),
        "n_images": sum(image_counts.values()),
        "duplicate_codes": duplicate_codes,
        "missing_folders": missing_folders,
        "extra_folders": extra_folders,
        "wrong_image_counts": wrong_image_counts,
        "problems": problems,
        "ok": len(problems) == 0,
    }

    if strict and problems:
        raise ValueError("Dataset validation failed:\n  - " + "\n  - ".join(problems))

    return report
