"""
Data Downloader for MovieLens Latest Small Dataset.
Downloads ml-latest-small.zip and extracts movies.csv and ratings.csv into the data/ directory.
"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

# URL for MovieLens Latest Small dataset (approx 1MB)
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download_and_extract_data(target_dir: Path | None = None) -> Path:
    """Downloads and extracts MovieLens Latest Small into target_dir."""
    if target_dir is None:
        target_dir = Path(__file__).resolve().parent

    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "ml-latest-small.zip"

    # 1. Download dataset if not already downloaded
    print(f"Downloading dataset from: {DATASET_URL} ...")
    try:
        urllib.request.urlretrieve(DATASET_URL, zip_path)
        print(f"Downloaded zip archive to {zip_path} ({zip_path.stat().st_size:,} bytes)")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download dataset from {DATASET_URL}. "
            "Please check network connection or manually place ml-latest-small.zip into data/."
        ) from exc

    # 2. Extract contents
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Extract files directly into target_dir (flattening ml-latest-small/ prefix)
        for member in zip_ref.namelist():
            filename = os.path.basename(member)
            if not filename:
                continue  # Skip directory entries
            source = zip_ref.open(member)
            target = open(target_dir / filename, "wb")
            with source, target:
                shutil.copyfileobj(source, target)
            print(f"  Extracted: {filename}")

    # Remove temporary zip file
    if zip_path.exists():
        zip_path.unlink()
        print("Cleaned up temporary zip file.")

    # 3. Verify critical files
    movies_csv = target_dir / "movies.csv"
    ratings_csv = target_dir / "ratings.csv"
    if not movies_csv.exists() or not ratings_csv.exists():
        raise FileNotFoundError(f"Extraction incomplete. Missing {movies_csv} or {ratings_csv}")

    print(f"\nSuccess! Dataset ready in: {target_dir}")
    print(f" - movies.csv: {movies_csv.stat().st_size:,} bytes")
    print(f" - ratings.csv: {ratings_csv.stat().st_size:,} bytes")
    return target_dir


if __name__ == "__main__":
    download_and_extract_data()
