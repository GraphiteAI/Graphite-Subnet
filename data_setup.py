import argparse
import shutil
from pathlib import Path
from graphite.data.dataset_utils import download_default_datasets

DATASET_DIR = Path(__file__).resolve().parent.joinpath("dataset")

def main(replace_files: bool):
    if replace_files and DATASET_DIR.exists():
        # Remove the dataset directory and all files in it
        shutil.rmtree(DATASET_DIR)
        print(f"Removed {DATASET_DIR}")
        
    # Download default datasets
    download_default_datasets()
    print("Datasets downloaded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handle dataset downloading and management.")
    parser.add_argument(
        "--replace_files",
        action="store_true",
        help="Remove the existing dataset directory and recreate all default files."
    )
    args = parser.parse_args()

    main(replace_files=args.replace_files)
