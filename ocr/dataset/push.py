import argparse
import csv
from pathlib import Path

from datasets import load_dataset


def make_metadata(dir_path: str):
    dir = Path(dir_path)
    if not dir.exists():
        raise FileNotFoundError(f"Directory {dir} does not exist")

    metadata = dir / "metadata.csv"
    metadata.unlink(missing_ok=True)
    with open(metadata, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "text"])

        for file in dir.glob("*"):
            if file.is_dir():
                continue
            if file.suffix not in [".jpg", ".jpeg", ".png"]:
                continue
            writer.writerow([file.name, file.stem])


def upload_dataset(dir_path: str, repo: str):
    dir = Path(dir_path)
    if not dir.exists():
        raise FileNotFoundError(f"Directory {dir} does not exist")

    print(f"Uploading dataset from {dir} to {repo}...")

    make_metadata(dir)

    dataset = load_dataset("imagefolder", data_dir=dir)
    dataset.push_to_hub(repo, private=True)
    print(f"Uploaded dataset to {repo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        help="Directory to upload",
        default="data",
    )
    parser.add_argument(
        "--repo",
        type=str,
        help="Repository to upload the dataset",
    )
    args = parser.parse_args()

    if args.repo is None:
        raise ValueError("Please specify the repository to upload the dataset")

    upload_dataset(args.dir, args.repo)


if __name__ == "__main__":
    main()
