from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_dataset

from .push import make_metadata


def load(dir_or_repo: str, test_size=0.2) -> DatasetDict:
    dir = Path(dir_or_repo)
    is_repo = not dir.exists()

    if is_repo:
        datasets: DatasetDict = load_dataset(dir_or_repo)
    else:
        make_metadata(dir)
        datasets: DatasetDict = load_dataset("imagefolder", data_dir=dir)

    splits = list(datasets.keys())
    dataset = concatenate_datasets([datasets[split] for split in splits])
    datasets = dataset.train_test_split(test_size=test_size, seed=1922)
    return datasets
