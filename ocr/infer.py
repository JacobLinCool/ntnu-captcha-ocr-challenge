import argparse
from typing import List

import torch
from accelerate import Accelerator
from datasets import Dataset, Image
from torch.utils.data import DataLoader

from .model.baseline import Baseline2024
from .utils import tensors_to_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Directory or repository to load the model",
        default="JacobLinCool/ntnu-captcha-ocr-challenge-baseline",
    )
    parser.add_argument(
        "captchas",
        type=str,
        nargs="+",
        help="Captcha images to infer",
    )
    args = parser.parse_args()

    accelerator = Accelerator()

    infer(args.model, accelerator, args.captchas)


def infer(model: str, accelerator: Accelerator, captchas: List[str]):
    model = Baseline2024.from_pretrained(model)
    model = model.to(accelerator.device)

    dataset = (
        Dataset.from_dict({"image": captchas, "path": captchas})
        .cast_column("image", Image())
        .with_format("torch")
    )
    loader = DataLoader(dataset, batch_size=16)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(accelerator.device).float() / 255.0
            output = model(image)
            texts = tensors_to_texts(output)
            for i, text in enumerate(texts):
                file = batch["path"][i]
                print(f"{file} : {text}")


if __name__ == "__main__":
    main()
