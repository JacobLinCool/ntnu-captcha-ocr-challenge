import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from huggingface_hub import HfApi

from .dataset.load import load
from .model.baseline import Baseline2024
from .utils import texts_to_tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="Directory or repository to load the dataset",
        default="GDSC-NTNU/ntnu-captcha-ocr-challenge-2024",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for training",
        default=64,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs for training",
        default=10000,
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate for training",
        default=1e-3,
    )
    parser.add_argument(
        "--wd",
        type=float,
        help="Weight decay for training",
        default=1e-3,
    )
    parser.add_argument(
        "--push",
        type=str,
        help="Repository to push the model",
    )
    args = parser.parse_args()

    accelerator = Accelerator()

    train(args, accelerator)


def train(args, accelerator: Accelerator):
    datasets = load(args.dataset).with_format("torch")

    train_loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False)

    model = Baseline2024()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    writer = SummaryWriter()
    sample_batch = next(iter(train_loader))
    writer.add_graph(model, sample_batch["image"].float().to(accelerator.device))

    epoch_tqdm = tqdm(range(args.epochs), desc="Training", unit="epoch")

    for epoch in epoch_tqdm:
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            x, y = batch["image"], batch["text"]
            x = x.float() / 255.0
            y = texts_to_tensors(y).to(x.device)
            y_hat = model(x)
            y_hat = y_hat.permute(0, 2, 1)
            loss = criterion(y_hat, y)
            accelerator.backward(loss)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["image"], batch["text"]
                x = x.float() / 255.0
                y = texts_to_tensors(y).to(x.device)
                y_hat = model(x)
                y_hat = y_hat.permute(0, 2, 1)
                loss = criterion(y_hat, y)
                val_loss += loss.item()

                _, predicted = torch.max(y_hat, dim=1)
                total += y.size(0)
                correct += (predicted == y).all(dim=1).sum().item()

        val_loss /= len(val_loader)
        accuracy = correct / total

        epoch_tqdm.set_postfix(
            {
                "Train Loss": f"{train_loss:.4f}",
                "Val Loss": f"{val_loss:.4f}",
                "Accuracy": f"{accuracy:.2f}",
            }
        )

        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Validation/Loss", val_loss, epoch)
        writer.add_scalar("Validation/Accuracy", accuracy, epoch)

    writer.close()

    if args.push is not None:
        accelerator.wait_for_everyone()
        model = accelerator.unwrap_model(model)
        model.push_to_hub(args.push, private=True)
        print(f"Pushed model to {args.push}")
        api = HfApi()
        api.upload_folder(
            repo_id=args.push,
            folder_path=writer.get_logdir(),
            path_in_repo=writer.get_logdir(),
        )

    else:
        model = accelerator.unwrap_model(model)
        model.save_pretrained("pretrained")


if __name__ == "__main__":
    main()
