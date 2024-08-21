"""
The baseline model is a simple convolutional neural network with 6 convolutional layers and 4 detection heads, one for each character in the captcha image.

To see the model architecture, parameters, and i/o shape, run this script with `python -m ocr.model.baseline`.

Parts of the model are derived from the implementation in https://github.com/aisu-programming/NTNU-Validation-Code-Recognition
"""

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor


class DetectionHeads(nn.Module):
    def __init__(self, input_dim: int, class_num: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.GELU(),
                    nn.Linear(64, 32),
                    nn.GELU(),
                    nn.Linear(32, 16),
                    nn.GELU(),
                    nn.Linear(16, 8),
                    nn.GELU(),
                )
                for _ in range(4)
            ]
        )
        self.proj = nn.Linear(8, class_num)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch_size, input_dim)
        # output: (batch_size, 4, class_num)
        y = torch.stack([self.proj(self.heads[i](x)) for i in range(4)], dim=1)
        return y


class Baseline2024(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        class_num: int = 26 + 10 + 3,
        n_channels: int = 32,
        p_dropout: float = 0.95,
    ):
        super().__init__()
        self.act = nn.GELU()
        self.pool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool1d = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0)

        self.conv1 = nn.Conv2d(3, n_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv2 = nn.Conv2d(
            n_channels, n_channels * 2, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(n_channels * 2)
        self.conv3 = nn.Conv2d(
            n_channels * 2, n_channels * 4, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(n_channels * 4)
        self.conv4 = nn.Conv2d(
            n_channels * 4, n_channels * 8, kernel_size=3, stride=1, padding=1
        )
        self.bn4 = nn.BatchNorm2d(n_channels * 8)
        self.conv5 = nn.Conv2d(
            n_channels * 8, n_channels * 16, kernel_size=3, stride=1, padding=1
        )
        self.bn5 = nn.BatchNorm2d(n_channels * 16)
        self.conv6 = nn.Conv2d(
            n_channels * 16, n_channels * 32, kernel_size=3, stride=1, padding=1
        )
        self.bn6 = nn.BatchNorm2d(n_channels * 32)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p_dropout)
        self.heads = DetectionHeads(n_channels * 32, class_num)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch_size, 3, 30, 108)
        # output: (batch_size, 4, class_num)
        x = self.conv1(x)
        x = self.act(x)
        x = self.pool2d(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.pool2d(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.act(x)
        x = self.pool2d(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.act(x)
        x = self.pool2d(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = self.act(x)
        x = self.pool1d(x)
        x = self.bn5(x)

        x = self.conv6(x)
        x = self.act(x)
        x = self.pool1d(x)
        x = self.bn6(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.heads(x)
        return x


if __name__ == "__main__":
    model = Baseline2024()
    print(model)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {params:,}")

    x = torch.randn(2, 3, 30, 108)
    print(f"Input Shape: {x.shape}")
    y = model(x)
    print(f"Output Shape: {y.shape}")
