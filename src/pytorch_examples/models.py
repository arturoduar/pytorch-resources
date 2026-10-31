"""Small model definitions used by the examples."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Fully connected network: ``depth`` hidden layers of width ``hidden`` with ReLU."""

    def __init__(self, in_features: int, hidden: int, out_features: int, depth: int = 2) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        layers: list[nn.Module] = []
        previous = in_features
        for _ in range(depth):
            layers += [nn.Linear(previous, hidden), nn.ReLU()]
            previous = hidden
        layers.append(nn.Linear(previous, out_features))
        self.net = nn.Sequential(*layers)
        self.in_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallCNN(nn.Module):
    """Two conv blocks (conv, ReLU, 2x2 max-pool) followed by a linear classifier."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2, image_size: int = 16) -> None:
        super().__init__()
        if image_size % 4 != 0:
            raise ValueError("image_size must be divisible by 4")
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        side = image_size // 4
        self.classifier = nn.Linear(16 * side * side, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(start_dim=1))


class LSTMClassifier(nn.Module):
    """Classify a sequence from the last hidden state of an LSTM."""

    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_classes: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])
