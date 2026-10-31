"""Synthetic datasets, so every example runs offline and finishes in seconds."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


def _generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def make_linear_data(
    n: int = 200,
    weight: float = 3.0,
    bias: float = 2.0,
    noise: float = 0.1,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``y = weight * x + bias + noise`` with ``x`` uniform in [-1, 1]. Shapes: (n, 1)."""
    g = _generator(seed)
    x = torch.rand(n, 1, generator=g) * 2 - 1
    y = weight * x + bias + noise * torch.randn(n, 1, generator=g)
    return x, y


def make_spirals(
    n_per_class: int = 100,
    n_classes: int = 3,
    noise: float = 0.2,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved 2-D spirals. Returns ``x`` (N, 2) float and ``y`` (N,) long."""
    g = _generator(seed)
    xs, ys = [], []
    for k in range(n_classes):
        radius = torch.linspace(0.05, 1.0, n_per_class)
        theta = torch.linspace(k * 4.0, (k + 1) * 4.0, n_per_class)
        theta = theta + noise * torch.randn(n_per_class, generator=g)
        xs.append(torch.stack([radius * torch.sin(theta), radius * torch.cos(theta)], dim=1))
        ys.append(torch.full((n_per_class,), k, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


def make_shape_images(
    n: int = 256,
    size: int = 16,
    noise: float = 0.3,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Noisy images of a horizontal bar (class 0) or a vertical bar (class 1).

    Returns ``images`` (n, 1, size, size) and ``labels`` (n,) long.
    """
    if size < 8:
        raise ValueError("size must be at least 8")
    g = _generator(seed)
    labels = torch.randint(0, 2, (n,), generator=g)
    positions = torch.randint(1, size - 2, (n,), generator=g)
    images = torch.zeros(n, 1, size, size)
    for i in range(n):
        p = int(positions[i])
        if int(labels[i]) == 0:
            images[i, 0, p : p + 2, :] = 1.0
        else:
            images[i, 0, :, p : p + 2] = 1.0
    images = images + noise * torch.randn(images.shape, generator=g)
    return images, labels


def make_sequences(
    n: int = 512,
    seq_len: int = 10,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random sequences labelled 1 when their sum is positive, else 0.

    Returns ``x`` (n, seq_len, 1) float and ``y`` (n,) long.
    """
    g = _generator(seed)
    x = torch.randn(n, seq_len, 1, generator=g)
    y = (x.sum(dim=(1, 2)) > 0).long()
    return x, y


def train_val_split(
    x: torch.Tensor,
    y: torch.Tensor,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shuffle and split into ``(x_train, y_train, x_val, y_val)``."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    perm = torch.randperm(x.shape[0], generator=_generator(seed))
    n_val = int(x.shape[0] * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 0,
) -> DataLoader:
    """Wrap tensors in a ``DataLoader`` with a seeded shuffle order."""
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=_generator(seed) if shuffle else None,
    )
