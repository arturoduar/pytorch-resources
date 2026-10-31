"""A small, readable training loop shared by the examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_examples.utils import get_device

MetricFn = Callable[[torch.Tensor, torch.Tensor], float]

_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


@dataclass
class History:
    """Per-epoch metrics recorded by :func:`fit`."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_metric: list[float] = field(default_factory=list)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of rows where the arg-max of ``logits`` equals ``targets``."""
    return (logits.argmax(dim=1) == targets).float().mean().item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    metric_fn: MetricFn | None = None,
) -> tuple[float, float | None]:
    """Return ``(mean_loss, mean_metric)`` over ``loader``; metric is None without ``metric_fn``."""
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    count = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        batch = xb.shape[0]
        total_loss += loss_fn(out, yb).item() * batch
        if metric_fn is not None:
            total_metric += metric_fn(out, yb) * batch
        count += batch
    return total_loss / count, (total_metric / count if metric_fn is not None else None)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    *,
    loss_fn: nn.Module,
    epochs: int = 20,
    lr: float = 1e-2,
    optimizer: str = "adam",
    device: str | torch.device | None = None,
    metric_fn: MetricFn | None = None,
    verbose: bool = False,
) -> History:
    """Train ``model`` in place and return the per-epoch :class:`History`."""
    if optimizer not in _OPTIMIZERS:
        raise ValueError(f"unknown optimizer {optimizer!r}; choose from {sorted(_OPTIMIZERS)}")
    device = get_device(device)
    model.to(device)
    opt = _OPTIMIZERS[optimizer](model.parameters(), lr=lr)
    history = History()

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.shape[0]
            seen += xb.shape[0]
        history.train_loss.append(running / seen)

        message = f"epoch {epoch:3d}/{epochs}  train_loss={history.train_loss[-1]:.4f}"
        if val_loader is not None:
            val_loss, val_metric = evaluate(model, val_loader, loss_fn, device, metric_fn)
            history.val_loss.append(val_loss)
            message += f"  val_loss={val_loss:.4f}"
            if val_metric is not None:
                history.val_metric.append(val_metric)
                message += f"  val_metric={val_metric:.3f}"
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(message)

    return history
