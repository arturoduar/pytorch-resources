"""Multi-class classification of 2-D spirals with an MLP."""

from __future__ import annotations

from torch import nn

from pytorch_examples.data import make_loader, make_spirals, train_val_split
from pytorch_examples.models import MLP
from pytorch_examples.train import accuracy, fit
from pytorch_examples.utils import count_parameters, set_seed

DESCRIPTION = "MLP classifier: separate three interleaved spirals (also the model behind `serve`)."
DEFAULT_EPOCHS = 150
IN_FEATURES = 2
NUM_CLASSES = 3
HIDDEN = 64
DEPTH = 2


def build_model() -> MLP:
    """Fresh, untrained model with the architecture used here and by the inference server."""
    return MLP(IN_FEATURES, HIDDEN, NUM_CLASSES, depth=DEPTH)


def train(
    *,
    epochs: int | None = None,
    seed: int = 0,
    device: str | None = None,
    verbose: bool = False,
) -> tuple[MLP, dict]:
    """Train the model and return ``(model_on_cpu, metrics)``."""
    epochs = DEFAULT_EPOCHS if epochs is None else epochs
    set_seed(seed)
    x, y = make_spirals(n_per_class=100, n_classes=NUM_CLASSES, seed=seed)
    x_tr, y_tr, x_val, y_val = train_val_split(x, y, val_fraction=0.2, seed=seed)
    train_loader = make_loader(x_tr, y_tr, batch_size=32, seed=seed)
    val_loader = make_loader(x_val, y_val, batch_size=64, shuffle=False)

    model = build_model()
    history = fit(
        model,
        train_loader,
        val_loader,
        loss_fn=nn.CrossEntropyLoss(),
        epochs=epochs,
        lr=1e-2,
        device=device,
        metric_fn=accuracy,
        verbose=verbose,
    )
    model = model.cpu()
    metrics = {
        "train_loss": history.train_loss[-1],
        "val_loss": history.val_loss[-1],
        "val_accuracy": history.val_metric[-1],
        "chance_accuracy": 1.0 / NUM_CLASSES,
        "parameters": count_parameters(model),
        "epochs": epochs,
    }
    return model, metrics


def run(
    *,
    epochs: int | None = None,
    seed: int = 0,
    device: str | None = None,
    verbose: bool = False,
) -> dict:
    _, metrics = train(epochs=epochs, seed=seed, device=device, verbose=verbose)
    return metrics
