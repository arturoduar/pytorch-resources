"""Image classification with a small CNN on synthetic bar images."""

from __future__ import annotations

from torch import nn

from pytorch_examples.data import make_loader, make_shape_images, train_val_split
from pytorch_examples.models import SmallCNN
from pytorch_examples.train import accuracy, fit
from pytorch_examples.utils import count_parameters, set_seed

DESCRIPTION = "CNN: tell horizontal bars from vertical bars in noisy 16x16 images."
DEFAULT_EPOCHS = 12
IMAGE_SIZE = 16


def run(
    *,
    epochs: int | None = None,
    seed: int = 0,
    device: str | None = None,
    verbose: bool = False,
) -> dict:
    epochs = DEFAULT_EPOCHS if epochs is None else epochs
    set_seed(seed)
    images, labels = make_shape_images(n=256, size=IMAGE_SIZE, seed=seed)
    x_tr, y_tr, x_val, y_val = train_val_split(images, labels, val_fraction=0.2, seed=seed)
    train_loader = make_loader(x_tr, y_tr, batch_size=32, seed=seed)
    val_loader = make_loader(x_val, y_val, batch_size=64, shuffle=False)

    model = SmallCNN(in_channels=1, num_classes=2, image_size=IMAGE_SIZE)
    history = fit(
        model,
        train_loader,
        val_loader,
        loss_fn=nn.CrossEntropyLoss(),
        epochs=epochs,
        lr=5e-3,
        device=device,
        metric_fn=accuracy,
        verbose=verbose,
    )
    return {
        "train_loss": history.train_loss[-1],
        "val_loss": history.val_loss[-1],
        "val_accuracy": history.val_metric[-1],
        "chance_accuracy": 0.5,
        "parameters": count_parameters(model),
        "epochs": epochs,
    }
