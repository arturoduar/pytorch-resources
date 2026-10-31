"""Linear regression with a single ``nn.Linear`` layer."""

from __future__ import annotations

from torch import nn

from pytorch_examples.data import make_linear_data, make_loader
from pytorch_examples.train import fit
from pytorch_examples.utils import set_seed

DESCRIPTION = "Linear regression: recover y = 3x + 2 from noisy data with one nn.Linear layer."
DEFAULT_EPOCHS = 60
TRUE_WEIGHT = 3.0
TRUE_BIAS = 2.0


def run(
    *,
    epochs: int | None = None,
    seed: int = 0,
    device: str | None = None,
    verbose: bool = False,
) -> dict:
    epochs = DEFAULT_EPOCHS if epochs is None else epochs
    set_seed(seed)
    x, y = make_linear_data(weight=TRUE_WEIGHT, bias=TRUE_BIAS, seed=seed)
    loader = make_loader(x, y, batch_size=32, seed=seed)

    model = nn.Linear(1, 1)
    history = fit(
        model, loader, loss_fn=nn.MSELoss(), epochs=epochs, lr=0.05, device=device, verbose=verbose
    )
    return {
        "weight": model.weight.item(),
        "bias": model.bias.item(),
        "true_weight": TRUE_WEIGHT,
        "true_bias": TRUE_BIAS,
        "final_loss": history.train_loss[-1],
        "epochs": epochs,
    }
