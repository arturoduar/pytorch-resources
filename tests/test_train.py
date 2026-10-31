import pytest
import torch
from torch import nn

from pytorch_examples.data import make_linear_data, make_loader
from pytorch_examples.train import accuracy, evaluate, fit


def test_accuracy():
    logits = torch.tensor([[2.0, 0.1], [0.1, 2.0], [3.0, 0.0], [0.0, 3.0]])
    targets = torch.tensor([0, 1, 1, 1])
    assert accuracy(logits, targets) == pytest.approx(0.75)


def test_fit_reduces_loss_and_records_history():
    x, y = make_linear_data(n=128)
    loader = make_loader(x, y, batch_size=32)
    history = fit(nn.Linear(1, 1), loader, loss_fn=nn.MSELoss(), epochs=15, lr=0.05, device="cpu")
    assert len(history.train_loss) == 15
    assert history.train_loss[-1] < history.train_loss[0]


def test_fit_records_validation_metrics():
    x, y = make_linear_data(n=64)
    loader = make_loader(x, y, batch_size=32)
    val_loader = make_loader(x, y, batch_size=32, shuffle=False)
    history = fit(
        nn.Linear(1, 1),
        loader,
        val_loader,
        loss_fn=nn.MSELoss(),
        epochs=3,
        device="cpu",
        metric_fn=lambda out, target: -nn.functional.mse_loss(out, target).item(),
    )
    assert len(history.val_loss) == 3
    assert len(history.val_metric) == 3


def test_fit_rejects_unknown_optimizer():
    x, y = make_linear_data(n=16)
    with pytest.raises(ValueError):
        fit(nn.Linear(1, 1), make_loader(x, y), loss_fn=nn.MSELoss(), optimizer="nope")


def test_evaluate_without_metric_returns_none():
    x, y = make_linear_data(n=16)
    loss, metric = evaluate(
        nn.Linear(1, 1), make_loader(x, y, shuffle=False), nn.MSELoss(), torch.device("cpu")
    )
    assert loss >= 0
    assert metric is None
