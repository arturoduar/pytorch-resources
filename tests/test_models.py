import pytest
import torch

from pytorch_examples.models import MLP, LSTMClassifier, SmallCNN


def test_mlp_output_shape():
    model = MLP(in_features=2, hidden=8, out_features=3, depth=2)
    assert model(torch.randn(5, 2)).shape == (5, 3)


def test_mlp_rejects_zero_depth():
    with pytest.raises(ValueError):
        MLP(2, 8, 3, depth=0)


def test_cnn_output_shape():
    model = SmallCNN(in_channels=1, num_classes=2, image_size=16)
    assert model(torch.randn(4, 1, 16, 16)).shape == (4, 2)


def test_cnn_rejects_image_size_not_divisible_by_four():
    with pytest.raises(ValueError):
        SmallCNN(image_size=18)


def test_lstm_output_shape():
    model = LSTMClassifier(input_size=1, hidden_size=8, num_classes=2)
    assert model(torch.randn(6, 10, 1)).shape == (6, 2)


@pytest.mark.parametrize(
    "model, batch",
    [
        (MLP(2, 8, 3), torch.randn(4, 2)),
        (SmallCNN(), torch.randn(4, 1, 16, 16)),
        (LSTMClassifier(hidden_size=8), torch.randn(4, 10, 1)),
    ],
)
def test_gradients_flow_to_every_parameter(model, batch):
    model(batch).sum().backward()
    assert all(p.grad is not None for p in model.parameters())
