import torch
from torch import nn

from pytorch_examples.utils import count_parameters, get_device, set_seed


def test_set_seed_makes_random_numbers_reproducible():
    set_seed(123)
    a = torch.rand(5)
    set_seed(123)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_get_device_prefers_explicit_choice():
    assert get_device("cpu") == torch.device("cpu")


def test_get_device_default_is_a_device():
    assert isinstance(get_device(), torch.device)


def test_count_parameters():
    assert count_parameters(nn.Linear(3, 2)) == 3 * 2 + 2


def test_count_parameters_ignores_frozen_weights():
    layer = nn.Linear(3, 2)
    layer.weight.requires_grad_(False)
    assert count_parameters(layer) == 2
