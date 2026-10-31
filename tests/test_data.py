import pytest
import torch

from pytorch_examples.data import (
    make_linear_data,
    make_loader,
    make_sequences,
    make_shape_images,
    make_spirals,
    train_val_split,
)


def test_linear_data_shapes():
    x, y = make_linear_data(n=50)
    assert x.shape == (50, 1)
    assert y.shape == (50, 1)


def test_linear_data_is_deterministic_per_seed():
    x1, y1 = make_linear_data(seed=7)
    x2, y2 = make_linear_data(seed=7)
    x3, _ = make_linear_data(seed=8)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert not torch.equal(x1, x3)


def test_spirals_shapes_and_labels():
    x, y = make_spirals(n_per_class=20, n_classes=3)
    assert x.shape == (60, 2)
    assert y.dtype == torch.long
    assert set(y.tolist()) == {0, 1, 2}


def test_shape_images_shapes_and_labels():
    images, labels = make_shape_images(n=32, size=16)
    assert images.shape == (32, 1, 16, 16)
    assert set(labels.tolist()) <= {0, 1}


def test_shape_images_rejects_tiny_size():
    with pytest.raises(ValueError):
        make_shape_images(size=4)


def test_sequences_labels_match_sign_of_sum():
    x, y = make_sequences(n=64, seq_len=8)
    assert x.shape == (64, 8, 1)
    assert torch.equal(y, (x.sum(dim=(1, 2)) > 0).long())


def test_train_val_split_partitions_the_data():
    x = torch.arange(100, dtype=torch.float32).unsqueeze(1)
    y = torch.arange(100)
    x_tr, y_tr, x_val, y_val = train_val_split(x, y, val_fraction=0.2)
    assert len(x_tr) == 80 and len(x_val) == 20
    assert set(y_tr.tolist()).isdisjoint(y_val.tolist())
    assert len(set(y_tr.tolist()) | set(y_val.tolist())) == 100


def test_train_val_split_rejects_bad_fraction():
    x, y = torch.zeros(10, 1), torch.zeros(10)
    with pytest.raises(ValueError):
        train_val_split(x, y, val_fraction=1.5)


def test_loader_batches():
    x, y = make_linear_data(n=70)
    batches = list(make_loader(x, y, batch_size=32))
    assert [len(xb) for xb, _ in batches] == [32, 32, 6]
