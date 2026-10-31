"""Registry of runnable examples.

Every example module exposes ``DESCRIPTION`` and ``run(*, epochs, seed, device, verbose) -> dict``
where the dict contains only JSON-serialisable values.
"""

from __future__ import annotations

from pytorch_examples.examples import (
    autograd_basics,
    cnn_images,
    linear_regression,
    lstm_sequences,
    mlp_classifier,
)

EXAMPLES = {
    "autograd_basics": autograd_basics,
    "linear_regression": linear_regression,
    "mlp_classifier": mlp_classifier,
    "cnn_images": cnn_images,
    "lstm_sequences": lstm_sequences,
}


def describe() -> dict[str, str]:
    """Map each example name to its one-line description."""
    return {name: module.DESCRIPTION for name, module in EXAMPLES.items()}
