"""Smoke tests for every example plus a quality check on each learning task.

Accuracy thresholds are deliberately far above chance but well below what the models reach,
so the tests catch broken training without being flaky.
"""

import json

import pytest

from pytorch_examples.examples import EXAMPLES, describe


def test_registry_has_descriptions_for_every_example():
    descriptions = describe()
    assert set(descriptions) == set(EXAMPLES)
    assert all(text for text in descriptions.values())


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_example_runs_and_returns_json_serialisable_metrics(name):
    result = EXAMPLES[name].run(epochs=2, seed=0, device="cpu")
    assert isinstance(result, dict) and result
    json.dumps(result)


def test_autograd_basics_matches_analytic_gradient_and_finds_minimum():
    result = EXAMPLES["autograd_basics"].run()
    assert result["grads_match"]
    assert result["minimiser"] == pytest.approx(5.0, abs=1e-3)


def test_linear_regression_recovers_true_parameters():
    result = EXAMPLES["linear_regression"].run(seed=0, device="cpu")
    assert result["weight"] == pytest.approx(3.0, abs=0.3)
    assert result["bias"] == pytest.approx(2.0, abs=0.3)


def test_mlp_classifier_beats_chance_by_a_wide_margin():
    result = EXAMPLES["mlp_classifier"].run(epochs=80, seed=0, device="cpu")
    assert result["val_accuracy"] > 0.6  # chance is 0.33


def test_cnn_images_learns_bar_orientation():
    result = EXAMPLES["cnn_images"].run(seed=0, device="cpu")
    assert result["val_accuracy"] > 0.8  # chance is 0.5


def test_lstm_sequences_beats_chance():
    result = EXAMPLES["lstm_sequences"].run(seed=0, device="cpu")
    assert result["val_accuracy"] > 0.7  # chance is 0.5


def test_runs_are_reproducible_for_a_fixed_seed():
    first = EXAMPLES["linear_regression"].run(epochs=5, seed=3, device="cpu")
    second = EXAMPLES["linear_regression"].run(epochs=5, seed=3, device="cpu")
    assert first == second
