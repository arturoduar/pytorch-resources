# Testing

```bash
pip install -e ".[dev,notebooks]"
pytest                       # everything
pytest -m "not notebooks"    # skip notebook execution
pytest tests/test_examples.py -k linear
```

## What is covered

| File | Checks |
| --- | --- |
| `test_utils.py` | seeding, device selection, parameter counting |
| `test_data.py` | dataset shapes, determinism per seed, splits, loaders |
| `test_models.py` | output shapes, input validation, gradients reach every parameter |
| `test_train.py` | `fit` reduces loss, records validation metrics, rejects bad options |
| `test_examples.py` | every example runs and returns JSON; each learning task beats a quality bar |
| `test_cli.py` | `list`, `info`, `run`, argument errors |
| `test_serve.py` | the predictor, the live HTTP endpoints (on a free port), checkpoint round trip |
| `test_notebooks.py` | notebook structure, and execution of each notebook |

## Quality thresholds

The learning tests assert accuracy well above chance (for example, more than 0.6 on a 3-class task where chance is 0.33). The bar is deliberately lower than what the models normally reach, so a broken training loop fails the test while ordinary run-to-run variation does not.

If you change a dataset or model and a threshold fails, check the metrics with `pytorch-examples run NAME --verbose` before deciding whether to move the bar.

## Continuous integration

`.github/workflows/ci.yml` runs the suite on Python 3.10 to 3.12, builds the docs with `--strict`, and builds and smoke-tests the Docker image.
