# Getting started

## Install

You need Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,notebooks,docs]"
```

`pip` installs the default PyTorch build for your platform. For a smaller CPU-only install on Linux:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,notebooks,docs]"
```

The extras are optional:

| Extra | Adds |
| --- | --- |
| `dev` | pytest |
| `notebooks` | JupyterLab, matplotlib, nbformat and nbclient (to execute notebooks in tests) |
| `docs` | mkdocs |

## Run an example

```bash
pytorch-examples list
pytorch-examples run linear_regression --verbose
pytorch-examples run mlp_classifier --epochs 100
pytorch-examples info          # PyTorch version and the device in use
```

Every `run` prints its metrics as JSON, so results are easy to script against.

Options for `run`:

| Option | Meaning |
| --- | --- |
| `--epochs N` | override the example's default epoch count |
| `--seed N` | random seed (default 0) |
| `--device D` | `cpu`, `cuda` or `mps` (default: best available) |
| `--verbose` | print training progress |

## Use it from Python

```python
from pytorch_examples.examples import EXAMPLES

result = EXAMPLES["linear_regression"].run(seed=1, device="cpu")
print(result["weight"], result["bias"])   # close to 3 and 2
```

## Build the docs

```bash
mkdocs serve          # live preview at http://127.0.0.1:8000
mkdocs build --strict
```

If the preview's port clashes with the inference server, use `mkdocs serve -a 127.0.0.1:8001`.
