# Notebooks

The `notebooks/` folder holds Jupyter notebooks that explain the same ideas interactively. Each one ends with an `assert` that checks its own result, so a notebook that runs cleanly is also a passing test.

| Notebook | Topic |
| --- | --- |
| `01_tensors_and_autograd.ipynb` | tensors, broadcasting, `backward()`, gradient descent by hand |
| `02_linear_regression.ipynb` | a training loop written out step by step, with a plot of the fit |
| `03_mlp_spirals.ipynb` | the `fit` helper, an MLP, and a decision-boundary plot |
| `04_cnn_shapes.ipynb` | image tensors, a CNN, and sample predictions |
| `05_serving_a_model.ipynb` | saving a checkpoint and calling the HTTP server from a notebook |

## Run them

```bash
pip install -e ".[notebooks]"
jupyter lab notebooks
```

The notebooks import `pytorch_examples`, so install the project into the same environment as the Jupyter kernel (the `pip install -e` above does this).

## Test them

The notebooks are checked in two ways by `tests/test_notebooks.py`:

- **Structure** (always runs): valid notebook JSON, some explanatory text, and every code cell parses.
- **Execution** (marker `notebooks`): each notebook runs top to bottom in a fresh kernel. This is skipped when the `notebooks` extra is not installed.

```bash
pytest tests/test_notebooks.py          # both checks
pytest -m "not notebooks"               # everything except notebook execution
```

If execution is skipped or fails with a kernel error, register the kernel for the active environment:

```bash
python -m ipykernel install --user --name python3
```

## Keep outputs out of git

The notebooks are committed without outputs, which keeps diffs small. Clear outputs before committing with *Edit → Clear Outputs of All Cells* in JupyterLab.
