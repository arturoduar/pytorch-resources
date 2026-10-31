# pytorch-resources

Runnable, tested and deployable PyTorch examples, with documentation and Jupyter notebooks.

| Example | What it shows |
| --- | --- |
| `autograd_basics` | gradients and gradient descent by hand |
| `linear_regression` | a training loop on a single `nn.Linear` layer |
| `mlp_classifier` | classifying interleaved spirals with an MLP |
| `cnn_images` | a small CNN on noisy bar images |
| `lstm_sequences` | an LSTM classifying sequences |

All of them train on synthetic data, so nothing is downloaded and each run takes seconds on a CPU.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebooks,docs]"

pytorch-examples list
pytorch-examples run linear_regression --verbose
pytest
```

## What is in the box

```
pytorch_resources/
├── src/pytorch_resources/
│   ├── examples/          five runnable examples + registry
│   ├── data.py            synthetic datasets and loaders
│   ├── models.py          MLP, SmallCNN, LSTMClassifier
│   ├── train.py           fit / evaluate / accuracy
│   ├── serve.py           HTTP inference server (standard library only)
│   ├── cli.py             `pytorch-examples list | info | run | serve`
│   └── utils.py           seeding, device selection
├── tests/                 pytest suite, including notebook checks
├── notebooks/             example .ipynb notebooks
├── docs/ + mkdocs.yml     documentation site
├── Dockerfile             CPU-only image that serves the model
├── docker-compose.yml
├── Makefile               install, test, docs, lab, serve, docker targets
└── .github/workflows/     CI: tests, docs build, Docker smoke test
```

## Common tasks

| Task | Command |
| --- | --- |
| run all tests | `make test` or `pytest` |
| skip notebook execution | `pytest -m "not notebooks"` |
| open the notebooks | `make lab` |
| preview the docs | `make docs-serve` |
| serve the model | `pytorch-examples serve --checkpoint model.pt` |
| run in Docker | `docker build -t pytorch-examples . && docker run --rm -p 8000:8000 pytorch-examples` |

Try the server once it is running:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" -d '{"inputs": [[0.1, 0.2]]}'
```

See the `docs/` folder (or run `make docs-serve`) for the full guide.
