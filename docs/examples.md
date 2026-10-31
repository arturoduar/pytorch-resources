# Examples

Every example lives in `src/pytorch_examples/examples/`, exposes `run(...)`, and uses synthetic data from `pytorch_examples.data`.

| Name | Model | Task | Concepts |
| --- | --- | --- | --- |
| `autograd_basics` | none | gradients and a hand-written gradient descent | `requires_grad`, `backward()`, `torch.no_grad()` |
| `linear_regression` | `nn.Linear` | recover `y = 3x + 2` from noisy data | loss, optimizer, `DataLoader`, training loop |
| `mlp_classifier` | `MLP` | separate three interleaved spirals | non-linear models, cross-entropy, validation accuracy |
| `cnn_images` | `SmallCNN` | horizontal vs vertical bars in noisy images | convolutions, pooling, image tensors `(N, C, H, W)` |
| `lstm_sequences` | `LSTMClassifier` | classify sequences by the sign of their sum | recurrent layers, `batch_first`, last hidden state |

## Common signature

```python
run(*, epochs=None, seed=0, device=None, verbose=False) -> dict
```

- `epochs=None` uses the example's `DEFAULT_EPOCHS`. For `autograd_basics` it is the number of gradient-descent steps.
- The returned dict only holds JSON-serialisable values, so the CLI can print it directly.
- `seed` fixes the data, the weight initialisation and the shuffle order. The same seed on CPU gives the same result.

## What each example reports

- `autograd_basics`: `autograd_grad`, `analytic_grad`, `grads_match`, `minimiser`.
- `linear_regression`: learned `weight` and `bias` next to the true values, and `final_loss`.
- `mlp_classifier`, `cnn_images`, `lstm_sequences`: `train_loss`, `val_loss`, `val_accuracy`, `chance_accuracy` and the parameter count.

## Adding your own example

1. Create `src/pytorch_examples/examples/my_example.py` with a `DESCRIPTION` string and a `run(...)` function using the signature above.
2. Register it in `examples/__init__.py`:

    ```python
    EXAMPLES = {
        ...,
        "my_example": my_example,
    }
    ```

3. It now appears in `pytorch-examples list`, runs with `pytorch-examples run my_example`, and the smoke test in `tests/test_examples.py` covers it automatically. Add a quality test next to it.
