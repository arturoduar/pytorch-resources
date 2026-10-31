# API reference

## `pytorch_examples.utils`

| Function | Description |
| --- | --- |
| `set_seed(seed=0)` | seed Python, NumPy and PyTorch (including CUDA) |
| `get_device(prefer=None)` | `prefer` if given, otherwise CUDA, then MPS, then CPU |
| `count_parameters(model)` | number of trainable parameters |

## `pytorch_examples.data`

All generators are deterministic for a given `seed`.

| Function | Returns |
| --- | --- |
| `make_linear_data(n=200, weight=3.0, bias=2.0, noise=0.1, seed=0)` | `x`, `y`, both `(n, 1)` |
| `make_spirals(n_per_class=100, n_classes=3, noise=0.2, seed=0)` | `x` `(N, 2)`, `y` `(N,)` long |
| `make_shape_images(n=256, size=16, noise=0.3, seed=0)` | images `(n, 1, size, size)`, labels `(n,)`; class 0 is a horizontal bar, class 1 a vertical bar |
| `make_sequences(n=512, seq_len=10, seed=0)` | `x` `(n, seq_len, 1)`, `y` `(n,)`; label 1 if the sum is positive |
| `train_val_split(x, y, val_fraction=0.2, seed=0)` | `x_train, y_train, x_val, y_val` |
| `make_loader(x, y, batch_size=32, shuffle=True, seed=0)` | a `DataLoader` with a seeded shuffle order |

## `pytorch_examples.models`

| Class | Constructor | Input to output |
| --- | --- | --- |
| `MLP` | `MLP(in_features, hidden, out_features, depth=2)` | `(N, in_features)` to `(N, out_features)` |
| `SmallCNN` | `SmallCNN(in_channels=1, num_classes=2, image_size=16)` | `(N, C, H, W)` to `(N, num_classes)`; `image_size` must be divisible by 4 |
| `LSTMClassifier` | `LSTMClassifier(input_size=1, hidden_size=32, num_classes=2)` | `(N, T, input_size)` to `(N, num_classes)` |

## `pytorch_examples.train`

`fit(model, train_loader, val_loader=None, *, loss_fn, epochs=20, lr=1e-2, optimizer="adam", device=None, metric_fn=None, verbose=False) -> History`

Trains `model` in place. `optimizer` is one of `adam`, `adamw` or `sgd`. `History` has the lists `train_loss`, `val_loss` and `val_metric`, with one entry per epoch.

`evaluate(model, loader, loss_fn, device, metric_fn=None) -> (loss, metric_or_None)`

`accuracy(logits, targets) -> float`

## `pytorch_examples.examples`

`EXAMPLES` maps names to example modules; `describe()` maps names to one-line descriptions. See [Examples](examples.md).

## `pytorch_examples.serve`

| Name | Description |
| --- | --- |
| `Predictor(model, device="cpu")` | `predict(inputs)` takes a list of feature lists and returns `{"class", "probabilities"}` dicts; raises `ValueError` for bad shapes |
| `build_predictor(checkpoint=None, epochs=None, seed=0)` | load from `checkpoint` if it exists, otherwise train (and save when a path is given) |
| `make_server(predictor, host="127.0.0.1", port=8000)` | returns a `ThreadingHTTPServer`; pass `port=0` to pick a free port |
