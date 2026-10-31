# Deployment

The `serve` command wraps the `mlp_classifier` model in a small HTTP server built on Python's standard library, so deployment needs nothing beyond PyTorch itself.

## Run locally

```bash
pytorch-examples serve --checkpoint model.pt
```

On first start the server trains the model and saves the weights to `model.pt`. Later starts load that file and skip training.

| Option | Meaning |
| --- | --- |
| `--host`, `--port` | bind address (default `127.0.0.1:8000`) |
| `--checkpoint PATH` | load weights from this file if it exists, otherwise train and save here |
| `--epochs N` | training epochs when there is no checkpoint |

## API

`GET /health`

```bash
curl http://127.0.0.1:8000/health
# {"status": "ok"}
```

`POST /predict` takes a list of 2-number points and returns one prediction for each:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[0.1, 0.2], [0.5, -0.5]]}'
```

```json
{"predictions": [
  {"class": 0, "probabilities": [0.91, 0.05, 0.04]},
  {"class": 2, "probabilities": [0.03, 0.12, 0.85]}
]}
```

The numbers above are illustrative. Invalid input (wrong number of features, a missing `inputs` key, malformed JSON) returns HTTP 400 with an `error` message. Any other path returns 404.

## Docker

```bash
docker build -t pytorch-examples .
docker run --rm -p 8000:8000 pytorch-examples
```

The image uses CPU-only PyTorch, runs as a non-root user, and has a health check that calls `/health`. Because the container trains at startup, the first health check is delayed by 60 seconds.

Other commands run in the same image:

```bash
docker run --rm pytorch-examples list
docker run --rm pytorch-examples run linear_regression
```

## Docker Compose

```bash
docker compose up --build
```

Compose mounts a volume at `/models` and passes `--checkpoint /models/mlp.pt`, so the trained weights survive restarts.

## Production notes

This server is a teaching example. Before exposing it publicly, consider:

- putting it behind a reverse proxy that adds TLS and authentication;
- using a production server such as FastAPI with uvicorn if you need concurrency, request validation or OpenAPI docs;
- limiting request body size, since the current handler reads the whole body into memory;
- pinning exact dependency versions in the image.
