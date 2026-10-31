"""A dependency-free HTTP inference server around the MLP classifier.

Endpoints:
    GET  /health   -> {"status": "ok"}
    POST /predict  -> body {"inputs": [[x1, x2], ...]}
                      reply {"predictions": [{"class": int, "probabilities": [...]}, ...]}
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from torch import nn

from pytorch_examples.examples import mlp_classifier


class Predictor:
    """Wraps a trained classifier and turns lists of feature vectors into predictions."""

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.in_features = int(model.in_features)

    def predict(self, inputs: list[list[float]]) -> list[dict]:
        x = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(f"inputs must be a non-empty list of {self.in_features}-number lists")
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)
        classes = probs.argmax(dim=1).tolist()
        return [
            {"class": c, "probabilities": p} for c, p in zip(classes, probs.tolist(), strict=True)
        ]


def build_predictor(
    checkpoint: str | Path | None = None,
    epochs: int | None = None,
    seed: int = 0,
) -> Predictor:
    """Load weights from ``checkpoint`` if it exists, otherwise train (and save if a path is given)."""
    if checkpoint is not None and Path(checkpoint).exists():
        model = mlp_classifier.build_model()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    else:
        model, _ = mlp_classifier.train(epochs=epochs, seed=seed, device="cpu")
        if checkpoint is not None:
            torch.save(model.state_dict(), checkpoint)
    return Predictor(model)


def make_server(
    predictor: Predictor, host: str = "127.0.0.1", port: int = 8000
) -> ThreadingHTTPServer:
    """Create (but do not start) an HTTP server. Use port 0 to let the OS pick a free port."""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (name fixed by BaseHTTPRequestHandler)
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/predict":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                predictions = predictor.predict(payload["inputs"])
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                self._send(400, {"error": f"bad request: {exc}"})
                return
            self._send(200, {"predictions": predictions})

    return ThreadingHTTPServer((host, port), Handler)
