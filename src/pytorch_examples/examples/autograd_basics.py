"""Autograd basics: analytic vs. automatic gradients, and gradient descent by hand."""

from __future__ import annotations

import torch

DESCRIPTION = "Autograd basics: compute dy/dx automatically, then minimise (x - 5)^2 by hand."
DEFAULT_EPOCHS = 100  # number of gradient-descent steps


def run(
    *,
    epochs: int | None = None,
    seed: int = 0,  # unused; kept so every example shares one signature
    device: str | None = None,  # unused: this example is tiny and always runs on CPU
    verbose: bool = False,
) -> dict:
    steps = DEFAULT_EPOCHS if epochs is None else epochs

    # 1) Automatic differentiation: y = x^2 + 3x, so dy/dx = 2x + 3 (= 7 at x = 2).
    x = torch.tensor(2.0, requires_grad=True)
    y = x**2 + 3 * x
    y.backward()
    autograd_grad = x.grad.item()
    analytic_grad = 2 * 2.0 + 3

    # 2) Gradient descent on f(p) = (p - 5)^2, starting from p = 0.
    p = torch.tensor(0.0, requires_grad=True)
    lr = 0.1
    for _ in range(steps):
        loss = (p - 5.0) ** 2
        loss.backward()
        with torch.no_grad():
            p -= lr * p.grad
        p.grad.zero_()
    if verbose:
        print(f"dy/dx at x=2: autograd={autograd_grad}, analytic={analytic_grad}")
        print(f"minimiser after {steps} steps: {p.item():.6f} (true value 5)")

    return {
        "autograd_grad": autograd_grad,
        "analytic_grad": analytic_grad,
        "grads_match": abs(autograd_grad - analytic_grad) < 1e-6,
        "minimiser": p.item(),
        "true_minimiser": 5.0,
        "steps": steps,
    }
