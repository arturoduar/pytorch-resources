"""Command line interface: ``pytorch-examples list | info | run NAME | serve``."""

from __future__ import annotations

import argparse
import json
import sys

import torch

from pytorch_examples import __version__
from pytorch_examples.examples import EXAMPLES, describe
from pytorch_examples.utils import get_device


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pytorch-examples", description="Runnable, tested and deployable PyTorch examples."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list the available examples")
    sub.add_parser("info", help="show PyTorch version and the device that would be used")

    run = sub.add_parser("run", help="run one example and print its metrics as JSON")
    run.add_argument("name", choices=sorted(EXAMPLES))
    run.add_argument("--epochs", type=int, default=None, help="override the default epoch count")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--device", default=None, help="e.g. cpu, cuda, mps (default: auto)")
    run.add_argument("--verbose", action="store_true", help="print progress while training")

    serve = sub.add_parser("serve", help="serve the MLP classifier over HTTP")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--epochs", type=int, default=None, help="training epochs at startup")
    serve.add_argument("--seed", type=int, default=0)
    serve.add_argument(
        "--checkpoint",
        default=None,
        help="load weights from this file if it exists; otherwise train and save it here",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "list":
        for name, description in describe().items():
            print(f"{name:20} {description}")
        return 0

    if args.command == "info":
        info = {
            "pytorch_examples": __version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(get_device()),
        }
        print(json.dumps(info, indent=2))
        return 0

    if args.command == "run":
        result = EXAMPLES[args.name].run(
            epochs=args.epochs, seed=args.seed, device=args.device, verbose=args.verbose
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "serve":
        from pytorch_examples.serve import build_predictor, make_server

        print("Preparing model...", file=sys.stderr, flush=True)
        predictor = build_predictor(args.checkpoint, epochs=args.epochs, seed=args.seed)
        server = make_server(predictor, args.host, args.port)
        print(f"Serving on http://{args.host}:{args.port}  (Ctrl+C to stop)", file=sys.stderr)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0

    return 2  # unreachable: argparse enforces a valid command


if __name__ == "__main__":
    raise SystemExit(main())
