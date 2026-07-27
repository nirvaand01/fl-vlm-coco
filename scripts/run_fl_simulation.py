"""Entry point: run the Flower FedAvg simulation for one model.

Usage:
    python scripts/run_fl_simulation.py --model blip --rounds 8 --epochs 2
    python scripts/run_fl_simulation.py --model clip --rounds 8 --epochs 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `data`/`models`/`fl` imports

import torch
from flwr.client import ClientApp
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.simulation import run_simulation

from fl.client import make_client_fn
from fl.server import build_strategy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["blip", "clip"], required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--num-clients", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2, help="local epochs per round (E)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--metrics-log", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = args.checkpoint_dir or f"checkpoints/{args.model}"
    metrics_log = args.metrics_log or f"logs/{args.model}_fl_metrics.json"

    client_fn = make_client_fn(args.model, args.data_dir, device, args.epochs, args.batch_size, args.lr)
    client_app = ClientApp(client_fn=client_fn)

    def server_fn(context):
        strategy = build_strategy(
            args.model, args.data_dir, device, args.num_clients, checkpoint_dir, metrics_log,
        )
        return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=args.rounds))

    server_app = ServerApp(server_fn=server_fn)

    backend_config = {"client_resources": {"num_cpus": 2, "num_gpus": 1.0 if device == "cuda" else 0.0}}

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=args.num_clients,
        backend_config=backend_config,
    )


if __name__ == "__main__":
    main()
