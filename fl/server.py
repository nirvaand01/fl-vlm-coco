"""FedAvg strategy with centralized (server-side) evaluation.

Evaluation runs once per round against the shared held-out val split, rather
than on every client -- the val set isn't partitioned across clients, so
per-client evaluate() would just repeat the same work K times.
"""
import json
from pathlib import Path

from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg

from data.dataset import load_split


def _flatten_metrics(metrics: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in metrics.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten_metrics(v, prefix=f"{key}_"))
        else:
            flat[key] = float(v)
    return flat


def build_strategy(model_name: str, data_dir: str, device: str, num_clients: int,
                    checkpoint_dir: str, metrics_log_path: str):
    if model_name == "blip":
        from models import blip_model as model_module
    elif model_name == "clip":
        from models import clip_model as model_module
    else:
        raise ValueError(f"unknown model_name {model_name!r}")

    eval_model, eval_processor = model_module.load_model_and_processor(device)
    val_items = load_split(f"{data_dir}/val.json")
    initial_parameters = ndarrays_to_parameters(model_module.get_parameters(eval_model))

    Path(metrics_log_path).parent.mkdir(parents=True, exist_ok=True)
    history = []

    def evaluate_fn(server_round, parameters, config):
        if server_round == 0:
            return None  # skip eval of the untouched initial parameters
        model_module.set_parameters(eval_model, parameters)
        metrics = model_module.evaluate(eval_model, eval_processor, val_items, device)
        flat = _flatten_metrics(metrics)
        print(f"[round {server_round}] {model_name} val metrics: {flat}")

        ckpt_path = f"{checkpoint_dir}/round{server_round}.pt"
        model_module.save_checkpoint(eval_model, ckpt_path)

        history.append({"round": server_round, **flat})
        with open(metrics_log_path, "w") as f:
            json.dump(history, f, indent=2)

        return 0.0, flat

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,  # centralized eval via evaluate_fn instead
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
    )
    return strategy
