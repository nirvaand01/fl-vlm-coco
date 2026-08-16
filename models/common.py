"""FedAvg parameter (de)serialization + checkpointing, shared by both model modules."""
from pathlib import Path

import torch


def get_parameters(model) -> list:
    return [v.cpu().numpy() for v in model.state_dict().values()]


def set_parameters(model, parameters: list) -> None:
    keys = list(model.state_dict().keys())
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(state_dict, strict=True)


def save_checkpoint(model, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
