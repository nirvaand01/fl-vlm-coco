"""Generic Flower NumPyClient shared by both model types.

Each simulated client loads its shard of the train split, runs a few local
epochs when asked (fit), and reports its held-out eval metrics on request
(evaluate). Model construction/train/eval logic is delegated to the model
module (models/blip_model.py or models/clip_model.py) so this class stays
model-agnostic.
"""
import torch
import flwr as fl

from data.dataset import load_split


class VLMClient(fl.client.NumPyClient):
    def __init__(self, model_module, client_id: int, data_dir: str, device: str,
                 epochs: int, batch_size: int, lr: float):
        self.model_module = model_module
        self.client_id = client_id
        self.data_dir = data_dir
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.model, self.processor = model_module.load_model_and_processor(device)
        self.train_items = load_split(f"{data_dir}/train_client{client_id}.json")

    def get_parameters(self, config):
        return self.model_module.get_parameters(self.model)

    def fit(self, parameters, config):
        self.model_module.set_parameters(self.model, parameters)
        n_examples = self.model_module.train_one_client(
            self.model, self.processor, self.train_items, self.device,
            epochs=self.epochs, batch_size=self.batch_size, lr=self.lr,
        )
        return self.model_module.get_parameters(self.model), n_examples, {}

    # Evaluation is done centrally on the server (see fl/server.py's evaluate_fn)
    # against a single shared held-out val set, rather than redundantly on every
    # client -- 4x cheaper and the val set isn't federated anyway.


def make_client_fn(model_name: str, data_dir: str, device: str, epochs: int, batch_size: int, lr: float):
    if model_name == "blip":
        from models import blip_model as model_module
    elif model_name == "clip":
        from models import clip_model as model_module
    else:
        raise ValueError(f"unknown model_name {model_name!r}")

    def client_fn(context: fl.common.Context) -> fl.client.Client:
        client_id = int(context.node_config["partition-id"])
        client = VLMClient(model_module, client_id, data_dir, device, epochs, batch_size, lr)
        return client.to_client()

    return client_fn
