"""BLIP image-captioning model wrapper: local training + evaluation.

Model: Salesforce/blip-image-captioning-base (HuggingFace).
Eval metrics (BLEU-4, CIDEr) computed via pycocoevalcap, same tooling
salesforce/BLIP's train_caption.py uses -- our eval loop is written fresh
against the model's generate() API rather than borrowing BLIP's training loop.
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BlipForConditionalGeneration, BlipProcessor

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from data.dataset import CaptionEvalDataset, CaptionTrainDataset

MODEL_ID = "Salesforce/blip-image-captioning-base"


def load_model_and_processor(device: str):
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    return model, processor


def _collate_train(batch, processor):
    images, captions = zip(*batch)
    inputs = processor(images=list(images), text=list(captions), padding=True, return_tensors="pt")
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs


def train_one_client(model, processor, items: list, device: str, epochs: int, batch_size: int, lr: float) -> int:
    """Local training loop. Returns number of training examples seen (for FedAvg weighting)."""
    dataset = CaptionTrainDataset(items)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: _collate_train(b, processor),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"    [blip] epoch {epoch + 1}/{epochs} avg loss {total_loss / max(len(loader), 1):.4f}")
    return len(dataset)


@torch.no_grad()
def evaluate(model, processor, items: list, device: str, batch_size: int = 16, max_new_tokens: int = 30) -> dict:
    """Generate captions for the eval split and score with BLEU-4 / CIDEr."""
    dataset = CaptionEvalDataset(items)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: (list(zip(*b))[0], list(zip(*b))[1]),
    )
    model.eval()
    hypotheses = {}
    for images, cocoids in loader:
        inputs = processor(images=list(images), return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        for cocoid, caption in zip(cocoids, captions):
            hypotheses[cocoid] = [caption.strip()]

    references = {k: v for k, v in dataset.references().items() if k in hypotheses}

    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize({str(k): [{"caption": c} for c in v] for k, v in references.items()})
    res = tokenizer.tokenize({str(k): [{"caption": c} for c in v] for k, v in hypotheses.items()})

    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(gts, res)
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)

    return {
        "bleu4": bleu_scores[3],
        "cider": cider_score,
    }


def get_parameters(model) -> list:
    return [v.cpu().numpy() for v in model.state_dict().values()]


def set_parameters(model, parameters: list) -> None:
    keys = list(model.state_dict().keys())
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(state_dict, strict=True)


def save_checkpoint(model, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
