"""CLIP (ViT-B/32) image-text retrieval model wrapper: local training + evaluation.

Model: openai/clip-vit-base-patch32 (HuggingFace).
Training loop: plain symmetric InfoNCE contrastive loss over in-batch negatives,
following the pattern in ylaxor/clip-like's fine-tune-clip.ipynb (NOT BLIP's
train_retrieval.py, which adds ITC+ITM losses for a different architecture).
Eval (R@1/5/10, i2t and t2i): standard COCO retrieval protocol -- N images vs
5N captions with an img2txt/txt2img ground-truth mapping, same convention BLIP's
train_retrieval.py uses, reimplemented here directly against CLIP's embeddings.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from data.dataset import RetrievalEvalDataset, RetrievalTrainDataset
from models.common import get_parameters, set_parameters, save_checkpoint  # noqa: F401

MODEL_ID = "openai/clip-vit-base-patch32"


def load_model_and_processor(device: str):
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    return model, processor


def _collate_train(batch, processor):
    images, captions = zip(*batch)
    inputs = processor(
        images=list(images), text=list(captions),
        padding=True, truncation=True, return_tensors="pt",
    )
    return inputs


def train_one_client(model, processor, items: list, device: str, epochs: int, batch_size: int, lr: float) -> int:
    dataset = RetrievalTrainDataset(items)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        collate_fn=lambda b: _collate_train(b, processor),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            labels = torch.arange(logits_per_image.size(0), device=device)
            loss_i = F.cross_entropy(logits_per_image, labels)
            loss_t = F.cross_entropy(logits_per_text, labels)
            loss = (loss_i + loss_t) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"    [clip] epoch {epoch + 1}/{epochs} avg loss {total_loss / max(len(loader), 1):.4f}")
    return len(dataset)


@torch.no_grad()
def evaluate(model, processor, items: list, device: str, batch_size: int = 32) -> dict:
    # transformers>=4.5x: get_image_features/get_text_features return a
    # BaseModelOutputWithPooling, not a bare tensor -- the projected embedding
    # is in .pooler_output (already 512-d, matching config.projection_dim).
    dataset = RetrievalEvalDataset(items)
    model.eval()

    image_embeds = []
    for i in range(0, len(dataset.images), batch_size):
        images = [dataset.get_image(j) for j in range(i, min(i + batch_size, len(dataset.images)))]
        inputs = processor(images=images, return_tensors="pt").to(device)
        feats = model.get_image_features(**inputs).pooler_output
        image_embeds.append(F.normalize(feats, dim=-1).cpu())
    image_embeds = torch.cat(image_embeds, dim=0)

    text_embeds = []
    for i in range(0, len(dataset.texts), batch_size):
        texts = dataset.texts[i:i + batch_size]
        inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(device)
        feats = model.get_text_features(**inputs).pooler_output
        text_embeds.append(F.normalize(feats, dim=-1).cpu())
    text_embeds = torch.cat(text_embeds, dim=0)

    sim_i2t = (image_embeds @ text_embeds.T).numpy()  # [N_img, N_txt]
    sim_t2i = sim_i2t.T  # [N_txt, N_img]

    def recall_at_k(sim: np.ndarray, gt: dict, ks=(1, 5, 10)) -> dict:
        n = sim.shape[0]
        ranks = np.zeros(n)
        for idx in range(n):
            order = np.argsort(-sim[idx])
            targets = gt[idx] if isinstance(gt[idx], list) else [gt[idx]]
            rank = min(np.where(order == t)[0][0] for t in targets)
            ranks[idx] = rank
        return {f"R@{k}": float(100.0 * np.mean(ranks < k)) for k in ks}

    i2t = recall_at_k(sim_i2t, dataset.img2txt)
    t2i = recall_at_k(sim_t2i, dataset.txt2img)
    return {"i2t": i2t, "t2i": t2i}
