"""Shared dataset classes for the BLIP captioning and CLIP retrieval tasks.

Both models consume the same underlying JSON format produced by
scripts/prepare_data.py: a list of
    {"cocoid": int, "filename": str, "local_path": str, "captions": [str, ...]}

Adapted from the structure of salesforce/BLIP's data/coco_karpathy_dataset.py
(random caption per image during training; all captions kept for eval scoring).
"""
import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


def load_split(json_path) -> list:
    with open(json_path) as f:
        return json.load(f)


class CaptionTrainDataset(Dataset):
    """One (image, caption) pair per item; a random caption is drawn each access,
    mirroring BLIP's train_caption dataset so the effective supervision varies epoch
    to epoch even though we cache one caption per __getitem__ call."""

    def __init__(self, items: list):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item["local_path"]).convert("RGB")
        caption = random.choice(item["captions"])
        return image, caption


class CaptionEvalDataset(Dataset):
    """Returns one image per item (no caption) for generation; ground-truth
    captions are fetched separately via `references()` for pycocoevalcap scoring."""

    def __init__(self, items: list):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item["local_path"]).convert("RGB")
        return image, item["cocoid"]

    def references(self) -> dict:
        """cocoid -> list[str] ground-truth captions, for pycocoevalcap."""
        return {item["cocoid"]: item["captions"] for item in self.items}


class RetrievalTrainDataset(Dataset):
    """Same (image, random caption) pattern as CaptionTrainDataset -- kept as a
    separate class since CLIP's collate/processor differs from BLIP's."""

    def __init__(self, items: list):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item["local_path"]).convert("RGB")
        caption = random.choice(item["captions"])
        return image, caption


class RetrievalEvalDataset(Dataset):
    """Standard COCO retrieval eval protocol: N images, 5N captions (all of them),
    with an img2txt / txt2img ground-truth index mapping -- same convention as
    BLIP's train_retrieval.py eval loop."""

    def __init__(self, items: list):
        self.items = items
        self.images = [item["local_path"] for item in items]
        self.texts = []
        self.img2txt = {}
        self.txt2img = {}
        txt_id = 0
        for img_id, item in enumerate(items):
            self.img2txt[img_id] = []
            for caption in item["captions"]:
                self.texts.append(caption)
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.images)

    def get_image(self, idx):
        return Image.open(self.images[idx]).convert("RGB")
