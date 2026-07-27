"""Centralized fine-tuning baseline: same full train set, no FL split.

Default epoch count is rounds * local_epochs, matching the total number of
passes each FL client makes over its shard across the whole FL run (E per
round * R rounds) -- so this row of the comparison table isolates "cost of
federation" rather than "cost of federation + less total training."

Usage:
    python scripts/train_centralized.py --model blip --rounds 8 --epochs 2
    python scripts/train_centralized.py --model clip --rounds 8 --epochs 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from data.dataset import load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["blip", "clip"], required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--rounds", type=int, default=8, help="R from the FL run, used to derive total epochs")
    ap.add_argument("--epochs", type=int, default=2, help="E from the FL run (local epochs per round)")
    ap.add_argument("--total-epochs", type=int, default=None, help="override rounds*epochs directly")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--checkpoint-out", default=None)
    args = ap.parse_args()

    if args.model == "blip":
        from models import blip_model as model_module
    else:
        from models import clip_model as model_module

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = model_module.load_model_and_processor(device)

    total_epochs = args.total_epochs or (args.rounds * args.epochs)
    train_items = load_split(f"{args.data_dir}/train.json")
    print(f"Centralized fine-tuning {args.model} on {len(train_items)} images for {total_epochs} epochs")

    model_module.train_one_client(
        model, processor, train_items, device,
        epochs=total_epochs, batch_size=args.batch_size, lr=args.lr,
    )

    ckpt_path = args.checkpoint_out or f"checkpoints/{args.model}/centralized.pt"
    model_module.save_checkpoint(model, ckpt_path)
    print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()
