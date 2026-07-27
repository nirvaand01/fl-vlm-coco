"""Evaluate a model (pretrained, or a fine-tuned checkpoint) on a data split.

Used for all three rows of the baseline comparison table:
  - zero-shot:    --checkpoint none
  - centralized:  --checkpoint checkpoints/<model>/centralized.pt
  - federated:    --checkpoint checkpoints/<model>/roundR.pt   (last FL round)

Usage:
    python scripts/eval_checkpoint.py --model blip --checkpoint none --split test
    python scripts/eval_checkpoint.py --model clip --checkpoint checkpoints/clip/round8.pt --split test
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from data.dataset import load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["blip", "clip"], required=True)
    ap.add_argument("--checkpoint", default="none", help="path to a .pt state_dict, or 'none' for zero-shot")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--out", default=None, help="where to write the result JSON")
    args = ap.parse_args()

    if args.model == "blip":
        from models import blip_model as model_module
    else:
        from models import clip_model as model_module

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = model_module.load_model_and_processor(device)

    if args.checkpoint != "none":
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        label = args.checkpoint
    else:
        label = "zero-shot (pretrained)"

    items = load_split(f"{args.data_dir}/{args.split}.json")
    metrics = model_module.evaluate(model, processor, items, device)
    print(f"[{args.model}] {label} on {args.split} ({len(items)} images): {metrics}")

    out_path = args.out or f"logs/{args.model}_{Path(args.checkpoint).stem if args.checkpoint != 'none' else 'zeroshot'}_{args.split}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "checkpoint": label, "split": args.split, "metrics": metrics}, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
