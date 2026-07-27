"""Assemble the 3-row baseline comparison table (zero-shot / centralized / federated)
from the JSON files eval_checkpoint.py writes.

Usage:
    python scripts/build_comparison_table.py --model blip \
        --zero-shot logs/blip_zeroshot_test.json \
        --centralized logs/blip_centralized_test.json \
        --federated logs/blip_round8_test.json
"""
import argparse
import json


def flatten(metrics: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in metrics.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(flatten(v, prefix=f"{key}_"))
        else:
            flat[key] = v
    return flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["blip", "clip"], required=True)
    ap.add_argument("--zero-shot", required=True)
    ap.add_argument("--centralized", required=True)
    ap.add_argument("--federated", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = {
        "Zero-shot (pretrained)": flatten(json.load(open(args.zero_shot))["metrics"]),
        "Centralized fine-tune": flatten(json.load(open(args.centralized))["metrics"]),
        "Federated (FedAvg)": flatten(json.load(open(args.federated))["metrics"]),
    }
    columns = list(next(iter(rows.values())).keys())

    lines = [f"### {args.model.upper()} — zero-shot vs. centralized vs. federated", ""]
    lines.append("| Setting | " + " | ".join(columns) + " |")
    lines.append("|---" * (len(columns) + 1) + "|")
    for name, metrics in rows.items():
        values = " | ".join(f"{metrics.get(c, float('nan')):.3f}" for c in columns)
        lines.append(f"| {name} | {values} |")

    table = "\n".join(lines)
    print(table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
