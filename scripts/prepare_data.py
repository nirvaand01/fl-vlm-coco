"""
Download the Karpathy split of MSCOCO captions and a subsampled set of images,
then partition the train split across K simulated FL clients (IID).

Annotation source: Karpathy split JSON (dataset_coco.json), same file used by
salesforce/BLIP's data/coco_karpathy_dataset.py.
Image source: images.cocodataset.org (the official COCO image CDN).

NOTE: images.cocodataset.org serves a mismatched TLS cert for its custom
hostname (SNI/SAN mismatch against the underlying S3 bucket) as of writing.
Plain HTTP works fine and is what we use here -- this is a data CDN quirk,
not a security-sensitive endpoint.
"""
import argparse
import json
import os
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

KARPATHY_SPLIT_URL = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
COCO_IMAGE_BASE = "http://images.cocodataset.org"


def download(url: str, dest: Path, timeout: int = 30) -> None:
    req = Request(url, headers={"User-Agent": "fl-vlm-data-prep/1.0"})
    with urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def fetch_karpathy_json(data_dir: Path) -> dict:
    zip_path = data_dir / "caption_datasets.zip"
    json_path = data_dir / "dataset_coco.json"
    if json_path.exists():
        print(f"[skip] {json_path} already present")
    else:
        print(f"Downloading Karpathy split annotations from {KARPATHY_SPLIT_URL}")
        download(KARPATHY_SPLIT_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("dataset_coco.json", data_dir)
        zip_path.unlink()
    with open(json_path) as f:
        return json.load(f)


def build_split_lists(karpathy: dict, n_train: int, n_val: int, n_test: int, seed: int):
    train_items, val_items, test_items = [], [], []
    for img in karpathy["images"]:
        entry = {
            "cocoid": img["cocoid"],
            "filepath": img["filepath"],  # "train2014" or "val2014"
            "filename": img["filename"],
            "captions": [s["raw"].strip() for s in img["sentences"]],
        }
        split = img["split"]
        if split in ("train", "restval"):
            train_items.append(entry)
        elif split == "val":
            val_items.append(entry)
        elif split == "test":
            test_items.append(entry)

    rng = random.Random(seed)
    rng.shuffle(train_items)
    rng.shuffle(val_items)
    rng.shuffle(test_items)

    return train_items[:n_train], val_items[:n_val], test_items[:n_test]


def download_images(items: list, images_dir: Path, max_workers: int = 16) -> list:
    images_dir.mkdir(parents=True, exist_ok=True)
    kept = []

    def _fetch(item):
        local_path = images_dir / item["filename"]
        if not local_path.exists():
            url = f"{COCO_IMAGE_BASE}/{item['filepath']}/{item['filename']}"
            try:
                download(url, local_path)
            except (URLError, HTTPError) as e:
                return item, False, str(e)
        return item, True, None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_fetch, item) for item in items]
        done = 0
        for fut in as_completed(futures):
            item, ok, err = fut.result()
            done += 1
            if ok:
                item["local_path"] = str(images_dir / item["filename"])
                kept.append(item)
            else:
                print(f"  [warn] failed to fetch {item['filename']}: {err}")
            if done % 500 == 0 or done == len(items):
                print(f"  fetched {done}/{len(items)} ({len(kept)} ok)")
    return kept


def partition_iid(items: list, num_clients: int, seed: int) -> list:
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    return [shuffled[i::num_clients] for i in range(num_clients)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--num-clients", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-workers", type=int, default=16)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir = data_dir / "images"

    karpathy = fetch_karpathy_json(data_dir)
    train_items, val_items, test_items = build_split_lists(
        karpathy, args.n_train, args.n_val, args.n_test, args.seed
    )
    print(f"Selected {len(train_items)} train / {len(val_items)} val / {len(test_items)} test images")

    for name, items in [("train", train_items), ("val", val_items), ("test", test_items)]:
        print(f"Downloading {name} images...")
        kept = download_images(items, images_dir, args.max_workers)
        out_path = data_dir / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(kept, f)
        print(f"  wrote {len(kept)} items -> {out_path}")

        if name == "train":
            client_shards = partition_iid(kept, args.num_clients, args.seed)
            for i, shard in enumerate(client_shards):
                client_path = data_dir / f"train_client{i}.json"
                with open(client_path, "w") as f:
                    json.dump(shard, f)
                print(f"  client {i}: {len(shard)} images -> {client_path}")


if __name__ == "__main__":
    main()
