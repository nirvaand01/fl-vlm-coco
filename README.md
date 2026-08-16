# Federated Learning on Vision-Language Models (MSCOCO)

Federated fine-tuning of two vision-language models on a subsampled Karpathy
split of MSCOCO, using [Flower](https://flower.ai) for FL simulation:

- **BLIP** (`Salesforce/blip-image-captioning-base`) — image captioning. Metrics: BLEU-4, CIDEr.
- **CLIP ViT-B/32** (`openai/clip-vit-base-patch32`) — image-text retrieval. Metrics: Recall@1/5/10 (i2t and t2i).

See [CLAUDE.md](CLAUDE.md) for the full task spec, model/dataset decisions, and reasoning.

**Results from the completed R=5/E=2 run, including the CLIP instability found
and fixed along the way, are in [logs/results.md](logs/results.md).**

## Provenance

- Data-loading conventions (Karpathy split, random-caption-per-epoch training
  sampling, all-captions-kept eval) are adapted from
  [salesforce/BLIP](https://github.com/salesforce/BLIP)'s
  `data/coco_karpathy_dataset.py`, the same source
  [ID_VL_Pruning](https://github.com/Nofear18/ID_VL_Pruning) (the assigner's
  reference repo) vendors its own `data/` from.
- Retrieval R@k eval convention (N images vs. 5N captions with an img2txt/txt2img
  ground-truth mapping) follows BLIP's `train_retrieval.py`, reimplemented here
  directly against CLIP's embeddings (BLIP's retrieval training loop itself,
  ITC+ITM losses, is not used — wrong architecture for a plain CLIP model).
- The CLIP contrastive training loop (symmetric InfoNCE over in-batch negatives)
  follows the pattern in
  [ylaxor/clip-like](https://github.com/ylaxor/clip-like)'s `fine-tune-clip.ipynb`.
- FL simulation uses Flower's built-in `FedAvg` strategy via `NumPyClient` +
  `flwr.simulation.run_simulation`.

## Setup

```bash
pip install -r requirements.txt
apt-get install -y default-jre-headless   # pycocoevalcap's PTBTokenizer shells out to java
```

Requires a CUDA GPU (tested on a 23GB A10; both models comfortably fit with
room for larger batch sizes).

### Environment quirks worth knowing

- `images.cocodataset.org` serves a TLS cert that doesn't match its own
  hostname (SNI/SAN mismatch with the underlying S3 bucket) — `scripts/prepare_data.py`
  downloads images over plain HTTP, not HTTPS. It's a public, non-sensitive
  image CDN, so this is a reasonable tradeoff.
- `transformers>=4.5x` changed `CLIPModel.get_image_features` /
  `get_text_features` to return a `BaseModelOutputWithPooling` instead of a bare
  tensor — the projected embedding is in `.pooler_output`. `models/clip_model.py`
  already handles this; worth knowing if you bump the transformers version further.
- `flwr[simulation]` (not plain `flwr`) is required for the Ray-based simulation backend.

## Pipeline

### 1. Prepare data

Downloads the Karpathy split annotations + a subsampled set of images, and
partitions the train split into K IID client shards.

```bash
python scripts/prepare_data.py --data-dir data \
    --n-train 10000 --n-val 1000 --n-test 1000 --num-clients 4
```

Produces `data/{train,val,test}.json` and `data/train_client{0..3}.json`, plus
the actual images under `data/images/`.

### 2. Run the federated simulation

```bash
python scripts/run_fl_simulation.py --model blip --rounds 8 --epochs 2 --batch-size 8
python scripts/run_fl_simulation.py --model clip --rounds 8 --epochs 2 --batch-size 8
```

- K = 4 clients (`--num-clients`), FedAvg, weighted by number of local examples.
- Evaluation runs centrally on the server each round against the shared
  `val.json` split (not redistributed to clients — it isn't federated data, and
  evaluating on 4 clients per round would just repeat the same work 4x).
- Per-round metrics are logged to `logs/<model>_fl_metrics.json`; checkpoints are
  saved to `checkpoints/<model>/round<N>.pt` after every round.
- On a single GPU, Ray schedules the 4 simulated clients sequentially each round
  (`num_gpus: 1.0` per client in the backend config).

### 3. Baselines, for the comparison table

```bash
# zero-shot (pretrained, no fine-tuning)
python scripts/eval_checkpoint.py --model blip --checkpoint none --split test \
    --out logs/blip_zeroshot_test.json

# federated result (last FL round's checkpoint)
python scripts/eval_checkpoint.py --model blip --checkpoint checkpoints/blip/round8.pt --split test \
    --out logs/blip_federated_test.json
```

Repeat with `--model clip` for the CLIP retrieval table.

### 4. Build the comparison table

```bash
python scripts/build_comparison_table.py --model blip \
    --zero-shot logs/blip_zeroshot_test.json \
    --federated logs/blip_federated_test.json \
    --out logs/blip_comparison_table.md
```

## Repo layout

```
data/dataset.py           Dataset classes shared by both models
models/blip_model.py      BLIP load/train/eval + FedAvg param (de)serialization
models/clip_model.py      CLIP load/train/eval + FedAvg param (de)serialization
fl/client.py               Flower NumPyClient (model-agnostic, delegates to models/*)
fl/server.py               FedAvg strategy + centralized per-round evaluation
scripts/prepare_data.py    Karpathy split download + IID client partitioning
scripts/run_fl_simulation.py   FL simulation entry point
scripts/eval_checkpoint.py     Eval a checkpoint (or zero-shot) on val/test
scripts/build_comparison_table.py  Assemble the comparison markdown table
```

## Notes / open items

- Went with a COCO-only interpretation of "at least 2 models" (BLIP_COCO +
  CLIP(ViT-B/32)_COCO) since the assignment prose specifies MSCOCO even though
  the model list includes Flickr30k/TextVQA entries.
- Non-IID client split (e.g. by COCO supercategory) is a stretch goal, not yet
  implemented — `scripts/prepare_data.py`'s `partition_iid` would need a
  supercategory-aware counterpart; skipped for time and GPU compute (it'd mean
  another full 5-round FL run per model on top of the ones already done).
- Centralized fine-tuning baseline (train once on the full un-partitioned
  `train.json` for `rounds * epochs` total epochs, no FedAvg split) is a
  natural extension, not currently implemented — would directly quantify
  "cost of federation" against the federated result in
  [logs/results.md](logs/results.md), skipped here for time and GPU compute.
- `/workspace` on the current vast.ai instance is not a persistent volume —
  sync checkpoints/code out periodically (git push, HF Hub, rclone).
