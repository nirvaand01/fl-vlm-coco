# Task: Federated Learning on Vision-Language Models (MSCOCO)

## Assignment

Preliminary task: implement at least 2 models from this list in a federated learning (FL) setup, using MSCOCO (scaling down datasets if GPU-constrained is explicitly allowed):

- BLIP_COCO
- BUTD_COCO
- FLaVA_TextVQA
- LLaVA_TextVQA
- CLIP (ViT-L/14)_Flickr30k
- CLIP (ViT-B/32)_COCO
- CLIP (ViT-B/32)_Flickr30k

Reference repo given by assigner: https://github.com/Nofear18/ID_VL_Pruning (ICML'24 paper "Exploring Intrinsic Dimension for Vision-Language Model Pruning" — the pruning code itself is NOT relevant to this task; only its data-loading and eval scaffolding is useful, see below).

## Decision made: BLIP_COCO + CLIP (ViT-B/32)_COCO

Reasoning:

- Both are on COCO specifically (assigner's prose said "MSCOCO dataset" even though the list includes Flickr30k/TextVQA entries).
- Two different task types = more interesting FL comparison: BLIP is generative (captioning), CLIP is contrastive matching (retrieval) — shows breadth rather than doing the same recipe twice.
- BUTD ruled out: needs a separate Faster R-CNN region-feature extraction pipeline (multi-hour preprocessing job, old 2018-era stack) — too much plumbing overhead for the value.
- FLaVA/LLaVA (VQA) ruled out: TextVQA eval is fiddly (OCR tokens), LLaVA is 7B+ params, not realistic for federated fine-tuning on a constrained GPU.

## Task type reference

- **BLIP_COCO** = image captioning. Model generates a caption for an image. Metrics: BLEU-4, CIDEr (via `pycocoevalcap`).
- **CLIP (ViT-B/32)_COCO** = image-text retrieval. Model embeds images/text into a shared space via contrastive loss (InfoNCE); no generation. Metrics: Recall@1/5/10, both directions (image→text and text→image).

## Code/data sources to pull from (NOT full pipelines — cherry-pick pieces)

- **salesforce/BLIP** (official repo, not the pruning fork) — `data/coco_karpathy_dataset.py` for the COCO caption Dataset class; `train_caption.py` for the eval loop (pycocoevalcap wiring). This is also what ID_VL_Pruning's own `data/` folder is vendored from.
- **ylaxor/clip-like** (`fine-tune-clip.ipynb`) — CLIP fine-tuning on COCO via HuggingFace `VisionTextDualEncoder`, plain contrastive loss. Use this as the CLIP training loop base, NOT BLIP's `train_retrieval.py` (that fine-tunes BLIP for retrieval with ITC+ITM losses — wrong architecture for our CLIP entry).
- Retrieval R@k computation is architecture-agnostic — can borrow eval math from BLIP's `train_retrieval.py` even though we're not using its training loop.
- HuggingFace model IDs: `Salesforce/blip-image-captioning-base`, `openai/clip-vit-base-patch32`.
- FL simulation: Flower (`flwr`), using `NumPyClient` + built-in `FedAvg` strategy + `flwr.simulation.start_simulation()`.

## Dataset sizing (Karpathy split)

Full COCO: 113,287 train images / 5,000 val / 5,000 test, 5 captions/image (~566k train captions).

Scaled-down target:

- Train: 8,000–15,000 images (random subsample, keep all 5 captions per image)
- Val: 1,000 images
- Test: 1,000 images (from Karpathy test split — held out, final metrics only)

## Federated setup

- K = 4 simulated clients (sequential on single GPU is fine, no real distributed hardware needed)
- IID split: random partition of train images across clients
- Optional/stretch: one non-IID split (e.g. cluster by COCO supercategory) to show non-IID performance gap — the more research-interesting result
- E = 2–3 local epochs per round, R = 5–10 communication rounds
- FedAvg: weighted average of client weights by num_examples, standard

## Metrics & reporting plan

- BLIP: BLEU-4, CIDEr (optionally METEOR/ROUGE-L if free from pycocoevalcap)
- CLIP: R@1/5/10 for i2t and t2i separately
- Report metrics per communication round, not just final (convergence chart)
- Three-row comparison table per model: (1) zero-shot pretrained baseline, (2) centralized fine-tuning (same data/epochs, no FL split) — shows "cost of federation," (3) federated result — the actual deliverable
- Cite data-loading/eval provenance in README (e.g. "adapted from ID_VL_Pruning / BLIP official repo")

## Open items / things to double check with assigner if possible

- Whether "at least 2 models" strictly means 2 COCO-only entries, or any 2 from the full list — went with COCO-only interpretation

## Environment

- Running on vast.ai, 1x A10 GPU (23GB VRAM), PyTorch template
- **`/workspace` is NOT a persistent volume on this instance** — sync code/checkpoints out (git push, HF Hub, rclone) periodically; a recycle/destroy wipes it
- Checkpoint after every FL round (cheap insurance against instance/session interruption)
- Libraries needed: `transformers`, `flwr[simulation]` (needs the `[simulation]` extra for the Ray backend), `pycocoevalcap`, `pillow`, `torchvision`, `datasets` (HF)
- `pycocoevalcap`'s PTBTokenizer shells out to Java — install `default-jre-headless` via apt, or BLIP eval crashes with `FileNotFoundError: java`
- `images.cocodataset.org` serves a mismatched TLS cert for its custom hostname (SNI/SAN mismatch against the S3 bucket behind it) — data prep downloads images over plain HTTP, not HTTPS
- transformers>=4.5x: `CLIPModel.get_image_features`/`get_text_features` return a `BaseModelOutputWithPooling`, not a bare tensor — grab `.pooler_output` (already projected to `config.projection_dim`)
