# Full run results

The actual full federated run (as opposed to `pipeline_validation.md`'s
smoke/timing tests). Run date: 2026-08-16, on a single vast.ai A10 (24GB).

## Settings

| Param | Value |
|---|---|
| Data | Karpathy split, 10,000 train / 1,000 val / 1,000 test images, all 5 captions/image kept |
| Clients (K) | 4, IID random partition (2,500 images/client) |
| Rounds (R) | 5 |
| Local epochs/round (E) | 2 (10 total epochs of exposure per client across the whole FL run) |
| Strategy | FedAvg (Flower `NumPyClient` + `run_simulation`), weighted by client example count |
| Optimizer | AdamW, lr=5e-5 |
| Batch size | BLIP: 8. CLIP: 32 (see "CLIP instability" below for why these differ) |
| Eval | Server-side only, once per round, against the shared `val.json` (1,000 images) |

**Deviation from the original plan:** R was cut from 8 to 5 to save
wall-clock time. Final tables are **zero-shot vs. federated**.

## Results (test split, 1,000 images, held out)

### BLIP (captioning)

| Setting | BLEU-4 | CIDEr |
|---|---|---|
| Zero-shot (pretrained) | 0.299 | 0.993 |
| Federated (FedAvg, R=5/E=2) | **0.343** | **1.190** |

Clean, unambiguous improvement from federated fine-tuning. No issues.

Per-round val metrics (server-side eval, `logs/blip_fl_metrics.json`):

| Round | BLEU-4 | CIDEr |
|---|---|---|
| 1 | 0.343 | 1.184 |
| 2 | 0.349 | 1.211 |
| 3 | 0.357 | 1.213 |
| 4 | 0.347 | 1.199 |
| 5 | 0.354 | 1.218 |

Monotonic-ish improvement with normal round-to-round noise (round 4 dips
slightly then recovers). Nothing concerning here.

### CLIP (ViT-B/32 retrieval)

| Setting | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 |
|---|---|---|---|---|---|---|
| Zero-shot (pretrained) | 72.4 | 92.4 | 95.9 | 50.8 | 79.4 | 88.9 |
| Federated (FedAvg, R=5/E=2, batch=32) | 60.5 | 84.7 | 91.3 | 46.5 | 79.8 | 89.8 |

Federated CLIP ends up **below zero-shot on every metric except t2i R@5/R@10**
(where it's roughly flat). See "Not satisfactory" below — this is the one
open issue in the run.

Per-round val metrics, the batch=32 (fixed) run (`logs/clip_fl_metrics.json`):

| Round | i2t R@1 | t2i R@1 |
|---|---|---|
| 1 | 67.6 | 52.3 |
| 2 | 64.4 | 47.4 |
| 3 | 62.6 | 47.9 |
| 4 | 61.4 | 46.2 |
| 5 | 59.3 | 46.3 |

Steady, gentle decline round over round — never recovers back toward
zero-shot, but degrades gracefully rather than collapsing.

## Not satisfactory: CLIP's federated result stays below zero-shot

**What happened, in order:**

1. **First attempt** used the same defaults as BLIP (`batch_size=8`,
   `lr=5e-5`) since that's what the script defaults to. Result was a genuine
   collapse, not just "a bit worse": i2t R@1 went 72.4 (zero-shot) → 41.3
   (round 1) → 27.2 (round 5), and t2i R@1 similarly 50.8 → 20.1. Preserved
   for reference in `checkpoints/clip_bs8_unstable/` and
   `logs/clip_fl_run_bs8_unstable.log` / `logs/clip_fl_metrics_bs8_unstable.json`.
   - **Diagnosis:** CLIP's contrastive InfoNCE loss uses the *other examples
     in the batch* as negatives. `batch_size=8` gives only 7 negatives per
     anchor — far too weak a signal to fine-tune an already well-calibrated
     embedding space without actively damaging it. BLIP has no such
     dependency (its loss is per-token captioning cross-entropy), so the same
     hyperparameters were fine for one model and broken for the other.
2. **Fix applied:** reran with `batch_size=32` (31 negatives/anchor),
   `lr=5e-5` unchanged. This is the run reported in the table above.
   Substantially more stable — round 1 t2i R@1 (52.3) even slightly *exceeds*
   zero-shot (50.8) — and it degrades gracefully instead of collapsing. Also
   ran ~30% *faster* wall-clock (19 min vs 28 min for 5 rounds), since
   CLIP-B/32 is small enough that batch=8 was leaving the A10 underused;
   fewer, larger steps was a straight win with no time cost.

**Why it's still not fully satisfactory:** even with the fix, federated CLIP
never recovers to zero-shot and drifts down every round rather than
plateauing or improving. Working hypotheses, untested:

- **LR still on the high side for fine-tuning.** `5e-5` is more typical for
  training CLIP from scratch on huge data; published CLIP fine-tuning recipes
  often use 1e-6–1e-5. Combined with only 4 clients × 2,500 images each, each
  local update may still be an aggressive step relative to how little new
  information is in the local shard.
- **FedAvg parameter averaging vs. contrastive geometry.** Averaging 4
  independently-drifted sets of weights (rather than gradients) may not
  recompose cleanly for a model whose useful signal lives in a similarity
  geometry between towers, in a way it doesn't for BLIP's per-token
  generative loss. Not confirmed, just a plausible mechanism.
- **Not enough local data per client for contrastive learning specifically.**
  2,500 images/client (12,500 image-caption pairs with 5 captions each) is
  small for InfoNCE-style training, independent of batch size.

None of this was run down further — flagging honestly rather than guessing.
If it matters for the writeup, the next concrete experiment would be a
low-LR (1e-5) rerun at the same batch=32, holding everything else fixed, to
see whether it flattens the decline.

## Runtime notes

- Data prep (12,000 images downloaded): 3m22s.
- Zero-shot eval: BLIP 47s, CLIP 25s (both on the 1,000-image test split).
- BLIP FL (5 rounds): ~1h43m end-to-end (~20-21 min/round). One operational
  hiccup along the way: the first attempt died mid-round-2 when a client
  laptop closed and dropped the SSH session, SIGTERM-ing the background Ray
  worker (only round 1's checkpoint survived, no resume support in the
  script, so it was restarted from scratch). Relaunched fully detached
  (`setsid`+`nohup`+`disown`) so it's immune to session drops going forward.
- CLIP FL (5 rounds): 28 min at batch=8 (the unstable run), 19 min at
  batch=32 (the fixed run, faster despite being "more work" per the naive
  batch-size intuition).
- A one-off full-dataset training-speed measurement (BLIP, batch=8) found
  **~11 min/epoch** for BLIP on this GPU, which is what confirmed FL round
  time is dominated by training compute, not per-round eval or Flower/Ray
  overhead as originally guessed.

## Code cleanup done alongside this run

- `models/common.py` added: `get_parameters`/`set_parameters`/`save_checkpoint`
  were byte-identical duplicates between `models/blip_model.py` and
  `models/clip_model.py`; now defined once and imported by both.
- `requirements.txt` trimmed: `pycocotools`, `datasets`, `scikit-learn`,
  `tqdm` were listed but never imported anywhere in the codebase.
- `scripts/build_comparison_table.py` and `scripts/eval_checkpoint.py`
  simplified to a plain zero-shot vs. federated comparison.

## Outstanding

- **Not backed up.** `/workspace` on this instance is not a persistent
  volume — checkpoints, logs, and this file are all lost on recycle/destroy.
  Not yet pushed anywhere durable (git remote, HF Hub, rclone).
