# Pipeline validation log

Records from validating the FL pipeline end-to-end before committing to the full
run. No trained checkpoints were kept from these — they were smoke/timing tests
only, deleted after each run. Real full training (R=8, E=2) has not happened yet.

## 1. Wiring smoke tests (tiny data, 6 images/client, batch_size=2, 1 round, 1 epoch)

Purpose: confirm the full Flower FedAvg loop (client fit → aggregate → centralized
eval → checkpoint) runs without errors for both models, on trivially small data.

**CLIP** — all recall trivially 100% as expected with only 6 val images:
```
i2t: R@1=100.0  R@5=100.0  R@10=100.0
t2i: R@1=100.0  R@5=100.0  R@10=100.0
```

**BLIP**:
```
bleu4 = 0.4760
cider = 2.1842
```

Bugs found and fixed during this pass:
- `flwr[simulation]` extra (Ray backend) was missing — plain `flwr` isn't enough.
- `pycocoevalcap`'s PTBTokenizer shells out to Java — needed `apt-get install default-jre-headless`.
- `transformers>=4.5x`: `CLIPModel.get_image_features`/`get_text_features` now
  return `BaseModelOutputWithPooling`, not a bare tensor — fixed by reading `.pooler_output`.

## 2. Real-data timing + sanity check (2,500 images/client, batch_size=8, 1 round, E=2)

Purpose: get an actual wall-clock number to extrapolate the full R=8 run, and sanity-check
that metrics move in a reasonable direction after real training (not just wiring-correct).

Run date: 2026-07-27, on this instance (1x A10, 23GB VRAM).

| Model | Wall time (1 round) | Metrics after round 1 |
|---|---|---|
| BLIP  | **21m 18.7s** | BLEU-4 = 0.3509, CIDEr = 1.2120 |
| CLIP  | **6m 16.1s**  | i2t: R@1=41.4, R@5=69.0, R@10=81.0 — t2i: R@1=29.89, R@5=60.03, R@10=75.20 |

Sanity read: CLIP i2t R@1=41.4% after a single round is a reasonable early number
(fully-trained CLIP ViT-B/32 on full COCO is typically ~50s for R@1). BLIP CIDEr
~1.2 after one round is also plausible early progress toward a fully-trained
CIDEr comfortably above 1.0.

### Extrapolation to the planned R=8 full run

- BLIP: ~21 min/round x 8 rounds ~= **2.8 hours**
- CLIP: ~6 min/round x 8 rounds ~= **50 min**
- Zero-shot baselines: a few minutes each
- **All-in estimate: ~3-3.5 hours, dominated by BLIP's per-round eval** (autoregressive
  caption generation over 1,000 val images every round is the slow part; CLIP's eval is
  just embedding lookups).

Levers to cut this down if needed: fewer rounds (R=5), evaluate BLIP every N rounds
instead of every round, or a larger batch size.
