# Reproducing the project

## Core tests without data

Use the setup commands in README.md. `scripts/setup_official_source.py` clones the official source at commit `075fc5f5a52d11077f9dc2b074644618f26939e2` only if the checkout does not exist; it refuses to silently change an existing dirty or mismatched checkout.

Set PYTHONPATH as shown in README.md because the research workflows import helpers from both project scripts and official scripts. `uv sync --locked` uses the tracksdata commit resolved in `uv.lock`; using an unconstrained update is a different environment.

## Dataset and weights

Obtain the dataset from the [competition data page](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/data) under its access conditions. Expected training root:

```text
data/raw/competition/biohub-cell-tracking-during-development/train/
```

The split manifest contains dataset identifiers and split metadata, not microscopy images or annotation graphs. Model checkpoints are not distributed by this repository. Training requires separately obtained upstream initialization weights as applicable; inspect `--help` for explicit training options.

```bash
uv run --no-sync python scripts/train_unet_transformer_mps.py --help
uv run --no-sync python scripts/predict_unet_transformer_mps.py --help
```

These are research workflows, not a one-command download-and-train product. Do not treat CLI defaults as the exact Warm2 training recipe.

## Official Fold0 reproduction

The existing reproduction gate requires the two exact Warm2 prediction sets under:

```text
predictions/heqiuyan/official_baseline_fold0_warm2_lr1e5_det0995/split_0/
predictions/heqiuyan/official_baseline_fold0_warm2_lr1e5_det09975/split_0/
```

These directory names are historical artifact identifiers; they do not depend on an OS username. With those prediction GEFFs and the training ground truth available:

```bash
uv run --no-sync python scripts/verify_frozen_v9.py --progress
```

It verifies the 40 datasets, exact candidate/intervention counts, overall component scores, and per-prefix summaries to tolerance 5e-7. It does not train a model or rerun inference. Without those artifacts, the repository supports inspection and core tests, but cannot independently regenerate the recorded official result.

## Research archives

Stage11/Stage12 scripts may need historical reports, action tables, or checkpoints not shipped here. Source-derived providers include explicit source hashes and reject mismatches. Resolve provenance with evidence before changing contracts or promoting an experimental strategy.
