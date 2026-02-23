# Eye Anemia Detection (MedSigLIP Linear Probe)

## What this project is
A lightweight eye‑anemia classifier built on top of a frozen MedSigLIP vision
encoder. We extract image embeddings and train a logistic‑regression linear
probe to separate **Anemia** vs **Non‑Anemia**.

## What we did
- Built multiple CSV splits for eye‑anemia experiments (full‑training, full‑eye,
  main, segmented).
- Trained and tuned a linear‑probe pipeline with CV, threshold calibration,
  and optional preprocessing.

## How we did it (high level)
1. Load images from CSV splits (absolute paths).
2. Extract frozen MedSigLIP embeddings.
3. Train a logistic‑regression probe.
4. Evaluate on validation/test splits and select thresholds.

## Dataset overview (short)
The eye‑anemia dataset is a mix of:
- **Dataset/dataset anemia** (India/Italy eye images)
- **Dataset/CP‑AnemiC dataset** (additional eye images)
- **Conjuctiva/Training** (augmented full‑training set)

The repo stores only **CSV splits**, not the raw images.

## How to reproduce
1. Place MedSigLIP weights locally (e.g., `medsiglip/`).
2. Ensure the datasets referenced by the CSVs exist at the same absolute paths.
3. Run the training script with your chosen split:

```bash
uv run python src/main/linear_probe_tuned.py \
  --train-csv src/full-training/train.csv \
  --val-csv src/full-training/val.csv \
  --test-csv src/full-training/test.csv \
  --model-dir medsiglip \
  --batch-size 16 \
  --max-iter 5000 \
  --wandb --wandb-entity sidhu1743 --wandb-project Medgemma
```

## File structure (key paths)
```
src/
  main/
    linear_probe_tuned.py     # main training + tuning pipeline
    train.csv                 # main split (train)
    val.csv                   # main split (val)
    test.csv                  # main split (test)
  full-eye/
    train.csv                 # full-eye split (train)
    val.csv                   # full-eye split (val)
    test.csv                  # full-eye split (test)
  full-training/
    train.csv                 # full-training split (train, augmented)
    val.csv                   # full-training split (val)
    test.csv                  # full-training split (test)
  segmented/
    train.csv                 # ROI-segmented split (train)
    val.csv                   # ROI-segmented split (val)
    test.csv                  # ROI-segmented split (test)

overview.md                  # dataset split comparison summary
```

## Notes
- CSVs contain absolute image paths.
- Images and model weights are not committed here.
