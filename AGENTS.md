# Repository Guidelines

## Project Structure & Module Organization
- `src/`: actively maintained runnable scripts (for example, `src/quick_start_with_hugging_face.py`).
- `src/baseline/`: baseline-only zero-shot run/evaluation scripts (`baseline-model-running.py`, `baseline-model-evaluation.py`).
- Repository root: reference notebooks/scripts exported from Colab (for example, `train_data_efficient_classifier.py`, `google-official-quick-start.py`) and project docs (`*_overview.md`, `*_report.md`).
- `medsiglip/`: local Hugging Face model assets (`config.json`, tokenizer and processor files).
- `Dataset/`: local image datasets used for experimentation and classifier training.
  - Expected anemia task layout: separate class folders such as `Dataset/.../Anemia/` and `Dataset/.../Non-Anemia/`.
- `model.safetensors`: large model weights file used by local inference scripts.

## Build, Test, and Development Commands
- `uv sync`: install project dependencies from `pyproject.toml` / `uv.lock`.
- `uv run python src/quick_start_with_hugging_face.py --model-dir medsiglip --images <path>`: run local MedSigLIP inference.
- `uv run python src/train_linear_probe_anemia.py --train-csv "Dataset/dataset anemia/train.csv" --test-csv "Dataset/dataset anemia/test.csv" --model-dir medsiglip --batch-size 16 --epochs 400 --output-dir results/linear_probe_anemia_full`: run end-to-end embedding + linear-probe training/evaluation.
- Add `--wandb` (optional) to log to Weights & Biases; override `--wandb-entity` and `--wandb-project` as needed.
- Add `--no-local-save` to skip local artifacts/plots/metrics (W&B only).
- `uv run python src/helpers/image_checking.py --input-csv "Dataset/dataset anemia/train.csv" --only-png --max-images 200 --output-dir results/image_checking`: inspect exact image decode failures.
- `uv run python src/baseline/baseline-model-running.py --input-csv "Dataset/dataset anemia/test.csv" --model-dir medsiglip --batch-size 16 --output-dir results/baseline_zero_shot`: run zero-shot baseline (no training).
- `uv run python src/baseline/baseline-model-evaluation.py --predictions-csv results/baseline_zero_shot/baseline_predictions.csv --output-dir results/baseline_zero_shot`: compute baseline metrics (accuracy, AUC, F1, confusion matrix).
- `uv run python -m py_compile src/*.py`: quick syntax validation for source scripts.
- `uv run python main.py`: minimal project entry point.
- `uv run python src/optimized/linear_probe_tuned.py --train-csv "src/segmented/train.csv" --val-csv "src/segmented/val.csv" --test-csv "src/segmented/test.csv" --model-dir medsiglip --batch-size 16 --wandb --wandb-entity sidhu1743 --wandb-project Medgemma`: optimized segmented split + grid search + recall-threshold calibration.
- `uv run python src/optimized/linear_probe_tuned.py --train-csv "src/full-training/train.csv" --val-csv "src/full-training/val.csv" --test-csv "src/full-training/test.csv" --model-dir medsiglip --batch-size 16 --max-iter 5000 --wandb --wandb-entity sidhu1743 --wandb-project Medgemma`: full-dataset training (all Dataset/ images, excluding full-eye images from dataset anemia).

## Coding Style & Naming Conventions
- Target Python `>=3.12`.
- Follow PEP 8: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes, constants in `UPPER_SNAKE_CASE`.
- Prefer type hints in new/edited code.
- Keep scripts CLI-friendly (`argparse`) and avoid notebook-only syntax (for example, `!wget`) in `src/`.
- File naming: use descriptive lowercase names with underscores (for example, `train_linear_probe_anemia.py`).

## Testing Guidelines
- No formal test framework is configured yet.
- For now, validate changes with:
  - `uv run python -m py_compile src/*.py`
  - at least one end-to-end script run on a sample image.
- When adding tests, use `pytest` with files named `tests/test_*.py` and focus on data loading, preprocessing, and metric computation.

## Modeling Baseline (Project-Specific)
- Follow Google’s data-efficient workflow used in `train_data_efficient_classifier.py`: extract frozen MedSigLIP embeddings, then train a lightweight linear probe (logistic regression) as the first baseline.
- Use folder-based labels for the anemia task: `Anemia -> 1`, `Non-Anemia -> 0`.
- Current split targets: `Anemia (train=300, test=80)`, `Non-Anemia (train=400, test=82)` via CSV files.
- Treat zero-shot prompts as a quick check, not the primary performance target for this repository.
- For anemia vs non-anemia work, report at minimum accuracy, F1, ROC-AUC, and confusion matrix on a held-out test split.
- Log ROC-based optimal threshold (Youden’s J) and its TPR/FPR for each run.
- Log metrics at the ROC-best threshold, the best-F1 threshold (test metrics), and a threshold sweep table (accuracy/precision/recall/F1 vs threshold) to W&B.
- Log a recall-target sweep table (0.90/0.93/0.95 by default) and pick the best trade-off by F1.
- Keep `0.5` threshold as default; tuning threshold is a separate explicit experiment.
- Baseline zero-shot mode is inference-only: do not update model weights or train a classifier.
- Baseline prompt strategy follows official Google zero-shot style with contrastive prompt sets (multiple prompts per class, class-wise averaged text embeddings).
- Linear probe preprocessing: resize to `448x448` with bilinear interpolation and `antialias=False`, convert to RGB, normalize to `[-1, 1]`.
- Linear probe classifier: scikit-learn `LogisticRegression` with `solver="saga"` (data-efficient).
- Optimized tuning uses explicit segmented train/val/test CSVs under `src/segmented/` with CP-AnemiC augmentation and recall-target calibration.
  - Segmented set filter: filename contains `forniceal`, `forniceal_palpebral`, or `palpebral`.
  - CP-AnemiC images are all treated as segmented.
- Full training CSVs live under `src/full-training/` and include all anemia/non-anemia images across `Dataset/`, excluding full-eye images from `Dataset/dataset anemia`.

## Image Loading Notes
- Some segmented PNGs fail in Pillow with `PIL.UnidentifiedImageError` at open time.
- Use OpenCV fallback in `src/train_linear_probe_anemia.py` for those files.
- Handle OpenCV `uint16` PNG decodes by scaling to `uint8` before creating PIL images.
- Save image-status CSVs (`train_image_status.csv`, `test_image_status.csv`) and track `opencv_rescued` counts in `metrics.json`.
- Save local evaluation plots for each run:
  - Linear probe: `runs/<run_id>/confusion_matrix.png`, `runs/<run_id>/roc_curve.png`, `runs/<run_id>/score_histogram.png`, `runs/<run_id>/top_weight_dimensions.png`, `runs/<run_id>/probe_parameters.json`
  - Track runs in `run_history.csv` (includes metrics + learned probe bias/weight norm), and `latest_run.txt`.
  - Zero-shot baseline eval: `baseline_confusion_matrix.png`, `baseline_roc_curve.png`, `baseline_score_histogram.png`

## Baseline Results
- Previous run (before OpenCV fallback, many PNGs skipped):
  - Train/Test usable counts: train `555` (skipped `145`), test `120` (skipped `42`).
  - Metrics: accuracy `0.558333`, precision `0.500000`, recall `0.962264`, F1 `0.658065`, ROC-AUC `0.692763`, confusion matrix `[[16, 51], [2, 51]]`.
- Current run (after OpenCV fallback, no skipped images):
  - Train/Test counts: train `700` (opencv_rescued `145`), test `162` (opencv_rescued `42`).
  - Metrics: accuracy `0.598765`, precision `0.556391`, recall `0.925000`, F1 `0.694836`, ROC-AUC `0.652896`, confusion matrix `[[23, 59], [6, 74]]`.
- Zero-shot baseline results should be tracked separately under `results/baseline_zero_shot/`.

## Hardware & GPU Usage
- Available hardware: **2 x NVIDIA RTX 3060 (12 GB VRAM each)**.
- Run MedSigLIP embedding extraction on a single GPU (`cuda:0`).
- Keep linear-probe training on CPU unless dataset size makes GPU acceleration necessary.

## Commit & Pull Request Guidelines
- No Git history conventions exist yet (repository has no commits).
- Use concise imperative commit messages, e.g., `Add anemia linear probe training script`.
- PRs should include:
  - what changed and why,
  - exact run commands,
  - key outputs/metrics,
  - any dataset/model path assumptions.

## Security & Configuration Tips
- Do not commit patient-identifiable data, secrets, or tokens.
- Keep large weights and datasets out of Git unless explicitly required.
- Use local paths via CLI flags instead of hardcoding environment-specific directories.

## Work Log
- `2026-02-12 13:16:43 IST`: Created subject-safe dataset splits and CSVs (`train.csv`, `test.csv`) for anemia vs non-anemia.
- `2026-02-12 13:16:43 IST`: Implemented `src/train_linear_probe_anemia.py` for embedding extraction + linear probe evaluation with metrics and saved artifacts.
- `2026-02-12 13:16:43 IST`: Added image-status reporting (`train_image_status.csv`, `test_image_status.csv`) and skipped/used counts.
- `2026-02-12 13:16:43 IST`: Identified Pillow PNG decode failures on segmented images; added OpenCV fallback with `uint16 -> uint8` conversion.
- `2026-02-12 13:16:43 IST`: Added baseline-only zero-shot pipeline in `src/baseline/` (`baseline-model-running.py`, `baseline-model-evaluation.py`) using contrastive prompt sets.
- `2026-02-12 13:16:43 IST`: Added evaluation outputs with accuracy, precision, recall, F1, AUC, and confusion matrix; consolidated results in `results.md`.
- `2026-02-14 19:40:00 IST`: Added segmented-only CSV builder (`src/segmented/build_segmented_splits.py`) combining dataset anemia + CP-AnemiC and balanced 800/100/100 splits.
- `2026-02-14 22:53:00 IST`: Added full-dataset CSVs under `src/full-training/` (all Dataset/ images, excluding full-eye images from dataset anemia) and ran full-dataset linear probe tuning.
