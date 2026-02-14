# Results Summary

## 1) Linear Probe (Before OpenCV Fallback)
- Train/Test usable counts: train `555` (skipped `145`), test `120` (skipped `42`)
- Accuracy: `0.558333`
- Precision: `0.500000`
- Recall: `0.962264`
- F1: `0.658065`
- ROC-AUC: `0.692763`
- Confusion Matrix: `[[16, 51], [2, 51]]`

## 2) Linear Probe (After OpenCV Fallback)
- Train/Test counts: train `700` (opencv_rescued `145`, skipped `0`), test `162` (opencv_rescued `42`, skipped `0`)
- Accuracy: `0.598765`
- Precision: `0.556391`
- Recall: `0.925000`
- F1: `0.694836`
- ROC-AUC: `0.652896`
- Confusion Matrix: `[[23, 59], [6, 74]]`

## 3) Zero-Shot Baseline (Official-Style Prompting, No Training)
- Examples evaluated: `162`
- Image loading: predicted `162`, skipped `0`, opencv_rescued `42`
- Accuracy: `0.487654`
- Precision: `0.490446`
- Recall: `0.962500`
- F1: `0.649789`
- AUC: `0.562348`
- Confusion Matrix: `[[2, 80], [3, 77]]`

## 4) Linear Probe (Full Cleaned Dataset)
- Dataset: 217 patients (91 Anemia, 126 Non-Anemia)
- Train/Test counts: train `686`, test `172`
- Image loading: train `686` (opencv_rescued `157`), test `172` (opencv_rescued `30`)
- Accuracy: `0.715116`
- Precision: `0.714286`
- Recall: `0.670732`
- F1: `0.691824`
- ROC-AUC: `0.783740`
- Confusion Matrix: `[[68, 22], [27, 55]]`
- Parameters: `batch_size=16, epochs=400, lr=0.01, weight_decay=0.0001`

## 5) Linear Probe (TF Resize + SAGA)
- Train/Test counts: train `686`, test `172`
- Accuracy: `0.686047`
- Precision: `0.694444`
- Recall: `0.609756`
- F1: `0.649351`
- ROC-AUC: `0.812195`
- Confusion Matrix: `[[68, 22], [32, 50]]`
- ROC best threshold (Youden’s J): `0.1475936` (TPR `0.829268`, FPR `0.333333`)

## 6) Linear Probe (Auto-PIL + Torch)
- Train/Test counts: train `686`, test `172`
- Accuracy: `0.726744`
- Precision: `0.733333`
- Recall: `0.670732`
- F1: `0.700637`
- ROC-AUC: `0.784688`
- Confusion Matrix: `[[70, 20], [27, 55]]`
- ROC best threshold (Youden’s J): `0.5850275` (TPR `0.670732`, FPR `0.200000`)

## 7) Optimized Split + Grid Search + Recall Calibration (Letterbox, Segmented-Only)
- Train/Val/Test counts: `800 / 100 / 100`
- Class balance (segmented + CP-AnemiC):
  - Train: Anemia `400`, Non-Anemia `400`
  - Val: Anemia `50`, Non-Anemia `50`
  - Test: Anemia `50`, Non-Anemia `50`
- Best C (grid): `3`
- Val AUC: `0.836400`
- Test AUC: `0.836800`
- Recall target: `0.90`
- Threshold (val-calibrated): `0.05`
- Resize mode: `letterbox` padding (no stretching)
- Metrics at threshold:
  - Accuracy `0.71`
  - Precision `0.652174`
  - Recall `0.90`
  - F1 `0.756303`
  - Confusion Matrix `[[26, 24], [5, 45]]`
- Best-F1 threshold (val) applied to test:
  - Threshold `0.23`
  - Accuracy `0.76`
  - Precision `0.716667`
  - Recall `0.86`
  - F1 `0.781818`
  - Confusion Matrix `[[33, 17], [7, 43]]`
- Recall-target sweep (val thresholds → test metrics): best by F1 at target `0.91`:
  - Threshold `0.04`
  - Accuracy `0.72`
  - Precision `0.657143`
  - Recall `0.92`
  - F1 `0.766667`
  - Confusion Matrix `[[26, 24], [4, 46]]`
- ROC best threshold (Youden’s J): `0.505267` (TPR `0.78`, FPR `0.18`)

## 8) Full-Dataset Linear Probe (All Dataset/ Images, Excluding Full-Eye from dataset anemia) - **Latest Run**
- Train/Val/Test counts: `9283 / 1160 / 1162`
- Class balance:
  - Train: Anemia `4730`, Non-Anemia `4553`
  - Val: Anemia `591`, Non-Anemia `569`
  - Test: Anemia `592`, Non-Anemia `570`
- Best C (grid): `1`
- Val AUC: `0.909260`
- Test AUC: `0.912441`
- Recall target: `0.90`
- Threshold (val-calibrated): `0.31`
- Resize mode: `letterbox` padding (no stretching)
- Metrics at threshold:
  - Accuracy `0.823580`
  - Precision `0.775249`
  - Recall `0.920608`
  - F1 `0.841699`
  - Confusion Matrix `[[412, 158], [47, 545]]`
- Best-F1 threshold (val) applied to test:
  - Threshold `0.44`
  - Accuracy `0.840792`
  - Precision `0.823529`
  - Recall `0.875000`
  - F1 `0.848485`
  - Confusion Matrix `[[459, 111], [74, 518]]`
- Recall-target sweep (val thresholds → test metrics): best by F1 at target `0.90`:
  - Threshold `0.31`
  - Accuracy `0.812931`
  - Precision `0.769452`
  - Recall `0.903553`
  - F1 `0.831128`
  - Confusion Matrix `[[409, 160], [57, 534]]`
- ROC best threshold (Youden’s J): `0.594325` (TPR `0.795262`, FPR `0.114236`)

## Key Takeaway
- For this dataset, the trained linear probe outperforms zero-shot baseline on overall classification quality (especially accuracy and AUC), while both setups currently show high recall for anemia and low specificity for non-anemia.

## Run Artifacts (Plots + Parameters)
- Linear probe runs now save local plots and parameter summary under `runs/<run_id>/`:
  - `confusion_matrix.png`
  - `roc_curve.png`
  - `score_histogram.png`
  - `top_weight_dimensions.png`
  - `probe_parameters.json` (includes learned bias and top weight dimensions)
- Linear probe run tracking files in the output root:
  - `run_history.csv` (each run: metrics + learned bias/weight norm)
  - `latest_run.txt` (points to the newest run id)
- Zero-shot baseline evaluation now saves:
  - `baseline_confusion_matrix.png`
  - `baseline_roc_curve.png`
  - `baseline_score_histogram.png`
codex resume 019c576d-e02c-74e1-9d2a-0d54fb7f836c
