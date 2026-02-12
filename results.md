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

## Key Takeaway
- For this dataset, the trained linear probe outperforms zero-shot baseline on overall classification quality (especially accuracy and AUC), while both setups currently show high recall for anemia and low specificity for non-anemia.
