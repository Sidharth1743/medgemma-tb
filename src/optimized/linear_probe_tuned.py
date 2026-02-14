from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import torch
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression
import cv2
from transformers import SiglipVisionModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Sample:
    path: Path
    label: int
    patient_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimized MedSigLIP linear probe with patient-safe split, C grid search, and threshold calibration."
    )
    parser.add_argument(
        "--data-dir",
        default="Dataset/CP-AnemiC dataset",
        help="Root folder containing class subfolders.",
    )
    parser.add_argument(
        "--class-dirs",
        nargs="+",
        default=["Anemia", "Non-Anemia"],
        help="Class subfolder names under --data-dir (order maps to labels).",
    )
    parser.add_argument(
        "--model-dir",
        default="medsiglip",
        help="Local MedSigLIP model directory.",
    )
    parser.add_argument(
        "--train-csv",
        default=None,
        help="Optional CSV for train split (image_path,label).",
    )
    parser.add_argument(
        "--val-csv",
        default=None,
        help="Optional CSV for val split (image_path,label).",
    )
    parser.add_argument(
        "--test-csv",
        default=None,
        help="Optional CSV for test split (image_path,label).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding extraction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--c-grid",
        nargs="+",
        type=float,
        default=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
        help="Grid of C values for LogisticRegression.",
    )
    parser.add_argument(
        "--recall-target",
        type=float,
        default=0.9,
        help="Minimum recall target for threshold calibration on validation.",
    )
    parser.add_argument(
        "--recall-targets",
        nargs="+",
        type=float,
        default=[0.9, 0.93, 0.95],
        help="Recall targets to sweep for threshold calibration on validation.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help="Maximum iterations for LogisticRegression (epoch-like).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/optimized_linear_probe",
        help="Directory for metrics JSON.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-entity",
        default="sidhu1743",
        help="Weights & Biases entity/team.",
    )
    parser.add_argument(
        "--wandb-project",
        default="Medgemma",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=None,
        help="Optional wandb tags.",
    )
    parser.add_argument(
        "--no-local-save",
        action="store_true",
        help="Do not write local artifacts/metrics (W&B only).",
    )
    parser.add_argument(
        "--save-local",
        action="store_true",
        help="Force local artifacts even when W&B is enabled.",
    )
    parser.add_argument(
        "--resize-mode",
        choices=["stretch", "letterbox"],
        default="letterbox",
        help="Resize mode: stretch to 448x448 or letterbox padding.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def iter_images(root: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def infer_patient_id(path: Path, class_dir: Path) -> str:
    rel = path.relative_to(class_dir)
    parts = rel.parts
    if len(parts) >= 2:
        return parts[0]
    return path.stem


def load_samples(data_dir: Path, class_dirs: list[str]) -> list[Sample]:
    samples: list[Sample] = []
    for label, class_name in enumerate(class_dirs):
        class_path = data_dir / class_name
        if not class_path.exists():
            raise FileNotFoundError(f"Missing class folder: {class_path}")
        for img_path in iter_images(class_path):
            pid = infer_patient_id(img_path, class_path)
            samples.append(Sample(img_path, label, pid))
    if not samples:
        raise ValueError(f"No images found under {data_dir}")
    return samples


def load_csv_samples(path: Path) -> list[Sample]:
    import csv
    rows: list[Sample] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not {"image_path", "label"}.issubset(set(r.fieldnames or [])):
            raise ValueError(f"{path} must contain image_path,label columns")
        for row in r:
            img_path = Path(row["image_path"]).expanduser()
            if not img_path.is_absolute():
                img_path = PROJECT_ROOT / img_path
            rows.append(Sample(img_path.resolve(), int(row["label"]), img_path.stem))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _init_wandb(args: argparse.Namespace) -> Any | None:
    if not args.wandb:
        return None
    try:
        import wandb  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "wandb is enabled but not installed. Install it with `uv sync`."
        ) from exc
    config = {
        "data_dir": str(args.data_dir),
        "class_dirs": args.class_dirs,
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "c_grid": args.c_grid,
        "recall_target": float(args.recall_target),
        "recall_targets": args.recall_targets,
        "max_iter": int(args.max_iter),
        "model_dir": str(args.model_dir),
    }
    return wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        config=config,
        tags=args.wandb_tags,
    )


def patient_safe_split(
    samples: list[Sample], train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    patient_ids = sorted({s.patient_id for s in samples})
    rng = random.Random(seed)
    rng.shuffle(patient_ids)
    n = len(patient_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_ids = set(patient_ids[:n_train])
    val_ids = set(patient_ids[n_train : n_train + n_val])
    test_ids = set(patient_ids[n_train + n_val :])
    train = [s for s in samples if s.patient_id in train_ids]
    val = [s for s in samples if s.patient_id in val_ids]
    test = [s for s in samples if s.patient_id in test_ids]
    return train, val, test


def _letterbox_tf(img: tf.Tensor, size: int = 448) -> tf.Tensor:
    with tf.device("/CPU:0"):
        h = tf.shape(img)[0]
        w = tf.shape(img)[1]
        scale = tf.minimum(
            tf.cast(size, tf.float32) / tf.cast(h, tf.float32),
            tf.cast(size, tf.float32) / tf.cast(w, tf.float32),
        )
        new_h = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
        new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)
        resized = tf.image.resize(img, [new_h, new_w], method="bilinear", antialias=False)
        pad_h = size - new_h
        pad_w = size - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padded = tf.pad(
            resized, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0
        )
        return padded


def preprocess_images(
    imgs: list[Image.Image],
    device: str,
    dtype: torch.dtype,
    resize_mode: str,
) -> torch.Tensor:
    processed: list[torch.Tensor] = []
    for img in imgs:
        img = img.convert("RGB")
        np_img = np.asarray(img, dtype=np.uint8)
        with tf.device("/CPU:0"):
            tf_img = tf.convert_to_tensor(np_img)
            if resize_mode == "letterbox":
                tf_img = _letterbox_tf(tf_img, size=448)
            else:
                tf_img = tf.image.resize(
                    tf_img, [448, 448], method="bilinear", antialias=False
                )
            tf_img = tf.cast(tf_img, tf.float32) / 255.0
            tf_img = tf_img * 2.0 - 1.0
            np_out = tf_img.numpy()
        tensor = torch.from_numpy(np_out).permute(2, 0, 1)
        processed.append(tensor)
    batch = torch.stack(processed, dim=0).to(device=device, dtype=dtype)
    return batch


def _load_image_for_viz(path: Path) -> Image.Image | None:
    if not path.exists():
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        cv_img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if cv_img is None:
            return None
        if cv_img.dtype == np.uint16:
            cv_img = ((cv_img.astype(np.float32) / 65535.0) * 255.0).clip(
                0, 255
            ).astype(np.uint8)
        elif cv_img.dtype != np.uint8:
            cv_img = cv2.normalize(cv_img, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )
        if cv_img.ndim == 2:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        elif cv_img.shape[2] == 4:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB)
        else:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv_img)


def _make_tf_viz_images(img: Image.Image, resize_mode: str) -> dict[str, Image.Image]:
    rgb = img.convert("RGB")
    np_img = np.asarray(rgb, dtype=np.uint8)
    tf_img = tf.convert_to_tensor(np_img)
    if resize_mode == "letterbox":
        tf_resized = _letterbox_tf(tf_img, size=448)
    else:
        tf_resized = tf.image.resize(
            tf_img,
            [448, 448],
            method="bilinear",
            antialias=False,
        )
    tf_float = tf.cast(tf_resized, tf.float32) / 255.0
    tf_norm = tf_float * 2.0 - 1.0
    resized = Image.fromarray(tf_resized.numpy().astype(np.uint8))
    norm_vis = ((tf_norm + 1.0) / 2.0 * 255.0).numpy().clip(0, 255).astype(np.uint8)
    normalized = Image.fromarray(norm_vis)
    return {
        "original": img,
        "rgb": rgb,
        "resized_448": resized,
        "normalized": normalized,
    }


def extract_embeddings(
    samples: list[Sample],
    model: torch.nn.Module,
    device: str,
    dtype: torch.dtype,
    batch_size: int,
    resize_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    all_embeds: list[np.ndarray] = []
    all_labels: list[int] = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        imgs = []
        kept_labels = []
        for s in batch:
            img = _load_image_for_viz(s.path)
            if img is None:
                continue
            imgs.append(img)
            kept_labels.append(s.label)
        if not imgs:
            continue
        with torch.no_grad():
            pixel_values = preprocess_images(imgs, device, dtype, resize_mode)
            outputs = model(pixel_values=pixel_values)
            embeddings = outputs.pooler_output
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        all_embeds.append(embeddings.float().cpu().numpy())
        all_labels.extend(kept_labels)
    x = np.concatenate(all_embeds, axis=0)
    y = np.asarray(all_labels, dtype=np.int64)
    return x, y


def fit_logreg(
    x_train: np.ndarray, y_train: np.ndarray, c_val: float, max_iter: int
) -> tuple[dict[str, np.ndarray], LogisticRegression]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_train_std = (x_train - mean) / std
    model = LogisticRegression(
        solver="saga",
        C=c_val,
        max_iter=max_iter,
        class_weight="balanced",
    )
    model.fit(x_train_std, y_train)
    scaler = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    return scaler, model


def predict_prob(
    x: np.ndarray, scaler: dict[str, np.ndarray], model: LogisticRegression
) -> np.ndarray:
    x_std = (x - scaler["mean"]) / scaler["std"]
    probs = model.predict_proba(x_std)[:, 1]
    return probs.astype(np.float32)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    tpr = tp / n_pos
    fpr = fp / n_neg
    tpr = np.concatenate(([0.0], tpr, [1.0]))
    fpr = np.concatenate(([0.0], fpr, [1.0]))
    return float(np.trapezoid(tpr, fpr))


def roc_threshold_stats(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return {
            "best_threshold": 0.5,
            "best_tpr": float("nan"),
            "best_fpr": float("nan"),
            "best_j": float("nan"),
        }
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    scores_sorted = y_score[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    tpr = tp / n_pos
    fpr = fp / n_neg
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    best_thresh = float(scores_sorted[best_idx])
    return {
        "best_threshold": best_thresh,
        "best_tpr": float(tpr[best_idx]),
        "best_fpr": float(fpr[best_idx]),
        "best_j": float(j[best_idx]),
    }


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, steps: int = 101) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(steps):
        t = i / (steps - 1)
        rows.append(compute_metrics(y_true, y_prob, threshold=float(t)) | {"threshold": t})
    return rows


def best_threshold_for_recall(
    y_true: np.ndarray, y_prob: np.ndarray, recall_target: float
) -> float:
    thresholds = np.linspace(0, 1, 101)
    best_t = 0.5
    best_precision = -1.0
    for t in thresholds:
        metrics = compute_metrics(y_true, y_prob, threshold=float(t))
        if metrics["recall"] >= recall_target:
            if metrics["precision"] > best_precision:
                best_precision = metrics["precision"]
                best_t = float(t)
    return best_t


def sweep_recall_targets(
    y_true: np.ndarray, y_prob: np.ndarray, recall_targets: list[float]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in recall_targets:
        threshold = best_threshold_for_recall(y_true, y_prob, target)
        metrics = compute_metrics(y_true, y_prob, threshold=threshold)
        rows.append(
            {
                "recall_target": float(target),
                "threshold": float(threshold),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "confusion_matrix": metrics["confusion_matrix"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.wandb and not args.save_local:
        args.no_local_save = True
    data_dir = resolve_path(args.data_dir)
    model_dir = resolve_path(args.model_dir)
    output_dir = resolve_path(args.output_dir)
    if not args.no_local_save:
        output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _init_wandb(args)

    if args.train_csv and args.val_csv and args.test_csv:
        train = load_csv_samples(resolve_path(args.train_csv))
        val = load_csv_samples(resolve_path(args.val_csv))
        test = load_csv_samples(resolve_path(args.test_csv))
    else:
        samples = load_samples(data_dir, args.class_dirs)
        train, val, test = patient_safe_split(
            samples, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    vision_model = SiglipVisionModel.from_pretrained(model_dir, dtype=dtype).to(device)
    vision_model.eval()

    x_train, y_train = extract_embeddings(
        train, vision_model, device, dtype, args.batch_size, args.resize_mode
    )
    x_val, y_val = extract_embeddings(
        val, vision_model, device, dtype, args.batch_size, args.resize_mode
    )
    x_test, y_test = extract_embeddings(
        test, vision_model, device, dtype, args.batch_size, args.resize_mode
    )

    best_c = None
    best_auc = -1.0
    best_scaler = None
    best_model = None
    c_rows: list[dict[str, Any]] = []
    for c in args.c_grid:
        scaler, model = fit_logreg(x_train, y_train, c, args.max_iter)
        val_probs = predict_prob(x_val, scaler, model)
        auc = roc_auc(y_val, val_probs)
        c_rows.append({"c": float(c), "val_auc": float(auc)})
        if auc > best_auc:
            best_auc = auc
            best_c = c
            best_scaler = scaler
            best_model = model

    assert best_model is not None and best_scaler is not None and best_c is not None

    val_probs = predict_prob(x_val, best_scaler, best_model)
    test_probs = predict_prob(x_test, best_scaler, best_model)

    val_auc = roc_auc(y_val, val_probs)
    test_auc = roc_auc(y_test, test_probs)

    threshold = best_threshold_for_recall(y_val, val_probs, args.recall_target)
    test_metrics = compute_metrics(y_test, test_probs, threshold=threshold)
    roc_stats = roc_threshold_stats(y_val, val_probs)
    sweep_rows = threshold_sweep(y_val, val_probs, steps=101)
    best_thresholds: dict[str, dict[str, float]] = {}
    for metric_name in ("accuracy", "precision", "recall", "f1"):
        best_row = max(sweep_rows, key=lambda r: r[metric_name])
        best_thresholds[metric_name] = {
            "threshold": float(best_row["threshold"]),
            metric_name: float(best_row[metric_name]),
        }
    best_f1_threshold = best_thresholds["f1"]["threshold"]
    test_metrics_best_f1 = compute_metrics(
        y_test, test_probs, threshold=best_f1_threshold
    )
    recall_sweep = sweep_recall_targets(y_val, val_probs, args.recall_targets)
    best_recall_row = max(recall_sweep, key=lambda r: r["f1"])

    summary = {
        "data": {
            "train_images": len(train),
            "val_images": len(val),
            "test_images": len(test),
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "class_dirs": args.class_dirs,
        },
        "best_c": best_c,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "threshold": threshold,
        "metrics_at_threshold": test_metrics,
        "metrics_at_best_f1_threshold_test": test_metrics_best_f1,
        "best_f1_threshold": best_f1_threshold,
        "recall_target": args.recall_target,
        "recall_target_sweep": recall_sweep,
        "recall_target_best_by_f1": best_recall_row,
        "roc_threshold": roc_stats,
        "best_thresholds": best_thresholds,
    }

    if not args.no_local_save:
        out_path = output_dir / "summary.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.log(
            {
                "metrics/val_auc": val_auc,
                "metrics/test_auc": test_auc,
                "metrics/threshold": threshold,
                "metrics/threshold_accuracy": test_metrics["accuracy"],
                "metrics/threshold_precision": test_metrics["precision"],
                "metrics/threshold_recall": test_metrics["recall"],
                "metrics/threshold_f1": test_metrics["f1"],
                "metrics/best_f1_threshold": best_f1_threshold,
                "metrics/best_f1_threshold_accuracy": test_metrics_best_f1["accuracy"],
                "metrics/best_f1_threshold_precision": test_metrics_best_f1["precision"],
                "metrics/best_f1_threshold_recall": test_metrics_best_f1["recall"],
                "metrics/best_f1_threshold_f1": test_metrics_best_f1["f1"],
                "metrics/confusion_matrix": test_metrics["confusion_matrix"],
                "metrics/roc_best_threshold": roc_stats["best_threshold"],
                "metrics/roc_best_j": roc_stats["best_j"],
                "metrics/roc_best_tpr": roc_stats["best_tpr"],
                "metrics/roc_best_fpr": roc_stats["best_fpr"],
                "metrics/best_threshold_accuracy": best_thresholds["accuracy"]["threshold"],
                "metrics/best_threshold_precision": best_thresholds["precision"]["threshold"],
                "metrics/best_threshold_recall": best_thresholds["recall"]["threshold"],
                "metrics/best_threshold_f1": best_thresholds["f1"]["threshold"],
                "metrics/recall_sweep_best_target": best_recall_row["recall_target"],
                "metrics/recall_sweep_best_threshold": best_recall_row["threshold"],
                "metrics/recall_sweep_best_accuracy": best_recall_row["accuracy"],
                "metrics/recall_sweep_best_precision": best_recall_row["precision"],
                "metrics/recall_sweep_best_recall": best_recall_row["recall"],
                "metrics/recall_sweep_best_f1": best_recall_row["f1"],
            }
        )
        try:
            import wandb  # type: ignore

            cm = test_metrics["confusion_matrix"]
            cm_img = Image.new("RGB", (760, 640), color=(255, 255, 255))
            draw = ImageDraw.Draw(cm_img)
            draw.text((24, 20), "Optimized Confusion Matrix", fill=(0, 0, 0))
            grid_left, grid_top = 150, 110
            cell = 170
            labels = ["Non-Anemia", "Anemia"]
            arr = np.asarray(cm, dtype=np.float32)
            min_v = float(arr.min())
            max_v = float(arr.max())
            norm = (arr - min_v) / (max_v - min_v + 1e-6)
            for r in range(2):
                for c in range(2):
                    val = int(arr[r, c])
                    shade = int(255 - 170 * float(norm[r, c]))
                    color = (shade, shade, 255)
                    x0 = grid_left + c * cell
                    y0 = grid_top + r * cell
                    x1 = x0 + cell
                    y1 = y0 + cell
                    draw.rectangle(
                        [(x0, y0), (x1, y1)],
                        fill=color,
                        outline=(80, 80, 80),
                        width=2,
                    )
                    draw.text((x0 + 70, y0 + 78), str(val), fill=(0, 0, 0))
            draw.text((grid_left + 120, grid_top + 365), "Predicted", fill=(0, 0, 0))
            draw.text((35, grid_top + 140), "True", fill=(0, 0, 0))
            for i, lbl in enumerate(labels):
                draw.text((grid_left + i * cell + 42, grid_top - 28), lbl, fill=(0, 0, 0))
                draw.text((grid_left - 110, grid_top + i * cell + 78), lbl, fill=(0, 0, 0))
            wandb_run.log({"plots/confusion_matrix": wandb.Image(cm_img)})
            c_table = wandb.Table(columns=["C", "val_auc"], data=[[r["c"], r["val_auc"]] for r in c_rows])
            wandb_run.log({"tables/c_grid": c_table})
            try:
                wandb_run.log(
                    {"plots/val_auc_vs_c": wandb.plot.line(c_table, "C", "val_auc", title="Val AUC vs C")}
                )
            except Exception:
                pass
            sweep_table = wandb.Table(
                columns=["threshold", "accuracy", "precision", "recall", "f1"],
                data=[
                    [r["threshold"], r["accuracy"], r["precision"], r["recall"], r["f1"]]
                    for r in sweep_rows
                ],
            )
            wandb_run.log({"tables/threshold_sweep": sweep_table})
            recall_table = wandb.Table(
                columns=["recall_target", "threshold", "accuracy", "precision", "recall", "f1"],
                data=[
                    [
                        r["recall_target"],
                        r["threshold"],
                        r["accuracy"],
                        r["precision"],
                        r["recall"],
                        r["f1"],
                    ]
                    for r in recall_sweep
                ],
            )
            wandb_run.log({"tables/recall_target_sweep": recall_table})
        except Exception:
            pass
        try:
            import wandb  # type: ignore

            train_samples = train[:2]
            test_samples = test[:2]
            for s, split in (
                [(x, "train") for x in train_samples] + [(x, "test") for x in test_samples]
            ):
                img = _load_image_for_viz(s.path)
                if img is None:
                    continue
                viz = _make_tf_viz_images(img, args.resize_mode)
                wandb_run.log(
                    {
                        f"images/{split}_{s.path.name}_original": wandb.Image(viz["original"]),
                        f"images/{split}_{s.path.name}_rgb": wandb.Image(viz["rgb"]),
                        f"images/{split}_{s.path.name}_resized": wandb.Image(viz["resized_448"]),
                        f"images/{split}_{s.path.name}_normalized": wandb.Image(viz["normalized"]),
                    }
                )
        except Exception:
            pass
        wandb_run.finish()

    print(json.dumps(summary, indent=2))
    if not args.no_local_save:
        print(f"Saved summary: {out_path}")


if __name__ == "__main__":
    main()
