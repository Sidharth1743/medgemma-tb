from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw
from transformers import AutoProcessor, SiglipVisionModel

from helpers.plotting import (
    save_confusion_matrix_png,
    save_roc_curve_png,
    save_score_histogram_png,
    save_top_weight_dimensions_png,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate anemia linear probe on frozen MedSigLIP embeddings."
    )
    parser.add_argument(
        "--train-csv",
        default="Dataset/dataset anemia/train.csv",
        help="Path to train CSV with columns: image_path,label",
    )
    parser.add_argument(
        "--test-csv",
        default="Dataset/dataset anemia/test.csv",
        help="Path to test CSV with columns: image_path,label",
    )
    parser.add_argument(
        "--model-dir",
        default="medsiglip",
        help="Local MedSigLIP model directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding extraction.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/linear_probe_anemia",
        help="Directory to write metrics and artifacts.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=400,
        help="Training epochs for linear logistic probe.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate for linear logistic probe.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for linear logistic probe.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap for train rows (debug only).",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="Optional cap for test rows (debug only).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Optional stable run id for tracking. "
            "If omitted, UTC timestamp is used."
        ),
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
        help="Do not write local artifacts/plots/metrics (W&B only).",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def read_csv_rows(csv_path: Path, limit: int | None = None) -> list[tuple[Path, int]]:
    rows: list[tuple[Path, int]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"image_path", "label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{csv_path} must contain columns: image_path,label")
        for row in reader:
            rows.append((resolve_path(row["image_path"]), int(row["label"])))
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def load_batch_images(paths: list[Path]) -> tuple[list[Image.Image], list[int], int]:
    imgs: list[Image.Image] = []
    kept_indices: list[int] = []
    rescued_with_opencv = 0
    for i, p in enumerate(paths):
        if not p.exists():
            print(f"Skipping missing image: {p}")
            continue
        try:
            imgs.append(Image.open(p).convert("RGB"))
            kept_indices.append(i)
        except Exception:
            # Fallback path: OpenCV can decode some PNGs that Pillow rejects.
            cv_img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if cv_img is None:
                print(f"Skipping unreadable image: {p}")
                continue
            if cv_img.dtype == np.uint16:
                # MedSigLIP processor expects regular image intensities; convert
                # 16-bit fallback decodes to 8-bit RGB consistently.
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
            imgs.append(Image.fromarray(cv_img))
            kept_indices.append(i)
            rescued_with_opencv += 1
            print(f"Recovered with OpenCV fallback: {p}")
    return imgs, kept_indices, rescued_with_opencv


def extract_embeddings(
    rows: list[tuple[Path, int]],
    processor: AutoProcessor,
    model: SiglipVisionModel,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[Path], list[Path], int]:
    all_embeddings: list[np.ndarray] = []
    kept_labels: list[int] = []
    kept_paths: list[Path] = []
    skipped_paths: list[Path] = []
    rescued_total = 0
    total = len(rows)
    for i in range(0, total, batch_size):
        batch_rows = rows[i : i + batch_size]
        batch_paths = [r[0] for r in batch_rows]
        batch_labels = [r[1] for r in batch_rows]
        imgs, kept_idx, rescued_batch = load_batch_images(batch_paths)
        rescued_total += rescued_batch
        skipped_paths.extend(
            [batch_paths[j] for j in range(len(batch_paths)) if j not in set(kept_idx)]
        )
        if not imgs:
            print(f"Embedded {min(i + batch_size, total)}/{total} images")
            continue
        kept_labels.extend([batch_labels[j] for j in kept_idx])
        kept_paths.extend([batch_paths[j] for j in kept_idx])
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.pooler_output
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        all_embeddings.append(embeddings.float().cpu().numpy())
        print(f"Embedded {min(i + batch_size, total)}/{total} images")
    if not all_embeddings:
        raise RuntimeError("No valid images were embedded.")
    x = np.concatenate(all_embeddings, axis=0)
    y = np.asarray(kept_labels, dtype=np.int64)
    return x, y, kept_paths, skipped_paths, rescued_total


def save_split_path_report(
    split_name: str,
    rows: list[tuple[Path, int]],
    used_paths: list[Path],
    skipped_paths: list[Path],
    output_dir: Path,
) -> dict[str, int]:
    used_set = set(used_paths)
    skipped_set = set(skipped_paths)

    report_csv = output_dir / f"{split_name}_image_status.csv"
    with report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "status"])
        for image_path, label in rows:
            if image_path in used_set:
                status = "used"
            elif image_path in skipped_set:
                status = "skipped"
            else:
                status = "unknown"
            writer.writerow([str(image_path), int(label), status])

    counts = {
        "total": len(rows),
        "used": len(used_paths),
        "skipped": len(skipped_paths),
    }
    print(
        f"{split_name.title()} image report: total={counts['total']}, used={counts['used']}, skipped={counts['skipped']} | {report_csv}"
    )
    return counts


def split_counts(
    rows: list[tuple[Path, int]],
    used_paths: list[Path],
    skipped_paths: list[Path],
) -> dict[str, int]:
    return {
        "total": len(rows),
        "used": len(used_paths),
        "skipped": len(skipped_paths),
    }


def binary_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return [[tn, fp], [fn, tp]]


def binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
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


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    y_pred = (y_prob >= 0.5).astype(np.int64)
    cm = binary_confusion_matrix(y_true, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(binary_roc_auc(y_true, y_prob)),
        "confusion_matrix": cm,
    }
    return metrics


def _init_wandb(args: argparse.Namespace, run_id: str) -> Any | None:
    if not args.wandb:
        return None
    try:
        import wandb  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "wandb is enabled but not installed. Install it with `uv sync`."
        ) from exc

    config = {
        "train_csv": str(args.train_csv),
        "test_csv": str(args.test_csv),
        "model_dir": str(args.model_dir),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "max_train_rows": args.max_train_rows,
        "max_test_rows": args.max_test_rows,
        "run_id": run_id,
    }
    return wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=run_id,
        config=config,
        tags=args.wandb_tags,
    )


def _confusion_matrix_image(cm: list[list[int]], title: str) -> Image.Image:
    arr = np.asarray(cm, dtype=np.float32)
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v - min_v < 1e-6:
        norm = np.zeros_like(arr, dtype=np.float32)
    else:
        norm = (arr - min_v) / (max_v - min_v)

    img = Image.new("RGB", (760, 640), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((24, 20), title, fill=(0, 0, 0))
    grid_left, grid_top = 150, 110
    cell = 170
    labels = ["Non-Anemia", "Anemia"]
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
    return img


def _roc_curve_image(y_true: np.ndarray, y_score: np.ndarray, title: str) -> Image.Image:
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        fpr = np.asarray([0.0, 1.0], dtype=np.float32)
        tpr = np.asarray([0.0, 1.0], dtype=np.float32)
    else:
        order = np.argsort(-y_score)
        y_sorted = y_true[order]
        tp = np.cumsum(y_sorted == 1)
        fp = np.cumsum(y_sorted == 0)
        tpr = tp / n_pos
        fpr = fp / n_neg
        tpr = np.concatenate(([0.0], tpr, [1.0]))
        fpr = np.concatenate(([0.0], fpr, [1.0]))

    img = Image.new("RGB", (760, 640), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((24, 20), title, fill=(0, 0, 0))
    left, top, width, height = 90, 100, 600, 470
    x0 = left
    y0 = top + height
    draw.line([(x0, y0), (x0 + width, y0)], fill=(30, 30, 30), width=2)
    draw.line([(x0, y0), (x0, y0 - height)], fill=(30, 30, 30), width=2)
    draw.line([(x0, y0), (x0 + width, y0 - height)], fill=(150, 150, 150), width=1)

    points: list[tuple[int, int]] = []
    for x, y in zip(fpr, tpr):
        px = x0 + int(float(x) * width)
        py = y0 - int(float(y) * height)
        points.append((px, py))
    if len(points) >= 2:
        draw.line(points, fill=(20, 120, 30), width=3)

    draw.text((x0 + width // 2 - 40, y0 + 34), "False Positive Rate", fill=(0, 0, 0))
    draw.text((20, y0 - height // 2), "True Positive Rate", fill=(0, 0, 0))
    return img


def fit_torch_logistic_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float,
    log_fn: Any | None = None,
) -> tuple[dict[str, np.ndarray], torch.nn.Module]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_train_std = (x_train - mean) / std

    x = torch.from_numpy(x_train_std).float().to("cpu")
    y = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1).to("cpu")

    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    pos_weight_val = (n_neg / n_pos) if n_pos > 0 else 1.0

    model = torch.nn.Linear(x.shape[1], 1).to("cpu")
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if log_fn is not None:
            log_fn({"train/loss": float(loss.item()), "epoch": epoch})
        if epoch == 1 or epoch % 100 == 0 or epoch == epochs:
            print(f"Linear probe epoch {epoch}/{epochs} loss={loss.item():.6f}")

    scaler = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    return scaler, model


def predict_prob_torch(
    x_input: np.ndarray, scaler: dict[str, np.ndarray], model: torch.nn.Module
) -> np.ndarray:
    model.eval()
    x_std = (x_input - scaler["mean"]) / scaler["std"]
    x = torch.from_numpy(x_std).float().to("cpu")
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()
    return probs.astype(np.float32)


def summarize_probe_parameters(weights: np.ndarray, bias: float, top_k: int = 10) -> dict[str, Any]:
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    pos_order = np.argsort(-w)
    neg_order = np.argsort(w)
    top_pos = [
        {"dimension": int(i), "weight": float(w[i])}
        for i in pos_order[:top_k]
    ]
    top_neg = [
        {"dimension": int(i), "weight": float(w[i])}
        for i in neg_order[:top_k]
    ]
    return {
        "num_dimensions": int(w.size),
        "bias": float(bias),
        "weight_l2_norm": float(np.linalg.norm(w)),
        "weight_mean": float(w.mean()) if w.size > 0 else 0.0,
        "weight_std": float(w.std()) if w.size > 0 else 0.0,
        "top_positive_weights": top_pos,
        "top_negative_weights": top_neg,
    }


def append_run_history(
    history_csv: Path,
    run_id: str,
    metrics: dict[str, Any],
    probe_summary: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not history_csv.exists()
    with history_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "timestamp_utc",
                    "run_id",
                    "epochs",
                    "lr",
                    "weight_decay",
                    "batch_size",
                    "train_total",
                    "train_used",
                    "train_skipped",
                    "train_opencv_rescued",
                    "test_total",
                    "test_used",
                    "test_skipped",
                    "test_opencv_rescued",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "probe_bias",
                    "probe_weight_l2_norm",
                ]
            )
        writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                run_id,
                int(args.epochs),
                float(args.lr),
                float(args.weight_decay),
                int(args.batch_size),
                int(metrics["train_images"]["total"]),
                int(metrics["train_images"]["used"]),
                int(metrics["train_images"]["skipped"]),
                int(metrics["train_images"]["opencv_rescued"]),
                int(metrics["test_images"]["total"]),
                int(metrics["test_images"]["used"]),
                int(metrics["test_images"]["skipped"]),
                int(metrics["test_images"]["opencv_rescued"]),
                float(metrics["accuracy"]),
                float(metrics["precision"]),
                float(metrics["recall"]),
                float(metrics["f1"]),
                float(metrics["roc_auc"]),
                float(probe_summary["bias"]),
                float(probe_summary["weight_l2_norm"]),
            ]
        )


def main() -> None:
    args = parse_args()
    train_csv = resolve_path(args.train_csv)
    test_csv = resolve_path(args.test_csv)
    model_dir = resolve_path(args.model_dir)
    output_dir = resolve_path(args.output_dir)
    if args.run_name:
        run_id = str(args.run_name).strip()
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / "runs" / run_id
    if not args.no_local_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _init_wandb(args, run_id)

    train_rows = read_csv_rows(train_csv, limit=args.max_train_rows)
    test_rows = read_csv_rows(test_csv, limit=args.max_test_rows)

    print(f"Train rows: {len(train_rows)} | Test rows: {len(test_rows)}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    print(f"Using device={device}, dtype={dtype}, model_dir={model_dir}")

    processor = AutoProcessor.from_pretrained(model_dir)
    vision_model = SiglipVisionModel.from_pretrained(model_dir, dtype=dtype).to(device)
    vision_model.eval()

    print("Extracting train embeddings...")
    x_train, y_train, train_paths_kept, train_paths_skipped, train_rescued = extract_embeddings(
        train_rows, processor, vision_model, device, args.batch_size
    )
    print("Extracting test embeddings...")
    x_test, y_test, test_paths_kept, test_paths_skipped, test_rescued = extract_embeddings(
        test_rows, processor, vision_model, device, args.batch_size
    )
    if args.no_local_save:
        train_counts = split_counts(train_rows, train_paths_kept, train_paths_skipped)
        test_counts = split_counts(test_rows, test_paths_kept, test_paths_skipped)
    else:
        train_counts = save_split_path_report(
            "train", train_rows, train_paths_kept, train_paths_skipped, run_dir
        )
        test_counts = save_split_path_report(
            "test", test_rows, test_paths_kept, test_paths_skipped, run_dir
        )
    print(f"Embedding shape train={x_train.shape}, test={x_test.shape}")
    print(
        f"Train class counts: non_anemia={(y_train == 0).sum()}, anemia={(y_train == 1).sum()} | skipped={train_counts['skipped']} | opencv_rescued={train_rescued}"
    )
    print(
        f"Test class counts: non_anemia={(y_test == 0).sum()}, anemia={(y_test == 1).sum()} | skipped={test_counts['skipped']} | opencv_rescued={test_rescued}"
    )
    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Train split has only one class after filtering; cannot train probe.")
    if len(np.unique(y_test)) < 2:
        raise RuntimeError("Test split has only one class after filtering; metrics are invalid.")

    scaler, probe = fit_torch_logistic_probe(
        x_train,
        y_train,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_fn=wandb_run.log if wandb_run is not None else None,
    )
    y_prob = predict_prob_torch(x_test, scaler, probe)
    metrics = compute_metrics(y_test, y_prob)
    metrics["train_images"] = train_counts
    metrics["test_images"] = test_counts
    metrics["train_images"]["opencv_rescued"] = int(train_rescued)
    metrics["test_images"]["opencv_rescued"] = int(test_rescued)

    weights = probe.weight.detach().cpu().numpy().reshape(-1)
    bias = float(probe.bias.detach().cpu().item())
    probe_summary = summarize_probe_parameters(weights, bias, top_k=12)
    if not args.no_local_save:
        metrics_path = run_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        probe_summary_path = run_dir / "probe_parameters.json"
        probe_summary_path.write_text(
            json.dumps(probe_summary, indent=2), encoding="utf-8"
        )

        np.savez_compressed(
            run_dir / "artifacts.npz",
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            y_prob=y_prob,
            train_paths=np.asarray([str(p) for p in train_paths_kept]),
            test_paths=np.asarray([str(p) for p in test_paths_kept]),
            scaler_mean=scaler["mean"],
            scaler_std=scaler["std"],
        )
        torch.save(probe.state_dict(), run_dir / "linear_probe.pt")

        cm_plot = run_dir / "confusion_matrix.png"
        roc_plot = run_dir / "roc_curve.png"
        hist_plot = run_dir / "score_histogram.png"
        top_w_plot = run_dir / "top_weight_dimensions.png"
        save_confusion_matrix_png(
            metrics["confusion_matrix"],
            cm_plot,
            title="Linear Probe Confusion Matrix",
        )
        save_roc_curve_png(y_test, y_prob, roc_plot, title="Linear Probe ROC Curve")
        save_score_histogram_png(
            y_test,
            y_prob,
            hist_plot,
            title="Linear Probe Predicted Probability Histogram",
        )
        save_top_weight_dimensions_png(
            weights,
            top_w_plot,
            title="Linear Probe Top Weight Dimensions",
        )
        latest_run_path = output_dir / "latest_run.txt"
        latest_run_path.write_text(f"{run_id}\n", encoding="utf-8")
        append_run_history(
            output_dir / "run_history.csv",
            run_id,
            metrics,
            probe_summary,
            args,
        )
    if wandb_run is not None:
        wandb_run.log(
            {
                "metrics/accuracy": metrics["accuracy"],
                "metrics/precision": metrics["precision"],
                "metrics/recall": metrics["recall"],
                "metrics/f1": metrics["f1"],
                "metrics/roc_auc": metrics["roc_auc"],
                "probe/bias": probe_summary["bias"],
                "probe/weight_l2_norm": probe_summary["weight_l2_norm"],
                "data/train_used": metrics["train_images"]["used"],
                "data/test_used": metrics["test_images"]["used"],
            }
        )
        try:
            import wandb  # type: ignore

            if args.no_local_save:
                cm_img = _confusion_matrix_image(
                    metrics["confusion_matrix"],
                    title="Linear Probe Confusion Matrix",
                )
                roc_img = _roc_curve_image(
                    y_test,
                    y_prob,
                    title="Linear Probe ROC Curve",
                )
                wandb_run.log(
                    {
                        "plots/confusion_matrix": wandb.Image(cm_img),
                        "plots/roc_curve": wandb.Image(roc_img),
                    }
                )
            else:
                wandb_run.log(
                    {
                        "plots/confusion_matrix": wandb.Image(str(cm_plot)),
                        "plots/roc_curve": wandb.Image(str(roc_plot)),
                        "plots/score_histogram": wandb.Image(str(hist_plot)),
                        "plots/top_weight_dimensions": wandb.Image(str(top_w_plot)),
                    }
                )
        except Exception:
            pass
        wandb_run.finish()

    print("Evaluation metrics:")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  {k}: {metrics[k]:.6f}")
    print(f"  confusion_matrix: {metrics['confusion_matrix']}")
    print(f"Run id: {run_id}")
    print(f"Run directory: {run_dir}")
    if not args.no_local_save:
        print(f"Saved metrics: {metrics_path}")
        print(f"Saved probe parameter summary: {probe_summary_path}")
        print(f"Saved artifacts: {run_dir / 'artifacts.npz'}")
        print(f"Saved plots: {cm_plot}, {roc_plot}, {hist_plot}, {top_w_plot}")
        print(f"Updated latest run marker: {latest_run_path}")
        print(f"Updated run history: {output_dir / 'run_history.csv'}")


if __name__ == "__main__":
    main()
