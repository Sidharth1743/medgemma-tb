from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, SiglipVisionModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Legacy PIL/AutoProcessor linear probe for comparison."
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
        default="results/linear_probe_anemia_pil",
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
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def read_csv_rows(csv_path: Path) -> list[tuple[Path, int]]:
    rows: list[tuple[Path, int]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"image_path", "label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{csv_path} must contain columns: image_path,label")
        for row in reader:
            rows.append((resolve_path(row["image_path"]), int(row["label"])))
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
            cv_img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if cv_img is None:
                print(f"Skipping unreadable image: {p}")
                continue
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


def roc_threshold_stats(
    y_true: np.ndarray, y_score: np.ndarray
) -> dict[str, Any]:
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


def compute_metrics_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(np.int64)
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
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
    }


def threshold_sweep(
    y_true: np.ndarray, y_prob: np.ndarray, steps: int = 101
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(steps):
        t = i / (steps - 1)
        rows.append(compute_metrics_at_threshold(y_true, y_prob, t))
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
        "train_csv": str(args.train_csv),
        "test_csv": str(args.test_csv),
        "model_dir": str(args.model_dir),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "pipeline": "legacy_auto_processor_pil",
    }
    return wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        config=config,
        tags=args.wandb_tags,
    )


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


def _make_pil_viz_images(img: Image.Image) -> dict[str, Image.Image]:
    rgb = img.convert("RGB")
    resized = rgb.resize((448, 448), resample=Image.BILINEAR)
    np_img = np.asarray(resized, dtype=np.float32) / 255.0
    norm = np_img * 2.0 - 1.0
    norm_vis = ((norm + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    normalized = Image.fromarray(norm_vis)
    return {
        "original": img,
        "rgb": rgb,
        "resized_448": resized,
        "normalized": normalized,
    }


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


def main() -> None:
    args = parse_args()
    if args.wandb and not args.save_local:
        args.no_local_save = True
    train_csv = resolve_path(args.train_csv)
    test_csv = resolve_path(args.test_csv)
    model_dir = resolve_path(args.model_dir)
    output_dir = resolve_path(args.output_dir)
    if not args.no_local_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_csv_rows(train_csv)
    test_rows = read_csv_rows(test_csv)

    print(f"Train rows: {len(train_rows)} | Test rows: {len(test_rows)}")
    wandb_run = _init_wandb(args)

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

    print(f"Embedding shape train={x_train.shape}, test={x_test.shape}")
    print(
        f"Train class counts: non_anemia={(y_train == 0).sum()}, anemia={(y_train == 1).sum()} | skipped={len(train_paths_skipped)} | opencv_rescued={train_rescued}"
    )
    print(
        f"Test class counts: non_anemia={(y_test == 0).sum()}, anemia={(y_test == 1).sum()} | skipped={len(test_paths_skipped)} | opencv_rescued={test_rescued}"
    )

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
    roc_stats = roc_threshold_stats(y_test, y_prob)
    metrics["roc_threshold"] = roc_stats
    best_metrics = compute_metrics_at_threshold(
        y_test, y_prob, roc_stats["best_threshold"]
    )
    metrics["roc_threshold"]["metrics_at_best"] = best_metrics

    if not args.no_local_save:
        metrics_path = output_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    if wandb_run is not None:
        wandb_run.log(
            {
                "metrics/accuracy": metrics["accuracy"],
                "metrics/precision": metrics["precision"],
                "metrics/recall": metrics["recall"],
                "metrics/f1": metrics["f1"],
                "metrics/roc_auc": metrics["roc_auc"],
                "metrics/confusion_matrix": metrics["confusion_matrix"],
                "metrics/roc_best_threshold": roc_stats["best_threshold"],
                "metrics/roc_best_j": roc_stats["best_j"],
                "metrics/roc_best_tpr": roc_stats["best_tpr"],
                "metrics/roc_best_fpr": roc_stats["best_fpr"],
                "metrics/best_threshold_accuracy": best_metrics["accuracy"],
                "metrics/best_threshold_precision": best_metrics["precision"],
                "metrics/best_threshold_recall": best_metrics["recall"],
                "metrics/best_threshold_f1": best_metrics["f1"],
                "epoch": int(args.epochs),
            }
        )
        try:
            import wandb  # type: ignore

            cm = metrics["confusion_matrix"]
            cm_img = Image.new("RGB", (760, 640), color=(255, 255, 255))
            draw = ImageDraw.Draw(cm_img)
            draw.text((24, 20), "Legacy PIL Confusion Matrix", fill=(0, 0, 0))
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
            cm_table = wandb.Table(columns=["pred_0", "pred_1"], data=cm)
            wandb_run.log({"tables/confusion_matrix": cm_table})
            sweep_rows = threshold_sweep(y_test, y_prob, steps=101)
            sweep_table = wandb.Table(
                columns=["threshold", "accuracy", "precision", "recall", "f1"],
                data=[
                    [
                        r["threshold"],
                        r["accuracy"],
                        r["precision"],
                        r["recall"],
                        r["f1"],
                    ]
                    for r in sweep_rows
                ],
            )
            wandb_run.log({"tables/threshold_sweep": sweep_table})
        except Exception:
            pass
        try:
            import wandb  # type: ignore

            train_samples = train_paths_kept[:2]
            test_samples = test_paths_kept[:2]
            for path, split in (
                [(p, "train") for p in train_samples]
                + [(p, "test") for p in test_samples]
            ):
                img = _load_image_for_viz(path)
                if img is None:
                    continue
                viz = _make_pil_viz_images(img)
                wandb_run.log(
                    {
                        f"images/{split}_{path.name}_original": wandb.Image(viz["original"]),
                        f"images/{split}_{path.name}_rgb": wandb.Image(viz["rgb"]),
                        f"images/{split}_{path.name}_resized": wandb.Image(viz["resized_448"]),
                        f"images/{split}_{path.name}_normalized": wandb.Image(viz["normalized"]),
                    }
                )
        except Exception:
            pass
        wandb_run.finish()

    print("Evaluation metrics:")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  {k}: {metrics[k]:.6f}")
    print(f"  confusion_matrix: {metrics['confusion_matrix']}")
    if not args.no_local_save:
        print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
