from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x0 = left
    y0 = top + height
    x1 = left + width
    y1 = top
    draw.line([(x0, y0), (x1, y0)], fill=(30, 30, 30), width=2)
    draw.line([(x0, y0), (x0, y1)], fill=(30, 30, 30), width=2)
    return x0, y0, x1, y1


def _safe_intensity(values: np.ndarray) -> np.ndarray:
    min_v = float(values.min())
    max_v = float(values.max())
    if max_v - min_v < 1e-9:
        return np.full_like(values, 0.5, dtype=np.float32)
    return ((values - min_v) / (max_v - min_v)).astype(np.float32)


def save_confusion_matrix_png(
    cm: list[list[int]],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    arr = np.asarray(cm, dtype=np.float32)
    normalized = _safe_intensity(arr)

    img = Image.new("RGB", (760, 640), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((24, 20), title, fill=(0, 0, 0))

    grid_left, grid_top = 150, 110
    cell = 170
    labels = ["Non-Anemia", "Anemia"]

    for r in range(2):
        for c in range(2):
            val = int(arr[r, c])
            shade = int(255 - 170 * float(normalized[r, c]))
            color = (shade, shade, 255)
            x0 = grid_left + c * cell
            y0 = grid_top + r * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle([(x0, y0), (x1, y1)], fill=color, outline=(80, 80, 80), width=2)
            draw.text((x0 + 70, y0 + 78), str(val), fill=(0, 0, 0))

    draw.text((grid_left + 120, grid_top + 365), "Predicted", fill=(0, 0, 0))
    draw.text((35, grid_top + 140), "True", fill=(0, 0, 0))
    for i, lbl in enumerate(labels):
        draw.text((grid_left + i * cell + 42, grid_top - 28), lbl, fill=(0, 0, 0))
        draw.text((grid_left - 110, grid_top + i * cell + 78), lbl, fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def save_score_histogram_png(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Path,
    title: str = "Predicted Anemia Probability Histogram",
) -> None:
    bins = np.linspace(0.0, 1.0, 11)
    neg = y_score[y_true == 0]
    pos = y_score[y_true == 1]
    h_neg, _ = np.histogram(neg, bins=bins)
    h_pos, _ = np.histogram(pos, bins=bins)
    h_max = int(max(int(h_neg.max(initial=0)), int(h_pos.max(initial=0)), 1))

    img = Image.new("RGB", (940, 640), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((22, 20), title, fill=(0, 0, 0))
    draw.text((22, 48), "Blue: non-anemia (label=0)  |  Red: anemia (label=1)", fill=(0, 0, 0))

    left, top, width, height = 90, 100, 800, 470
    x0, y0, x1, y1 = _draw_axes(draw, left, top, width, height)

    for i in range(11):
        x = x0 + int(width * (i / 10))
        draw.line([(x, y0), (x, y0 + 5)], fill=(30, 30, 30), width=1)
        draw.text((x - 8, y0 + 10), f"{i/10:.1f}", fill=(0, 0, 0))

    bar_w = width // 10
    for i in range(10):
        base_x = x0 + i * bar_w
        h0 = int((h_neg[i] / h_max) * (height - 20))
        h1 = int((h_pos[i] / h_max) * (height - 20))

        draw.rectangle(
            [(base_x + 3, y0 - h0), (base_x + bar_w // 2 - 2, y0)],
            fill=(70, 130, 220),
            outline=(50, 90, 160),
        )
        draw.rectangle(
            [(base_x + bar_w // 2 + 2, y0 - h1), (base_x + bar_w - 3, y0)],
            fill=(220, 80, 80),
            outline=(150, 40, 40),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def _roc_points(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.asarray([0.0, 1.0], dtype=np.float32), np.asarray(
            [0.0, 1.0], dtype=np.float32
        )
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    tpr = tp / n_pos
    fpr = fp / n_neg
    tpr = np.concatenate(([0.0], tpr, [1.0]))
    fpr = np.concatenate(([0.0], fpr, [1.0]))
    return fpr.astype(np.float32), tpr.astype(np.float32)


def save_roc_curve_png(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Path,
    title: str = "ROC Curve",
) -> None:
    fpr, tpr = _roc_points(y_true, y_score)

    img = Image.new("RGB", (760, 640), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((24, 20), title, fill=(0, 0, 0))

    left, top, width, height = 90, 100, 600, 470
    x0, y0, _, _ = _draw_axes(draw, left, top, width, height)

    # Chance diagonal.
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def save_top_weight_dimensions_png(
    weights: np.ndarray,
    output_path: Path,
    title: str = "Top Probe Weight Dimensions",
    top_k: int = 20,
) -> None:
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    top_idx = np.argsort(np.abs(w))[::-1][:top_k]
    top_w = w[top_idx]
    max_abs = float(np.max(np.abs(top_w))) if len(top_w) else 1.0
    max_abs = max(max_abs, 1e-6)

    img = Image.new("RGB", (980, 720), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((24, 20), title, fill=(0, 0, 0))
    draw.text((24, 48), "Green: positive (anemia), Orange: negative (non-anemia)", fill=(0, 0, 0))

    left, top, width, height = 100, 100, 840, 560
    x0, y0, _, _ = _draw_axes(draw, left, top, width, height)
    mid_y = y0 - height // 2
    draw.line([(x0, mid_y), (x0 + width, mid_y)], fill=(100, 100, 100), width=1)

    if len(top_w) == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        return

    bar_w = max(1, width // len(top_w))
    half_h = height // 2 - 16
    for i, (idx, val) in enumerate(zip(top_idx.tolist(), top_w.tolist())):
        x_left = x0 + i * bar_w + 2
        x_right = x0 + (i + 1) * bar_w - 2
        scaled = int((abs(float(val)) / max_abs) * half_h)
        if val >= 0:
            y_top = mid_y - scaled
            y_bottom = mid_y
            fill = (70, 170, 90)
            outline = (40, 120, 60)
        else:
            y_top = mid_y
            y_bottom = mid_y + scaled
            fill = (230, 150, 60)
            outline = (170, 100, 30)
        draw.rectangle([(x_left, y_top), (x_right, y_bottom)], fill=fill, outline=outline)
        if i % 2 == 0:
            draw.text((x_left, y0 + 8), str(idx), fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
