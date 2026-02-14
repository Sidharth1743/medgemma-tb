from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Iterable, List, Tuple

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SEGMENT_KEYS = ("forniceal", "forniceal_palpebral", "palpebral")


def _is_segmented(path: Path) -> bool:
    name = path.name.lower()
    return any(key in name for key in SEGMENT_KEYS)


def _iter_images(folder: Path) -> Iterable[Path]:
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def _collect(folder: Path, label: int, segmented_only: bool) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    for path in _iter_images(folder):
        if not segmented_only or _is_segmented(path):
            items.append((str(path.resolve()), label))
    return items


def _split_fixed(
    items: List[Tuple[str, int]],
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    total_needed = train_count + val_count + test_count
    if len(items) < total_needed:
        raise ValueError(f"Not enough items: need {total_needed}, have {len(items)}")
    rng = random.Random(seed)
    rng.shuffle(items)
    train = items[:train_count]
    val = items[train_count : train_count + val_count]
    test = items[train_count + val_count : train_count + val_count + test_count]
    return train, val, test


def _write_csv(path: Path, rows: List[Tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(rows)


def _count(rows: List[Tuple[str, int]]) -> Tuple[int, int]:
    anemia = sum(1 for _, label in rows if label == 1)
    non_anemia = sum(1 for _, label in rows if label == 0)
    return anemia, non_anemia


def main() -> None:
    parser = argparse.ArgumentParser(description="Create segmented-only train/val/test CSVs.")
    parser.add_argument(
        "--anemia-dir",
        default="Dataset/dataset anemia/Anemia",
        help="Folder with anemia images (segmented-only by filename)",
    )
    parser.add_argument(
        "--non-anemia-dir",
        default="Dataset/dataset anemia/Non-Anemia",
        help="Folder with non-anemia images (segmented-only by filename)",
    )
    parser.add_argument(
        "--cp-anemia-dir",
        default="Dataset/CP-AnemiC dataset/Anemic",
        help="Folder with CP-AnemiC anemia images (all considered segmented)",
    )
    parser.add_argument(
        "--cp-non-anemia-dir",
        default="Dataset/CP-AnemiC dataset/Non-anemic",
        help="Folder with CP-AnemiC non-anemia images (all considered segmented)",
    )
    parser.add_argument(
        "--out-dir",
        default="src/segmented",
        help="Output directory for train/val/test CSVs",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=400)
    parser.add_argument("--val-count", type=int, default=50)
    parser.add_argument("--test-count", type=int, default=50)
    args = parser.parse_args()

    anemia_items = _collect(Path(args.anemia_dir), 1, segmented_only=True)
    anemia_items += _collect(Path(args.cp_anemia_dir), 1, segmented_only=False)
    non_anemia_items = _collect(Path(args.non_anemia_dir), 0, segmented_only=True)
    non_anemia_items += _collect(Path(args.cp_non_anemia_dir), 0, segmented_only=False)

    a_train, a_val, a_test = _split_fixed(
        anemia_items,
        args.train_count,
        args.val_count,
        args.test_count,
        args.seed,
    )
    n_train, n_val, n_test = _split_fixed(
        non_anemia_items,
        args.train_count,
        args.val_count,
        args.test_count,
        args.seed,
    )

    train = a_train + n_train
    val = a_val + n_val
    test = a_test + n_test

    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "train.csv", train)
    _write_csv(out_dir / "val.csv", val)
    _write_csv(out_dir / "test.csv", test)

    train_counts = _count(train)
    val_counts = _count(val)
    test_counts = _count(test)

    print(f"Train: anemia={train_counts[0]} non_anemia={train_counts[1]} total={len(train)}")
    print(f"Val: anemia={val_counts[0]} non_anemia={val_counts[1]} total={len(val)}")
    print(f"Test: anemia={test_counts[0]} non_anemia={test_counts[1]} total={len(test)}")


if __name__ == "__main__":
    main()
