from __future__ import annotations

import argparse
import csv
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check images and log exact Pillow errors and failure stage."
    )
    parser.add_argument(
        "--input-csv",
        default="Dataset/dataset anemia/train.csv",
        help="CSV with column image_path. If omitted, --image-dir is required.",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Directory to scan recursively for images if CSV is not used.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=120,
        help="Maximum number of images to check.",
    )
    parser.add_argument(
        "--only-png",
        action="store_true",
        help="Restrict checks to PNG files only.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/image_checking",
        help="Directory for logs.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def read_paths_from_csv(csv_path: Path, max_images: int) -> list[Path]:
    paths: list[Path] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "image_path" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} must contain 'image_path' column.")
        for row in reader:
            paths.append(resolve_path(row["image_path"]))
            if len(paths) >= max_images:
                break
    return paths


def read_paths_from_dir(image_dir: Path, max_images: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    paths = [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    paths.sort()
    return paths[:max_images]


def run_step(step_name: str, fn: Any) -> tuple[bool, str | None, str | None, str | None]:
    try:
        fn()
        return True, None, None, None
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        return False, step_name, type(exc).__name__, tb


def check_one_image(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "image_path": str(path),
        "exists": path.exists(),
        "suffix": path.suffix.lower(),
        "file_size_bytes": path.stat().st_size if path.exists() else -1,
        "open_ok": False,
        "verify_ok": False,
        "convert_ok": False,
        "fallback_convert_ok": False,
        "first_failed_stage": "",
        "error_type": "",
        "error_message": "",
        "full_traceback": "",
    }

    if not path.exists():
        row["first_failed_stage"] = "exists"
        row["error_type"] = "FileNotFoundError"
        row["error_message"] = f"File not found: {path}"
        row["full_traceback"] = "File does not exist."
        return row

    def step_open() -> None:
        with Image.open(path) as img:
            _ = (img.format, img.mode, img.size)

    def step_verify() -> None:
        with Image.open(path) as img:
            img.verify()

    def step_convert() -> None:
        with Image.open(path) as img:
            _ = img.convert("RGB")

    ok, stage, etype, tb = run_step("open", step_open)
    row["open_ok"] = ok
    if not ok:
        row["first_failed_stage"] = stage or ""
        row["error_type"] = etype or ""
        row["error_message"] = tb.splitlines()[-1] if tb else ""
        row["full_traceback"] = tb or ""
        return row

    ok, stage, etype, tb = run_step("verify", step_verify)
    row["verify_ok"] = ok
    if not ok:
        row["first_failed_stage"] = stage or ""
        row["error_type"] = etype or ""
        row["error_message"] = tb.splitlines()[-1] if tb else ""
        row["full_traceback"] = tb or ""
        return row

    ok, stage, etype, tb = run_step("convert", step_convert)
    row["convert_ok"] = ok
    if ok:
        return row

    row["first_failed_stage"] = stage or ""
    row["error_type"] = etype or ""
    row["error_message"] = tb.splitlines()[-1] if tb else ""
    row["full_traceback"] = tb or ""

    # Retry with Pillow truncated-image fallback to identify salvageable files.
    old_flag = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        def step_fallback_convert() -> None:
            with Image.open(path) as img:
                _ = img.convert("RGB")

        ok2, _, _, _ = run_step("fallback_convert", step_fallback_convert)
        row["fallback_convert_ok"] = ok2
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = old_flag

    return row


def write_logs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "image_check_report.csv"
    txt_path = output_dir / "image_check_errors.txt"

    fieldnames = [
        "timestamp_utc",
        "image_path",
        "exists",
        "suffix",
        "file_size_bytes",
        "open_ok",
        "verify_ok",
        "convert_ok",
        "fallback_convert_ok",
        "first_failed_stage",
        "error_type",
        "error_message",
        "full_traceback",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with txt_path.open("w", encoding="utf-8") as f:
        for row in rows:
            if row["first_failed_stage"]:
                f.write("=" * 80 + "\n")
                f.write(f"path: {row['image_path']}\n")
                f.write(f"stage: {row['first_failed_stage']}\n")
                f.write(f"error_type: {row['error_type']}\n")
                f.write(f"error_message: {row['error_message']}\n")
                f.write("traceback:\n")
                f.write(f"{row['full_traceback']}\n")

    total = len(rows)
    failed = sum(1 for r in rows if r["first_failed_stage"])
    rescued = sum(1 for r in rows if (r["first_failed_stage"] and r["fallback_convert_ok"]))
    print(f"Checked images: {total}")
    print(f"Failures: {failed}")
    print(f"Failures rescued by fallback: {rescued}")
    print(f"CSV log: {csv_path}")
    print(f"Text log: {txt_path}")


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)

    if args.image_dir:
        image_dir = resolve_path(args.image_dir)
        paths = read_paths_from_dir(image_dir, max_images=args.max_images)
    else:
        csv_path = resolve_path(args.input_csv)
        paths = read_paths_from_csv(csv_path, max_images=args.max_images)

    if args.only_png:
        paths = [p for p in paths if p.suffix.lower() == ".png"]

    if not paths:
        raise ValueError("No images found to check. Adjust --input-csv/--image-dir.")

    rows = [check_one_image(p) for p in paths]
    write_logs(rows, output_dir)


if __name__ == "__main__":
    main()
