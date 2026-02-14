from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Anemia thresholds (WHO criteria)
ANEMIA_THRESHOLD = {"F": 12.0, "M": 13.0}


def classify_anemia(hgb: float, gender: str) -> int:
    """Classify anemia based on Hgb level and gender."""
    if gender == "F":
        return 1 if hgb < ANEMIA_THRESHOLD["F"] else 0
    else:  # Male
        return 1 if hgb < ANEMIA_THRESHOLD["M"] else 0


def load_excel_data(excel_path: Path) -> dict[str, dict[str, Any]]:
    """Load patient data from Excel file and return dict by patient number."""
    df = pd.read_excel(excel_path)
    df["Hgb"] = pd.to_numeric(df["Hgb"], errors="coerce")

    patients = {}
    for _, row in df.iterrows():
        patient_num = int(row["Number"])
        hgb = row["Hgb"]
        gender = row["Gender"]

        if pd.isna(hgb):
            print(f"Warning: Patient {patient_num} has missing Hgb, skipping")
            continue

        # Use prefix to distinguish India vs Italy patients
        if "India" in str(excel_path):
            key = f"india_{patient_num}"
        else:
            key = f"italy_{patient_num}"

        patients[key] = {
            "hgb": hgb,
            "gender": gender,
            "anemia": classify_anemia(hgb, gender),
        }

    return patients


def find_images_in_dataset(data_dir: Path) -> dict[str, list[Path]]:
    """Scan Anemia/ and Non-Anemia folders and collect all images by patient."""
    anemia_dir = data_dir / "Anemia"
    non_anemia_dir = data_dir / "Non-Anemia"

    patient_data: dict[str, list[Path]] = {}

    # Scan Anemia folders (both India and Italy patients)
    print("Scanning Anemia folders...")
    anemia_count = 0
    for patient_folder in anemia_dir.iterdir():
        if not patient_folder.is_dir():
            continue

        images = list(patient_folder.glob("*.png")) + list(patient_folder.glob("*.jpg"))
        if images:
            patient_data[patient_folder.name] = [(img, 1) for img in sorted(images)]
            anemia_count += 1

    print(f"  Found {anemia_count} patients")

    # Scan Non-Anemia folders
    print("\nScanning Non-Anemia folders...")
    non_anemia_count = 0
    for patient_folder in non_anemia_dir.iterdir():
        if not patient_folder.is_dir():
            continue

        images = list(patient_folder.glob("*.png")) + list(patient_folder.glob("*.jpg"))
        if images:
            patient_data[patient_folder.name] = [(img, 0) for img in sorted(images)]
            non_anemia_count += 1

    print(f"  Found {non_anemia_count} patients")

    return patient_data


def create_dataset_splits(
    data_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Create train.csv, val.csv, and test.csv with 80/10/10 split."""

    # Collect all patient images with labels
    patient_data = find_images_in_dataset(data_dir)

    if not patient_data:
        raise ValueError(f"No images found in {data_dir}")
        return

    # Get patient IDs
    patient_ids = list(patient_data.keys())
    random.seed(seed)
    random.shuffle(patient_ids)

    # Calculate split sizes
    total = len(patient_ids)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    # Split patient IDs
    train_ids = set(patient_ids[:n_train])
    val_ids = set(patient_ids[n_train : n_train + n_val])
    test_ids = set(patient_ids[n_train + n_val :])

    # Collect images for each split
    train_rows: list[tuple[str, int]] = []
    val_rows: list[tuple[str, int]] = []
    test_rows: list[tuple[str, int]] = []

    for pid in patient_ids:
        if pid in train_ids:
            target_rows = train_rows
            split = "train"
        elif pid in val_ids:
            target_rows = val_rows
            split = "val"
        elif pid in test_ids:
            target_rows = test_rows
            split = "test"
        else:
            continue

        for img_path, label in patient_data[pid]:
            target_rows.append((str(img_path), label))

    # Count rows
    print(f"\nTotal patients: {total}")
    print(f"Train: {len(train_ids)} patients ({len(train_rows)} images)")
    print(f"Val:   {len(val_ids)} patients ({len(val_rows)} images)")
    print(f"Test:  {len(test_ids)} patients ({len(test_rows)} images)")

    # Save CSVs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save train.csv
    train_csv = output_dir / "train.csv"
    with train_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(train_rows)

    # Save val.csv
    val_csv = output_dir / "val.csv"
    with val_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(val_rows)

    # Save test.csv
    test_csv = output_dir / "test.csv"
    with test_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(test_rows)

    print(f"\nSaved:")
    print(f"  {train_csv}")
    print(f"  {val_csv}")
    print(f"  {test_csv}")


def main() -> None:
    data_dir = PROJECT_ROOT / "Dataset" / "dataset anemia"
    output_dir = PROJECT_ROOT / "Dataset" / "dataset anemia"

    create_dataset_splits(data_dir, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
