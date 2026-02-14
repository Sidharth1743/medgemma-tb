from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Anemia thresholds (WHO criteria)
ANEMIA_THRESHOLD = {"F": 12.0, "M": 13.0}


def classify_anemia(hgb: float, gender: str) -> int:
    """Classify anemia based on Hgb level and gender."""
    if gender == "F":
        return 1 if hgb < ANEMIA_THRESHOLD["F"] else 0
    else:  # Male
        return 1 if hgb < ANEMIA_THRESHOLD["M"] else 0


def load_excel_data_with_fix(excel_path: Path) -> dict[str, dict[str, Any]]:
    """Load patient data from Excel file, fixing comma decimals."""
    df = pd.read_excel(excel_path)

    patients = {}
    for _, row in df.iterrows():
        patient_num = int(row["Number"])
        hgb_raw = row["Hgb"]
        gender = row["Gender"]

        # Fix comma decimal (e.g., "15,1" -> 15.1)
        if isinstance(hgb_raw, str) and "," in str(hgb_raw):
            hgb = float(hgb_raw.replace(",", "."))
        else:
            hgb = pd.to_numeric(hgb_raw, errors="coerce")

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


def create_csvs_from_folders(
    dataset_path: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> None:
    """Create train.csv and test.csv from Anemia/Non-Anemia folder structure."""

    anemia_dir = dataset_path / "Anemia"
    non_anemia_dir = dataset_path / "Non-Anemia"

    # Load patient labels from Excel files
    print("Loading patient data from Excel files...")
    india_excel_path = dataset_path / "India" / "India.xlsx"
    italy_excel_path = dataset_path / "Italy" / "Italy.xlsx"

    india_patients = load_excel_data_with_fix(india_excel_path)
    italy_patients = load_excel_data_with_fix(italy_excel_path)

    all_patients = {**india_patients, **italy_patients}
    print(f"  Loaded {len(all_patients)} patients from Excel")

    # Collect all patient folders with their labels
    patient_data: dict[str, list[tuple[Path, int]]] = {}

    # Anemia folders
    print("\nScanning Anemia folders...")
    anemia_count = 0
    for patient_folder in anemia_dir.iterdir():
        if not patient_folder.is_dir():
            continue

        folder_name = patient_folder.name

        # Get label from Excel data
        if folder_name in all_patients:
            label = all_patients[folder_name]["anemia"]
            # Verify label matches folder location
            if label != 1:
                print(f"  WARNING: {folder_name} in Anemia/ but label={label} (expected 1)")
        else:
            print(f"  WARNING: {folder_name} not found in Excel, using folder-based label=1")
            label = 1

        images = list(patient_folder.glob("*.png")) + list(patient_folder.glob("*.jpg"))
        if images:
            patient_data[folder_name] = [(img, label) for img in sorted(images)]
            anemia_count += 1

    print(f"  Found {anemia_count} patients")

    # Non-Anemia folders
    print("\nScanning Non-Anemia folders...")
    non_anemia_count = 0
    for patient_folder in non_anemia_dir.iterdir():
        if not patient_folder.is_dir():
            continue

        folder_name = patient_folder.name

        # Get label from Excel data
        if folder_name in all_patients:
            label = all_patients[folder_name]["anemia"]
            # Verify label matches folder location
            if label != 0:
                print(f"  WARNING: {folder_name} in Non-Anemia/ but label={label} (expected 0)")
        else:
            print(f"  WARNING: {folder_name} not found in Excel, using folder-based label=0")
            label = 0

        images = list(patient_folder.glob("*.png")) + list(patient_folder.glob("*.jpg"))
        if images:
            patient_data[folder_name] = [(img, label) for img in sorted(images)]
            non_anemia_count += 1

    print(f"  Found {non_anemia_count} patients")

    # Count total images
    total_images = sum(len(imgs) for imgs in patient_data.values())
    total_patients = len(patient_data)
    print(f"\nTotal patients: {total_patients}")
    print(f"Total images: {total_images}")

    # Count labels
    anemia_imgs = sum(1 for imgs in patient_data.values() for _, label in imgs if label == 1)
    non_anemia_imgs = total_images - anemia_imgs
    print(f"Anemia images: {anemia_imgs}")
    print(f"Non-Anemia images: {non_anemia_imgs}")

    # Subject-safe split
    patient_ids = list(patient_data.keys())
    random.seed(seed)
    random.shuffle(patient_ids)

    split_idx = int(len(patient_ids) * train_ratio)
    train_patient_ids = patient_ids[:split_idx]
    test_patient_ids = patient_ids[split_idx:]

    # Collect train and test images
    train_rows: list[tuple[str, int]] = []
    test_rows: list[tuple[str, int]] = []

    for pid in train_patient_ids:
        for img_path, label in patient_data[pid]:
            train_rows.append((str(img_path), label))

    for pid in test_patient_ids:
        for img_path, label in patient_data[pid]:
            test_rows.append((str(img_path), label))

    # Save train.csv
    output_dir.mkdir(parents=True, exist_ok=True)
    train_csv = output_dir / "train.csv"
    with train_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(train_rows)

    # Save test.csv
    test_csv = output_dir / "test.csv"
    with test_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(test_rows)

    print(f"\n=== Split Summary ===")
    print(f"Train patients: {len(train_patient_ids)}")
    print(f"Test patients: {len(test_patient_ids)}")
    print(f"Train images: {len(train_rows)}")
    print(f"Test images: {len(test_rows)}")

    train_anemia = sum(1 for _, label in train_rows if label == 1)
    test_anemia = sum(1 for _, label in test_rows if label == 1)
    print(f"\nTrain: Anemia={train_anemia}, Non-Anemia={len(train_rows) - train_anemia}")
    print(f"Test: Anemia={test_anemia}, Non-Anemia={len(test_rows) - test_anemia}")

    print(f"\nSaved:")
    print(f"  {train_csv}")
    print(f"  {test_csv}")


def main() -> None:
    dataset_path = PROJECT_ROOT / "Dataset" / "dataset anemia"
    output_dir = dataset_path

    create_csvs_from_folders(dataset_path, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
