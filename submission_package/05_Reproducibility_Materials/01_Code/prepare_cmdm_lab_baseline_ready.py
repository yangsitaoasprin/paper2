from __future__ import annotations

import csv
from pathlib import Path

from bundle_paths import DATA_DIR


CANONICAL_PATH = DATA_DIR / "cmdm_lab_cyp450_root_canonical.csv"
SEQUENCE_SOURCE_PATH = DATA_DIR / "cyp450_real.csv"
OUTPUT_PATH = DATA_DIR / "cmdm_lab_cyp450_baseline_ready.csv"


def load_enzyme_sequence_map(file_path: Path) -> dict[str, str]:
    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        sequence_map: dict[str, str] = {}
        for row in reader:
            enzyme_name = row["enzyme_name"].strip()
            enzyme_seq = row["enzyme_seq"].strip()
            if enzyme_name in sequence_map and sequence_map[enzyme_name] != enzyme_seq:
                raise ValueError(f"Multiple sequences found for {enzyme_name}")
            sequence_map[enzyme_name] = enzyme_seq
    return sequence_map


def build_rows(sequence_map: dict[str, str]) -> list[dict[str, str]]:
    with CANONICAL_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            isoform_name = row["isoform_name"].strip()
            if isoform_name not in sequence_map:
                raise ValueError(f"No enzyme sequence found for {isoform_name}")

            rows.append(
                {
                    "enzyme_name": isoform_name,
                    "enzyme_seq": sequence_map[isoform_name],
                    "drug_name": row["chemical_name_raw"].strip(),
                    "drug_smiles": row["smiles_raw"].strip(),
                    "label": row["label"].strip(),
                    "split": row["split_source"].strip(),
                    "example_id": row["example_id"].strip(),
                    "source_dataset": row["source_dataset"].strip(),
                    "data_source_raw": row["data_source_raw"].strip(),
                    "row_origin_file": row["row_origin_file"].strip(),
                }
            )
    return rows


def write_rows(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "enzyme_name",
        "enzyme_seq",
        "drug_name",
        "drug_smiles",
        "label",
        "split",
        "example_id",
        "source_dataset",
        "data_source_raw",
        "row_origin_file",
    ]
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    sequence_map = load_enzyme_sequence_map(SEQUENCE_SOURCE_PATH)
    rows = build_rows(sequence_map)
    write_rows(rows)

    total_rows = len(rows)
    total_positive = sum(1 for row in rows if row["label"] == "1")
    total_negative = sum(1 for row in rows if row["label"] == "0")
    total_train = sum(1 for row in rows if row["split"] == "train")
    total_test = sum(1 for row in rows if row["split"] == "test")

    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Total rows: {total_rows}")
    print(f"Label=1: {total_positive}")
    print(f"Label=0: {total_negative}")
    print(f"Train rows: {total_train}")
    print(f"Test rows: {total_test}")


if __name__ == "__main__":
    main()

