from __future__ import annotations

import csv
from pathlib import Path

from bundle_paths import DATA_DIR


SOURCE_DIR = DATA_DIR / "external_cmdm_root_csv"
OUTPUT_PATH = DATA_DIR / "cmdm_lab_cyp450_root_canonical.csv"

EXPECTED_COLUMNS = ["Name", "SMILES", "Label", "Source"]
SOURCE_DATASET = "CMDM-Lab/CYP450"


def parse_file_name(file_path: Path) -> tuple[str, str]:
    stem = file_path.stem
    parts = stem.split("_")
    if len(parts) != 2:
        raise ValueError(f"Unexpected file name format: {file_path.name}")

    isoform_name, split_token = parts
    split_map = {
        "trainingset": "train",
        "testingset": "test",
    }
    if split_token not in split_map:
        raise ValueError(f"Unexpected split token in file name: {file_path.name}")

    return isoform_name, split_map[split_token]


def load_rows(file_path: Path) -> list[dict[str, str]]:
    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                f"{file_path.name} has unexpected columns: {reader.fieldnames}"
            )
        return list(reader)


def build_canonical_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    files = sorted(
        file_path
        for file_path in SOURCE_DIR.glob("*.csv")
        if file_path.stem.count("_") == 1
        and file_path.stem.split("_")[1] in {"trainingset", "testingset"}
        and file_path.stem.split("_")[0].upper().startswith("CYP")
    )
    canonical_rows: list[dict[str, str]] = []
    file_summaries: list[dict[str, str]] = []

    for file_path in files:
        isoform_name, split_source = parse_file_name(file_path)
        rows = load_rows(file_path)

        label_zero = 0
        label_one = 0

        for idx, row in enumerate(rows, start=1):
            label_raw = row["Label"].strip()
            if label_raw not in {"0", "1"}:
                raise ValueError(
                    f"{file_path.name} contains unexpected Label value: {label_raw}"
                )

            if label_raw == "0":
                label_zero += 1
            else:
                label_one += 1

            canonical_rows.append(
                {
                    "example_id": f"CMDM_{isoform_name}_{split_source.upper()}_{idx:06d}",
                    "source_dataset": SOURCE_DATASET,
                    "isoform_name": isoform_name,
                    "split_source": split_source,
                    "chemical_name_raw": row["Name"].strip(),
                    "smiles_raw": row["SMILES"].strip(),
                    "label_raw": label_raw,
                    "label": label_raw,
                    "data_source_raw": row["Source"].strip(),
                    "row_origin_file": file_path.name,
                }
            )

        file_summaries.append(
            {
                "file": file_path.name,
                "isoform_name": isoform_name,
                "split_source": split_source,
                "rows": str(len(rows)),
                "label_0": str(label_zero),
                "label_1": str(label_one),
            }
        )

    return canonical_rows, file_summaries


def write_canonical_table(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "example_id",
        "source_dataset",
        "isoform_name",
        "split_source",
        "chemical_name_raw",
        "smiles_raw",
        "label_raw",
        "label",
        "data_source_raw",
        "row_origin_file",
    ]
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows, file_summaries = build_canonical_rows()
    write_canonical_table(rows)

    total_rows = len(rows)
    total_positive = sum(int(item["label_1"]) for item in file_summaries)
    total_negative = sum(int(item["label_0"]) for item in file_summaries)

    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Total rows: {total_rows}")
    print(f"Label=1: {total_positive}")
    print(f"Label=0: {total_negative}")
    print("Per-file summary:")
    for item in file_summaries:
        print(
            f"- {item['file']}: rows={item['rows']}, "
            f"label_0={item['label_0']}, label_1={item['label_1']}"
        )


if __name__ == "__main__":
    main()

