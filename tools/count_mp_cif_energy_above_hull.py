from __future__ import annotations

import csv
import math
from pathlib import Path


SOURCE_CSV = Path.home() / "run" / "mp-cif" / "mp_all_summary.csv"
CIF_ROOT = Path.home() / "run" / "mp-cif"


def parse_finite_float(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def has_cif(material_id: str) -> bool:
    return (CIF_ROOT / f"{material_id}.cif").is_file() or (
        CIF_ROOT / material_id / f"{material_id}.cif"
    ).is_file()


if __name__ == "__main__":
    total_rows = 0
    valid_numeric = 0
    valid_numeric_with_cif = 0
    invalid_target = 0
    missing_cif = 0

    with SOURCE_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "material_id" not in (reader.fieldnames or []):
            raise ValueError("Missing required column: material_id")
        if "energy_above_hull" not in (reader.fieldnames or []):
            raise ValueError("Missing required column: energy_above_hull")
        for row in reader:
            total_rows += 1
            material_id = (row.get("material_id") or "").strip()
            value = parse_finite_float(row.get("energy_above_hull") or "")
            if not material_id or value is None:
                invalid_target += 1
                continue
            valid_numeric += 1
            if has_cif(material_id):
                valid_numeric_with_cif += 1
            else:
                missing_cif += 1

    print(f"source_csv={SOURCE_CSV}")
    print(f"cif_root={CIF_ROOT}")
    print(f"total_data_rows={total_rows}")
    print(f"valid_numeric_energy_above_hull={valid_numeric}")
    print(f"valid_numeric_energy_above_hull_with_cif={valid_numeric_with_cif}")
    print(f"invalid_or_missing_energy_above_hull={invalid_target}")
    print(f"valid_numeric_rows_missing_cif={missing_cif}")
