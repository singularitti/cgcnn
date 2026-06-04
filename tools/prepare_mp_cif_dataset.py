from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


DEFAULT_MP_CIF_ROOT = Path("/work2/04996/tg842951/stampede3/run/mp-cif")
DEFAULT_SUMMARY_CSV = DEFAULT_MP_CIF_ROOT / "mp_all_summary.csv"
DEFAULT_ATOM_INIT = Path("data/sample-regression/atom_init.json")
DEFAULT_REPO_DATASET_LINK = Path("data/mp-all-formation-energy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten a Materials Project CIF export into the dataset structure "
            "expected by CGCNN."
        )
    )
    parser.add_argument(
        "--mp-cif-root",
        type=Path,
        default=DEFAULT_MP_CIF_ROOT,
        help="Directory containing mp_all_summary.csv and material_id subdirectories.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="CSV containing material_id and formation_energy_per_atom.",
    )
    parser.add_argument(
        "--atom-init-source",
        type=Path,
        default=DEFAULT_ATOM_INIT,
        help="Existing atom_init.json to symlink into the dataset root.",
    )
    parser.add_argument(
        "--repo-dataset-link",
        type=Path,
        default=DEFAULT_REPO_DATASET_LINK,
        help="Symlink to create inside the repo that points at the prepared dataset.",
    )
    return parser.parse_args()


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() and link_path.resolve() == target_path.resolve():
            return
        raise FileExistsError(f"{link_path} already exists and does not match {target_path}")
    link_path.symlink_to(target_path)


def flatten_cifs(root: Path) -> tuple[int, int, int]:
    moved = 0
    removed_dirs = 0
    skipped_existing = 0

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        cif_path = entry / f"{entry.name}.cif"
        flat_cif_path = root / f"{entry.name}.cif"

        if cif_path.exists():
            if flat_cif_path.exists():
                if flat_cif_path.stat().st_size != cif_path.stat().st_size:
                    raise FileExistsError(
                        f"Conflicting CIF files found for {entry.name}: "
                        f"{flat_cif_path} and {cif_path}"
                    )
                skipped_existing += 1
            else:
                os.replace(cif_path, flat_cif_path)
                moved += 1

        remaining = list(entry.iterdir())
        if remaining:
            raise RuntimeError(
                f"Directory {entry} is not empty after flattening: "
                f"{', '.join(item.name for item in remaining[:5])}"
            )
        entry.rmdir()
        removed_dirs += 1

    return moved, removed_dirs, skipped_existing


def write_id_prop_csv(root: Path, summary_csv: Path) -> tuple[int, int]:
    id_prop_path = root / "id_prop.csv"
    kept = 0
    skipped_missing_cif = 0

    with summary_csv.open(newline="") as source, id_prop_path.open("w", newline="") as dest:
        reader = csv.DictReader(source)
        if "material_id" not in reader.fieldnames or "formation_energy_per_atom" not in reader.fieldnames:
            raise ValueError(
                f"{summary_csv} must contain material_id and formation_energy_per_atom columns."
            )
        writer = csv.writer(dest)
        for row in reader:
            material_id = row["material_id"].strip()
            target = row["formation_energy_per_atom"].strip()
            cif_path = root / f"{material_id}.cif"
            if not cif_path.exists():
                skipped_missing_cif += 1
                continue
            if not target:
                raise ValueError(f"Missing formation_energy_per_atom for {material_id}")
            writer.writerow([material_id, target])
            kept += 1

    if kept == 0:
        raise RuntimeError("No training rows were written to id_prop.csv")
    return kept, skipped_missing_cif


def main() -> None:
    args = parse_args()
    mp_cif_root = args.mp_cif_root.resolve()
    summary_csv = args.summary_csv.resolve()
    atom_init_source = args.atom_init_source.resolve()
    repo_dataset_link = args.repo_dataset_link.resolve()

    if not mp_cif_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {mp_cif_root}")
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV does not exist: {summary_csv}")
    if not atom_init_source.exists():
        raise FileNotFoundError(f"atom_init.json source does not exist: {atom_init_source}")

    moved, removed_dirs, skipped_existing = flatten_cifs(mp_cif_root)
    kept_rows, skipped_missing_cif = write_id_prop_csv(mp_cif_root, summary_csv)

    atom_init_link = mp_cif_root / "atom_init.json"
    ensure_symlink(atom_init_link, atom_init_source)

    if repo_dataset_link.exists() or repo_dataset_link.is_symlink():
        if not repo_dataset_link.is_symlink() or repo_dataset_link.resolve() != mp_cif_root:
            raise FileExistsError(
                f"{repo_dataset_link} already exists and does not point to {mp_cif_root}"
            )
    else:
        repo_dataset_link.parent.mkdir(parents=True, exist_ok=True)
        repo_dataset_link.symlink_to(mp_cif_root, target_is_directory=True)

    print(f"dataset_root={mp_cif_root}")
    print(f"moved_cifs={moved}")
    print(f"removed_subdirs={removed_dirs}")
    print(f"skipped_existing_flat_cifs={skipped_existing}")
    print(f"id_prop_rows={kept_rows}")
    print(f"summary_rows_missing_cif={skipped_missing_cif}")
    print(f"atom_init_link={atom_init_link}")
    print(f"repo_dataset_link={repo_dataset_link}")


if __name__ == "__main__":
    main()
