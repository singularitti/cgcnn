from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from pymatgen.core import Lattice, Structure

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_mptrj_module():
    module_path = REPO_ROOT / "tools" / "build_mptrj_graph_cache.py"
    spec = importlib.util.spec_from_file_location("build_mptrj_graph_cache", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MPtrjGraphCacheTests(unittest.TestCase):
    def test_stream_records_uses_graph_id_and_ef_per_atom_target(self):
        module = load_mptrj_module()
        structure = Structure(
            Lattice.cubic(3.0),
            ["Li"],
            [[0.0, 0.0, 0.0]],
        )
        payload = {
            "mp-1": {
                "mp-1-0-0": {
                    "structure": structure.as_dict(),
                    "energy_per_atom": -1.0,
                    "ef_per_atom": -0.25,
                    "e_per_atom_relaxed": -1.1,
                    "ef_per_atom_relaxed": -1.1,
                    "force": [[0.0, 0.0, 0.0]],
                    "stress": [[0.0, 0.0, 0.0]] * 3,
                    "magmom": None,
                    "bandgap": None,
                    "mp_id": "mp-1",
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "tiny.json"
            path.write_text(json.dumps(payload))
            rows = list(module.stream_records(path, "ef_per_atom"))

        self.assertEqual(len(rows), 1)
        _, records, frame_count, nonfinite_targets = rows[0]
        self.assertEqual(frame_count, 1)
        self.assertEqual(nonfinite_targets, 0)
        self.assertEqual(records[0].graph_id, "mp-1::mp-1-0-0")
        self.assertEqual(records[0].target, -0.25)
        self.assertEqual(records[0].n_atoms, 1)

    def test_group_splits_do_not_split_mp_id_across_splits(self):
        module = load_mptrj_module()
        graph_rows = [
            ("mp-1::a", "mp-1", -0.1),
            ("mp-1::b", "mp-1", -0.2),
            ("mp-2::a", "mp-2", -0.3),
            ("mp-3::a", "mp-3", -0.4),
            ("mp-4::a", "mp-4", -0.5),
        ]

        splits = module.make_group_splits(
            graph_rows,
            seed=7,
            train_ratio=0.5,
            val_ratio=0.25,
        )
        module.validate_split_groups(graph_rows, splits)

        locations = {}
        for split_name, ids in splits.items():
            for graph_id in ids:
                locations[graph_id.split("::")[0]] = split_name
        self.assertEqual(locations["mp-1"], locations["mp-1"])
        self.assertEqual(sum(len(ids) for ids in splits.values()), len(graph_rows))


if __name__ == "__main__":
    unittest.main()
