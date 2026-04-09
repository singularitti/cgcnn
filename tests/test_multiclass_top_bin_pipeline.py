from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cgcnn.inference import predict_model
from cgcnn.model import CrystalGraphConvNet


def load_threshold_split_module():
    module_path = REPO_ROOT / "tools" / "train_magnetization_multiclass_top_bin.py"
    spec = importlib.util.spec_from_file_location(
        "train_magnetization_multiclass_top_bin", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DummyCrystalDataset:
    def __init__(self):
        self.n_targets = 1
        self.rows = []
        for index in range(3):
            atom_fea = torch.tensor([[1.0, 0.0, float(index)]], dtype=torch.float)
            nbr_fea = torch.zeros((1, 2, 4), dtype=torch.float)
            nbr_fea_idx = torch.zeros((1, 2), dtype=torch.long)
            target = torch.tensor([index % 4], dtype=torch.float)
            self.rows.append(((atom_fea, nbr_fea, nbr_fea_idx), target, f"id_{index}"))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class MulticlassTopBinPipelineTests(unittest.TestCase):
    def test_multiclass_checkpoint_reload_and_csv_shape(self):
        dataset = DummyCrystalDataset()
        model = CrystalGraphConvNet(
            orig_atom_fea_len=3,
            nbr_fea_len=4,
            atom_fea_len=8,
            h_fea_len=16,
            n_conv=1,
            n_h=1,
            classification=True,
            n_classes=4,
        )
        self.assertEqual(model.fc_out.out_features, 4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            checkpoint_path = tmp_path / "model_best.pth.tar"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "normalizer": {"mean": torch.tensor([0.0]), "std": torch.tensor([1.0])},
                    "args": {
                        "task": "classification",
                        "atom_fea_len": 8,
                        "n_conv": 1,
                        "h_fea_len": 16,
                        "n_h": 1,
                        "n_targets": 1,
                        "n_classes": 4,
                    },
                },
                checkpoint_path,
            )

            output_csv = tmp_path / "predictions.csv"
            predict_model(
                dataset=dataset,
                task="classification",
                modelpath=str(checkpoint_path),
                batch_size=2,
                workers=0,
                cuda=False,
                output_csv=str(output_csv),
            )

            with output_csv.open(newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(len(rows), len(dataset))
            self.assertTrue(all(len(row) == 7 for row in rows))
            self.assertTrue(all(int(float(row[2])) in {0, 1, 2, 3} for row in rows))

    def test_compute_class_weights_uses_only_training_ids(self):
        module = load_threshold_split_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            records = []
            values_by_id = {
                "train_zero": 0.0,
                "train_tiny": 1e-8,
                "train_small": 1e-4,
                "train_large": 1e-2,
                "holdout_large_a": 2e-2,
                "holdout_large_b": 3e-2,
            }
            for material_id, value in values_by_id.items():
                cif_path = tmp_path / f"{material_id}.cif"
                cif_path.write_text("data_test\n")
                records.append(module.MaterialRecord(material_id, value, cif_path))

            weights, counts = module.compute_class_weights(
                records,
                ["train_zero", "train_tiny", "train_small", "train_large"],
            )

            self.assertEqual(
                counts,
                {
                    "zero": 1,
                    "tiny_positive": 1,
                    "small_positive": 1,
                    "large_positive": 1,
                },
            )
            self.assertEqual(weights, [1.0, 1.0, 1.0, 1.0])

    def test_write_stage_dataset_and_top_bin_subset(self):
        module = load_threshold_split_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            records = []
            values_by_id = {
                "zero": 0.0,
                "tiny": 5e-9,
                "small": 5e-5,
                "large": 5e-3,
            }
            for material_id, value in values_by_id.items():
                cif_path = tmp_path / f"{material_id}.cif"
                cif_path.write_text("data_test\n")
                records.append(module.MaterialRecord(material_id, value, cif_path))

            classifier_dir = tmp_path / "classifier_dataset"
            top_regressor_dir = tmp_path / "top_regressor_dataset"
            record_map = {record.material_id: record for record in records}
            top_bin_ids = module.subset_ids_by_class(
                list(values_by_id.keys()),
                record_map,
                module.CLASS_LARGE,
            )

            module.write_stage_dataset(classifier_dir, records, mode="classification")
            module.write_stage_dataset(
                top_regressor_dir,
                [record_map[material_id] for material_id in top_bin_ids],
                mode="regression",
            )

            with (classifier_dir / "id_prop.csv").open(newline="") as handle:
                classifier_rows = list(csv.reader(handle))
            self.assertEqual(
                {row[0]: int(row[1]) for row in classifier_rows},
                {
                    "zero": 0,
                    "tiny": 1,
                    "small": 2,
                    "large": 3,
                },
            )
            with (top_regressor_dir / "id_prop.csv").open(newline="") as handle:
                regressor_rows = list(csv.reader(handle))
            self.assertEqual(regressor_rows, [["large", "0.005"]])

    def test_resolve_stage_resume_prefers_checkpoint_for_incomplete_stage(self):
        module = load_threshold_split_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            stage_dir = Path(tmp_dir)
            history_path = stage_dir / "training_history.json"
            checkpoint_path = stage_dir / "checkpoint.pth.tar"
            checkpoint_path.write_text("checkpoint")
            history_path.write_text(json.dumps([{"epoch": 3}, {"epoch": 5}]))

            resume_path, completed = module.resolve_stage_resume(stage_dir, epochs=30)

            self.assertEqual(resume_path, checkpoint_path)
            self.assertFalse(completed)

    def test_resolve_stage_resume_uses_best_model_for_completed_stage(self):
        module = load_threshold_split_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            stage_dir = Path(tmp_dir)
            history_path = stage_dir / "training_history.json"
            checkpoint_path = stage_dir / "checkpoint.pth.tar"
            best_model_path = stage_dir / "model_best.pth.tar"
            checkpoint_path.write_text("checkpoint")
            best_model_path.write_text("best")
            history_path.write_text(json.dumps([{"epoch": 30}]))

            resume_path, completed = module.resolve_stage_resume(stage_dir, epochs=30)

            self.assertEqual(resume_path, best_model_path)
            self.assertTrue(completed)


if __name__ == "__main__":
    unittest.main()
