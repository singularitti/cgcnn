import csv
import sys
import tempfile
import unittest
from pathlib import Path

import torch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cgcnn.data import collate_pool, collate_pool_vectorized
from cgcnn.inference import predict_model, predict_regression_models
from cgcnn.model import CrystalGraphConvNet
from cgcnn.utils import Normalizer


class TinyGraphDataset:
    n_targets = 1

    def __init__(self):
        self.items = []
        for index, atom_count in enumerate([2, 3, 1, 4]):
            atom_fea = torch.arange(atom_count * 3, dtype=torch.float32).reshape(
                atom_count, 3
            ) / 10
            nbr_fea = torch.ones(atom_count, 2, 4, dtype=torch.float32) * (index + 1)
            nbr_fea_idx = torch.arange(atom_count, dtype=torch.long).view(-1, 1).repeat(1, 2)
            self.items.append(
                (
                    (atom_fea, nbr_fea, nbr_fea_idx),
                    torch.tensor([float(index)]),
                    f"id_{index}",
                )
            )

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)


class MultiModelInferenceTests(unittest.TestCase):
    def test_shared_graph_predictions_allow_no_models(self):
        self.assertEqual(
            predict_regression_models(
                TinyGraphDataset(),
                {},
                output_csvs={},
                device="cpu",
            ),
            {},
        )

    def test_vectorized_pooling_matches_list_pooling(self):
        dataset = TinyGraphDataset()
        list_inputs, _, _ = collate_pool(dataset.items)
        vector_inputs, _, _ = collate_pool_vectorized(dataset.items)
        model = CrystalGraphConvNet(3, 4, atom_fea_len=5)
        atom_fea = torch.randn(sum(item[0][0].shape[0] for item in dataset.items), 5)

        expected = model.pooling(atom_fea, list_inputs[3])
        actual = model.pooling(atom_fea, vector_inputs[3])

        torch.testing.assert_close(actual, expected)

    def test_shared_graph_predictions_match_single_model_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            torch.manual_seed(7)
            dataset = TinyGraphDataset()
            model = CrystalGraphConvNet(
                3,
                4,
                atom_fea_len=6,
                n_conv=2,
                h_fea_len=8,
                n_h=1,
                n_targets=1,
            )
            normalizer = Normalizer(torch.tensor([[1.0], [2.0], [3.0]]))
            checkpoint = tmp_path / "model.pth.tar"
            torch.save(
                {
                    "args": {
                        "task": "regression",
                        "atom_fea_len": 6,
                        "n_conv": 2,
                        "h_fea_len": 8,
                        "n_h": 1,
                        "n_targets": 1,
                    },
                    "state_dict": model.state_dict(),
                    "normalizer": normalizer.state_dict(),
                },
                checkpoint,
            )

            single_csv = tmp_path / "single.csv"
            shared_csv = tmp_path / "shared.csv"
            predict_model(
                dataset,
                modelpath=str(checkpoint),
                batch_size=2,
                workers=0,
                device="cpu",
                print_freq=100,
                output_csv=str(single_csv),
            )
            predict_regression_models(
                dataset,
                {"model": checkpoint},
                output_csvs={"model": shared_csv},
                batch_size=2,
                workers=0,
                device="cpu",
                print_freq=100,
            )

            with single_csv.open(newline="") as handle:
                single_rows = list(csv.reader(handle))
            with shared_csv.open(newline="") as handle:
                shared_rows = list(csv.reader(handle))
            self.assertEqual(
                [row[0] for row in shared_rows],
                [row[0] for row in single_rows],
            )
            torch.testing.assert_close(
                torch.tensor([float(row[2]) for row in shared_rows]),
                torch.tensor([float(row[2]) for row in single_rows]),
            )

            modelpaths = {f"model_{index}": checkpoint for index in range(4)}
            output_csvs = {
                key: tmp_path / f"{key}.csv" for key in modelpaths
            }
            outputs = predict_regression_models(
                dataset,
                modelpaths,
                output_csvs=output_csvs,
                batch_size=2,
                workers=0,
                device="cpu",
                print_freq=100,
            )
            self.assertEqual(set(outputs), set(modelpaths))
            for output_csv in output_csvs.values():
                with output_csv.open(newline="") as handle:
                    self.assertEqual(len(list(csv.reader(handle))), len(dataset))
