"""
Inference utilities for CGCNN; refactors predict.py into importable functions.
"""

import os
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .data import CIFData, collate_pool
from .model import CrystalGraphConvNet
from .utils import Normalizer, mae, class_eval, AverageMeter, _validate

__all__ = ["predict_model"]


def predict_model(
    dataset: CIFData,
    task: str | None = None,
    atom_fea_len: int | None = None,
    n_conv: int | None = None,
    h_fea_len: int | None = None,
    n_h: int | None = None,
    n_targets: int | None = None,
    model: CrystalGraphConvNet | None = None,
    normalizer: Normalizer | None = None,
    modelpath: str | None = None,
    batch_size: int = 256,
    workers: int = 0,
    cuda: bool | None = None,
    print_freq: int = 10,
):
    """Load a model from a saved checkpoint or use provided model and predict on CIF files in `dataset`.

    Returns the path to the CSV file written (`test_results.csv`) or None on failure.
    """
    if cuda is None:
        cuda = torch.cuda.is_available()
    if model is None:
        if modelpath is None or not os.path.isfile(modelpath):
            raise ValueError("Either model or valid modelpath must be provided")
        checkpoint = torch.load(modelpath, map_location=(lambda s, l: s))
        args = checkpoint.get("args", {})
        task = task or args.get("task", "regression")
        atom_fea_len = atom_fea_len or args.get("atom_fea_len", 64)
        n_conv = n_conv or args.get("n_conv", 3)
        h_fea_len = h_fea_len or args.get("h_fea_len", 128)
        n_h = n_h or args.get("n_h", 1)
        n_targets = n_targets or args.get("n_targets", dataset.n_targets)
    else:
        checkpoint = None
        task = task or "regression"
        atom_fea_len = atom_fea_len or 64
        n_conv = n_conv or 3
        h_fea_len = h_fea_len or 128
        n_h = n_h or 1
        n_targets = n_targets or dataset.n_targets

    collate_fn = collate_pool
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=cuda,
    )
    model_target_dim = n_targets
    if task == "regression" and dataset.n_targets != model_target_dim:
        raise ValueError("Dataset target dimensionality doesn't match model.")

    if model is None:
        structures, _, _ = dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = CrystalGraphConvNet(
            orig_atom_fea_len,
            nbr_fea_len,
            atom_fea_len=atom_fea_len,
            n_conv=n_conv,
            h_fea_len=h_fea_len,
            n_h=n_h,
            classification=True if task == "classification" else False,
            n_targets=model_target_dim,
        )
        model.load_state_dict(checkpoint["state_dict"])  # will raise on mismatch
    if cuda:
        model.cuda()

    if task == "classification":
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()

    if normalizer is None:
        if checkpoint is None:
            raise ValueError(
                "Normalizer must be provided if not loading from checkpoint"
            )
        normalizer = Normalizer(
            torch.zeros(model_target_dim if task == "regression" else 1)
        )
        normalizer.load_state_dict(
            checkpoint.get("normalizer", normalizer.state_dict())
        )

    out_csv = _validate(
        test_loader,
        model,
        criterion,
        normalizer,
        cuda,
        task,
        test=True,
        print_freq=print_freq,
    )
    return out_csv
