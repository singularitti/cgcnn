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
from .utils import Normalizer, mae, class_eval, AverageMeter

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
        test_loader, model, criterion, normalizer, cuda, print_freq, task
    )
    return out_csv


def _validate(
    val_loader, model, criterion, normalizer, cuda, print_freq, task, test=True
):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # Always initialize metrics and testing lists to avoid warnings
    mae_errors = AverageMeter()
    accuracies = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    fscores = AverageMeter()
    auc_scores = AverageMeter()
    test_targets = []
    test_preds = []
    test_cif_ids = []

    model.eval()
    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
            if cuda:
                input_var = (
                    Variable(input[0].cuda(non_blocking=True)),
                    Variable(input[1].cuda(non_blocking=True)),
                    input[2].cuda(non_blocking=True),
                    [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
                )
            else:
                input_var = (Variable(input[0]), Variable(input[1]), input[2], input[3])
        if task == "regression":
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        with torch.no_grad():
            if cuda:
                target_var = Variable(target_normed.cuda(non_blocking=True))
            else:
                target_var = Variable(target_normed)

        output = model(*input_var)
        loss = criterion(output, target_var)

        if task == "regression":
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.tolist()
                test_targets += test_target.tolist()
                test_cif_ids += batch_cif_ids
        else:
            accuracy, precision, recall, fscore, auc_score = class_eval(
                output.data.cpu(), target
            )
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0:
            if task == "regression":
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        mae_errors=mae_errors,
                    )
                )
            else:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accu {accu.val:.3f} ({accu.avg:.3f})\t"
                    "Precision {prec.val:.3f} ({prec.avg:.3f})\t"
                    "Recall {recall.val:.3f} ({recall.avg:.3f})\t"
                    "F1 {f1.val:.3f} ({f1.avg:.3f})\t"
                    "AUC {auc.val:.3f} ({auc.avg:.3f})".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        accu=accuracies,
                        prec=precisions,
                        recall=recalls,
                        f1=fscores,
                        auc=auc_scores,
                    )
                )

    if test:
        out_csv = "test_results.csv"
        import csv

        with open(out_csv, "w") as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow(
                    [cif_id] + list(map(float, target)) + list(map(float, pred))
                )
        return out_csv
    else:
        if task == "regression":
            print(" MAE {mae_errors.avg:.3f}".format(mae_errors=mae_errors))
            return mae_errors.avg
        else:
            print(" AUC {auc.avg:.3f}".format(auc=auc_scores))
            return auc_scores.avg
