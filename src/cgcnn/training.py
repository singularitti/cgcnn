"""
Training utilities and high-level train function for CGCNN.
This refactors the previous `main.py` logic into importable functions.
"""

import os
import sys
import time
import warnings
from random import sample

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from .data import CIFData, collate_pool, get_train_val_test_loader
from .model import CrystalGraphConvNet
from .utils import (
    Normalizer,
    mae,
    class_eval,
    AverageMeter,
    save_checkpoint,
)

__all__ = ["train_model"]


def _to_device(obj, cuda: bool):
    if cuda:
        if isinstance(obj, tuple):
            return tuple((_to_device(x, cuda) for x in obj))
        try:
            return obj.cuda(non_blocking=True)
        except Exception:
            return obj
    return obj


def train_model(
    root_dir: str,
    task: str = "regression",
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 0.01,
    lr_milestones: list | None = None,
    optim_name: str = "SGD",
    atom_fea_len: int = 64,
    h_fea_len: int = 128,
    n_conv: int = 3,
    n_h: int = 1,
    cuda: bool | None = None,
    workers: int = 0,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    start_epoch: int = 0,
    print_freq: int = 10,
    train_ratio: float | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    train_size: int | None = None,
    val_size: int | None = None,
    test_size: int | None = None,
    resume: str | None = None,
    save_dir: str | None = None,
):
    """Train a CGCNN model.

    Parameters
    ----------
    root_dir: str
        Path to CIF dataset directory (root of dataset)
    task: str
        'regression' or 'classification'
    Other args are analogous to the previous CLI arguments.

    Returns
    -------
    path to the best saved model file (model_best.pth.tar) if saved, else None.
    """
    if cuda is None:
        cuda = torch.cuda.is_available()
    if lr_milestones is None:
        lr_milestones = [100]
    dataset = CIFData(root_dir)
    collate_fn = collate_pool
    returned_loaders = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        train_ratio=train_ratio,
        num_workers=workers,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        pin_memory=cuda,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        return_test=True,
    )
    if isinstance(returned_loaders, tuple) and len(returned_loaders) == 3:
        train_loader, val_loader, test_loader = returned_loaders
    else:
        train_loader, val_loader = returned_loaders
        test_loader = None
    n_targets = dataset.n_targets

    # obtain target value normalizer
    if task == "classification":
        normalizer = Normalizer(torch.zeros(n_targets))
        normalizer.load_state_dict({"mean": 0.0, "std": 1.0})
    else:
        if len(dataset) < 500:
            warnings.warn(
                "Dataset has less than 500 data points. Lower accuracy is expected. "
            )
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in sample(range(len(dataset)), 500)]
        _, sample_target, _ = collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)

    # build model
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
        n_targets=n_targets,
    )
    if cuda:
        model.cuda()

    # define loss func and optimizer
    if task == "classification":
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()
    if optim_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr, momentum=momentum, weight_decay=weight_decay
        )
    elif optim_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    else:
        raise NameError("Only SGD or Adam is allowed as optim_name")

    # optionally resume from a checkpoint
    best_mae_error = 1e10 if task == "regression" else 0.0
    if resume:
        if os.path.isfile(resume):
            checkpoint = torch.load(resume, map_location=(lambda s, l: s))
            start_epoch = checkpoint.get("epoch", start_epoch)
            best_mae_error = checkpoint.get("best_mae_error", best_mae_error)
            model.load_state_dict(checkpoint["state_dict"])  # raises if mismatch
            optimizer.load_state_dict(checkpoint["optimizer"])
            normalizer.load_state_dict(checkpoint["normalizer"])

    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

    # training loop
    best_checkpoint_path = None
    for epoch in range(start_epoch, epochs):
        _train_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            normalizer,
            cuda,
            task,
            print_freq,
        )
        mae_error = _validate(val_loader, model, criterion, normalizer, cuda, task)
        if mae_error != mae_error:
            sys.exit(1)
        scheduler.step()
        if task == "regression":
            is_best = mae_error < best_mae_error
            best_mae_error = min(mae_error, best_mae_error)
        else:
            is_best = mae_error > best_mae_error
            best_mae_error = max(mae_error, best_mae_error)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_mae_error": best_mae_error,
                "optimizer": optimizer.state_dict(),
                "normalizer": normalizer.state_dict(),
                "args": {},
            },
            is_best,
        )
        if is_best:
            best_checkpoint_path = os.path.abspath("model_best.pth.tar")

    # test best model if requested
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=(lambda s, l: s))
        model.load_state_dict(checkpoint["state_dict"])
        _validate(test_loader, model, criterion, normalizer, cuda, task, test=True)

    return best_checkpoint_path


def _train_epoch(
    train_loader, model, criterion, optimizer, epoch, normalizer, cuda, task, print_freq
):
    # Simple version of original main.train that uses local arguments
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # initialize all metrics so they're available for both tasks
    mae_errors = AverageMeter()
    accuracies = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    fscores = AverageMeter()
    auc_scores = AverageMeter()

    model.train()
    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            if task == "regression":
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})".format(
                        epoch,
                        i,
                        len(train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        mae_errors=mae_errors,
                    )
                )
            else:
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accu {accu.val:.3f} ({accu.avg:.3f})\t"
                    "Precision {prec.val:.3f} ({prec.avg:.3f})\t"
                    "Recall {recall.val:.3f} ({recall.avg:.3f})\t"
                    "F1 {f1.val:.3f} ({f1.avg:.3f})\t"
                    "AUC {auc.val:.3f} ({auc.avg:.3f})".format(
                        epoch,
                        i,
                        len(train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        accu=accuracies,
                        prec=precisions,
                        recall=recalls,
                        f1=fscores,
                        auc=auc_scores,
                    )
                )


def _validate(val_loader, model, criterion, normalizer, cuda, task, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # initialize all metrics so they're available for both tasks
    mae_errors = AverageMeter()
    accuracies = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    fscores = AverageMeter()
    auc_scores = AverageMeter()
    # prepare test result lists (always initialize to avoid warnings)
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
        if i % 10 == 0:
            # Print some validation info
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
        star_label = "**"
        import csv

        with open("test_results.csv", "w") as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow(
                    [cif_id] + list(map(float, target)) + list(map(float, pred))
                )
    else:
        star_label = "*"
    if task == "regression":
        print(
            " {star} MAE {mae_errors.avg:.3f}".format(
                star=star_label, mae_errors=mae_errors
            )
        )
        return mae_errors.avg
    else:
        print(" {star} AUC {auc.avg:.3f}".format(star=star_label, auc=auc_scores))
        return auc_scores.avg
