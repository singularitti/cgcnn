"""
Shared utilities used by training and inference modules.
This file contains Normalizer, mae, class_eval, AverageMeter, save_checkpoint, and _validate.
"""

import shutil
import time

import numpy as np
import torch
from sklearn import metrics
from torch.autograd import Variable

__all__ = [
    "AverageMeter",
    "Normalizer",
    "_validate",
    "class_eval",
    "mae",
    "save_checkpoint",
]


class Normalizer:
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor: torch.Tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        tensor = tensor.float()
        if tensor.ndim == 0:
            tensor = tensor.view(1, 1)
        elif tensor.ndim == 1:
            tensor = tensor.view(1, -1)
        self.mean = torch.mean(tensor, dim=0)
        self.std = torch.std(tensor, dim=0, unbiased=False)
        # Avoid division-by-zero for features with 0 std
        self.std[self.std == 0] = 1.0

    def norm(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor: torch.Tensor) -> torch.Tensor:
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


def mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, n_targets)
    target: torch.Tensor (N, n_targets)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction: torch.Tensor, target: torch.Tensor):
    """Evaluate classification predictions and return accuracy, precision,
    recall, fscore, and auc.
    """
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.atleast_1d(np.squeeze(target))
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average="binary"
        )
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError("Only binary classification supported by class_eval")
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def _prepare_inputs_targets(input, target, normalizer, cuda, task):
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
    return input_var, target_var


def _forward_and_loss(model, input_var, target_var, criterion):
    output = model(*input_var)
    loss = criterion(output, target_var)
    return output, loss


def _update_metrics(
    loss,
    output,
    target,
    task,
    losses,
    mae_errors,
    accuracies,
    precisions,
    recalls,
    fscores,
    auc_scores,
    test,
    test_preds,
    test_targets,
    test_cif_ids,
    batch_cif_ids,
    normalizer,
):
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


def _print_progress(
    i,
    len_loader,
    batch_time,
    losses,
    mae_errors,
    accuracies,
    precisions,
    recalls,
    fscores,
    auc_scores,
    task,
    print_freq,
    prefix="Test",
):
    if i % print_freq == 0:
        if task == "regression":
            print(
                f"{prefix}: [{i}/{len_loader}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                f"MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})"
            )
        else:
            print(
                f"{prefix}: [{i}/{len_loader}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                f"Accu {accuracies.val:.3f} ({accuracies.avg:.3f})\t"
                f"Precision {precisions.val:.3f} ({precisions.avg:.3f})\t"
                f"Recall {recalls.val:.3f} ({recalls.avg:.3f})\t"
                f"F1 {fscores.val:.3f} ({fscores.avg:.3f})\t"
                f"AUC {auc_scores.val:.3f} ({auc_scores.avg:.3f})"
            )


def _validate(
    val_loader,
    model,
    criterion,
    normalizer,
    cuda,
    task,
    test=False,
    print_freq=10,
    output_csv="test_results.csv",
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
        input_var, target_var = _prepare_inputs_targets(
            input, target, normalizer, cuda, task
        )
        output, loss = _forward_and_loss(model, input_var, target_var, criterion)
        _update_metrics(
            loss,
            output,
            target,
            task,
            losses,
            mae_errors,
            accuracies,
            precisions,
            recalls,
            fscores,
            auc_scores,
            test,
            test_preds,
            test_targets,
            test_cif_ids,
            batch_cif_ids,
            normalizer,
        )
        batch_time.update(time.time() - end)
        end = time.time()
        _print_progress(
            i,
            len(val_loader),
            batch_time,
            losses,
            mae_errors,
            accuracies,
            precisions,
            recalls,
            fscores,
            auc_scores,
            task,
            print_freq,
        )

    if test:
        import csv

        with open(output_csv, "w") as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets, test_preds):
                if not isinstance(target, (list, tuple)):
                    target = [target]
                if not isinstance(pred, (list, tuple)):
                    pred = [pred]
                writer.writerow(
                    [cif_id] + list(map(float, target)) + list(map(float, pred))
                )
        return output_csv
    else:
        if task == "regression":
            print(f" MAE {mae_errors.avg:.3f}")
            return mae_errors.avg
        else:
            print(f" AUC {auc_scores.avg:.3f}")
            return auc_scores.avg
