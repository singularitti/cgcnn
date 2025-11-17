"""
Shared utilities used by training and inference modules.
This file contains Normalizer, mae, class_eval, AverageMeter, and save_checkpoint.
"""
from __future__ import annotations

import shutil
import numpy as np
import torch
from sklearn import metrics

__all__ = [
    "Normalizer",
    "mae",
    "class_eval",
    "AverageMeter",
    "save_checkpoint",
]


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

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
        self.std[self.std == 0] = 1.

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
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average="binary"
        )
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError("Only binary classification supported by class_eval")
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
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
