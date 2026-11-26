import numpy as np
from sklearn import metrics
import torch


def hh_rejection_at_20(logits, true, weight, selection_info):
    """Inclusive positive-MC-weight rejection at 20% HH efficiency from jet preselection."""
    p = logits.softmax(dim=1).detach().cpu().numpy()
    truth = true.detach().cpu().numpy() == 0
    jets, signed = selection_info.detach().cpu().numpy().T
    w = signed.astype(np.float64)
    pre = (jets >= 1) & (w > 0)
    sr = pre & (p[:, 2] < 0.8)
    signal = sr & truth
    total_signal = w[pre & truth].sum()
    if not signal.any() or total_signal <= 0:
        return 0.0
    order = np.argsort(-p[signal, 0], kind="stable")
    cumulative = np.cumsum(w[signal][order])
    index = np.searchsorted(cumulative, 0.2 * total_signal)
    if index >= len(order):
        return 0.0
    cut = p[signal, 0][order[index]]
    passing = w[sr & ~truth & (p[:, 0] >= cut)].sum()
    return float(w[pre & ~truth].sum() / passing) if passing > 0 else 0.0

def weighted_accuracy(logits: torch.Tensor, true: torch.Tensor, weight: torch.Tensor):
    """Compute weighted accuracy."""
    score = torch.softmax(logits, dim=1)
    y_pred = torch.argmax(score, dim=1)
    correct = (y_pred == true).float()
    weighted_correct = correct * weight
    return torch.sum(weighted_correct) / torch.sum(weight)

# def weighted_auc_2cls(logits: torch.Tensor, true: torch.Tensor,  weight: torch.Tensor):
#     """Compute weighted ROC curve accounting for negative weights."""
#     score = torch.softmax(logits, dim=1).numpy(force=True)
#     true = true.numpy(force=True)
#     weight = weight.numpy(force=True)
#     # Sort by score (descending)
#     sorted_indices = np.argsort(-score)
#     true = true[sorted_indices]
#     weight = weight[sorted_indices]
#     # Define signal and background masks
#     is_signal = true == 0
#     is_background = true == 1
#     # Compute total positive and negative weights
#     total_signal_weight = np.sum(weight[is_signal])
#     total_background_weight = np.sum(weight[is_background])
#     # Compute cumulative sums
#     tpr = np.cumsum(weight * is_signal) / total_signal_weight
#     fpr = np.cumsum(weight * is_background) / total_background_weight
#     tpr, fpr = np.concatenate(([0], tpr)), np.concatenate(([0], fpr))  # Ensure starting point (0,0)
#     auc = np.trapezoid(fpr, tpr)
#     return auc

def weighted_auc(logits: torch.Tensor, true: torch.Tensor,  weight: torch.Tensor):
    score = torch.softmax(logits, dim=1).numpy(force=True)
    true = true.numpy(force=True)
    weight = weight.numpy(force=True)
    if score.shape[1] == 2:
        return metrics.roc_auc_score(true, score[:, 1], sample_weight=weight)
    return metrics.roc_auc_score(true, score, sample_weight=weight, multi_class="ovr", average="macro")

# def weighted_confusion_matrix_2cls(logits: torch.Tensor, true: torch.Tensor, weight: torch.Tensor):
#     """Compute weighted confusion matrix accounting for negative weights."""
#     score = torch.softmax(logits, dim=1).numpy(force=True)
#     true = true.numpy(force=True)
#     weight = weight.numpy(force=True)
#     y_pred = score.argmax(1)
#     cm = np.zeros((2, 2), dtype=np.float32)
#     for i in range(len(true)):
#         if true[i] == 1:
#             if y_pred[i] == 1:
#                 cm[0, 0] += weight[i]  # True Positive
#             else:
#                 cm[0, 1] += weight[i]  # False Negative
#         else:
#             if y_pred[i] == 0:
#                 cm[1, 1] += weight[i]  # True Negative
#             else:
#                 cm[1, 0] += weight[i]  # False Positive
#     # Normalizes confusion matrix over the true (rows)
#     cm[0, :] /= cm[0, :].sum() if cm[0, :].sum() > 0 else 1
#     cm[1, :] /= cm[1, :].sum() if cm[1, :].sum() > 0 else 1
#
#     return cm

def weighted_confusion_matrix(logits: torch.Tensor, true: torch.Tensor, weight: torch.Tensor):
    """Compute weighted confusion matrix."""
    score = torch.softmax(logits, dim=1).numpy(force=True)
    true = true.numpy(force=True)
    weight = weight.numpy(force=True)
    y_pred = score.argmax(1)
    cm = metrics.confusion_matrix(true, y_pred, sample_weight=weight, normalize="true")

    return cm

