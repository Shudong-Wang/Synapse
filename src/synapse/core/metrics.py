import numpy as np
from sklearn import metrics
import torch

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
    if score.shape[1] == 2:
        # For binary classification, use the first column as the positive class score
        pred = score[:, 1]
    true = true.numpy(force=True)
    weight = weight.numpy(force=True)
    auc_score = metrics.roc_auc_score(true, pred, sample_weight=weight, multi_class='ovo')
    return auc_score

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

