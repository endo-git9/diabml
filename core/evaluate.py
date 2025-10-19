# =========================================================
# 🧠 Evaluation Utilities (Final Clean Version)
# =========================================================
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, average_precision_score,
    roc_curve, precision_recall_curve
)

def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute classification metrics safely."""
    y_true = np.array(y_true).ravel()
    y_prob = np.array(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}
    try:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["brier"] = float(brier_score_loss(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception as e:
        metrics["error"] = str(e)
    return metrics

def get_curves(y_true, y_prob):
    """Return ROC and Precision-Recall curve values."""
    y_true = np.array(y_true).ravel()
    y_prob = np.array(y_prob).ravel()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return {"roc": (fpr, tpr), "pr": (precision, recall)}
