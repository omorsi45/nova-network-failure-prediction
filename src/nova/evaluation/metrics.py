from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = y_true.astype(int)
    y_hat = (y_score >= threshold).astype(int)

    out = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["roc_auc"] = None
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        out["pr_auc"] = None

    out["f1"] = float(f1_score(y_true, y_hat)) if len(np.unique(y_true)) > 1 else None
    out["precision"] = float(precision_score(y_true, y_hat, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_hat, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    out["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return out
