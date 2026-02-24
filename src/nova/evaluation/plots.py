from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def save_roc(y_true: np.ndarray, y_score: np.ndarray, out_path: str | Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_pr(y_true: np.ndarray, y_score: np.ndarray, out_path: str | Path) -> None:
    p, r, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_score_hist(y_true: np.ndarray, y_score: np.ndarray, out_path: str | Path) -> None:
    plt.figure()
    plt.hist(y_score[y_true == 0], bins=50, alpha=0.6, label="normal")
    plt.hist(y_score[y_true == 1], bins=50, alpha=0.6, label="crisis")
    plt.legend()
    plt.xlabel("score")
    plt.ylabel("count")
    plt.title("Score distribution")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
