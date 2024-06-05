from sklearn import metrics
import numpy as np


def roc_auc(dm: np.ndarray,
            gt: np.ndarray):
    rows, cols = gt.shape

    gt = gt.reshape(rows * cols)
    dm = dm.reshape(rows * cols)

    fpr, tpr, _ = metrics.roc_curve(gt, dm)
    auc = metrics.auc(fpr, tpr)

    return fpr, tpr, auc



