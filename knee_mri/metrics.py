import numpy as np
from sklearn.metrics import roc_auc_score
from .constants import LABELS


def auc_metrics(target, probability):
    target, probability = np.asarray(target), np.asarray(probability)
    values = {}
    for i, name in enumerate(LABELS):
        valid = np.isfinite(target[:, i]) & np.isfinite(probability[:, i])
        y = target[valid, i] > .5
        values[name] = float(roc_auc_score(y, probability[valid, i])) if len(np.unique(y)) == 2 else None
    scores = [v for v in values.values() if v is not None]
    return {'macro_auc': float(np.mean(scores)) if scores else None, 'per_target_auc': values}
