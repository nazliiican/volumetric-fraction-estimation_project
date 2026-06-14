"""Shared evaluation metrics, used identically by every model.
Composition arrays are shaped ``(n_samples, n_materials)`` and each row is a
fraction vector that should sum to 1.
"""

import numpy as np

PRESENCE_TOL = 1e-8


def _arrays(y_true, y_pred):
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def mae(y_true, y_pred):
    """Mean absolute error over all entries."""
    yt, yp = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred):
    """Root mean squared error over all entries."""
    yt, yp = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def r2(y_true, y_pred):
    """Coefficient of determination, pooled over all composition entries."""
    yt, yp = _arrays(y_true, y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def per_material_mae(y_true, y_pred, materials):
    """MAE computed separately for each material column.

    Returns a dict ``{material: mae}``.
    """
    yt, yp = _arrays(y_true, y_pred)
    col_mae = np.mean(np.abs(yt - yp), axis=0)
    return {m: float(col_mae[i]) for i, m in enumerate(materials)}


def composition_sum_error(y_pred):
    """Mean absolute deviation of each predicted row's sum from 1.

    A model that outputs a valid composition (e.g. via softmax) scores ~0 here.
    """
    yp = np.asarray(y_pred, dtype=float)
    row_sums = yp.sum(axis=-1)
    return float(np.mean(np.abs(row_sums - 1.0)))


def to_presence(y, threshold=0.05):
    """Binarize a composition / probability array into a presence array.

    Entries strictly greater than ``threshold`` become 1, the rest 0. Does not
    mutate ``y``.
    """
    return (np.asarray(y, dtype=float) > threshold).astype(int)


def presence_f1(y_true_presence, y_pred_presence):
    """Micro-averaged F1 over a binary multi-hot presence array."""
    yt = np.asarray(y_true_presence).astype(bool)
    yp = np.asarray(y_pred_presence).astype(bool)

    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp))

    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return float(2 * precision * recall / (precision + recall))


def false_positive_absent_percentage(y_true, y_pred):
    """Percentage of predicted composition mass placed on absent materials.
    For each sample, sum the predicted fractions where the true fraction is 0,
    divide by the total predicted mass, then average over samples and express
    as a percentage. This quantifies the "leakage onto absent materials".
    """
    yt, yp = _arrays(y_true, y_pred)
    absent = yt <= PRESENCE_TOL
    # Predicted mass that lands on truly-absent materials, per sample, as a
    # share of the total predicted mass (each prediction row sums to 1).
    leaked = np.where(absent, yp, 0.0).sum(axis=-1)
    total = yp.sum(axis=-1)
    return float(np.mean(leaked / total) * 100.0)


def evaluate_composition(y_true, y_pred, materials, presence_threshold=0.05):
    """Every shared metric in one dict."""
    yt, yp = _arrays(y_true, y_pred)
    true_presence = to_presence(yt, threshold=PRESENCE_TOL)
    pred_presence = to_presence(yp, threshold=presence_threshold)

    results = {
        "MAE": mae(yt, yp),
        "RMSE": rmse(yt, yp),
        "R2": r2(yt, yp),
        "composition_sum_error": composition_sum_error(yp),
        "presence_F1": presence_f1(true_presence, pred_presence),
        "false_positive_absent_%": false_positive_absent_percentage(yt, yp),
    }
    per_mat = per_material_mae(yt, yp, materials)
    for material, value in per_mat.items():
        results[f"MAE_{material}"] = value
    return results
