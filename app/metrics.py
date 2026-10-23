"""Unclipped regression metrics with explicit completeness and reproducible intervals."""
import numpy as np


def metrics(predictions, labels, sex, *, bootstrap_samples=1000,
            bootstrap_seed=20261023, prediction_min=0.0, prediction_max=360.0):
    """Keep every supplied case in ``n`` and every finite prediction in analysis.

    Negative/out-of-range predictions are never clipped or silently discarded.
    Non-finite prediction/label pairs have no meaningful error, so their absence
    is reported and ``complete`` is false. A finite-only MAE is not evidence that
    the complete cohort met a performance target. Intervals resample images;
    patient independence cannot be inferred from them.
    """
    p = np.asarray(predictions, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    raw_sex = np.asarray(sex).reshape(-1)
    if not (len(p) == len(y) == len(raw_sex)):
        raise ValueError("predictions, labels and sex must contain the same number of cases")
    if not isinstance(bootstrap_samples, (int, np.integer)) or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    if not np.isfinite([prediction_min, prediction_max]).all() or prediction_min >= prediction_max:
        raise ValueError("prediction range must be finite and increasing")
    sexes = np.full(len(p), -1, dtype=np.int8)
    for index, value in enumerate(raw_sex):
        if value in (1, 1.0, "M", "m", "1", "1.0"):
            sexes[index] = 1
        elif value in (0, 0.0, "F", "f", "0", "0.0"):
            sexes[index] = 0
    rng = np.random.default_rng(bootstrap_seed)

    def rounded(value):
        return round(float(value), 4)

    def subset(mask):
        pred, truth = p[mask], y[mask]
        finite_pred, finite_truth = np.isfinite(pred), np.isfinite(truth)
        finite = finite_pred & finite_truth
        error = pred[finite] - truth[finite]
        absolute = np.abs(error)
        interval = None
        if len(error):
            # Limit the allocation even for large independently supplied cohorts.
            estimates = np.empty(bootstrap_samples, dtype=np.float64)
            chunk = max(1, min(128, 1_000_000 // len(error)))
            for start in range(0, bootstrap_samples, chunk):
                count = min(chunk, bootstrap_samples - start)
                indices = rng.integers(0, len(error), size=(count, len(error)))
                estimates[start:start + count] = absolute[indices].mean(axis=1)
            interval = [rounded(v) for v in np.percentile(estimates, (2.5, 97.5))]
        return {
            "n": int(mask.sum()),
            "finite_n": int(finite.sum()),
            "finite_prediction_n": int(finite_pred.sum()),
            "finite_label_n": int(finite_truth.sum()),
            "invalid_n": int((~finite).sum()),
            "nonfinite_prediction_n": int((~finite_pred).sum()),
            "nonfinite_label_n": int((~finite_truth).sum()),
            "out_of_range_n": int((finite_pred & ((pred < prediction_min) | (pred > prediction_max))).sum()),
            "complete": bool(len(error) == len(pred) and len(pred) > 0),
            "mae_months": rounded(absolute.mean()) if len(error) else None,
            "rmse_months": rounded(np.sqrt(np.mean(error ** 2))) if len(error) else None,
            "median_absolute_error": rounded(np.median(absolute)) if len(error) else None,
            "p90_absolute_error": rounded(np.percentile(absolute, 90)) if len(error) else None,
            "signed_bias_months": rounded(error.mean()) if len(error) else None,
            "mae_ci95_months": interval,
        }

    result = subset(np.ones(len(p), dtype=bool))
    result["unknown_sex_n"] = int((sexes == -1).sum())
    result["by_sex"] = {"M": subset(sexes == 1), "F": subset(sexes == 0)}
    result["by_boneage"] = {}
    for low, high in ((0, 24), (24, 72), (72, 120), (120, 168), (168, 1000)):
        result["by_boneage"][f'{low}-{high if high < 1000 else "+"}'] = subset((y >= low) & (y < high))
    result["bootstrap"] = {"samples": int(bootstrap_samples), "seed": int(bootstrap_seed),
                           "confidence": 0.95, "unit": "image", "method": "percentile"}
    result["prediction_range_months"] = [float(prediction_min), float(prediction_max)]
    result["metrics_use_unclipped_predictions"] = True
    return result
