"""Quantitative specificity diagnostics, independent of SL checkpoint selection."""

import numpy as np
import torch


def numeric_metrics(
    truth,
    prediction,
    *,
    fitting_mean,
    wrong_gene_prediction=None,
    predicted_additive=None,
):
    y, p = np.asarray(truth, np.float64), np.asarray(prediction, np.float64)
    baseline = np.broadcast_to(np.asarray(fitting_mean, np.float64), y.shape)
    if (
        y.shape != p.shape
        or not y.size
        or not all(np.isfinite(v).all() for v in (y, p, baseline))
    ):
        raise ValueError("Require complete finite matched quantitative observations")
    mse = float(np.square(y - p).mean())
    base_mse = float(np.square(y - baseline).mean())
    result = {
        "measurements": int(y.size),
        "mse": mse,
        "fitting_mean_mse": base_mse,
        "mse_skill_over_fitting_mean": 1 - mse / base_mse if base_mse else None,
        "truth_std": float(y.std()),
        "prediction_std": float(p.std()),
        "pearson": float(np.corrcoef(y.ravel(), p.ravel())[0, 1])
        if y.std() > 0 and p.std() > 0
        else None,
    }
    if wrong_gene_prediction is not None:
        wrong = np.asarray(wrong_gene_prediction, np.float64)
        if wrong.shape != y.shape or not np.isfinite(wrong).all():
            raise ValueError("Wrong-gene predictions are not matched")
        wrong_mse = float(np.square(y - wrong).mean())
        result.update(
            wrong_gene_mse=wrong_mse,
            wrong_minus_correct_mse=wrong_mse - mse,
            prediction_change_mse=float(np.square(p - wrong).mean()),
        )
    if predicted_additive is not None:
        additive = np.asarray(predicted_additive, np.float64)
        if additive.shape != y.shape or not np.isfinite(additive).all():
            raise ValueError("Additive predictions are not matched")
        result["predicted_additive_mse"] = float(np.square(y - additive).mean())
    return result


@torch.inference_mode()
def four_conditions(model, batch):
    """Diagnostic f(0), f(A), f(B), f(AB); never the mandatory SL path.

    A caller must specify whether its assay uses additive log effects or a
    multiplicative raw-fitness null. This helper imposes neither equation.
    """
    if batch["action_mask"].shape[1] != 2 or not batch["action_mask"].all():
        raise ValueError("Four-condition diagnostic requires two actions")
    result = {}
    for name, mask in [
        ("control", [False, False]),
        ("a", [True, False]),
        ("b", [False, True]),
        ("ab", [True, True]),
    ]:
        query = dict(batch)
        query["action_mask"] = torch.tensor(
            mask, device=batch["action_mask"].device
        ).expand_as(batch["action_mask"])
        result[name] = model(query)["location"].float()
    return result
