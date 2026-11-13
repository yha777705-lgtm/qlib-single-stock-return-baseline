"""Recalculate validation metrics from saved predictions without retraining."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate(actual, predicted):
    """Return MAE and RMSE in fractional-return units, without multiplying by 100."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Expected non-empty, one-dimensional arrays of equal shape.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual returns and predictions must be finite.")

    error = predicted - actual
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    return {"mae": float(mae), "rmse": float(rmse)}


def main():
    # Resolve paths relative to this script so it works from another directory too.
    reports = Path(__file__).resolve().parent / "reports"
    meta = json.loads((reports / "split_summary.json").read_text(encoding="utf-8"))
    reference = json.loads(
        (reports / "lightgbm_validation.json").read_text(encoding="utf-8")
    )
    if reference["evaluation_set"] != "validation" or reference["test_evaluated"]:
        raise ValueError("Expected a validation-only reference report.")

    df = pd.read_csv(
        reports / "lightgbm_validation_predictions.csv",
        parse_dates=["feature_date", "target_date"],
    )
    if len(df) != meta["splits"]["validation"]["rows"]:
        raise ValueError("Validation row count does not match the split record.")
    if df[["feature_date", "target_date"]].isna().any().any():
        raise ValueError("Prediction dates must not be missing.")
    if not df["feature_date"].is_unique:
        raise ValueError("Duplicate validation feature dates.")
    start = pd.Timestamp(meta["validation_start"])
    end = pd.Timestamp(meta["test_start"])
    if not (
        df["feature_date"].ge(start).all()
        and df["feature_date"].lt(end).all()
        and df["target_date"].gt(df["feature_date"]).all()
        and df["target_date"].lt(end).all()
    ):
        raise ValueError("Prediction dates cross the validation boundaries.")

    actual = df["actual_return"].to_numpy()
    predicted = df["prediction"].to_numpy()
    model_result = evaluate(actual, predicted)
    zero_result = evaluate(actual, np.zeros_like(actual))

    # Check full precision against the existing experiment, not rounded table values.
    for name, result in (("LIGHTGBM_FIXED", model_result), ("ZERO", zero_result)):
        for metric, value in result.items():
            expected = reference["validation_comparison"][name][metric]
            if not np.isclose(value, expected, rtol=0, atol=1e-12):
                raise ValueError(f"Metric mismatch: {name}.{metric}")

    model_rmse = model_result["rmse"]
    zero_rmse = zero_result["rmse"]
    if zero_rmse == 0:
        raise ValueError("Relative RMSE gain is undefined when ZERO RMSE is zero.")
    gain_pct = (zero_rmse - model_rmse) / zero_rmse * 100

    print("样本数：", len(df))
    print("LightGBM：", model_result)
    print("零预测：", zero_result)
    print(f"验证集 RMSE 相对改善：{gain_pct:.4f}%")
    if model_rmse < zero_rmse:
        print("模型更好")
    elif model_rmse > zero_rmse:
        print("零预测更好")
    else:
        print("两者 RMSE 相同")
    print("指标单位：收益率小数；换算为百分点请乘100。")
    print("REFERENCE_METRICS_MATCH: True")
    print("TEST_EVALUATED: False")


if __name__ == "__main__":
    main()
