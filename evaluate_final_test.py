"""Run the fixed final test evaluation exactly as recorded in TEST_PLAN.md."""

import hashlib
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
PLAN_PATH = REPORTS / "final_test_plan.json"
OUTPUT_REPORT = REPORTS / "final_test_evaluation.json"
OUTPUT_PREDICTIONS = REPORTS / "final_test_predictions.csv"


def evaluate(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Expected non-empty one-dimensional arrays with equal shape.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual values and predictions must be finite.")
    error = predicted - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
    }


def sha256(relative_path):
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


def main():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if plan["status"] not in {"planned_not_evaluated", "evaluated"}:
        raise ValueError("The final test plan has an invalid status.")
    if plan["test_split"]["rows"] != 478:
        raise ValueError("Unexpected test row count in the plan.")

    # Verify every artifact and source file recorded before test scoring.
    for relative_path, expected_hash in plan["input_sha256"].items():
        actual_hash = sha256(relative_path)
        if actual_hash != expected_hash:
            raise ValueError(f"Input fingerprint changed: {relative_path}")

    split = json.loads((REPORTS / "split_summary.json").read_text(encoding="utf-8"))
    if split["features"] != plan["features"] or split["target"] != plan["target"]:
        raise ValueError("Feature or target definition differs from the fixed plan.")
    if split["splits"]["test"] != plan["test_split"]:
        raise ValueError("Test split differs from the fixed plan.")

    test = pd.read_csv(
        ROOT / "data/splits/test.csv",
        parse_dates=["feature_date", "target_date"],
    )
    features = plan["features"]
    target = plan["target"]
    if len(test) != plan["test_split"]["rows"]:
        raise ValueError("Test row count mismatch.")
    if not test["feature_date"].is_unique:
        raise ValueError("Duplicate test feature dates.")
    if not np.isfinite(test[features + [target]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite test values.")
    test_start = pd.Timestamp(split["test_start"])
    if not (
        test["feature_date"].ge(test_start).all()
        and test["target_date"].gt(test["feature_date"]).all()
    ):
        raise ValueError("Test date boundaries are invalid.")

    actual = test[target].to_numpy(dtype=float)
    predictions = test[["feature_date", "target_date", target]].rename(
        columns={target: "actual_return"}
    )
    models = {}
    models["ZERO"] = joblib.load(ROOT / "artifacts/baselines/zero.joblib")
    models["TRAIN_MEAN"] = joblib.load(ROOT / "artifacts/baselines/train_mean.joblib")
    models["RIDGE"] = joblib.load(ROOT / "artifacts/baselines/ridge.joblib")
    models["LIGHTGBM_FIXED"] = joblib.load(ROOT / "artifacts/lightgbm_fixed.joblib")

    results = {}
    for name, model in models.items():
        if name in {"ZERO", "TRAIN_MEAN", "RIDGE"}:
            predicted = model.predict(test[features])
        else:
            predicted = model.predict(test[features])
        predicted = np.asarray(predicted, dtype=float)
        result = evaluate(actual, predicted)
        results[name] = {
            **result,
            "rmse_gain_vs_zero_pct": None,
        }
        predictions[name] = predicted

    zero_rmse = results["ZERO"]["rmse"]
    if zero_rmse <= 0:
        raise ValueError("ZERO RMSE must be positive.")
    for result in results.values():
        result["rmse_gain_vs_zero_pct"] = (
            zero_rmse - result["rmse"]
        ) / zero_rmse * 100

    ranking = sorted(results, key=lambda name: results[name]["rmse"])
    output = {
        "evaluation_set": "test",
        "test_evaluated": True,
        "evaluation_date": "2026-11-14",
        "source_commit": plan["source_commit"],
        "current_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "features": features,
        "target": target,
        "rows": len(test),
        "feature_date_range": {
            "start": str(test["feature_date"].min().date()),
            "end": str(test["feature_date"].max().date()),
        },
        "target_date_range": {
            "start": str(test["target_date"].min().date()),
            "end": str(test["target_date"].max().date()),
        },
        "primary_metric": plan["primary_metric"],
        "secondary_metric": plan["secondary_metric"],
        "metric_units": {
            "stored": plan["stored_metric_units"],
            "display": plan["display_metric_units"],
        },
        "models": results,
        "rmse_ranking": ranking,
        "model_selection_after_test": False,
        "refit_on_validation": plan["refit_on_validation"],
        "rolling_retraining": plan["rolling_retraining"],
        "uncertainty_assessed": plan["uncertainty_assessment_planned"],
        "interpretation": "Descriptive result for one instrument and one fixed historical test period; no significance, stability, or trading-profit claim.",
        "input_sha256": plan["input_sha256"],
    }
    OUTPUT_REPORT.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    predictions.to_csv(OUTPUT_PREDICTIONS, index=False)

    print("TEST_ROWS:", len(test))
    print("MODEL  MAE_PP  RMSE_PP  RMSE_GAIN_VS_ZERO_PCT")
    for name in ["ZERO", "TRAIN_MEAN", "RIDGE", "LIGHTGBM_FIXED"]:
        result = results[name]
        print(
            f"{name}  {result['mae'] * 100:.6f}  "
            f"{result['rmse'] * 100:.6f}  "
            f"{result['rmse_gain_vs_zero_pct']:+.4f}%"
        )
    print("RMSE_RANKING:", " < ".join(ranking))
    print("MODEL_SELECTION_AFTER_TEST: False")
    print("REPORT:", OUTPUT_REPORT)


if __name__ == "__main__":
    main()
