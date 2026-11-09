from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

base = Path(__file__).resolve().parent
reports = base / "reports"
meta = json.loads((reports / "split_summary.json").read_text(encoding="utf-8"))
baseline = json.loads((reports / "baseline_validation.json").read_text(encoding="utf-8"))

features = meta["features"]
target = meta["target"]
train = pd.read_csv(base / "data/splits/train.csv")
valid = pd.read_csv(base / "data/splits/validation.csv")

assert features == baseline["features"]
assert len(train) == baseline["train_rows"]
assert len(valid) == baseline["validation_rows"]
assert np.isfinite(train[features + [target]].to_numpy(dtype=float)).all()
assert np.isfinite(valid[features + [target]].to_numpy(dtype=float)).all()

params = {
    "objective": "regression",
    "n_estimators": 100,
    "learning_rate": 0.03,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 50,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": 1,
    "verbosity": -1,
}
model = lgb.LGBMRegressor(**params)
model.fit(train[features], train[target])

def scores(actual, predicted):
    assert np.isfinite(predicted).all()
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }

train_scores = scores(train[target], model.predict(train[features]))
predicted = model.predict(valid[features])
valid_scores = scores(valid[target], predicted)

comparison = dict(baseline["models"])
comparison["LIGHTGBM_FIXED"] = valid_scores

print("VALIDATION COMPARISON:")
for name, result in comparison.items():
    print(
        f"{name}: MAE={result['mae'] * 100:.4f} pp, "
        f"RMSE={result['rmse'] * 100:.4f} pp"
    )

print(
    f"LIGHTGBM TRAIN: MAE={train_scores['mae'] * 100:.4f} pp, "
    f"RMSE={train_scores['rmse'] * 100:.4f} pp"
)

payload = {
    "primary_metric": "rmse",
    "evaluation_set": "validation",
    "test_evaluated": False,
    "features": features,
    "target": target,
    "parameters": params,
    "lightgbm_version": lgb.__version__,
    "lightgbm_train_scores": train_scores,
    "validation_comparison": comparison,
}
report = reports / "lightgbm_validation.json"
report.write_text(
    json.dumps(payload, indent=2, allow_nan=False),
    encoding="utf-8",
)

predictions = valid[["feature_date", "target_date"]].copy()
predictions["actual_return"] = valid[target]
predictions["prediction"] = predicted
predictions.to_csv(reports / "lightgbm_validation_predictions.csv", index=False)

model_dir = base / "artifacts"
model_dir.mkdir(exist_ok=True)
joblib.dump(model, model_dir / "lightgbm_fixed.joblib")

print("REPORT:", report)
print("TEST_EVALUATED: False")
