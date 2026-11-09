from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

base = Path(__file__).resolve().parent
meta = json.loads(
    (base / "reports" / "split_summary.json").read_text(encoding="utf-8")
)
features = meta["features"]
target = meta["target"]

train = pd.read_csv(base / "data" / "splits" / "train.csv")
valid = pd.read_csv(base / "data" / "splits" / "validation.csv")

assert len(train) == meta["splits"]["train"]["rows"]
assert len(valid) == meta["splits"]["validation"]["rows"]
assert np.isfinite(train[features + [target]].to_numpy(dtype=float)).all()
assert np.isfinite(valid[features + [target]].to_numpy(dtype=float)).all()

X_train, y_train = train[features], train[target]
X_valid, y_valid = valid[features], valid[target]

models = {
    "ZERO": DummyRegressor(strategy="constant", constant=0.0),
    "TRAIN_MEAN": DummyRegressor(strategy="mean"),
    "RIDGE": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
}

model_dir = base / "artifacts" / "baselines"
model_dir.mkdir(parents=True, exist_ok=True)

predictions = valid[["feature_date", "target_date"]].copy()
predictions["actual_return"] = y_valid
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predicted = model.predict(X_valid)
    assert np.isfinite(predicted).all()

    mae = float(mean_absolute_error(y_valid, predicted))
    rmse = float(np.sqrt(mean_squared_error(y_valid, predicted)))
    results[name] = {"mae": mae, "rmse": rmse}
    predictions[name] = predicted

    joblib.dump(model, model_dir / f"{name.lower()}.joblib")
    print(f"{name}: MAE={mae * 100:.4f} pp, RMSE={rmse * 100:.4f} pp")

report_dir = base / "reports"
report_dir.mkdir(parents=True, exist_ok=True)
report = report_dir / "baseline_validation.json"

payload = {
    "evaluation_set": "validation",
    "test_evaluated": False,
    "train_rows": len(train),
    "validation_rows": len(valid),
    "features": features,
    "target": target,
    "ridge_alpha": 1.0,
    "metric_units": "fractional return",
    "versions": {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    },
    "models": results,
}
report.write_text(
    json.dumps(payload, indent=2, allow_nan=False),
    encoding="utf-8",
)
predictions.to_csv(
    report_dir / "baseline_validation_predictions.csv",
    index=False,
)

print("REPORT:", report)
print("TEST_EVALUATED: False")
