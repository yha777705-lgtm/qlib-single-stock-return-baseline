from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

base = Path(__file__).resolve().parent
meta = json.loads(
    (base / "reports/split_summary.json").read_text(encoding="utf-8")
)
model = joblib.load(base / "artifacts/lightgbm_fixed.joblib")
features = meta["features"]
target = meta["target"]

results = {}

for name in ["train", "validation"]:
    data = pd.read_csv(base / "data/splits" / f"{name}.csv")
    assert len(data) == meta["splits"][name]["rows"]

    actual = data[target].to_numpy(dtype=float)
    predicted = model.predict(data[features])
    assert np.isfinite(actual).all()
    assert np.isfinite(predicted).all()

    zero_rmse = float(np.sqrt(np.mean(actual ** 2)))
    model_rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    assert zero_rmse > 0

    gain_pct = (zero_rmse - model_rmse) / zero_rmse * 100

    results[name] = {
        "rows": len(data),
        "return_std": float(np.std(actual, ddof=0)),
        "zero_rmse": zero_rmse,
        "lightgbm_rmse": model_rmse,
        "rmse_gain_pct": gain_pct,
    }

    print(name.upper())
    print(f"  RETURN_STD: {results[name]['return_std'] * 100:.4f} pp")
    print(f"  ZERO_RMSE: {zero_rmse * 100:.4f} pp")
    print(f"  LIGHTGBM_RMSE: {model_rmse * 100:.4f} pp")
    print(f"  RMSE_GAIN: {gain_pct:+.4f}%")

report = base / "reports/relative_baseline_diagnostics.json"
report.write_text(
    json.dumps({
        "new_training_performed": False,
        "test_evaluated": False,
        "results": results,
    }, indent=2, allow_nan=False),
    encoding="utf-8",
)
print("REPORT:", report)
