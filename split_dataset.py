from pathlib import Path
import hashlib
import json
import pandas as pd

base = Path(__file__).resolve().parent
source = base / "data" / "sh600000_daily.csv"

df = pd.read_csv(
    source,
    parse_dates=["feature_date", "target_date"],
).sort_values("feature_date").reset_index(drop=True)

assert df["feature_date"].is_unique
assert not df.isna().any().any()
assert (df["target_date"] > df["feature_date"]).all()

valid_start = pd.Timestamp("2017-01-01")
test_start = pd.Timestamp("2019-01-01")

train_mask = (
    (df["feature_date"] < valid_start)
    & (df["target_date"] < valid_start)
)
valid_mask = (
    (df["feature_date"] >= valid_start)
    & (df["feature_date"] < test_start)
    & (df["target_date"] < test_start)
)
test_mask = df["feature_date"] >= test_start

parts = {
    "train": df.loc[train_mask].copy(),
    "validation": df.loc[valid_mask].copy(),
    "test": df.loc[test_mask].copy(),
}

assert all(not part.empty for part in parts.values())
assert parts["train"]["target_date"].max() < parts["validation"]["feature_date"].min()
assert parts["validation"]["target_date"].max() < parts["test"]["feature_date"].min()

output_dir = base / "data" / "splits"
output_dir.mkdir(parents=True, exist_ok=True)

summary = {
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "features": ["return_1d", "return_5d", "volume_ratio_5d"],
    "target": "target_return_1d",
    "validation_start": str(valid_start.date()),
    "test_start": str(test_start.date()),
    "purged_boundary_rows": len(df) - sum(len(part) for part in parts.values()),
    "splits": {},
}

for name, part in parts.items():
    part.to_csv(output_dir / f"{name}.csv", index=False)
    info = {
        "rows": len(part),
        "feature_start": str(part["feature_date"].min().date()),
        "feature_end": str(part["feature_date"].max().date()),
        "last_target_date": str(part["target_date"].max().date()),
    }
    summary["splits"][name] = info
    print(name.upper(), info)

report = base / "reports" / "split_summary.json"
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("PURGED_BOUNDARY_ROWS:", summary["purged_boundary_rows"])
print("SPLIT_SUMMARY:", report)
