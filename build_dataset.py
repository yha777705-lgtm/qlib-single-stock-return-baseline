from pathlib import Path
import numpy as np
import pandas as pd
import qlib
from qlib.data import D
from config import DATA_DIR

qlib.init(
    provider_uri=str(DATA_DIR),
    region="cn",
    kernels=1,
    expression_cache=None,
    dataset_cache=None,
)

prices = D.features(
    ["SH600000"],
    ["$close", "$volume"],
    start_time="2010-01-01",
    end_time="2020-12-31",
    freq="day",
).reset_index().sort_values("datetime").reset_index(drop=True)

close = prices["$close"].where(prices["$close"] > 0)
volume = prices["$volume"]

samples = pd.DataFrame({
    "feature_date": prices["datetime"],
    "target_date": prices["datetime"].shift(-1),
    "return_1d": close.pct_change(fill_method=None),
    "return_5d": close.pct_change(periods=5, fill_method=None),
    "volume_ratio_5d": volume / volume.rolling(5).mean() - 1,
    "target_return_1d": close.shift(-1) / close - 1,
})

samples = samples.replace([np.inf, -np.inf], np.nan)
invalid_rows = samples.isna().any(axis=1)

print("RAW_MISSING:", prices[["$close", "$volume"]].isna().sum().to_dict())
print("MISSING_BY_COLUMN:", samples.isna().sum().to_dict())
print("INVALID_ROWS:", int(invalid_rows.sum()))
print(
    "FIRST_INVALID_DATES:",
    samples.loc[invalid_rows, "feature_date"]
    .head(10).dt.strftime("%Y-%m-%d").tolist()
)

samples = samples.dropna()

assert not samples.empty, "No usable samples."
assert samples["feature_date"].is_unique, "Duplicate dates."
assert (samples["target_date"] > samples["feature_date"]).all()

output = Path(__file__).resolve().parent / "data" / "sh600000_daily.csv"
output.parent.mkdir(parents=True, exist_ok=True)
samples.to_csv(output, index=False)

print(samples.head(3).to_string(index=False))
print("RAW_ROWS:", len(prices))
print("CLEAN_ROWS:", len(samples))
print("REMOVED_ROWS:", len(prices) - len(samples))
print("SAVED_TO:", output)
