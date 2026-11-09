
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

df = D.features(
    ["SH600000"],
    ["$close", "$volume"],
    start_time="2020-01-02",
    end_time="2020-01-10",
    freq="day",
)

print(df)
print("ROWS:", len(df))
print("NON_NULL:", df.notna().sum().to_dict())
