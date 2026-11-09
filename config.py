import os
from pathlib import Path

# 可以通过环境变量指定数据位置；默认使用用户的 Downloads 目录。
DATA_DIR = Path(
    os.environ.get(
        "QLIB_DATA_DIR",
        str(Path.home() / "Downloads" / "qlib_data_simple_cn_1d_0.9"),
    )
).expanduser()

for folder in ("calendars", "features"):
    if not (DATA_DIR / folder).is_dir():
        raise FileNotFoundError(
            f"Missing: {DATA_DIR / folder}. "
            "Set QLIB_DATA_DIR to your Qlib data directory."
        )