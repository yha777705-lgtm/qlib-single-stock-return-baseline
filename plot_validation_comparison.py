"""Create an honest, validation-only comparison figure from saved reports."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    reports = root / "reports"
    report_path = reports / "lightgbm_validation.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    if payload["evaluation_set"] != "validation" or payload["test_evaluated"]:
        raise ValueError("This figure requires a validation-only report.")

    comparison = payload["validation_comparison"]
    model_names = ["ZERO", "TRAIN_MEAN", "RIDGE", "LIGHTGBM_FIXED"]
    if list(comparison) != model_names:
        raise ValueError("Unexpected model order in validation report.")

    rmse_pp = np.array([comparison[name]["rmse"] * 100 for name in model_names])
    zero_rmse = comparison["ZERO"]["rmse"]
    if zero_rmse <= 0:
        raise ValueError("ZERO RMSE must be positive to calculate relative gain.")
    gain_pct = (zero_rmse - np.array([comparison[name]["rmse"] for name in model_names])) / zero_rmse * 100

    if not np.isfinite(np.r_[rmse_pp, gain_pct]).all() or (rmse_pp < 0).any():
        raise ValueError("Metrics must be finite.")

    labels = ["Zero", "Train mean", "Ridge", "LightGBM"]
    colors = ["#0072B2", "#56B4E9", "#E69F00", "#D55E00"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.10, 1, 0.90))
    fig.suptitle(
        "Validation model comparison (SH600000, 2017-2018)",
        fontsize=15,
        fontweight="bold",
    )

    ax = axes[0]
    bars = ax.bar(labels, rmse_pp, color=colors, edgecolor="#333333", linewidth=0.6)
    ax.set_ylabel("RMSE (percentage points; lower is better)")
    ax.set_title("Validation RMSE")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(0, max(rmse_pp) * 1.18)
    for bar, value in zip(bars, rmse_pp):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(rmse_pp) * 0.025,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax = axes[1]
    bars = ax.bar(labels, gain_pct, color=colors, edgecolor="#333333", linewidth=0.6)
    ax.axhline(0, color="#222222", linewidth=0.9)
    ax.set_ylabel("RMSE gain vs zero prediction (%)")
    ax.set_title("Relative RMSE gain")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=20)
    low = min(gain_pct.min() * 1.35, -0.05)
    high = max(gain_pct.max() * 1.35, 0.12)
    ax.set_ylim(low, high)
    for bar, value in zip(bars, gain_pct):
        offset = 0.035 if value >= 0 else -0.035
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.4f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )

    fig.text(
        0.5,
        0.025,
        "Primary metric: RMSE. Test set was not evaluated; bars are descriptive, not uncertainty estimates.",
        ha="center",
        fontsize=9,
        color="#444444",
    )

    output = reports / "validation_model_comparison.png"
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)

    print("REPORT:", report_path)
    print("OUTPUT:", output)
    print("MODELS:", ", ".join(model_names))
    print("TEST_EVALUATED: False")


if __name__ == "__main__":
    main()
