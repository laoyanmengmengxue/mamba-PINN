from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "editor2_formal_training_summary.csv"
OUT = ROOT / "editor2_formal_long_window_summary.png"


def read_rows() -> list[dict[str, str]]:
    with CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    rows = read_rows()
    cases = ["single", "three"]
    horizons = ["5_7", "5_10", "5_15"]
    labels = ["65-67 s", "65-70 s", "65-75 s"]
    colors = {"PINN": "#A7B6C2", "MI-PINN": "#2D6F8E"}

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.6), dpi=200)
    for ci, case in enumerate(cases):
        ax_r2 = axes[ci, 0]
        ax_rmse = axes[ci, 1]
        x = np.arange(len(horizons))
        width = 0.34
        for offset, method in [(-width / 2, "PINN"), (width / 2, "MI-PINN")]:
            row = next(r for r in rows if r["case"] == case and r["method"] == method)
            r2 = [float(row[f"speed_r2_{h}"]) for h in horizons]
            rmse = [float(row[f"speed_rmse_{h}"]) for h in horizons]
            ax_r2.bar(x + offset, r2, width=width, label=method, color=colors[method], edgecolor="white", linewidth=0.6)
            ax_rmse.bar(x + offset, rmse, width=width, label=method, color=colors[method], edgecolor="white", linewidth=0.6)
        ax_r2.axhline(0, color="#333333", lw=0.7)
        ax_r2.set_xticks(x, labels)
        ax_rmse.set_xticks(x, labels)
        ax_r2.set_ylabel("Speed-field $R^2$")
        ax_rmse.set_ylabel("Speed RMSE")
        ax_r2.set_title(f"{case.capitalize()} cylinder: $R^2$")
        ax_rmse.set_title(f"{case.capitalize()} cylinder: RMSE")
        ax_r2.grid(axis="y", alpha=0.22, lw=0.5)
        ax_rmse.grid(axis="y", alpha=0.22, lw=0.5)
        ax_r2.legend(frameon=False, fontsize=8)
        ax_rmse.legend(frameon=False, fontsize=8)
        ax_r2.set_ylim(min(-3.0, ax_r2.get_ylim()[0]), 1.08)
    fig.suptitle("Independent COMSOL long-window follow-up: PINN vs. MI-PINN", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
