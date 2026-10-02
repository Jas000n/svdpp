"""Plot the 5-fold mean test RMSE/MAE per epoch from results/*.json into curve.png."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = [  # (label, results file, color)
    ("SVD++", ROOT / "results" / "svdpp.json", "#2a78d6"),
    ("Baseline", ROOT / "results" / "baseline.json", "#eb6834"),
]
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
    for label, path, color in RUNS:
        folds = json.loads(path.read_text())["folds"]
        n_epochs = sum(len(f["train_time"]) for f in folds)
        train_time = sum(sum(f["train_time"]) for f in folds)
        test_time = sum(sum(f["test_time"]) for f in folds)
        print(f"{label}: total training time = {train_time:.3f}s, per epoch = {train_time / n_epochs:.4f}s; "
              f"total test time = {test_time:.3f}s, per evaluation = {test_time / (n_epochs + len(folds)):.4f}s")
        for ax, metric in zip(axes, ["rmse", "mae"]):
            curve = np.mean([f[metric] for f in folds], axis=0)
            ax.plot(curve, color=color, lw=2, marker="o", ms=4, label=label)
            ax.annotate(f"{curve[-1]:.4f}", (len(curve) - 1, curve[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", color=TEXT, fontsize=9)
            print(f"  {metric.upper()} = {curve[-1]:.4f}")

    for ax, name in zip(axes, ["RMSE", "MAE"]):
        ax.set_facecolor(SURFACE)
        ax.set_title(f"Test {name} (mean of 5 folds)", color=TEXT, loc="left")
        ax.set_xlabel("epoch", color=MUTED)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.tick_params(colors=MUTED)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.margins(x=0.12)
        ax.legend(frameon=False, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(ROOT / "curve.png", dpi=150)


if __name__ == "__main__":
    main()
