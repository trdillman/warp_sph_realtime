"""Plot saved receipts without rerunning physics: python -m scripts.plot_differentiable_pour."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


def main():
    root = Path("evidence/differentiable-pour-20261017")
    floor = json.loads((root / "results.json").read_text(encoding="utf-8"))
    deflected = json.loads((root / "deflected.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {"adam": "#d97706", "random": "#64748b", "coordinate": "#2563eb"}
    names = {"adam": "Gradients (Adam)", "random": "Random search", "coordinate": "Coordinate search"}
    for axis, report, title in zip(axes[0], (floor, deflected),
                                   ("Floor: gradients pass; translation-only task",
                                    "Obstacles: 3/6 gradient checks fail"), strict=True):
        for method, color in colors.items():
            traces = []
            units = np.arange(2, 25)
            for trial in report["trials"]:
                if trial["method"] == method:
                    traces.append([min(row["loss"] for row in trial["history"] if row["units"] <= n)
                                   for n in units])
            values = np.array(traces)
            axis.plot(units, np.median(values, axis=0), label=names[method], color=color)
            axis.fill_between(units, values.min(axis=0), values.max(axis=0), color=color, alpha=0.10)
        axis.set(title=title, xlabel="Nominal work units (forward + backward)",
                 ylabel="Mean squared horizontal distance (m²)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    for axis, vertical, title in zip(axes[1], (1, 2),
                                     ("Obstacle case, seed 0: top view", "Obstacle case, seed 0: side view"), strict=True):
        trials = [t for t in deflected["trials"] if t["seed"] == 0]
        initial = np.array(trials[0]["initial_positions"])
        axis.scatter(initial[:, 0], initial[:, vertical], c="#aaaaaa", s=10, alpha=0.5,
                     label="Unoptimized endpoint")
        for trial in trials:
            if trial["method"] in ("adam", "coordinate"):
                p = np.array(trial["final_positions"])
                axis.scatter(p[:, 0], p[:, vertical], color=colors[trial["method"]], s=12,
                             alpha=0.7, label=names[trial["method"]])
        bottom = 0.02 if vertical == 1 else 0.0
        axis.add_patch(Rectangle((0.12, bottom), 0.36, 0.36 if vertical == 1 else 0.18,
                                facecolor="#16a34a", edgecolor="#16a34a", alpha=0.18, label="Target region"))
        if vertical == 2:
            axis.add_patch(Circle((0, 0.55), 0.55, fill=False, linestyle="--", color="#555555",
                                  label="Sphere section"))
            axis.axhline(0, color="#333333", linewidth=1)
        axis.set(title=title, xlabel="x (m)", ylabel="y (m)" if vertical == 1 else "z (m)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.2)
    fig.suptitle("Current WCSPH solver: differentiability proof of concept\n"
                 "64 particles • five paired starts • 24 work units • bands show min–max, not confidence intervals",
                 fontsize=13)
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
