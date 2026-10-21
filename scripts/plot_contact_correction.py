"""Render charts and an actual retained-particle animation from the corrected receipt."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle


def main():
    root = Path("evidence/contact-correction-20261017")
    report = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    names = {"adam": "Automatic gradients", "fd_adam": "Finite-difference gradients",
             "random": "Random search", "coordinate": "Coordinate search"}
    colors = {"adam": "#d97706", "fd_adam": "#9333ea", "random": "#64748b", "coordinate": "#2563eb"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for method, label in names.items():
        trials = [t for t in report["trials"] if t["method"] == method]
        units = np.arange(5, 65)
        loss, coverage = [], []
        for trial in trials:
            loss.append([min(r["loss"] for r in trial["history"] if r["units"] <= n) for n in units])
            coverage.append([min((r for r in trial["history"] if r["units"] <= n),
                                 key=lambda r: r["loss"])["coverage"] for n in units])
        axes[0].plot(units, np.median(loss, axis=0), label=label, color=colors[method])
        axes[1].plot(units, np.median(coverage, axis=0) * 100, label=label, color=colors[method])
    axes[0].set(title="Match the receiving region", ylabel="Median objective (lower is better)", yscale="log")
    axes[1].set(title="Actual particles delivered", ylabel="Median coverage (%)", ylim=(0, 105))
    axes[1].axhline(75, color="#16a34a", linestyle="--", label="Success threshold: 75%")
    for axis in axes:
        axis.set_xlabel("Nominal work units: forward + backward")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Corrected contact and loop replay • 8 paired starts • identical forward physics", fontsize=12)
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)

    trials = {t["method"]: t for t in report["trials"] if t["seed"] == 0}
    datasets = [trials["adam"]["initial_frames"], trials["adam"]["selected_frames"],
                trials["coordinate"]["selected_frames"]]
    titles = ["Starting controls", "Automatic gradients", "Coordinate search"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    scatters = []
    for column, title in enumerate(titles):
        for row, vertical in enumerate((1, 2)):
            axis = axes[row, column]
            scatter = axis.scatter([], [], s=13, color=("#64748b", "#d97706", "#2563eb")[column])
            scatters.append((scatter, column, vertical))
            axis.add_patch(Rectangle((-1.02, -0.22 if vertical == 1 else 0), 0.44,
                                     0.44 if vertical == 1 else 0.18, color="#16a34a", alpha=0.2))
            axis.add_patch(Circle((0, 0 if vertical == 1 else 0.55), 0.55,
                                  fill=False, linestyle="--", color="#555555"))
            if vertical == 1:
                for x in (-0.85, 0.85):
                    for y in (-0.55, 0.55):
                        axis.add_patch(Circle((x, y), 0.18, fill=False, color="#555555"))
            else:
                axis.axhline(0, color="#555555", linewidth=1)
            axis.set(xlim=(-1.4, 0.6), ylim=(-0.8, 0.8) if vertical == 1 else (-0.03, 1.2),
                     xlabel="x (m)", ylabel="y (m)" if vertical == 1 else "z (m)",
                     title=title + (" — top" if vertical == 1 else " — side projection"))
            axis.set_aspect("equal")
    title_text = fig.suptitle("Seed 0: actual simulation states • green = receiving region")

    def update(frame):
        for scatter, column, vertical in scatters:
            p = np.asarray(datasets[column][frame])
            scatter.set_offsets(p[:, [0, vertical]])
        title_text.set_text(f"Seed 0 • simulated t={frame * 12 * report['workload']['dt']:.3f}s "
                            "• playback slowed for inspection")
        return [item[0] for item in scatters]

    animation = FuncAnimation(fig, update, frames=len(datasets[0]), interval=100)
    animation.save(root / "pour-comparison.gif", writer=PillowWriter(fps=10))
    update(len(datasets[0]) - 1)
    fig.savefig(root / "pour-endpoints.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
