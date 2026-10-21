"""Plot retained measured outcomes and actual particle states; no surrogate motion."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle

LABELS = {"adam": "Gradients", "coordinate": "Individual coordinate", "cem": "Cross-entropy",
          "grouped_coordinate": "Move whole path"}
COLORS = {"adam": "#2375bb", "coordinate": "#ef9a28", "cem": "#9864b4", "grouped_coordinate": "#278868"}


def summarize(data):
    result = {"methods": {}, "initial": {}, "aa": {}}
    initials = [next(t for t in data["trials"] if t["seed"] == seed)["history"][0]["candidate"] for seed in range(8)]
    result["initial"] = {"successes": sum(r["success"] for r in initials),
                         "median_coverage": float(np.median([r["coverage"] for r in initials])),
                         "median_spill": float(np.median([r["spill"] for r in initials]))}
    ratios = [r["ratio"] for r in data["aa"]]
    result["aa"] = {"minimum_ratio": min(ratios), "maximum_ratio": max(ratios),
                    "median_ratio": float(np.median(ratios)),
                    "within_predeclared_envelope": all(0.8 <= r <= 1.2 for r in ratios)}
    for method in data["contract"]["methods"]:
        rows = [t for t in data["trials"] if t["method"] == method]
        first = [next((r for r in t["history"] if r["best"]["success"]), None) for t in rows]
        successful = [r for r in first if r is not None]
        result["methods"][method] = {
            "budget_successes": sum(t["best"]["success"] for t in rows),
            "time_successes": sum(t["best_at_time_cutoff"]["success"] for t in rows),
            "median_budget_coverage": float(np.median([t["best"]["coverage"] for t in rows])),
            "median_budget_spill": float(np.median([t["best"]["spill"] for t in rows])),
            "median_time_coverage": float(np.median([t["best_at_time_cutoff"]["coverage"] for t in rows])),
            "median_time_spill": float(np.median([t["best_at_time_cutoff"]["spill"] for t in rows])),
            "median_search_seconds": float(np.median([t["wall_seconds"] for t in rows])),
            "median_budget_loss": float(np.median([t["best"]["loss"] for t in rows])),
            "first_success_count": len(successful),
            "median_first_success_units_among_successes": float(np.median([r["units"] for r in successful])) if successful else None,
            "median_first_success_seconds_among_successes": float(np.median([r["completed_seconds"] for r in successful])) if successful else None,
            "per_seed": [{"seed": t["seed"], "budget": t["best"], "time": t["best_at_time_cutoff"],
                          "first_success_units": f["units"] if f else None,
                          "first_success_seconds": f["completed_seconds"] if f else None}
                         for t, f in zip(rows, first, strict=True)],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    data = json.loads(args.receipt.read_text(encoding="utf-8"))
    if data["status"] != "complete":
        raise ValueError("Only a complete measured receipt may be plotted")
    directory = args.receipt.parent
    summary = summarize(data)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    methods = data["contract"]["methods"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    locations = np.arange(len(methods))
    names = [LABELS[m] for m in methods]
    for offset, boundary, title in ((-0.18, "best", "64 work units"), (0.18, "best_at_time_cutoff", "10 seconds")):
        counts = [sum(t[boundary]["success"] for t in data["trials"] if t["method"] == method) for method in methods]
        bars = axes[0].bar(locations + offset, counts, width=0.36, label=title)
        axes[0].bar_label(bars, padding=2)
    axes[0].set(ylabel="Constraint-satisfying outputs / 8 starts", ylim=(0, 9.2), title="Artist success: ≥90% target, ≤5% spill")
    axes[0].set_xticks(locations, names, rotation=20, ha="right")
    axes[0].legend(loc="upper right")
    for method in methods:
        trials = [t for t in data["trials"] if t["method"] == method]
        unit_grid = np.arange(2, data["contract"]["budget"] + 1, 2)
        time_grid = np.linspace(1, data["contract"]["time_limit"], 30)
        for ax, grid, key in ((axes[1], unit_grid, "units"), (axes[2], time_grid, "completed_seconds")):
            values = []
            for cutoff in grid:
                successes = 0
                for trial in trials:
                    available = [r for r in trial["history"] if r[key] <= cutoff]
                    successes += bool(available and available[-1]["best"]["success"])
                values.append(successes)
            ax.plot(grid, values, label=LABELS[method], color=COLORS[method], linewidth=2)
    axes[1].set(xlabel="Forward + reverse work units", ylabel="Successful selected outputs / 8", ylim=(-0.2, 8.5), title="Simulation budget")
    axes[2].set(xlabel="Elapsed seconds (completed evaluations)", ylabel="Successful selected outputs / 8", ylim=(-0.2, 8.5), title="Actual measured time")
    axes[2].legend(fontsize=8)
    fig.suptitle("Four-keyframe burst emitter • receiving aperture + protected surface • exploratory RTX4080SUPER")
    fig.tight_layout()
    chart = directory / "outcomes.png"
    fig.savefig(chart, dpi=160)
    plt.close(fig)

    chosen = [next(t for t in data["trials"] if t["seed"] == 0 and t["method"] == m)
              for m in ("adam", "grouped_coordinate")]
    collections = [data["initial_frames_seed0"]] + [t["frames"] for t in chosen]
    initials = next(t for t in data["trials"] if t["seed"] == 0)["history"][0]["candidate"]
    candidates = [initials] + [t["best"] for t in chosen]
    titles = ["Original artist keyframes", "Gradient result", "Move-whole-path result"]
    all_positions = np.concatenate([np.asarray(frame["positions"]) for frames in collections for frame in frames])
    minimum, maximum = all_positions.min(axis=0), all_positions.max(axis=0)
    x_limits = (min(-2.05, minimum[0] - 0.05), max(-0.63, maximum[0] + 0.05))
    y_limits = (min(-0.60, minimum[1] - 0.05), max(0.60, maximum[1] + 0.05))
    z_limits = (min(-0.01, minimum[2] - 0.01), max(0.60, maximum[2] + 0.05))
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    dots = []
    for column, (title, candidate) in enumerate(zip(titles, candidates, strict=True)):
        top, side = axes[0, column], axes[1, column]
        top.add_patch(Rectangle((-1.67, -0.22), 0.44, 0.44, fill=False, edgecolor="#15876b", linewidth=2,
                               label="Target aperture"))
        top.add_patch(Rectangle((-1.22, -0.40), 0.54, 0.80, facecolor="#d75353", alpha=0.13, edgecolor="#b22",
                               label="Protected volume"))
        for y in (-0.55, 0.55):
            top.add_patch(Circle((-0.85, y), 0.18, facecolor="#888", alpha=0.25))
        top.add_patch(Circle((0, 0), 0.55, facecolor="#888", alpha=0.25))
        keys = np.asarray(candidate["controls"]).reshape(4, 2) + [-1.45, 0]
        top.scatter(keys[:, 0], keys[:, 1], marker="x", s=40, c="black", label="Release keys")
        suffix = "" if column == 0 else " (64 units)"
        top.set(xlim=x_limits, ylim=y_limits, xlabel="x (m)", ylabel="y (m)",
                title=f"{title}{suffix}\nTarget {candidate['coverage']:.0%} • spill {candidate['spill']:.0%}")
        top.legend(loc="upper left", fontsize=6)
        top.set_aspect("equal")
        side.plot([-1.67, -1.23], [0.35, 0.35], color="#15876b", linewidth=3)
        side.add_patch(Rectangle((-1.22, 0.30), 0.54, 0.10, facecolor="#d75353", alpha=0.25))
        side.add_patch(Circle((0, 0.55), 0.55, facecolor="#888", alpha=0.25))
        side.axhline(0, color="gray", linewidth=1)
        side.set(xlim=x_limits, ylim=z_limits, xlabel="x (m)", ylabel="z (m)")
        dots.append((top.scatter([], [], s=10, c="#2375bb"), side.scatter([], [], s=10, c="#2375bb")))
    text = fig.suptitle("Actual retained SPH states • target counts first downward crossing • fluid continues below aperture")

    def update(frame):
        for column, frames in enumerate(collections):
            p = np.asarray(frames[frame]["positions"])
            dots[column][0].set_offsets(p[:, :2])
            dots[column][1].set_offsets(p[:, [0, 2]])
        step = collections[0][frame]["step"]
        text.set_text(f"Actual SPH • t={step * data['contract']['dt']:.3f}s • target counts first downward crossing; fluid continues")
        return [item for pair in dots for item in pair]

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    animation = FuncAnimation(fig, update, frames=len(collections[0]), interval=70)
    animation_path = directory / "artist-comparison.gif"
    animation.save(animation_path, writer=PillowWriter(fps=14), dpi=110)
    update(len(collections[0]) - 1)
    endpoint = directory / "artist-endpoints.png"
    fig.savefig(endpoint, dpi=150)
    plt.close(fig)
    manifest = {"receipt_sha256": hashlib.sha256(args.receipt.read_bytes()).hexdigest(),
                "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "artifacts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (chart, animation_path, endpoint)}}
    (directory / "plots.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Created", chart, animation_path, endpoint)


if __name__ == "__main__":
    main()
