"""Dam-break validation sweep: front vs Martin & Moyce (1952), energy, conservation and rest density.

Run from the repository root:  python -m scripts.validate_dambreak [--device cuda:0] [--output DIR]
Writes dambreak.png and summary.json (default evidence/physics-validation-<date>/, which is gitignored).
Method and the recorded numbers: docs/VALIDATION.md.
"""

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

from warp_sph_realtime.sim.sim_core import ARTIFICIAL_VISCOSITY_ALPHA, RealtimeWCSPHSolver
from warp_sph_realtime.sim.validation import (
    MARTIN_MOYCE_T,
    MARTIN_MOYCE_Z,
    DamBreakConfig,
    compare_front,
    conservation_report,
    rest_density_report,
    run_dam_break,
    run_rest,
)

DEFAULT_RESOLUTIONS = (6, 8, 12, 16, 24, 32, 48)  # particles across the column width
ENERGY_PLOT_CELLS = 24


def solver_parameters(device: str) -> dict:
    probe = RealtimeWCSPHSolver(num_particles=1, h=0.04, enable_obstacles=0, device=device)
    return {
        "rest_density": probe.rest_density,
        "stiffness": probe.stiffness,
        "sound_speed": float((7.0 * probe.stiffness / probe.rest_density) ** 0.5),
        "viscosity": probe.viscosity,
        "artificial_viscosity_alpha": float(ARTIFICIAL_VISCOSITY_ALPHA),
        "cfl_factor": probe.cfl_factor,
        "substeps": probe.substeps,
        "kernel": "cubic spline, h = 2 x spacing",
        "boundary": "soft wall layer (width = spacing) plus hard clamp, no wall friction",
    }


def plot(runs: dict, comparisons: dict, rest, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    front, error, energy, settle = axes.flat
    colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, len(runs)))
    for color, (cells, run) in zip(colors, runs.items(), strict=True):
        front.plot(run.T, run.Z, color=color, label=f"{cells} cells ({run.num_particles} particles)")
        error.plot(comparisons[cells]["T"], comparisons[cells]["error"], "o-", ms=3, color=color, label=f"{cells} cells")
    front.plot(MARTIN_MOYCE_T, MARTIN_MOYCE_Z, "ko", label="Martin & Moyce 1952, n$^2$=2")
    front.set(xlabel="T = t sqrt(2g/a)", ylabel="Z = x/a", title="Surge front")
    error.axhline(0.0, color="k", lw=0.8)
    error.set(xlabel="T", ylabel="Z$_{sim}$ - Z$_{ref}$ (positive = ahead of experiment)", title="Front error")
    front.legend(fontsize=7)
    error.legend(fontsize=7, ncol=2)
    ref = runs.get(ENERGY_PLOT_CELLS) or next(reversed(runs.values()))
    e0 = ref.energy[0]
    energy.plot(ref.T, ref.kinetic / e0, label="kinetic")
    energy.plot(ref.T, ref.potential / e0, label="potential")
    energy.plot(ref.T, ref.energy / e0, "k", label="total")
    energy.set(xlabel="T", ylabel="energy / initial energy", title=f"Dam-break energy ({ref.num_particles} particles)")
    energy.legend(fontsize=8)
    settle.plot(rest.time, rest.kinetic / rest.energy[0], label="kinetic")
    settle.plot(rest.time, rest.energy / rest.energy[0], "k", label="total")
    dev = rest_density_report(rest)
    settle.set(xlabel="t [s]", ylabel="energy / initial energy",
               title=f"Settling block ({rest.num_particles} particles): bulk |rho/rho0-1| = "
                     f"{100 * dev['mean_abs_density_error_bulk']:.2f} %")
    settle.legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.suptitle("warp_sph_realtime WCSPH validation", fontsize=13)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0" if wp.is_cuda_available() else "cpu")
    parser.add_argument("--resolutions", type=int, nargs="+", default=list(DEFAULT_RESOLUTIONS),
                        help="particles across the column width for each dam-break run")
    parser.add_argument("--thickness", type=int, default=4, help="particles across the slab")
    parser.add_argument("--rest-cells", type=int, default=16)
    parser.add_argument("--rest-seconds", type=float, default=1.0)
    parser.add_argument("--output", type=Path,
                        default=Path(f"evidence/physics-validation-{datetime.now(UTC):%Y%m%d}"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    runs, comparisons, per_run = {}, {}, {}
    for cells in args.resolutions:
        run = run_dam_break(DamBreakConfig(cells_across=cells, thickness_cells=args.thickness, device=args.device))
        runs[cells] = run
        comparisons[cells] = compare_front(run.T, run.Z)
        c = comparisons[cells]
        per_run[str(cells)] = {
            "particles": run.num_particles, "spacing_m": run.spacing, "steps": run.steps,
            "wall_seconds": run.wall_seconds, "front": c, "conservation": conservation_report(run),
            "T": run.T.tolist(), "Z": run.Z.tolist(),
        }
        print(f"{cells:3d} cells {run.num_particles:6d} particles  mean err {c['mean_error']:+.3f}  "
              f"rms {c['rms_error']:.3f}  max |err|/Z {100 * c['max_abs_relative_error']:.1f} %  "
              f"E_final/E0 {per_run[str(cells)]['conservation']['final_energy_fraction']:.3f}  "
              f"({run.wall_seconds:.1f} s)", flush=True)

    rest = run_rest(cells=args.rest_cells, seconds=args.rest_seconds, device=args.device)
    rest_summary = {
        "particles": rest.num_particles, "seconds": args.rest_seconds, "spacing_m": rest.spacing,
        "density": rest_density_report(rest), "conservation": conservation_report(rest),
    }
    print("rest block:", json.dumps(rest_summary["density"]), flush=True)

    plot(runs, comparisons, rest, args.output / "dambreak.png")
    summary = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "device": args.device,
        "device_name": wp.get_device(args.device).name,
        "warp": wp.__version__,
        "python": platform.python_version(),
        "solver": solver_parameters(args.device),
        "reference": {
            "source": "Martin & Moyce 1952, Phil. Trans. R. Soc. A 244:312-324, n^2 = 2 surge front "
                      "(values via Ocellaris scripts/plot_dam_break.py, cross-checked against PySPH digitisations)",
            "T": MARTIN_MOYCE_T.tolist(), "Z": MARTIN_MOYCE_Z.tolist(),
        },
        "dam_break": {"thickness_cells": args.thickness, "runs": per_run},
        "rest": rest_summary,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {args.output / 'dambreak.png'} and {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
