"""Localize a stable derivative discrepancy without changing physics/tolerances."""
import json
from pathlib import Path

from warp_sph_realtime.experimental.differentiable_pour import PourExperiment


def main():
    path = Path("evidence/contact-correction-20261017/horizon.json")
    if path.exists():
        raise FileExistsError(path)
    rows = []
    for steps in (64, 128, 192, 256, 320, 384):
        experiment = PourExperiment(steps, "reachable")
        checks = [experiment.check_gradient([1.4, 0.4], eps) for eps in (0.001, 0.0001)]
        row = {"steps": steps, "checks": checks, "trajectory": experiment.trajectory_diagnostics()}
        rows.append(row)
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(row, flush=True)


if __name__ == "__main__":
    main()
