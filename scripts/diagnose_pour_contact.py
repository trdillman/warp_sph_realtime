"""One bounded diagnostic of corrected contacts; preserve a fresh JSON receipt."""
import argparse
import json
from pathlib import Path

import numpy as np

from warp_sph_realtime.experimental.differentiable_pour import PourExperiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("evidence/contact-correction-20261017/diagnostics-refined.json"))
    args = parser.parse_args()
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    report = {"checks": [], "witness": None, "parity": []}
    experiment = PourExperiment(384, "reachable")
    witness = experiment.evaluate([0, 0])
    report["witness"] = {"controls": [0, 0], "loss": witness["loss"], "coverage": witness["coverage"],
                         "trajectory": experiment.trajectory_diagnostics()}
    print("witness", report["witness"], flush=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for controls in ([0, 0], [0.7, -0.2], [1.4, 0.4]):
        for epsilon in (0.002, 0.001):
            check = experiment.check_gradient(controls, epsilon)
            report["checks"].append(check)
            print(check, flush=True)
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    retained = experiment.evaluate([0.7, -0.2])
    report["trajectory"] = experiment.trajectory_diagnostics()
    for graph in (False, True):
        p, v = experiment.production_rollout([0.7, -0.2], graph)
        report["parity"].append({"graph": graph,
                                "position_max_abs": float(np.max(np.abs(retained["positions"] - p))),
                                "velocity_max_abs": float(np.max(np.abs(retained["velocities"] - v)))})
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("trajectory", report["trajectory"], flush=True)


if __name__ == "__main__":
    main()
