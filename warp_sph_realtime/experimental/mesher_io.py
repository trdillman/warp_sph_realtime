import subprocess

import numpy as np
import warp as wp


def export_particles_xyz(pos_wp: wp.array, filepath: str):
    """Exports GPU particle positions to disk for SplashSurf meshing or VTK inspection."""
    pts = pos_wp.numpy()
    np.savetxt(filepath, pts, fmt="%.5f")

def run_splashsurf_cli(input_xyz: str, output_obj: str, particle_radius: float = 0.02):
    """Invokes SplashSurf surface reconstruction CLI to produce a watertight mesh."""
    cmd = [
        "splashsurf", "reconstruct", input_xyz,
        "-o", output_obj,
        f"--particle-radius={particle_radius}",
        "--smoothing-length=2.0",
        "--cube-size=0.75",
    ]
    return subprocess.run(cmd, check=False, capture_output=True, text=True)
