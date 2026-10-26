"""Run the P2.9 Phase 1 runtime gates (scripts/p29_runtime_gates.py) in Blender, one process at a time.

    python scripts/run_p29_runtime_gates.py --blender PATH [--backends opengl,vulkan] [--particles 100000,1000000]
                                            [--warp-folder DIR] [--out DIR]

Uses the showcase runner's isolated profile (this checkout's add-on copied in) and lease, and starts Blender with
``--enable-event-simulate`` so the modal-transform gates can drive a real G grab. Each run writes its summary.json
to <out>/<backend>-<particles>/; <out>/summary.json collects them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_showcase as rs  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "p29_runtime_gates.py"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--blender")
    p.add_argument("--backends", default="opengl,vulkan")
    p.add_argument("--particles", default="100000,1000000")
    p.add_argument("--warp-folder")
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--out")
    args = p.parse_args(argv)
    blender = rs.find_blender(args.blender)
    out = Path(args.out or rs.ROOT / "evidence" / "p29-runtime" / datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
    profile = rs.ROOT / "evidence" / "showcase" / "_profile"
    env = dict(**__import__("os").environ, **rs.profile_env(profile))
    warp = rs.find_warp_folder(args.warp_folder)
    if warp:
        env[rs.WARP_FOLDER_ENV] = warp
    results = []
    with rs.runner_lease(), rs._KeepAwake():
        rs.install_addon(profile)
        blend = rs.empty_blend(blender, profile, env)
        for backend in [b.strip() for b in args.backends.split(",") if b.strip()]:
            for count in [int(c) for c in args.particles.split(",") if c.strip()]:
                if not rs.wait_for_other_blender(3600):
                    sys.exit("another Blender is still running")
                folder = out / f"{backend}-{count}"
                folder.mkdir(parents=True, exist_ok=True)
                command = [blender, "--factory-startup", "--online-mode", "--enable-event-simulate",
                           "--gpu-backend", backend, "--gpu-vsync", "off", "--window-geometry", *rs.WINDOW,
                           str(blend), "--python", str(SCRIPT)]
                print(f"[p29] {backend} {count:,} particles ...", flush=True)
                t0 = time.monotonic()
                with (folder / "blender.log").open("w", encoding="utf-8", errors="replace") as log:
                    try:
                        code = subprocess.run(command, env=dict(env, **{rs.OUT_ENV: str(folder),
                                              "P29_PARTICLES": str(count)}), stdin=subprocess.DEVNULL, stdout=log,
                                              stderr=subprocess.STDOUT, timeout=args.timeout, check=False).returncode
                    except subprocess.TimeoutExpired:
                        code = None
                summary_path = folder / "summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
                failed = sorted(k for k, c in summary.get("checks", {}).items() if not c.get("ok"))
                results.append({"backend": backend, "particles": count, "exit_code": code,
                                "seconds": round(time.monotonic() - t0, 1), "ok": bool(summary.get("ok")) and code == 0,
                                "failed_checks": failed, "error": summary.get("error")})
                print(f"[p29] {backend} {count:,}: {'PASS' if results[-1]['ok'] else 'FAIL'} {failed}", flush=True)
    (out / "summary.json").write_text(json.dumps({"blender": blender, "runs": results}, indent=2), encoding="utf-8")
    print(f"[p29] results in {out}")
    return 0 if results and all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
