"""Run the gravity regression through the showcase isolation/lease harness."""
import argparse
import os
import subprocess
from pathlib import Path

import render_showcase as harness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    with harness.runner_lease():
        if harness.other_blender_running():
            raise RuntimeError("another Blender is running; gravity probe not launched")
        if out.exists():
            raise RuntimeError("use a fresh output directory")
        out.mkdir(parents=True)
        env = dict(os.environ, **harness.profile_env(out / "_profile"), GRAVITY_PROBE_OUT=str(out))
        env["PYTHONPATH"] = ""
        import warp
        env["GRAVITY_WARP_PARENT"] = str(Path(warp.__file__).parent.parent)
        result = subprocess.run(
            [args.blender, "--background", "--factory-startup", "--python-exit-code", "1",
             "--python", str(harness.ROOT / "tests" / "runtime_cache_gravity.py")],
            env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
        (out / "blender.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        print(result.stdout[-3000:])
        print(result.stderr[-1000:])
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
