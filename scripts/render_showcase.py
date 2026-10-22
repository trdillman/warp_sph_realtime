"""Run the showcase demos (examples/showcase/) in Blender and collect their media.

    python scripts/render_showcase.py [--demos display,presets] [--backends vulkan,opengl] [--blender PATH]
                                      [--warp-folder DIR] [--timeout SECONDS] [--out DIR]

Each demo runs in its own Blender process, one at a time (the runner waits while another Blender is running, as
two GPU simulations would compete for the same card). The display demo runs once per GPU backend. Blender gets an
isolated user profile (evidence/showcase/_profile) with this checkout's add-on copied into it; Warp comes from
``--warp-folder`` (the add-on's "Local Folder" Warp source), else from WARPSPH_SHOWCASE_WARP_FOLDER, else from the
Python running this script. Blender is located from --blender, then the BLENDER environment variable, then PATH.

Outputs go to evidence/showcase/<timestamp>/ (gitignored): <demo>/ (display/<backend>/) with each demo's media and
summary.json, a summary.json for the whole run, contact_sheet.png with one key frame per demo, and an mp4 per
frame sequence when ffmpeg is on PATH (otherwise the PNG sequences stay as they are).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOWCASE = ROOT / "examples" / "showcase"
DEMOS = ("display", "validation", "emitter_mixer", "bake_mesh_export", "presets")
BACKENDS = ("vulkan", "opengl")
PER_BACKEND = {"display"}           # demos run once per GPU backend
DEFAULT_TIMEOUT = 1200              # seconds per Blender process
MIXER_TIMEOUT = 2700                # 2 x 900 s bakes, 600 s media, 300 s setup/teardown
ENCODE_TIMEOUT = 300
WARP_FOLDER_ENV = "WARPSPH_SHOWCASE_WARP_FOLDER"
OUT_ENV = "SHOWCASE_OUT"
TEXT_SUFFIXES = (".py", ".json", ".toml", ".txt", ".glsl")
WINDOW = ("0", "0", "1600", "900")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--demos", default=",".join(DEMOS), help=f"Comma-separated subset of {', '.join(DEMOS)}")
    parser.add_argument("--backends", default=",".join(BACKENDS),
                        help="GPU backends for the display demo; the other demos use the first one")
    parser.add_argument("--blender", help="Path to the Blender 5.2 executable")
    parser.add_argument("--warp-folder", help="Folder that contains the 'warp' package (Warp 1.17)")
    parser.add_argument("--timeout", type=float,
                        help="Seconds per Blender run (default 1200; emitter_mixer 2700 for its paired bakes)")
    parser.add_argument("--wait", type=float, default=3600.0,
                        help="Seconds to wait for another running Blender to exit before giving up")
    parser.add_argument("--out", help="Output folder (default evidence/showcase/<timestamp>)")
    args = parser.parse_args(argv)
    args.demos = _choices(args.demos, DEMOS, "demo", parser)
    args.backends = _choices(args.backends, BACKENDS, "backend", parser)
    if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("--timeout must be finite and positive")
    if not math.isfinite(args.wait) or args.wait < 0:
        parser.error("--wait must be finite and nonnegative")
    return args


def job_timeout(demo: str, requested: float | None) -> float:
    """Explicit user deadlines win; the paired-bake demo gets its full default stage budget."""
    return requested if requested is not None else MIXER_TIMEOUT if demo == "emitter_mixer" else DEFAULT_TIMEOUT


def _choices(value: str, allowed: tuple[str, ...], kind: str, parser: argparse.ArgumentParser) -> list[str]:
    chosen = [v.strip().lower() for v in value.split(",") if v.strip()]
    unknown = [v for v in chosen if v not in allowed]
    if unknown or not chosen:
        parser.error(f"unknown {kind} {', '.join(unknown) or '(none)'}; choose from {', '.join(allowed)}")
    return list(dict.fromkeys(chosen))  # keep the order, drop repeats


def plan(demos: list[str], backends: list[str]) -> list[dict]:
    """The Blender runs, in order: [{"demo", "backend", "folder"}]; folder is relative to the output root."""
    runs = []
    for demo in demos:
        if demo in PER_BACKEND:
            runs.extend({"demo": demo, "backend": b, "folder": f"{demo}/{b}"} for b in backends)
        else:
            runs.append({"demo": demo, "backend": backends[0], "folder": demo})
    return runs


def blender_running(tasklist_csv: str) -> bool:
    """True if Windows ``tasklist /FO CSV /NH`` output lists a blender.exe process."""
    for row in csv.reader(io.StringIO(tasklist_csv)):
        if row and row[0].strip().lower() == "blender.exe":
            return True
    return False


def other_blender_running() -> bool:
    if sys.platform == "win32":
        result = subprocess.run(["tasklist", "/FI", "IMAGENAME eq blender.exe", "/FO", "CSV", "/NH"],
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="replace",
                                check=False, timeout=10)
        if result.returncode != 0:
            raise RuntimeError("cannot establish whether another Blender is running")
        return blender_running(result.stdout)
    result = subprocess.run(["pgrep", "-x", "blender"], stdin=subprocess.DEVNULL, capture_output=True, check=False,
                            timeout=10)
    if result.returncode not in (0, 1):
        raise RuntimeError("cannot establish whether another Blender is running")
    return result.returncode == 0


def wait_for_other_blender(limit: float, poll: float = 10.0) -> bool:
    """Wait until no Blender is running; False if one is still running after ``limit`` seconds."""
    t0 = time.monotonic()
    announced = False
    while other_blender_running():
        if time.monotonic() - t0 > limit:
            return False
        if not announced:
            print("Another Blender is running; waiting for it to exit ...", flush=True)
            announced = True
        time.sleep(poll)
    return True


def run_result(run: dict, demo_summary: dict | None, exit_code: int | None, seconds: float,
               error: str | None = None) -> dict:
    """One row of the run summary from a demo's own summary.json (None if it wrote none)."""
    summary = demo_summary or {}
    checks = summary.get("checks", {})
    failed = sorted(name for name, check in checks.items() if not check.get("ok"))
    ok = (summary.get("ok") is True and bool(checks) and not failed
          and all(c.get("ok") is True for c in checks.values())
          and exit_code == 0 and error is None and not summary.get("error"))
    return {
        "demo": run["demo"], "backend": run["backend"], "folder": run["folder"], "ok": ok,
        "exit_code": exit_code, "seconds": round(seconds, 1),
        "particles": summary.get("particles"), "transport": summary.get("transport"),
        "modes": summary.get("modes", {}),
        "timings": summary.get("timings", {}), "failed_checks": failed, "checks": len(checks),
        "key_frame": summary.get("key_frame"), "sequences": summary.get("sequences", []),
        "video_fps": summary.get("video_fps", 24),
        "error": error or summary.get("error") or (None if demo_summary is not None else "no summary.json"),
    }


def run_summary(results: list[dict], started: str, blender: str, out: Path) -> dict:
    return {"started": started, "blender": blender, "output": str(out), "ok": bool(results) and all(
        r["ok"] for r in results), "passed": sum(r["ok"] for r in results), "runs": results}


def contact_sheet_spec(results: list[dict], output: Path, title: str, columns: int = 3) -> dict:
    """Input of ``_showcase.py contact-sheet``: one cell per run that produced a key frame."""
    cells = []
    for r in results:
        label = r["demo"] if r["demo"] not in PER_BACKEND else f"{r['demo']} ({r['backend']})"
        cells.append({"image": r.get("key_frame") or "", "label": f"{label}: {'PASS' if r['ok'] else 'FAIL'}"})
    return {"cells": cells, "output": str(output), "columns": min(columns, max(1, len(cells))),
            "cell_width": 480, "title": title}


def ffmpeg_command(ffmpeg: str, folder: Path, fps: float, target: Path) -> list[str]:
    """H.264 mp4 of a ``frame_0001.png`` ... sequence (even dimensions, yuv420p so every player opens it)."""
    return [ffmpeg, "-y", "-loglevel", "error", "-framerate", f"{fps:g}", "-i", str(folder / "frame_%04d.png"),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(target)]


def blender_command(blender: str, backend: str, blend: Path, script: Path) -> list[str]:
    return [blender, "--factory-startup", "--online-mode", "--gpu-backend", backend, "--gpu-vsync", "off",
            "--window-geometry", *WINDOW, str(blend), "--python", str(script)]


def find_blender(explicit: str | None) -> str:
    candidate = explicit or os.environ.get("BLENDER") or shutil.which("blender")
    if not candidate:
        sys.exit("Blender not found: pass --blender PATH or set the BLENDER environment variable.")
    return candidate


def find_warp_folder(explicit: str | None) -> str | None:
    """Folder containing the ``warp`` package: --warp-folder, the environment, else this Python's Warp."""
    folder = explicit or os.environ.get(WARP_FOLDER_ENV)
    if folder:
        return folder
    spec = importlib.util.find_spec("warp")
    if spec is not None and spec.origin:
        return str(Path(spec.origin).resolve().parents[1])
    return None


def profile_env(profile: Path) -> dict[str, str]:
    names = ("RESOURCES", "CONFIG", "SCRIPTS", "DATAFILES", "EXTENSIONS")
    return {f"BLENDER_USER_{n}": str(profile / n.lower()) for n in names}


def install_addon(profile: Path) -> Path:
    """Copy this checkout's add-on into the profile's user_default extension repository (LF line endings, as in
    the built package)."""
    target = profile / "extensions" / "user_default" / "warp_sph_realtime"
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(ROOT / "warp_sph_realtime", target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for path in target.rglob("*"):
        if path.suffix in TEXT_SUFFIXES:
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    return target


@contextmanager
def runner_lease(path: Path | None = None):
    """Cross-checkout OS lock, held through all Blender jobs. OS releases it after a crash.

    Never unlink the lock file: unlinking permits two processes to lock different inodes.
    Acquire before any profile/output mutation or check-then-launch sequence.
    """
    path = path or Path(tempfile.gettempdir()) / "warpsph-showcase-runner.lock"
    with path.open("a+b") as lease:
        lease.seek(0, 2)
        if lease.tell() == 0:
            lease.write(b"0")
            lease.flush()
        lease.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("another showcase runner holds the execution lease") from exc
        try:
            yield
        finally:
            lease.seek(0)
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(lease.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)


def read_summary(path: Path) -> dict:
    """Reject corrupt/partial receipts instead of aborting the batch or trusting truthy data."""
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or type(summary.get("ok")) is not bool:
        raise ValueError("summary must be an object with a boolean ok")
    checks = summary.get("checks")
    if not isinstance(checks, dict) or any(not isinstance(c, dict) or type(c.get("ok")) is not bool
                                           for c in checks.values()):
        raise ValueError("summary checks must map names to boolean check results")
    sequences = summary.get("sequences", [])
    if not isinstance(sequences, list) or any(not isinstance(s, str) for s in sequences):
        raise ValueError("summary sequences must be a list of paths")
    fps = summary.get("video_fps", 24)
    if type(fps) not in (int, float):
        raise ValueError("summary video_fps must be finite and positive")
    try:
        fps = float(fps)
    except OverflowError as exc:
        raise ValueError("summary video_fps must be finite and positive") from exc
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("summary video_fps must be finite and positive")
    summary["video_fps"] = fps
    return summary


def empty_blend(blender: str, profile: Path, env: dict[str, str]) -> Path:
    """A start file without objects (opening a file also skips the splash screen)."""
    blend = profile / "start.blend"
    if not blend.is_file():
        expr = ("import bpy\nfor o in list(bpy.data.objects): bpy.data.objects.remove(o)\n"
                f"bpy.ops.wm.save_as_mainfile(filepath={str(blend)!r})")
        subprocess.run([blender, "-b", "--factory-startup", "--python-expr", expr], env=env,
                       stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=300)
    return blend


class _KeepAwake:
    """Windows: keep the machine and display awake while Blender renders."""

    def __enter__(self):
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
        return self

    def __exit__(self, *exc):
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        return False


def run_demo(blender: str, run: dict, out: Path, blend: Path, env: dict[str, str], timeout: float,
             wait: float) -> dict:
    folder = out / run["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    if not wait_for_other_blender(wait):
        return run_result(run, None, None, 0.0, f"another Blender was still running after {wait:.0f} s")
    command = blender_command(blender, run["backend"], blend, SHOWCASE / f"{run['demo']}.py")
    print(f"== {run['demo']} ({run['backend']}) -> {folder}", flush=True)
    t0 = time.monotonic()
    code: int | None = None
    error = None
    with (folder / "blender-stdout.log").open("w", encoding="utf-8", errors="replace") as log, _KeepAwake():
        try:
            code = subprocess.run(command, env=dict(env, **{OUT_ENV: str(folder),
                                  "SHOWCASE_EXPECTED_BACKEND": run["backend"].upper()}), stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False).returncode
        except subprocess.TimeoutExpired:
            error = f"timed out after {timeout:.0f} s"
        except OSError as exc:
            error = f"Blender launch failed: {exc}"
    summary_path = folder / "summary.json"
    demo_summary = None
    try:
        demo_summary = read_summary(summary_path)
        if demo_summary.get("demo") != run["demo"]:
            raise ValueError("summary demo does not match requested job")
    except (OSError, ValueError, UnicodeError) as exc:
        error = error or f"invalid or missing summary.json: {exc}"
    result = run_result(run, demo_summary, code, time.monotonic() - t0, error)
    print(f"   {'PASS' if result['ok'] else 'FAIL'} in {result['seconds']} s"
          + (f": {result['error']}" if result["error"] else "")
          + (f" (failed checks: {', '.join(result['failed_checks'])})" if result["failed_checks"] else ""), flush=True)
    return result


def encode_videos(results: list[dict], timeout: float = ENCODE_TIMEOUT, errors: list | None = None) -> list[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg is not on PATH: keeping the PNG sequences only", flush=True)
        return []
    videos = []
    for r in results:
        for sequence in r["sequences"]:
            folder = Path(sequence)
            if not (folder / "frame_0001.png").is_file():
                continue
            target = folder.parent / f"{r['demo']}.mp4"
            try:
                done = subprocess.run(ffmpeg_command(ffmpeg, folder, r["video_fps"], target), stdin=subprocess.DEVNULL,
                                      capture_output=True, text=True, errors="replace", check=False, timeout=timeout)
                if done.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
                    raise RuntimeError(f"ffmpeg exit {done.returncode}: {done.stderr.strip()[-400:]}")
                videos.append(str(target))
            except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError, OverflowError, TypeError) as exc:
                detail = {"stage": "video", "sequence": str(folder), "error": str(exc)}
                if errors is not None:
                    errors.append(detail)
                print(f"ffmpeg failed for {folder}: {exc}", flush=True)
    return videos


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    blender = find_blender(args.blender)
    try:
        with runner_lease():
            return run_batch(args, blender)
    except (RuntimeError, OSError) as exc:
        print(f"showcase could not start: {exc}", flush=True)
        return 1


def write_report(out: Path, summary: dict) -> None:
    temporary = out / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(out / "summary.json")


def run_batch(args, blender: str) -> int:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    # Absolute: Blender resolves a relative render path against the drive root, not the working directory.
    out = (Path(args.out) if args.out else ROOT / "evidence" / "showcase" / stamp).resolve()
    # Preserve previous evidence; a supplied output path must be new.
    out.mkdir(parents=True, exist_ok=False)
    profile = out / "_profile"
    env = dict(os.environ, **profile_env(profile))
    warp_folder = find_warp_folder(args.warp_folder)
    if warp_folder:
        env[WARP_FOLDER_ENV] = warp_folder
    else:
        print("No Warp folder found: the add-on will use the Warp installed in the showcase profile", flush=True)
    runs = plan(args.demos, args.backends)
    results: list[dict] = []
    summary = run_summary(results, stamp, blender, out)
    summary["warp_folder"] = warp_folder
    summary.update(videos=[], media_errors=[], contact_sheet=None, complete=False)
    write_report(out, summary)
    setup_failed = False
    try:
        try:
            if not wait_for_other_blender(args.wait):
                raise RuntimeError(f"another Blender was still running after {args.wait:.0f} s")
            install_addon(profile)
            blend = empty_blend(blender, profile, env)
        except (OSError, RuntimeError, UnicodeError, subprocess.SubprocessError) as exc:
            setup_failed = True
            results.extend(run_result(run, None, None, 0, f"batch setup failed: {exc}") for run in runs)
        else:
            for run in runs:
                try:
                    result = run_demo(blender, run, out, blend, env, job_timeout(run["demo"], args.timeout), args.wait)
                except Exception as exc:
                    result = run_result(run, None, None, 0, f"job failed: {type(exc).__name__}: {exc}")
                results.append(result)
                summary.update(run_summary(results, stamp, blender, out))
                summary["ok"] = False  # incomplete checkpoints cannot qualify the full plan
                write_report(out, summary)
        summary.update(run_summary(results, stamp, blender, out))
        if setup_failed:
            summary.update(complete=True, media_skipped="batch setup failed")
            print(f"Batch setup failed; all {len(results)} result rows retained -> {out}", flush=True)
            return 1  # finally still publishes; do not wait again or launch media without a profile
        summary["videos"] = encode_videos(results, errors=summary["media_errors"])
        try:
            spec = contact_sheet_spec(results, out / "contact_sheet.png", f"WarpSPH showcase {stamp}")
            spec_path = out / "contact_sheet.json"
            spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
            if not wait_for_other_blender(args.wait):
                raise RuntimeError("another Blender is running; contact sheet not launched")
            sheet = subprocess.run([blender, "-b", "--factory-startup", "--python", str(SHOWCASE / "_showcase.py"),
                                    "--", "contact-sheet", str(spec_path)], env=env, stdin=subprocess.DEVNULL,
                                   capture_output=True, text=True, errors="replace", check=False, timeout=600)
            contact = out / "contact_sheet.png"
            if sheet.returncode != 0 or not contact.is_file():
                raise RuntimeError(f"contact sheet exit {sheet.returncode}: {(sheet.stdout + sheet.stderr)[-600:]}")
            summary["contact_sheet"] = str(contact)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            summary["media_errors"].append({"stage": "contact_sheet", "error": str(exc)})
        summary["complete"] = len(results) == len(runs)
    finally:
        summary["ok"] = (summary["complete"] and all(r["ok"] for r in results)
                         and not summary["media_errors"] and bool(summary["contact_sheet"]))
        write_report(out, summary)
    print(f"{summary['passed']}/{len(results)} runs passed -> {out}", flush=True)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
