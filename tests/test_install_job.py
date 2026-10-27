"""P1.4: background install job, cancel without partial files, thread safety, hardware check."""
from __future__ import annotations

import builtins
import hashlib
import io
import sys
import tempfile
import threading
import types
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from warp_sph_realtime import diagnostics, install_job
from warp_sph_realtime import warp_installer as installer

URL = "https://files.pythonhosted.org/packages/test.whl"


def wheel_bytes() -> bytes:
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as z:
        z.writestr("warp/__init__.py", '__version__ = "1.17.0"\n')
        z.writestr("warp_lang-1.17.0.dist-info/METADATA", "Name: warp-lang\nVersion: 1.17.0\n")
    return blob.getvalue()


class GatedResponse:
    """urlopen() stand-in: serves the first half, then waits for ``release`` before the rest."""

    def __init__(self, data: bytes, release: threading.Event, first_chunk_sent: threading.Event):
        self.data, self.release, self.sent = data, release, first_chunk_sent
        self.position = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def geturl(self):
        return URL

    def read(self, _size):
        half = len(self.data) // 2
        if self.position == 0:
            self.position = half
            self.sent.set()
            return self.data[:half]
        self.release.wait(10)
        chunk, self.position = self.data[self.position:], len(self.data)
        return chunk


class JobTestBase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.owned = self.base / "user-extension" / ".local"
        self.scratch = self.base / "scratch"  # TMPDIR for the job: must be empty again afterwards
        self.scratch.mkdir()
        patcher = mock.patch.object(tempfile, "tempdir", str(self.scratch))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.data = wheel_bytes()
        self.info = installer.WheelInfo("test.whl", hashlib.sha256(self.data).hexdigest(), len(self.data), URL,
                                        "windows-x64")
        self.plan = installer.InstallPlan(self.info, self.owned)
        patcher = mock.patch.object(installer, "get_wheel_info_for_platform", lambda platform=None: self.info)
        patcher.start()
        self.addCleanup(patcher.stop)
        saved = sys.modules.pop("warp", None)
        self.addCleanup(lambda: saved is not None and sys.modules.__setitem__("warp", saved))

    def run_job(self, job: install_job.InstallJob, timeout=20.0):
        job.start()
        job.join(timeout)
        snap = job.snapshot()
        self.assertTrue(snap.finished, "job did not finish")
        assert snap.result is not None
        return snap.result


class InstallJobTest(JobTestBase):
    def test_runs_off_thread_and_publishes_progress(self):
        seen = []

        def runner(plan, *, progress, cancel):
            seen.append(threading.current_thread() is not threading.main_thread())
            progress("Downloading", 0.5)
            return installer.InstallResult(True, "ok")

        job = install_job.InstallJob(self.plan, runner)
        self.assertFalse(job.snapshot().finished)
        result = self.run_job(job)
        self.assertTrue(result.success)
        self.assertEqual(seen, [True])
        snap = job.snapshot()
        self.assertEqual((snap.message, snap.fraction), ("Downloading", 0.5))

    def test_runner_exception_becomes_failed_result(self):
        def runner(plan, *, progress, cancel):
            raise RuntimeError("disk exploded")

        result = self.run_job(install_job.InstallJob(self.plan, runner))
        self.assertFalse(result.success)
        self.assertIn("disk exploded", result.message)

    def test_cancel_flag_reaches_runner(self):
        started, stopped = threading.Event(), threading.Event()

        def runner(plan, *, progress, cancel):
            started.set()
            stopped.wait(10)
            return installer.InstallResult(False, "stopped", cancelled=cancel())

        job = install_job.InstallJob(self.plan, runner)
        job.start()
        started.wait(10)
        self.assertFalse(job.snapshot().cancelling)
        job.cancel()
        self.assertTrue(job.snapshot().cancelling)
        stopped.set()
        job.join(10)
        result = job.snapshot().result
        assert result is not None
        self.assertTrue(result.cancelled)

    def test_start_is_refused_while_an_install_runs(self):
        started, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def runner(plan, *, progress, cancel):
            started.set()
            release.wait(10)
            return installer.InstallResult(True, "ok")

        job = install_job.InstallJob(self.plan, runner)
        job.start()
        self.assertTrue(started.wait(10))
        self.assertTrue(installer.install_running())
        # Start imports Warp through this check; it must not load files the worker is replacing.
        with self.assertRaisesRegex(RuntimeError, "install is still running"):
            installer.ensure_installed_warp_on_path()
        release.set()
        job.join(10)
        self.assertTrue(job.snapshot().finished)
        self.assertFalse(installer.install_running())


class RealPlanJobTest(JobTestBase):
    """The real ``run_plan`` on a thread, with only the network faked."""

    def start_gated_download(self):
        release, sent = threading.Event(), threading.Event()
        response = GatedResponse(self.data, release, sent)
        patcher = mock.patch.object(installer, "urlopen", return_value=response)
        patcher.start()
        self.addCleanup(patcher.stop)
        job = install_job.InstallJob(self.plan)
        job.start()
        self.assertTrue(sent.wait(10), "download never started")
        return job, release

    def test_download_installs_and_ui_thread_stays_free(self):
        job, release = self.start_gated_download()
        # The worker is parked inside the download; the caller (UI) can still poll instantly.
        snap = job.snapshot()
        self.assertFalse(snap.finished)
        self.assertIn("Downloading Warp 1.17.0", snap.message)
        self.assertFalse(self.owned.exists())
        release.set()
        job.join(20)
        result = job.snapshot().result
        assert result is not None
        self.assertTrue(result.success, result.message)
        self.assertEqual(job.snapshot().fraction, 1.0)
        self.assertTrue(installer._is_managed(self.owned))
        self.assertEqual(list(self.scratch.iterdir()), [])  # temp wheel cleaned up

    def test_cancel_during_download_leaves_no_partial_files(self):
        job, release = self.start_gated_download()
        job.cancel()
        release.set()
        job.join(20)
        result = job.snapshot().result
        assert result is not None
        self.assertFalse(result.success)
        self.assertTrue(result.cancelled)
        self.assertIn("cancelled", result.message)
        self.assertFalse(self.owned.exists())
        self.assertFalse(self.owned.parent.exists())
        self.assertEqual(list(self.scratch.iterdir()), [], "temp download left behind")

    def test_cancel_during_extraction_leaves_existing_install_alone(self):
        wheel = self.base / "w.whl"
        wheel.write_bytes(self.data)
        local = installer.InstallPlan(self.info, self.owned, local_wheel=wheel)
        self.assertTrue(installer.run_plan(local).success)
        before = sorted(p.relative_to(self.owned) for p in self.owned.rglob("*"))
        calls = {"n": 0}

        def cancel_in_staging_extraction():
            calls["n"] += 1
            return calls["n"] > 3  # copy, post-hash check, then the second archive member

        result = installer.run_plan(local, cancel=cancel_in_staging_extraction)
        self.assertTrue(result.cancelled, result.message)
        self.assertEqual(sorted(p.relative_to(self.owned) for p in self.owned.rglob("*")), before)
        self.assertTrue(installer._is_managed(self.owned))
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_worker_never_touches_bpy(self):
        violations: list[str] = []
        main = threading.main_thread()

        class Guard(types.ModuleType):
            def __getattribute__(self, name):
                if threading.current_thread() is not main:
                    violations.append(f"bpy.{name}")
                return super().__getattribute__(name)

        real_import = builtins.__import__

        def watching_import(name, *args, **kwargs):
            if name == "bpy" and threading.current_thread() is not main:
                violations.append("import bpy")
            return real_import(name, *args, **kwargs)

        release, sent = threading.Event(), threading.Event()
        release.set()
        guard = Guard("bpy")
        with mock.patch.dict(sys.modules, {"bpy": guard}), \
                mock.patch.object(installer, "urlopen", return_value=GatedResponse(self.data, release, sent)), \
                mock.patch.object(builtins, "__import__", watching_import):
            result = self.run_job(install_job.InstallJob(self.plan))
        self.assertTrue(result.success, result.message)
        self.assertEqual(violations, [])


class HardwareCheckTest(unittest.TestCase):
    SMI_ROWS = "NVIDIA GeForce RTX 4080 SUPER, 610.88, 16376, 8.9\n"
    SMI_HEADER = "| NVIDIA-SMI 610.88   KMD Version: 610.88   CUDA UMD Version: 13.3 |\n"

    def fake_run(self, rows=None, header=None, returncode=0):
        def run(cmd, **_kwargs):
            text = rows if any("--query-gpu" in part for part in cmd) else header
            return SimpleNamespace(returncode=returncode, stdout=text, stderr="boom")
        return run

    def probe(self, **kwargs):
        with mock.patch.object(diagnostics, "_smi_executable", lambda: "nvidia-smi"):
            return diagnostics.nvidia_smi_info(self.fake_run(**kwargs))

    def test_parses_gpu_driver_vram_and_cuda(self):
        info = self.probe(rows=self.SMI_ROWS, header=self.SMI_HEADER)
        self.assertEqual(info["gpu_name"], "NVIDIA GeForce RTX 4080 SUPER")
        self.assertEqual(info["driver_version"], "610.88")
        self.assertEqual(info["gpu_memory"], "16376 MB")
        self.assertEqual(info["cuda_driver"], "13.3")
        old_style = self.probe(rows=self.SMI_ROWS, header="CUDA Version: 12.9")
        self.assertEqual(old_style["cuda_driver"], "12.9")

    def test_multiple_gpus_are_counted(self):
        info = self.probe(rows=self.SMI_ROWS + "NVIDIA RTX A2000, 610.88, 6144, 8.6\n", header="")
        self.assertTrue(info["gpu_name"].endswith("(+1 more)"))

    def test_missing_nvidia_smi_is_reported_not_raised(self):
        with mock.patch.object(diagnostics, "_smi_executable", lambda: None):
            self.assertIn("not found", diagnostics.nvidia_smi_info()["error"])
        self.assertIn("nvidia-smi failed", self.probe(rows="", header="", returncode=9)["error"])

    def test_probe_exception_never_raises(self):
        def run(*_a, **_k):
            raise TimeoutError("hung")
        with mock.patch.object(diagnostics, "_smi_executable", lambda: "nvidia-smi"):
            self.assertIn("hung", diagnostics.nvidia_smi_info(run)["error"])

    def test_summary_before_warp_is_installed(self):
        info = self.probe(rows=self.SMI_ROWS, header=self.SMI_HEADER)
        summary = diagnostics.hardware_summary(info, "Vulkan")
        rows = dict(summary.rows)
        self.assertEqual(rows["GPU"], "NVIDIA GeForce RTX 4080 SUPER")
        self.assertEqual(rows["NVIDIA driver"], "610.88")
        self.assertEqual(rows["GPU memory (VRAM)"], "16376 MB")
        self.assertEqual(rows["Graphics backend"], "Vulkan")
        self.assertIs(summary.ready, True)

    def test_summary_states(self):
        self.assertIsNone(diagnostics.hardware_summary(None, "OpenGL").ready)
        old = diagnostics.hardware_summary({"gpu_name": "GTX", "driver_version": "450", "cuda_driver": "11.8"}, "OpenGL")
        self.assertIs(old.ready, False)
        self.assertIn("update the NVIDIA driver", old.verdict)
        none = diagnostics.hardware_summary({"error": "nvidia-smi not found"}, "OpenGL")
        self.assertIs(none.ready, False)
        self.assertIn("nvidia-smi not found", none.verdict)

    def test_active_warp_values_win_over_nvidia_smi(self):
        device = SimpleNamespace(name="Warp GPU", arch=89, total_memory=8 * 2**30)
        wp = SimpleNamespace(__version__="1.17.0", __file__="x", get_cuda_driver_version=lambda: (12, 9),
                             get_device=lambda _n: device)
        summary = diagnostics.hardware_summary({"gpu_name": "smi GPU", "gpu_memory": "1 MB"}, "Vulkan", wp)
        rows = dict(summary.rows)
        self.assertEqual(rows["GPU"], "Warp GPU")
        self.assertEqual(rows["GPU memory (VRAM)"], "8192 MB")
        self.assertEqual(rows["Driver supports CUDA"], "12.9")


if __name__ == "__main__":
    unittest.main()
