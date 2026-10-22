"""Negative controls for showcase acceptance and batch fault containment (no GPU)."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from test_showcase_runner import ROOT, _load, _summary, helper, runner


class ReportingFailures(unittest.TestCase):
    RUN = {"demo": "presets", "backend": "opengl", "folder": "presets"}

    def test_summary_cannot_override_failed_or_absent_checks(self):
        for checks in ({}, {"route": {"ok": False}}, {"route": {"ok": "yes"}}):
            with self.subTest(checks=checks):
                self.assertFalse(runner.run_result(self.RUN, _summary(checks=checks), 0, 1)["ok"])

    def test_malformed_and_unreadable_summary_become_failed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for content in ('{"ok":', '[]', '{"ok":true,"checks":[]}'):
                def child(*args, content=content, **kwargs):
                    (root / "presets" / "summary.json").write_text(content)
                    return SimpleNamespace(returncode=0)
                with patch.object(runner, "wait_for_other_blender", return_value=True), \
                        patch.object(runner.subprocess, "run", side_effect=child):
                    result = runner.run_demo("blender", self.RUN, root, root / "start.blend", {}, 1, 0)
                self.assertFalse(result["ok"])
                self.assertIn("summary", result["error"])
            with patch.object(runner, "wait_for_other_blender", return_value=True), \
                    patch.object(runner.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
                    patch.object(Path, "read_text", side_effect=OSError("unreadable")):
                result = runner.run_demo("blender", self.RUN, root, root / "s.blend", {}, 1, 0)
            self.assertFalse(result["ok"])
            self.assertIn("summary", result["error"])

    def test_spawn_failure_and_timeout_are_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for fault in (OSError("launch failed"), subprocess.TimeoutExpired("blender", 1)):
                with patch.object(runner, "wait_for_other_blender", return_value=True), \
                        patch.object(runner.subprocess, "run", side_effect=fault):
                    self.assertFalse(runner.run_demo("b", self.RUN, root, root / "s", {}, 1, 0)["ok"])

    def test_real_child_timeout_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(runner, "wait_for_other_blender", return_value=True), \
                patch.object(runner, "blender_command", return_value=[sys.executable, "-c", "import time; time.sleep(30)"]):
            root = Path(tmp)
            result = runner.run_demo("b", self.RUN, root, root / "s", {}, 0.3, 0)
            self.assertFalse(result["ok"])
            self.assertIn("timed out", result["error"])
            self.assertLess(result["seconds"], 5)

    def test_encoder_timeout_and_failure_preserve_next_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq = Path(tmp) / "frames"
            seq.mkdir()
            (seq / "frame_0001.png").touch()
            rows = [runner.run_result(self.RUN, _summary(sequences=[str(seq)]), 0, 1)] * 2
            calls = []
            def encoder(cmd, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    raise subprocess.TimeoutExpired(cmd, 1)
                Path(cmd[-1]).write_bytes(b"video")
                return SimpleNamespace(returncode=0, stderr="")
            errors = []
            with patch.object(runner.shutil, "which", return_value="ffmpeg"), \
                    patch.object(runner.subprocess, "run", side_effect=encoder):
                videos = runner.encode_videos(rows, timeout=1, errors=errors)
            self.assertEqual(len(videos), 1)
            self.assertEqual(len(errors), 1)
            self.assertTrue(all(c["timeout"] == 1 for c in calls))


class RouteControls(unittest.TestCase):
    def test_actual_backend_mode_route_table(self):
        with patch.dict(sys.modules, {"bpy": SimpleNamespace(), "_showcase": helper}), \
                patch.object(helper, "run"), \
                patch.object(helper, "module", return_value=SimpleNamespace(GL_LABEL="gl_texture", VULKAN_LABEL="vk")):
            display = _load("showcase_display", ROOT / "examples/showcase/display.py")
            for backend, expected in (("OPENGL", ("gl_texture", helper.GL_VBO_LABEL, helper.GL_VBO_LABEL)),
                                      ("VULKAN", ("vk", "vk", "vk")), ("UNKNOWN", (None, None, None))):
                self.assertEqual(tuple(display.expected_label(backend, mode) for mode in display.RENDER_MODES), expected)

    def test_wrong_route_and_disguised_readback_fail(self):
        for label, readback in (("none", False), ("wrong GPU interop", False), ("expected", True),
                                ("Readback (CPU copy)", True)):
            result = helper.transport_verdict({"label": label, "readback": readback}, "expected")
            self.assertFalse(result["accelerated"])
        result = helper.transport_verdict({"label": "Readback (CPU copy)", "readback": True}, "expected")
        self.assertEqual(result["classification"], "readback_fallback")
        self.assertTrue(helper.transport_verdict({"label": "expected", "readback": False}, "expected")["accelerated"])


class SetupFailure(unittest.TestCase):
    def test_setup_failure_writes_summary_and_quits(self):
        callbacks, quits = [], []
        fake = SimpleNamespace(app=SimpleNamespace(version_string="test", background=False,
                               timers=SimpleNamespace(register=lambda f, **kw: callbacks.append(f))),
                               ops=SimpleNamespace(wm=SimpleNamespace(quit_blender=lambda: quits.append(True))))
        def build():
            raise RuntimeError("injected setup failure")
        with tempfile.TemporaryDirectory() as tmp, patch.object(helper, "bpy", fake), \
                patch.object(helper, "SUMMARY", {"checks": {}, "timings": {}}), \
                patch.dict("os.environ", {helper.OUT_ENV: tmp}):
            helper.run("setup_control", lambda: iter(()), build=build)
            self.assertIsNone(callbacks[0]())
            result = json.loads((Path(tmp) / "summary.json").read_text())
            self.assertFalse(result["ok"])
            self.assertIn("injected setup failure", result["error"])
            self.assertEqual(quits, [True])


class PaddleControls(unittest.TestCase):
    def test_gravity_emission_and_array_reordering_do_not_prove_paddle(self):
        metrics = _load("showcase_metrics", ROOT / "examples/showcase/_metrics.py")
        ids = np.arange(40)
        theta = np.linspace(0, 2 * np.pi, len(ids), endpoint=False)
        pos = np.column_stack((0.15 * np.cos(theta), 0.15 * np.sin(theta), np.full(len(ids), 0.1)))
        velocity = np.tile([0, 0, -3.0], (len(ids), 1))  # strong gravity/emission, zero stirring
        frame = SimpleNamespace(position=pos, velocity=velocity, id=ids)
        shuffled = SimpleNamespace(position=pos[::-1], velocity=velocity[::-1], id=ids[::-1])
        null = metrics.paddle_influence(frame, shuffled)
        self.assertFalse(null["ok"])
        tangent = np.column_stack((-np.sin(theta), np.cos(theta), np.zeros(len(ids))))
        moving = SimpleNamespace(position=pos, velocity=velocity + 0.1 * tangent, id=ids)
        self.assertTrue(metrics.paddle_influence(frame, moving)["ok"])
        reverse = SimpleNamespace(position=pos, velocity=velocity - 0.1 * tangent, id=ids)
        self.assertFalse(metrics.paddle_influence(frame, reverse)["ok"])


class BatchControls(unittest.TestCase):
    def test_setup_decode_error_records_every_planned_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            args = runner.parse_args(["--demos", "presets,validation", "--out", str(out)])
            fault = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "injected")
            with patch.object(runner, "find_warp_folder", return_value=None), \
                    patch.object(runner, "wait_for_other_blender", side_effect=[fault, False]), \
                    patch.object(runner, "encode_videos", return_value=[]):
                self.assertEqual(runner.run_batch(args, "b"), 1)
            report = json.loads((out / "summary.json").read_text())
            self.assertEqual(len(report["runs"]), 2)
            self.assertTrue(all("batch setup failed" in r["error"] for r in report["runs"]))

    def test_successful_batch_preserves_media_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            args = runner.parse_args(["--demos", "presets", "--out", str(out)])
            def sheet(*args, **kwargs):
                (out / "contact_sheet.png").write_bytes(b"png")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            with patch.object(runner, "find_warp_folder", return_value=None), \
                    patch.object(runner, "wait_for_other_blender", return_value=True), \
                    patch.object(runner, "install_addon"), \
                    patch.object(runner, "empty_blend", return_value=Path("start.blend")), \
                    patch.object(runner, "run_demo", side_effect=lambda b, run, *a: runner.run_result(run, _summary(), 0, 1)), \
                    patch.object(runner, "encode_videos", return_value=["presets.mp4"]), \
                    patch.object(runner.subprocess, "run", side_effect=sheet):
                self.assertEqual(runner.run_batch(args, "b"), 0)
            report = json.loads((out / "summary.json").read_text())
            self.assertTrue(report["ok"])
            self.assertEqual(report["passed"], 1)
            self.assertEqual(report["videos"], ["presets.mp4"])

    def test_jobs_continue_and_contact_timeout_keeps_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            args = runner.parse_args(["--demos", "presets,validation", "--out", str(out), "--wait", "0"])
            def job(blender, run, *rest):
                if run["demo"] == "presets":
                    raise OSError("injected job failure")
                return runner.run_result(run, _summary(), 0, 1)
            with patch.object(runner, "find_warp_folder", return_value=None), \
                    patch.object(runner, "wait_for_other_blender", return_value=True), \
                    patch.object(runner, "install_addon"), \
                    patch.object(runner, "empty_blend", return_value=Path("start.blend")), \
                    patch.object(runner, "run_demo", side_effect=job), \
                    patch.object(runner, "encode_videos", return_value=[]), \
                    patch.object(runner.subprocess, "run", side_effect=subprocess.TimeoutExpired("sheet", 600)):
                self.assertEqual(runner.run_batch(args, "blender"), 1)
            report = json.loads((out / "summary.json").read_text())
            self.assertEqual([r["ok"] for r in report["runs"]], [False, True])
            self.assertTrue(report["complete"])
            self.assertFalse(report["ok"])
            self.assertEqual(report["media_errors"][0]["stage"], "contact_sheet")

    def test_busy_blender_never_mutates_profile_and_reports_all_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            args = runner.parse_args(["--demos", "display,presets", "--out", str(out), "--wait", "0"])
            with patch.object(runner, "find_warp_folder", return_value=None), \
                    patch.object(runner, "wait_for_other_blender", return_value=False), \
                    patch.object(runner, "install_addon") as install, \
                    patch.object(runner, "encode_videos", return_value=[]):
                self.assertEqual(runner.run_batch(args, "b"), 1)
                install.assert_not_called()
            report = json.loads((out / "summary.json").read_text())
            self.assertEqual(len(report["runs"]), 3)
            self.assertTrue(all(not r["ok"] for r in report["runs"]))
            self.assertFalse((out / "_profile").exists())

    def test_real_process_exclusion_and_release(self):
        # The contender imports the real runner in a different Python process.
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "lease"
            code = ("import importlib.util,pathlib; "
                    f"s=importlib.util.spec_from_file_location('runner',{str(ROOT / 'scripts/render_showcase.py')!r}); "
                    "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                    f"\nwith m.runner_lease(pathlib.Path({str(lock)!r})): print('acquired')")
            with runner.runner_lease(lock):
                child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
                self.assertNotEqual(child.returncode, 0)
                self.assertIn("execution lease", child.stderr)
            child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertIn("acquired", child.stdout)

    def test_lock_rejection_happens_before_any_batch_mutation(self):
        with patch.object(runner, "find_blender", return_value="b"), \
                patch.object(runner, "runner_lease", side_effect=RuntimeError("leased")), \
                patch.object(runner, "run_batch") as batch:
            self.assertEqual(runner.main([]), 1)
            batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
