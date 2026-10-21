"""Diagnostics report and package logging (no Blender or GPU needed)."""
from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from warp_sph_realtime import diagnostics

HOME = r"C:\Users\Someone"


def full_data():
    data = {}
    for _title, fields in diagnostics.SECTIONS:
        for key, _label in fields:
            data[key] = f"value-{key}"
    return data


def fake_wp():
    device = SimpleNamespace(name="NVIDIA Test GPU", arch=89, total_memory=8 * 2**30)
    return SimpleNamespace(
        __version__="1.17.0",
        __file__=HOME + r"\ext\warp\__init__.py",
        get_cuda_driver_version=lambda: (12, 9),
        get_device=lambda name: device,
        context=SimpleNamespace(runtime=SimpleNamespace(toolkit_version=(12, 9))),
    )


class ReportTest(unittest.TestCase):
    def test_every_section_and_field_present(self):
        report = diagnostics.build_report(full_data(), ["log line"])
        for title, fields in diagnostics.SECTIONS:
            self.assertIn(f"[{title}]", report)
            for key, label in fields:
                self.assertIn(f"{label}: value-{key}", report)
        self.assertIn("[Recent log]\nlog line", report)

    def test_probe_failures_never_raise(self):
        def boom():
            raise RuntimeError("probe broke")
        data = {k: boom for k in full_data()}
        report = diagnostics.build_report(data, boom)
        self.assertIn("unavailable: probe broke", report)
        self.assertEqual(diagnostics.build_report({}, None).count("unavailable: not collected"), len(full_data()))

    def test_warp_probes_without_activation_do_not_import_warp(self):
        probes = diagnostics.warp_probes(None)
        report = diagnostics.build_report({**full_data(), **probes})
        self.assertIn("Warp version: not loaded", report)
        self.assertIn("CUDA driver version: unavailable: Warp not activated", report)

    def test_warp_probes_with_module(self):
        report = diagnostics.build_report({**full_data(), **diagnostics.warp_probes(fake_wp())}, homes=(HOME,))
        for expected in ("Warp version: 1.17.0", "CUDA driver version: 12.9", "CUDA runtime (cudart): 12.9",
                         "GPU: NVIDIA Test GPU", "Compute capability: 8.9 (sm_89)", "GPU memory: 8192 MB",
                         "Warp location: ~\\ext\\warp"):
            self.assertIn(expected, report)

    def test_home_path_redacted_in_all_spellings(self):
        data = full_data()
        data["last_error"] = rf"failed at {HOME}\x and c:/users/someone/y"
        report = diagnostics.build_report(data, [f"log {HOME}\\z"], homes=(HOME,))
        self.assertNotIn("Someone", report)
        self.assertNotIn("someone", report)
        self.assertIn(r"failed at ~\x and ~/y", report)

    def test_log_tail_reads_last_lines_and_handles_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "x.log")
            path.write_text("\n".join(str(i) for i in range(100)), encoding="utf-8")
            self.assertEqual(diagnostics.read_log_tail(path, 3), ["97", "98", "99"])
            self.assertTrue(diagnostics.read_log_tail(Path(tmp, "none.log"))[0].startswith("unavailable"))
        self.assertEqual(diagnostics.read_log_tail(None), [])


class LoggingTest(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("warpsph_diag_test")
        self.addCleanup(diagnostics.teardown_logging, self.logger)

    def tagged(self):
        return [h for h in self.logger.handlers if getattr(h, diagnostics._HANDLER_TAG, False)]

    def test_setup_is_idempotent_and_teardown_removes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = None
            for _ in range(3):
                path = diagnostics.setup_logging(Path(tmp, "logs"), self.logger)
            assert path is not None
            self.assertEqual(len(self.tagged()), 1)
            self.assertEqual(diagnostics.log_path(self.logger), path)
            self.logger.info("hello world")
            self.tagged()[0].flush()
            self.assertIn("hello world", path.read_text(encoding="utf-8"))
            diagnostics.teardown_logging(self.logger)
            self.assertEqual(self.tagged(), [])
            self.assertIsNone(diagnostics.log_path(self.logger))

    def test_rotation_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            diagnostics.setup_logging(tmp, self.logger)
            handler = self.tagged()[0]
            self.assertEqual((handler.maxBytes, handler.backupCount), (256 * 1024, 1))
            diagnostics.teardown_logging(self.logger)

    def test_unwritable_directory_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp, "file")
            blocker.write_text("x")
            self.assertIsNone(diagnostics.setup_logging(blocker / "logs", self.logger))
            self.assertEqual(self.tagged(), [])


if __name__ == "__main__":
    unittest.main()
