"""Warp activation self-test, driver check and foreign-Warp naming."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from warp_sph_realtime import warp_installer as installer
from warp_sph_realtime import warp_runtime


class _FakeArray:
    def __init__(self, values):
        self.values = values

    def numpy(self):
        return self.values


def fake_warp(result, driver=(12, 9)):
    """Warp stand-in whose kernel launch yields ``result(n)``."""
    return SimpleNamespace(
        kernel=lambda fn: fn,
        int32=int,
        array=lambda dtype: None,
        tid=lambda: 0,
        zeros=lambda n, dtype, device: _FakeArray([0] * n),
        launch=lambda kernel, dim, inputs, device: setattr(inputs[0], "values", result(dim)),
        get_cuda_driver_version=lambda: driver,
    )


class SelfTestTest(unittest.TestCase):
    def setUp(self):
        warp_runtime._self_test_passed = False
        self.addCleanup(setattr, warp_runtime, "_self_test_passed", False)

    def test_pass_is_cached(self):
        wp = fake_warp(lambda n: [i * 2 + 1 for i in range(n)])
        warp_runtime.run_self_test(wp)
        wp.launch = mock.Mock(side_effect=AssertionError("re-ran"))
        warp_runtime.run_self_test(wp)

    def test_mismatch_raises_with_driver(self):
        wp = fake_warp(lambda n: [0] * n)
        with self.assertRaisesRegex(RuntimeError, r"self-test returned .*driver CUDA 12\.9"):
            warp_runtime.run_self_test(wp)
        self.assertFalse(warp_runtime._self_test_passed)

    def test_launch_failure_is_wrapped(self):
        wp = fake_warp(lambda n: [])
        wp.launch = mock.Mock(side_effect=OSError("boom"))
        with self.assertRaisesRegex(RuntimeError, "failed to run.*boom"):
            warp_runtime.run_self_test(wp)

    def test_old_driver_named(self):
        with self.assertRaisesRegex(RuntimeError, r"supports CUDA 11\.8 .* needs CUDA 12"):
            warp_runtime._check_driver(fake_warp(lambda n: [], driver=(11, 8)))

    def test_unknown_driver_not_fatal(self):
        warp_runtime._check_driver(fake_warp(lambda n: [], driver=None))

    def test_real_cuda_self_test(self):
        try:
            import warp as wp
            wp.init()
            if not wp.is_cuda_available():
                raise ImportError("no CUDA")
        except Exception:
            self.skipTest("Warp CUDA unavailable")
        warp_runtime.run_self_test(wp)
        self.assertTrue(warp_runtime._self_test_passed)


class ForeignWarpVersionTest(unittest.TestCase):
    def test_loaded_foreign_names_version_and_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            init = Path(tmp) / "foreign" / "warp" / "__init__.py"
            init.parent.mkdir(parents=True)
            init.write_text("", encoding="utf8")
            fake = SimpleNamespace(__file__=str(init), __version__="1.9.9")
            with mock.patch.dict(sys.modules, {"warp": fake}), \
                    mock.patch.object(installer, "_user_site_warp_path", lambda: None), \
                    mock.patch.object(installer, "_owned_candidates", lambda: [Path(tmp) / "owned"]), \
                    mock.patch.object(installer, "_importable_warp_file", lambda: init.resolve()):
                reason = installer.foreign_warp_block_reason()
            assert reason is not None
            self.assertIn("Warp 1.9.9", reason)
            self.assertIn(str(init.resolve()), reason)

    def test_version_from_dist_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            init = Path(tmp) / "warp" / "__init__.py"
            init.parent.mkdir()
            init.write_text("", encoding="utf8")
            (Path(tmp) / "warp_lang-1.5.0.dist-info").mkdir()
            with mock.patch.dict(sys.modules):
                sys.modules.pop("warp", None)
                self.assertEqual(installer._foreign_warp_version(init), "1.5.0")
            missing = Path(tmp) / "x" / "warp" / "__init__.py"
            self.assertEqual(installer._foreign_warp_version(missing), "unknown version")


if __name__ == "__main__":
    unittest.main()
