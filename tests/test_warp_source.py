"""P1.11: Warp source choice (download / local wheel / existing folder). No Blender, no GPU, no network."""
from __future__ import annotations

import hashlib
import io
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from warp_sph_realtime import warp_installer as installer
from warp_sph_realtime import warp_runtime

FAKE_WARP_INIT = '''
__version__ = "{version}"
class _Arr:
    def __init__(self, n): self.v = [0] * n
    def numpy(self): return self.v
class _Dev: is_cuda = True
def init(): pass
def is_cuda_available(): return True
def get_device(name): return _Dev()
def get_cuda_driver_version(): return {driver}
def kernel(fn): return fn
int32 = int
def array(dtype): return None
def tid(): return 0
def zeros(n, dtype, device): return _Arr(n)
def launch(kernel, dim, inputs, device): inputs[0].v[:] = [i * 2 + 1 {wrong} for i in range(dim)]
'''


def wheel_bytes() -> bytes:
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as z:
        z.writestr("warp/__init__.py", '__version__ = "1.17.0"\n')
        z.writestr("warp_lang-1.17.0.dist-info/METADATA", "Name: warp-lang\nVersion: 1.17.0\n")
        z.writestr("warp_lang-1.17.0.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-win_amd64\n")
    return blob.getvalue()


def make_folder(root: Path, *, version="1.17.0", dist_info=True, wrong="", driver="(12, 9)", init=None) -> Path:
    """A site-packages-like folder holding a fake but importable ``warp`` package."""
    (root / "warp").mkdir(parents=True)
    (root / "warp" / "__init__.py").write_text(
        init if init is not None else FAKE_WARP_INIT.format(version=version, wrong=wrong, driver=driver),
        encoding="utf8")
    (root / "warp" / "config.py").write_text(f'version: str = "{version}"\n', encoding="utf8")
    if dist_info:
        (root / f"warp_lang-{version}.dist-info").mkdir()
        (root / f"warp_lang-{version}.dist-info" / "METADATA").write_text(f"Name: warp-lang\nVersion: {version}\n")
    return root


def _loaded_warp_file() -> Path | None:
    file_name = getattr(sys.modules.get("warp"), "__file__", None)
    return Path(file_name).resolve() if file_name else None


_real_importable_warp_file = installer._importable_warp_file


def snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns) if p.is_file() else None
            for p in sorted(root.rglob("*"))}


class SourceTestBase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.owned = self.base / "user-extension" / ".local"
        self.data = wheel_bytes()
        self.info = installer.WheelInfo(
            filename="warp_lang-1.17.0-py3-none-win_amd64.whl", sha256=hashlib.sha256(self.data).hexdigest(),
            size=len(self.data), url="https://files.pythonhosted.org/packages/test.whl", platform="windows-x64")
        self.source = installer.WarpSource()
        # Isolate sys.modules/sys.path/self-test state: the folder tests import a fake warp.
        saved = sys.modules.pop("warp", None)
        self.addCleanup(lambda: sys.modules.pop("warp", None) or None)
        if saved is not None:
            self.addCleanup(sys.modules.__setitem__, "warp", saved)
        self.addCleanup(setattr, sys, "path", list(sys.path))
        # A developer machine may have a real Warp on sys.path; the product appends a chosen folder last.
        sys.path[:] = [e for e in sys.path if not Path(e or os.getcwd(), "warp", "__init__.py").is_file()]
        self.addCleanup(installer._ADDED_FOLDERS.clear)
        self.addCleanup(setattr, warp_runtime, "_self_test_passed", False)
        warp_runtime._self_test_passed = False
        self.online = mock.Mock(return_value=None)
        for name, value in {
            "_default_install_target": lambda create=False: self.owned,
            "get_wheel_info_for_platform": lambda platform=None: self.info,
            "_user_site_root": lambda: None,
            "_user_site_warp_path": lambda: None,
            "_blender_extensions_dir": lambda: None,
            "_importable_warp_file": _loaded_warp_file,  # not find_spec: the dev machine has a real Warp
            "_online_access_block_reason": lambda: self.online(),
            "active_source": lambda: self.source,
        }.items():
            patcher = mock.patch.object(installer, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.download = mock.Mock(side_effect=AssertionError("no network access expected"))
        patcher = mock.patch.object(installer, "_download_wheel", self.download)
        patcher.start()
        self.addCleanup(patcher.stop)

    def use_folder(self, folder: Path, allow=False):
        self.source = installer.WarpSource(installer.SOURCE_FOLDER, folder, allow)

    def write_wheel(self, data: bytes | None = None, name="warp_lang-1.17.0-py3-none-win_amd64.whl") -> Path:
        path = self.base / name
        path.write_bytes(self.data if data is None else data)
        return path


class LocalWheelTest(SourceTestBase):
    def test_matching_wheel_installs_offline(self):
        self.online.return_value = "Online access required to install Warp."
        wheel = self.write_wheel()
        before = wheel.read_bytes()
        result = installer.install_warp(local_wheel=wheel)
        self.assertTrue(result.success, result.message)
        self.download.assert_not_called()
        self.assertTrue(installer._is_managed(self.owned))
        self.assertTrue((self.owned / "warp" / "__init__.py").is_file())
        self.assertEqual(wheel.read_bytes(), before)
        self.assertTrue(warp_runtime.status().available)

    def test_download_still_refused_offline_but_wheel_is_not(self):
        self.online.return_value = "Online access required to install Warp."
        refused = installer.install_warp()
        self.assertFalse(refused.success)
        self.assertIn("Online access", refused.message)
        self.assertFalse(self.owned.exists())

    def test_tampered_wheel_same_size_refused(self):
        tampered = bytearray(self.data)
        tampered[len(tampered) // 2] ^= 0xFF
        result = installer.install_warp(local_wheel=self.write_wheel(bytes(tampered)))
        self.assertFalse(result.success)
        self.assertIn("SHA-256", result.message)
        self.assertFalse(self.owned.exists())

    def test_wrong_size_wheel_refused_before_hashing(self):
        result = installer.install_warp(local_wheel=self.write_wheel(self.data + b"x"))
        self.assertFalse(result.success)
        self.assertIn("not the certified Warp wheel", result.message)
        self.assertFalse(self.owned.exists())

    def test_missing_wheel_file_is_named(self):
        result = installer.install_warp(local_wheel=self.base / "nope.whl")
        self.assertFalse(result.success)
        self.assertIn("nope.whl", result.message)

    def test_wheel_swapped_after_selection_cannot_win(self):
        """The copy is verified, so changing the original afterwards has no effect on what is extracted."""
        wheel = self.write_wheel()
        plan = installer.plan_install(local_wheel=wheel)
        assert isinstance(plan, installer.InstallPlan)
        original = installer._promote_verified_wheel

        def swap_then_promote(path, *args, **kwargs):
            wheel.write_bytes(b"evil")
            return original(path, *args, **kwargs)

        with mock.patch.object(installer, "_promote_verified_wheel", swap_then_promote):
            result = installer.run_plan(plan)
        self.assertTrue(result.success, result.message)
        self.assertEqual((self.owned / "warp" / "__init__.py").read_text("utf8"), '__version__ = "1.17.0"\n')

    def test_wheel_install_is_idempotent(self):
        wheel = self.write_wheel()
        self.assertTrue(installer.install_warp(local_wheel=wheel).success)
        again = installer.install_warp(local_wheel=wheel)
        self.assertTrue(again.already_installed)


class ExistingFolderTest(SourceTestBase):
    def test_valid_folder_is_used_without_being_touched(self):
        folder = make_folder(self.base / "studio-site")
        before = snapshot(folder)
        self.use_folder(folder)
        self.online.return_value = "Online access required to install Warp."
        wp = warp_runtime.require_cuda_runtime()
        self.assertEqual(Path(wp.__file__).resolve().parent.parent, folder.resolve())
        self.assertEqual(sys.path[-1], str(folder))
        self.assertEqual(snapshot(folder), before)  # no __pycache__, no edits
        self.assertFalse(self.owned.exists())
        self.assertTrue(warp_runtime._self_test_passed)

    def test_status_reports_folder_without_importing(self):
        self.use_folder(make_folder(self.base / "site"))
        state = warp_runtime.status()
        self.assertTrue(state.available, state.message)
        self.assertEqual(state.version, "1.17.0")
        self.assertNotIn("warp", sys.modules)

    def test_any_1_17_x_is_accepted(self):
        folder = make_folder(self.base / "site", version="1.17.3")
        self.assertTrue(installer.check_existing_folder(folder).ok)

    def test_version_falls_back_to_config_py(self):
        folder = make_folder(self.base / "site", dist_info=False)
        self.assertEqual(installer.check_existing_folder(folder).version, "1.17.0")

    def test_no_folder_chosen(self):
        self.use_folder(None)  # type: ignore[arg-type]
        self.assertIn("Choose the folder", warp_runtime.status().message)

    def test_missing_folder(self):
        check = installer.check_existing_folder(self.base / "gone")
        self.assertFalse(check.ok)
        self.assertIn("not found", check.message)

    def test_folder_without_warp(self):
        empty = self.base / "empty"
        empty.mkdir()
        check = installer.check_existing_folder(empty)
        self.assertFalse(check.ok)
        self.assertIn("No 'warp' package", check.message)
        self.assertIn(str(empty), check.message)

    def test_warp_package_folder_chosen_by_mistake(self):
        folder = make_folder(self.base / "site")
        check = installer.check_existing_folder(folder / "warp")
        self.assertFalse(check.ok)
        self.assertIn("parent folder", check.message)

    def test_unknown_version(self):
        folder = make_folder(self.base / "site", dist_info=False)
        (folder / "warp" / "config.py").unlink()
        check = installer.check_existing_folder(folder)
        self.assertFalse(check.ok)
        self.assertIn("Cannot determine the Warp version", check.message)

    def test_wrong_version_refused_by_default_and_names_the_checkbox(self):
        for version in ("1.18.0", "1.16.2", "2.0.0"):
            with self.subTest(version=version):
                folder = make_folder(self.base / f"site-{version}", version=version)
                check = installer.check_existing_folder(folder)
                self.assertFalse(check.ok)
                self.assertEqual(check.version, version)
                self.assertIn(f"Warp {version} is not certified", check.message)
                self.assertIn("Allow uncertified version", check.message)
                self.use_folder(folder)
                with self.assertRaisesRegex(RuntimeError, "not certified"):
                    warp_runtime.require_cuda_runtime()
                self.assertNotIn("warp", sys.modules)

    def test_uncertified_version_allowed_with_checkbox_and_flagged(self):
        folder = make_folder(self.base / "site", version="1.18.0")
        self.use_folder(folder, allow=True)
        check = installer.check_existing_folder(folder, True)
        self.assertTrue(check.ok)
        self.assertTrue(check.uncertified)
        state = warp_runtime.status()
        self.assertTrue(state.available)
        self.assertIn("UNCERTIFIED", state.message)
        self.assertIn("UNCERTIFIED", installer.describe_source())
        wp = warp_runtime.require_cuda_runtime()
        self.assertEqual(wp.__version__, "1.18.0")

    def test_failing_known_answer_self_test_has_specific_message(self):
        folder = make_folder(self.base / "site", wrong="+ 1")
        self.use_folder(folder)
        with self.assertRaisesRegex(RuntimeError, r"Existing Warp folder .*site.*\(Warp 1\.17\.0\) failed the startup "
                                                  r"self-test: .*self-test returned"):
            warp_runtime.require_cuda_runtime()
        self.assertFalse(warp_runtime._self_test_passed)

    def test_old_driver_has_specific_message(self):
        self.use_folder(make_folder(self.base / "site", driver="(11, 8)"))
        with self.assertRaisesRegex(RuntimeError, r"failed the startup self-test: NVIDIA driver supports CUDA 11\.8"):
            warp_runtime.require_cuda_runtime()

    def test_unimportable_warp_has_specific_message(self):
        self.use_folder(make_folder(self.base / "site", init='raise ImportError("missing libwarp")'))
        with self.assertRaisesRegex(RuntimeError, r"Existing Warp folder .* could not be imported: missing libwarp"):
            warp_runtime.require_cuda_runtime()

    def test_foreign_guard_accepts_only_the_chosen_folder(self):
        chosen = make_folder(self.base / "chosen")
        other = make_folder(self.base / "other")
        self.use_folder(chosen)
        with mock.patch.object(installer, "_importable_warp_file", lambda: (other / "warp" / "__init__.py").resolve()):
            reason = installer.foreign_warp_block_reason()
        assert reason is not None
        self.assertIn("not the Warp folder you chose", reason)
        self.assertIn(str(other.resolve()), reason)
        with mock.patch.object(installer, "_importable_warp_file", lambda: (chosen / "warp" / "__init__.py").resolve()):
            self.assertIsNone(installer.foreign_warp_block_reason())

    def test_another_warp_earlier_on_sys_path_is_named_not_imported(self):
        chosen = make_folder(self.base / "chosen")
        other = make_folder(self.base / "other", version="1.9.9")
        self.use_folder(chosen)
        sys.path.insert(0, str(other))
        pattern = r"Warp 1\.9\.9 is already loaded from .*other.*not the Warp folder"
        with mock.patch.object(installer, "_importable_warp_file", _real_importable_warp_file):
            with self.assertRaisesRegex(RuntimeError, pattern):
                warp_runtime.require_cuda_runtime()
        self.assertNotIn("warp", sys.modules)

    def test_folder_is_appended_so_it_cannot_shadow_blenders_own_packages(self):
        folder = make_folder(self.base / "site-packages")
        self.use_folder(folder)
        installer.ensure_installed_warp_on_path()
        self.assertEqual(sys.path[-1], str(folder))

    def test_owned_copy_is_foreign_while_a_folder_is_chosen(self):
        installer.install_warp(local_wheel=self.write_wheel())
        self.use_folder(make_folder(self.base / "chosen"))
        with mock.patch.object(installer, "_importable_warp_file", lambda: (self.owned / "warp" / "__init__.py")):
            self.assertIn("not the Warp folder you chose", installer.foreign_warp_block_reason() or "")

    def test_user_site_warp_is_foreign_unless_it_is_the_chosen_folder(self):
        folder = make_folder(self.base / "user-site")
        init = (folder / "warp" / "__init__.py").resolve()
        self.use_folder(folder)
        with mock.patch.object(installer, "_user_site_warp_path", lambda: init):
            self.assertIsNone(installer.foreign_warp_block_reason())
        self.source = installer.WarpSource()
        with mock.patch.object(installer, "_user_site_warp_path", lambda: init):
            self.assertIn("user site-packages", installer.foreign_warp_block_reason() or "")

    def test_install_refused_while_folder_source_is_active(self):
        self.use_folder(make_folder(self.base / "site"))
        result = installer.install_warp()
        self.assertFalse(result.success)
        self.assertIn("existing Warp folder", result.message)
        self.assertFalse(self.owned.exists())

    def test_switching_back_to_download_restores_the_owned_install(self):
        self.assertTrue(installer.install_warp(local_wheel=self.write_wheel()).success)
        owned_before = snapshot(self.owned)
        folder = make_folder(self.base / "site")
        folder_before = snapshot(folder)

        self.use_folder(folder)
        installer.ensure_installed_warp_on_path()
        self.assertEqual(sys.path[-1], str(folder))
        self.assertNotIn(str(self.owned), sys.path)
        self.assertIn("existing folder", warp_runtime.status().message)

        self.source = installer.WarpSource()  # back to Download
        self.assertIsNone(installer.foreign_warp_block_reason())
        installer.ensure_installed_warp_on_path()
        self.assertIn(str(self.owned), sys.path)
        self.assertNotIn(str(folder), sys.path)
        state = warp_runtime.status()
        self.assertTrue(state.available, state.message)
        self.assertIn("installed", state.message)
        self.assertEqual(snapshot(self.owned), owned_before)
        self.assertEqual(snapshot(folder), folder_before)
        self.download.assert_not_called()
        self.assertTrue(installer.install_warp().already_installed)

    def test_folder_loaded_this_session_blocks_switch_with_restart_hint(self):
        folder = make_folder(self.base / "site")
        self.use_folder(folder)
        warp_runtime.require_cuda_runtime()
        self.source = installer.WarpSource()
        reason = installer.foreign_warp_block_reason()
        assert reason is not None
        self.assertIn("Restart Blender", reason)

    def test_works_with_online_access_off(self):
        self.online.return_value = "Online access required to install Warp."
        self.use_folder(make_folder(self.base / "site"))
        self.assertIsNotNone(warp_runtime.require_cuda_runtime())
        self.download.assert_not_called()

    def test_describe_source(self):
        self.assertEqual(installer.describe_source(), "Download: not installed")
        installer.install_warp(local_wheel=self.write_wheel())
        self.assertEqual(installer.describe_source(), "Download: owned (WarpSPH-managed install)")
        self.source = installer.WarpSource(installer.SOURCE_WHEEL)
        self.assertTrue(installer.describe_source().startswith("Install from local wheel file: owned"))
        folder = make_folder(self.base / "site")
        self.use_folder(folder)
        text = installer.describe_source()
        self.assertIn("Use existing Warp folder", text)
        self.assertIn(str(folder), text)
        self.assertIn("Warp 1.17.0", text)
        self.use_folder(self.base / "missing")
        self.assertIn("not found", installer.describe_source())


class SourcePreferenceTest(unittest.TestCase):
    def read(self, prefs):
        addons = {"warp_sph_realtime": SimpleNamespace(preferences=prefs)}
        bpy = SimpleNamespace(context=SimpleNamespace(preferences=SimpleNamespace(addons=addons)),
                              path=SimpleNamespace(abspath=lambda p: p))
        with mock.patch.dict(sys.modules, {"bpy": bpy}), mock.patch.object(installer, "__package__", "warp_sph_realtime"):
            return installer._read_source_preference()

    def test_defaults_to_download_without_blender_or_preferences(self):
        with mock.patch.dict(sys.modules, {"bpy": None}):
            self.assertEqual(installer._read_source_preference(), installer.WarpSource())
        self.assertEqual(self.read(SimpleNamespace()), installer.WarpSource())

    def test_reads_choice(self):
        source = self.read(SimpleNamespace(warp_source="FOLDER", warp_folder_path="  C:/studio/site  ",
                                           warp_allow_uncertified=True))
        self.assertTrue(source.is_folder)
        self.assertEqual(source.folder, Path("C:/studio/site"))
        self.assertTrue(source.allow_uncertified)
        self.assertEqual(self.read(SimpleNamespace(warp_source="WHEEL")).kind, "WHEEL")

    def test_unknown_values_fall_back_to_download(self):
        self.assertEqual(self.read(SimpleNamespace(warp_source="NOPE")).kind, "DOWNLOAD")
        self.assertEqual(self.read(SimpleNamespace(warp_source="FOLDER", warp_folder_path="")).folder, None)

    def test_folder_policy(self):
        self.assertEqual(warp_runtime.folder_version_support("1.17.0+cu12")[0], "certified")
        self.assertEqual(warp_runtime.folder_version_support("1.170.0")[0], "unsupported")
        self.assertEqual(warp_runtime.folder_version_support("1.18.0", True)[0], "uncertified")
        self.assertEqual(warp_runtime.folder_version_support("", True)[0], "unsupported")


if __name__ == "__main__":
    unittest.main()
