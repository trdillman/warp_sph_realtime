"""Isolation, restart, tamper, and cleanup tests requiring no CUDA/Blender."""
from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warp_installer as installer
import warp_runtime
_original_user_site_path = installer._user_site_warp_path
_original_download = installer._download_wheel


def valid_wheel():
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as z:
        z.writestr("warp/__init__.py", '__version__ = "1.17.0"\n')
        z.writestr("warp_lang-1.17.0.dist-info/METADATA", "Name: warp-lang\nVersion: 1.17.0\n")
        z.writestr("warp_lang-1.17.0.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-win_amd64\n")
    return blob.getvalue()


class ManagedWarpTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.owned = Path(self.temp.name) / "user-extension" / ".local"
        self.data = valid_wheel()
        self.wheel = installer.WheelInfo(
            filename="test.whl", sha256=hashlib.sha256(self.data).hexdigest(),
            size=len(self.data), url="https://files.pythonhosted.org/packages/test.whl",
            platform="windows-x64",
        )
        self.old_warp = sys.modules.pop("warp", None)
        self.addCleanup(self._restore_warp)
        self.patch("_default_install_target", lambda create=False: self.owned)
        self.patch("get_wheel_info_for_platform", lambda platform=None: self.wheel)
        self.patch("_user_site_root", lambda: None)
        self.patch("_user_site_warp_path", lambda: None)
        self.patch("_importable_warp_file", lambda: None)
        self.patch("_blender_extensions_dir", lambda: None)
        self.patch("_online_access_block_reason", lambda: None)
        self.download = mock.Mock(side_effect=lambda url, dest, **kwargs: dest.write_bytes(self.data))
        self.patch("_download_wheel", self.download)

    def _restore_warp(self):
        if self.old_warp is None:
            sys.modules.pop("warp", None)
        else:
            sys.modules["warp"] = self.old_warp

    def patch(self, name, replacement):
        p = mock.patch.object(installer, name, replacement)
        p.start()
        self.addCleanup(p.stop)

    def install(self):
        result = installer.install_warp()
        self.assertTrue(result.success, result.message)
        return result

    def test_lock_is_pinned_and_platform_specific(self):
        lock = json.loads(Path(installer.__file__).with_name("warp_wheels.lock.json").read_text("utf8"))
        self.assertEqual(lock["warp_version"], "1.17.0")
        self.assertEqual(lock["distribution_version"], "1.17.0+cu12")
        self.assertEqual(set(e["platform"] for e in lock["files"].values()), {"windows-x64", "linux-x64"})
        self.assertTrue(all(len(v["sha256"]) == 64 and v["url"].startswith("https://") for v in lock["files"].values()))

    def test_install_and_idempotency(self):
        self.install()
        self.assertTrue((self.owned / "warp" / "__init__.py").is_file())
        self.assertTrue(installer._is_managed(self.owned))
        before = {x.relative_to(self.owned): x.read_bytes() for x in self.owned.rglob("*") if x.is_file()}
        second = self.install()
        self.assertTrue(second.already_installed)
        self.download.assert_called_once()
        self.assertEqual(before, {x.relative_to(self.owned): x.read_bytes() for x in self.owned.rglob("*") if x.is_file()})

    def test_download_rejects_bad_host_and_redirect(self):
        destination=Path(self.temp.name)/"download.whl"
        with self.assertRaisesRegex(RuntimeError, "untrusted"):
            _original_download(
                "https://evil.example.com/fake.whl",destination,
                expected_sha256=self.wheel.sha256,expected_size=self.wheel.size
            )
        class FakeRedirect:
            def __enter__(self): return self
            def __exit__(self,*_args): return False
            def geturl(self): return "http://files.pythonhosted.org/insecure"
        with mock.patch.object(installer,"urlopen",return_value=FakeRedirect()):
            with self.assertRaisesRegex(RuntimeError,"redirect"):
                _original_download(
                    self.wheel.url,destination,
                    expected_sha256=self.wheel.sha256,expected_size=self.wheel.size
                )
        self.assertFalse(destination.exists())

    def test_download_aborts_oversized_response(self):
        destination=Path(self.temp.name)/"download.whl"
        class FakeResponse:
            def __init__(self): self.reads=0
            def __enter__(self): return self
            def __exit__(self,*_args): return False
            def geturl(self): return "https://files.pythonhosted.org/packages/test.whl"
            def read(self,_size):
                self.reads+=1
                return b"X"*(self_size+1) if self.reads==1 else b""
        self_size=self.wheel.size
        with mock.patch.object(installer,"urlopen",return_value=FakeResponse()):
            with self.assertRaisesRegex(RuntimeError,"locked byte size"):
                _original_download(
                    self.wheel.url,destination,
                    expected_sha256=self.wheel.sha256,expected_size=self.wheel.size
                )
        self.assertFalse(destination.exists())
        self.assertFalse(destination.with_name(destination.name+".part").exists())

    def test_offline_does_not_download_or_create(self):
        self.patch("_online_access_block_reason", lambda: "Offline mode")
        result = installer.install_warp()
        self.assertFalse(result.success)
        self.assertIn("Offline", result.message)
        self.download.assert_not_called()
        self.assertFalse(self.owned.exists())

    def test_bad_checksum_does_not_mutate_target(self):
        self.patch("_download_wheel", lambda url, dest, **kwargs: dest.write_bytes(b"tampered"))
        result = installer.install_warp()
        self.assertFalse(result.success)
        self.assertFalse((self.owned / "warp").exists())

    def test_foreign_loaded_is_preserved_and_blocks(self):
        foreign = Path(self.temp.name) / "foreign" / "warp" / "__init__.py"
        foreign.parent.mkdir(parents=True)
        foreign.write_text("foreign", encoding="utf8")
        self.patch("_importable_warp_file", lambda: foreign)
        result = installer.install_warp()
        self.assertFalse(result.success)
        self.assertEqual(foreign.read_text("utf8"), "foreign")
        self.download.assert_not_called()

    def test_disabled_user_site_does_not_block_isolated_blender(self):
        import site
        root=Path(self.temp.name)/"user-site"
        foreign=root/"warp"/"__init__.py"
        foreign.parent.mkdir(parents=True)
        foreign.write_text("other users install", encoding="utf8")
        real_check=installer._user_site_warp_path
        # Restore original implementation from this module, not the setUp stub.
        from inspect import getsource
        with mock.patch.object(installer, "_user_site_root", return_value=root):
            # Simulates BLENDER_USER_CONFIG and PYTHONNOUSERSITE profile isolation.
            with mock.patch.object(site, "ENABLE_USER_SITE", False):
                self.assertIsNone(_original_user_site_path())
        self.install()
        self.assertEqual(foreign.read_text("utf8"), "other users install")

    def test_unmanaged_existing_files_never_overwritten(self):
        (self.owned / "warp").mkdir(parents=True)
        (self.owned / "warp" / "__init__.py").write_text("unmanaged", encoding="utf8")
        result = installer.install_warp()
        self.assertFalse(result.success)
        self.assertEqual((self.owned / "warp" / "__init__.py").read_text("utf8"), "unmanaged")
        self.assertFalse(installer._is_managed(self.owned))

    def test_cleanup_refuses_unmanaged_and_preserves_unrelated(self):
        self.owned.mkdir(parents=True)
        keep = self.owned / "other-extension.txt"
        keep.write_text("preserve", encoding="utf8")
        with self.assertRaisesRegex(RuntimeError, "ownership"):
            installer._clear_owned_warp(self.owned)
        self.install()
        installer._clear_owned_warp(self.owned)
        self.assertEqual(keep.read_text("utf8"), "preserve")
        self.assertFalse((self.owned / "warp").exists())

    def test_loaded_repair_stages_until_restart(self):
        self.install()
        path = self.owned / "warp" / "__init__.py"
        initial = path.read_bytes()
        sys.modules["warp"] = SimpleNamespace(__file__=str(path))
        result = installer.install_warp(force_reinstall=True)
        self.assertTrue(result.success, result.message)
        self.assertIn("Restart Blender", result.message)
        self.assertTrue(installer._pending_repair_path(self.owned).is_file())
        self.assertEqual(path.read_bytes(), initial)
        with self.assertRaisesRegex(RuntimeError, "restart"):
            installer._apply_pending_repair(self.owned)
        sys.modules.pop("warp")
        installer._pending_repair_pid_path(self.owned).write_text("0", encoding="ascii")
        self.assertTrue(installer._apply_pending_repair(self.owned))
        self.assertEqual(path.read_bytes(), initial)
        self.assertFalse(installer._pending_repair_path(self.owned).exists())
        self.assertTrue(installer._is_managed(self.owned))

    def test_tampered_pending_repair_preserves_original(self):
        self.install()
        before = (self.owned / "warp" / "__init__.py").read_bytes()
        sys.modules["warp"] = SimpleNamespace(__file__=str(self.owned / "warp" / "__init__.py"))
        result = installer.install_warp(force_reinstall=True)
        self.assertTrue(result.success, result.message)
        sys.modules.pop("warp")
        installer._pending_repair_pid_path(self.owned).write_text("0")
        installer._pending_repair_path(self.owned).write_bytes(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "SHA-256"):
            installer._apply_pending_repair(self.owned)
        self.assertEqual((self.owned / "warp" / "__init__.py").read_bytes(), before)

    def test_repair_not_modify_native_after_metadata_deleted(self):
        self.install()
        (self.owned / "warp_lang-1.17.0.dist-info" / "METADATA").unlink()
        sys.modules["warp"] = SimpleNamespace(__file__=str(self.owned / "warp" / "__init__.py"))
        result = installer.install_warp(force_reinstall=True)
        self.assertTrue(result.success, result.message)
        self.assertTrue(installer._pending_repair_path(self.owned).exists())

    def test_zip_traversal_symlink_and_unexpected_roots(self):
        samples = ["../escaped.txt", "/abs.txt", "warp/../../escape", "other/evil"]
        for name in samples:
            with self.subTest(name=name):
                path = Path(self.temp.name) / "bad.whl"
                with zipfile.ZipFile(path, "w") as z:
                    z.writestr(name, "bad")
                with self.assertRaisesRegex(RuntimeError, "Unsafe"):
                    installer._extract_wheel(path, self.owned)
        path = Path(self.temp.name) / "symlink.whl"
        with zipfile.ZipFile(path, "w") as z:
            i = zipfile.ZipInfo("warp/escape")
            i.create_system = 3
            i.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(i, "../../escape")
        with self.assertRaisesRegex(RuntimeError, "Symlink"):
            installer._extract_wheel(path, self.owned)

    def test_failed_backup_move_keeps_original_receipt(self):
        self.install()
        before = (self.owned/"warp"/"__init__.py").read_bytes()
        wheel=Path(self.temp.name)/"repair.whl"
        wheel.write_bytes(self.data)
        original=installer.shutil.move
        def fail_receipt(source,dest,*args,**kwargs):
            if str(source).endswith(".warpsph-owned.json"):
                raise OSError("simulate backup receipt failure")
            return original(source,dest,*args,**kwargs)
        with mock.patch.object(installer.shutil,"move",side_effect=fail_receipt):
            with self.assertRaisesRegex(OSError,"receipt failure"):
                installer._promote_verified_wheel(wheel,self.owned,self.wheel)
        self.assertTrue(installer._is_managed(self.owned))
        self.assertEqual((self.owned/"warp"/"__init__.py").read_bytes(),before)

    @unittest.skipUnless(sys.platform=="win32","Windows long-path guard")
    def test_long_path_extract_and_cleanup(self):
        deep=Path(self.temp.name)/("warpsph-deep-"*17)/".local"
        wheel=Path(self.temp.name)/"long.whl"
        with zipfile.ZipFile(wheel,"w") as z:
            z.writestr("warp/__init__.py",'__version__="1.17.0"')
            z.writestr("warp_lang-1.17.0.dist-info/METADATA","Name: warp-lang\nVersion: 1.17.0\n")
            z.writestr("warp_lang-1.17.0.dist-info/licenses/"+
                       ("very-long-license-folder-"*3)+"/LICENSE.txt","MIT license")
        self.assertGreater(len(str(deep/"warp_lang-1.17.0.dist-info"/"licenses"/("very-long-license-folder-"*3)/"LICENSE.txt")),260)
        installer._extract_wheel(wheel,deep)
        self.assertTrue((installer._filesystem_path(deep)/"warp"/"__init__.py").is_file())
        license_path=installer._filesystem_path(deep)/"warp_lang-1.17.0.dist-info"/"licenses"/("very-long-license-folder-"*3)/"LICENSE.txt"
        self.assertEqual(license_path.read_text("utf8"),"MIT license")

    def test_promote_failed_move_restores_owned_install(self):
        self.install()
        initial = (self.owned / "warp" / "__init__.py").read_bytes()
        source = Path(self.temp.name) / "repair.whl"
        source.write_bytes(self.data)
        original_move = installer.shutil.move
        fail_once = {"count": 0}
        def simulated_failure(a, b, *args, **kwargs):
            if "staging" in str(a) and fail_once["count"] == 0:
                fail_once["count"] += 1
                raise OSError("disk failure")
            return original_move(a, b, *args, **kwargs)
        with mock.patch.object(installer.shutil, "move", side_effect=simulated_failure):
            with self.assertRaisesRegex(OSError, "disk failure"):
                installer._promote_verified_wheel(source, self.owned, self.wheel)
        self.assertEqual((self.owned / "warp" / "__init__.py").read_bytes(), initial)
        self.assertTrue(installer._is_managed(self.owned))

    def test_runtime_status_refuses_unmanaged(self):
        self.owned.mkdir(parents=True, exist_ok=True)
        (self.owned / "warp").mkdir()
        (self.owned / "warp" / "__init__.py").write_text("not managed")
        (self.owned / "warp_lang-1.17.0.dist-info").mkdir()
        self.assertFalse(warp_runtime.status().available)
        self.assertEqual(warp_runtime.warp_version_support("1.18.0")[0], "unsupported")
        self.assertEqual(warp_runtime.warp_version_support("1.16.0")[0], "unsupported")


if __name__ == "__main__":
    unittest.main()
