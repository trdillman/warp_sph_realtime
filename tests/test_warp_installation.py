"""Isolation, restart, tamper, and cleanup tests requiring no CUDA/Blender."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from warp_sph_realtime import warp_installer as installer
from warp_sph_realtime import warp_runtime

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
        # Restore original implementation from this module, not the setUp stub.
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

    # ---- promotion rollback and crash recovery ----------------------------------------------

    def _repair_wheel(self):
        wheel = Path(self.temp.name) / "repair.whl"
        wheel.write_bytes(self.data)
        return wheel

    def _snapshot(self):
        return {p.relative_to(self.owned).as_posix(): (p.read_bytes() if p.is_file() else None)
                for p in self.owned.rglob("*")}

    def _install_marked(self):
        """Install, then tag the copy so a restore is provably the previous install, not the new one."""
        self.install()
        (self.owned / "warp" / "original.txt").write_text("previous install", encoding="utf8")
        return self._snapshot()

    @staticmethod
    def _failing_rename(should_fail):
        """Patch os.rename; ``should_fail(src, dst, calls)`` raises a locked-file error for chosen moves."""
        real = os.rename
        calls = []

        def rename(src, dst):
            calls.append((Path(src), Path(dst)))
            if should_fail(Path(src), Path(dst), calls):
                raise PermissionError("simulated locked file")
            return real(src, dst)
        return mock.patch.object(installer.os, "rename", side_effect=rename)

    @staticmethod
    def _nth_move_from(folder, n):
        def fail(src, _dst, calls):
            return src.parent.name == folder and sum(c[0].parent.name == folder for c in calls) == n
        return fail

    def test_partial_promotion_failure_restores_previous_install_exactly(self):
        # The second move of each phase fails, so one entry has already moved: the partial-move path.
        phases = {
            "backup": lambda src, dst, calls: dst.parent.name == "backup"
            and sum(c[1].parent.name == "backup" for c in calls) == 2,
            "staging": self._nth_move_from("staging", 2),
        }
        for phase, fail in phases.items():
            with self.subTest(phase=phase):
                before = self._install_marked()
                with self._failing_rename(fail), self.assertRaisesRegex(OSError, "locked"):
                    installer._promote_verified_wheel(self._repair_wheel(), self.owned, self.wheel)
                self.assertEqual(self._snapshot(), before)  # no journal, no nesting, original bytes
                self.assertFalse((self.owned / "warp" / "warp").exists())
                self.assertTrue(installer._is_managed(self.owned))

    def test_locked_file_during_rollback_keeps_backup_until_recovered(self):
        before = self._install_marked()
        first_staging_fail = self._nth_move_from("staging", 2)
        locked = {"on": True}

        def fail(src, dst, calls):  # the install fails, then the new 'warp' cannot be moved aside
            return first_staging_fail(src, dst, calls) or (locked["on"] and dst.parent.name == "trash")
        real_rmtree = shutil.rmtree

        def rmtree(path, ignore_errors=False, **kwargs):  # a DLL held open by a scanner
            if locked["on"] and not ignore_errors:
                raise PermissionError("simulated locked DLL")
            return real_rmtree(path, ignore_errors=ignore_errors, **kwargs)
        with self._failing_rename(fail), mock.patch.object(installer.shutil, "rmtree", side_effect=rmtree):
            with self.assertRaisesRegex(RuntimeError, "rollback did not finish"):
                installer._promote_verified_wheel(self._repair_wheel(), self.owned, self.wheel)
        backup = self.owned / ".warpsph-promote" / "backup"
        self.assertEqual((backup / "warp" / "original.txt").read_text("utf8"), "previous install")
        locked["on"] = False
        result = installer.install_warp()  # the next press restores the previous install first
        self.assertTrue(result.already_installed, result.message)
        self.assertEqual(self._snapshot(), before)

    def test_committed_journal_with_locked_leftover_never_rolls_back_good_install(self):
        good = self._install_marked()
        work = self.owned / ".warpsph-promote"
        (work / "backup" / "warp").mkdir(parents=True)
        (work / "backup" / "warp" / "locked.dll").write_bytes(b"old")
        for marker in ("BACKED_UP", "COMMITTED"):
            (work / marker).touch()
        def rmtree(path, ignore_errors=False, **_kwargs):  # best-effort delete; the locked DLL survives
            for p in [*sorted(Path(path).rglob("*"), reverse=True), Path(path)]:
                with contextlib.suppress(OSError):
                    if p.is_dir():
                        p.rmdir()  # fails while not empty, like Windows
                    elif p.name != "locked.dll":
                        p.unlink()

        def discard(_path):  # renaming a directory with a locked file inside fails
            raise PermissionError("simulated locked DLL")
        with mock.patch.object(installer.shutil, "rmtree", side_effect=rmtree), \
                mock.patch.object(installer, "_discard", side_effect=discard):
            for _ in range(2):
                self.assertTrue(installer._recover_interrupted_promotion(self.owned))
        self.assertTrue((work / "COMMITTED").exists())
        installer._recover_interrupted_promotion(self.owned)  # the lock is gone: journal cleared
        self.assertEqual(self._snapshot(), good)

    def test_plan_install_refuses_while_an_install_runs(self):
        installer.begin_install_activity()
        self.addCleanup(installer.end_install_activity)
        result = installer.plan_install()
        self.assertIsInstance(result, installer.InstallResult)
        self.assertFalse(result.success)
        self.assertIn("already running", result.message)

    def test_crash_mid_promotion_is_repaired_by_next_install_or_start(self):
        for existing in (False, True):
            with self.subTest(existing=existing):
                self.owned = Path(self.temp.name) / f"crash-{existing}" / ".local"
                before = self._install_marked() if existing else None
                # The process dies after one new entry moved in: no rollback runs.
                with self._failing_rename(self._nth_move_from("staging", 2)), \
                        mock.patch.object(installer, "_recover_interrupted_promotion", return_value=False), \
                        self.assertRaises(OSError):
                    installer._promote_verified_wheel(self._repair_wheel(), self.owned, self.wheel)
                self.assertFalse(installer._is_managed(self.owned))
                if existing:
                    installer.apply_pending_repair_on_startup()
                    self.assertEqual(self._snapshot(), before)
                else:
                    result = installer.install_warp()
                    self.assertTrue(result.success, result.message)
                    self.assertTrue(installer._is_managed(self.owned))
                    self.assertFalse((self.owned / ".warpsph-promote").exists())

    def test_download_reads_small_chunks_with_short_timeout(self):
        destination = Path(self.temp.name) / "download.whl"
        data, sizes, timeouts = self.data, [], []

        class Response:
            position = 0
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def geturl(self): return "https://files.pythonhosted.org/packages/test.whl"
            def read(self, size):
                sizes.append(size)
                chunk = data[self.position:self.position + size]
                self.position += len(chunk)
                return chunk

        def fake_urlopen(_url, timeout):
            timeouts.append(timeout)
            return Response()
        with mock.patch.object(installer, "urlopen", side_effect=fake_urlopen):
            _original_download(self.wheel.url, destination,
                               expected_sha256=self.wheel.sha256, expected_size=self.wheel.size)
        self.assertEqual(destination.read_bytes(), self.data)
        self.assertLessEqual(max(sizes), 64 * 1024)  # Cancel is checked between small reads
        self.assertLessEqual(timeouts[0], 10)        # a stalled read gives up quickly

    def test_cancel_during_stalled_read_is_reported_as_cancelled(self):
        pressed = {"cancel": False}

        def stalled(url, dest, **kwargs):
            pressed["cancel"] = True  # Cancel pressed while the read was blocked; it then times out
            raise TimeoutError("The read operation timed out")
        self.patch("_download_wheel", stalled)
        plan = installer.plan_install()
        assert isinstance(plan, installer.InstallPlan)
        result = installer.run_plan(plan, cancel=lambda: pressed["cancel"])
        self.assertTrue(result.cancelled, result.message)
        self.assertFalse(self.owned.exists())

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
