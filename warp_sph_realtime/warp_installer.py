"""Install the certified NVIDIA Warp wheel into this extension's own directory.

The pin is ``warp_wheels.lock.json`` next to this module.
The installer downloads only that URL and accepts the file only when its size
and SHA-256 match the lock.

Blender installs bundled extension wheels by extracting them into a
site-packages directory (see the 5.2 manual, "Python Wheels"). That shared
``extensions/.local`` directory belongs to every extension, and Blender's
wheel sync removes packages that no enabled extension declares. This button
does not write there, and it does not write into the directory a Warp module
was imported from.

The only directory it creates is the one Blender assigns to this extension
via ``bpy.utils.extension_path_user`` (see the 5.2 manual, "Add-ons"):
``<extensions>/.user/<repository>/<extension id>/.local``. The wheel
is extracted into that directory the same way Blender extracts a wheel,
without pip. A second press when that owned copy is already Warp 1.17.0
reports success and does not download or write.

Install steps that read Blender (``plan_install``) run on the main thread; the
slow steps (``run_plan``: download or local-wheel copy, hash check, extraction,
promotion) touch no ``bpy`` state, so ``install_job`` can run them on a worker
thread and cancel them. Nothing reaches the owned directory until the wheel is
fully downloaded and verified.

Warp source (Preferences): *Download* and *Install from local wheel file* both
end in that owned directory. *Use existing Warp folder* leaves the owned install
alone and imports Warp from a folder the user chose; WarpSPH only reads it.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import platform as _platform
import re
import shutil
import stat
import sys
import tempfile
import threading
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from .warp_runtime import CERTIFIED_WARP_VERSION, folder_version_support, warp_version_support

_CHUNK_SIZE = 1024 * 1024
# Small reads and a short socket timeout bound how long Cancel waits on a slow or stalled download.
_DOWNLOAD_CHUNK_SIZE = 64 * 1024
_DOWNLOAD_TIMEOUT = 10
_LOCK_PATH = Path(__file__).with_name("warp_wheels.lock.json")
_OWNED_LOCAL_DIR = ".local"
_RECEIPT = ".warpsph-owned.json"
_PENDING_REPAIR_WHEEL = ".warpsph-warp-repair.whl"
_PENDING_REPAIR_PID = ".warpsph-warp-repair.pid"
# Promotion journal inside the owned directory (same volume, so every move is an atomic rename).
_PROMOTE_DIR = ".warpsph-promote"
_DISCARD_PREFIX = ".warpsph-discard-"
_BACKED_UP = "BACKED_UP"    # every previous Warp entry is in backup/; destination Warp entries are new
_COMMITTED = "COMMITTED"    # the new install and its receipt are in place; backup/ may be deleted

_active_installs = 0
_active_installs_lock = threading.Lock()


@dataclass(frozen=True)
class WheelInfo:
    """Metadata for one locked Warp wheel."""

    filename: str
    sha256: str
    size: int
    url: str
    platform: str


@dataclass
class InstallResult:
    """Result of a Warp installation attempt."""

    success: bool
    message: str
    version: str = ""
    already_installed: bool = False
    cancelled: bool = False


class InstallCancelled(Exception):
    """The user cancelled; raised inside the worker and turned into a result."""


@dataclass(frozen=True)
class InstallPlan:
    """Everything ``run_plan`` needs, resolved on the main thread (no ``bpy`` afterwards)."""

    wheel_info: WheelInfo
    destination: Path
    stage_repair: bool = False
    local_wheel: Path | None = None


ProgressFn = Callable[[str, float], None]  # (phase message, overall fraction 0..1)
CancelFn = Callable[[], bool]


def _check_cancel(cancel: CancelFn | None) -> None:
    if cancel is not None and cancel():
        raise InstallCancelled()


def begin_install_activity() -> None:
    """Mark an install as running (from plan to result) so Start cannot import Warp meanwhile."""
    global _active_installs
    with _active_installs_lock:
        _active_installs += 1


def end_install_activity() -> None:
    global _active_installs
    with _active_installs_lock:
        _active_installs = max(_active_installs - 1, 0)


def install_running() -> bool:
    with _active_installs_lock:
        return _active_installs > 0


def _host_platform() -> str | None:
    machine = _platform.machine().lower()
    if machine not in ("amd64", "x86_64"):
        return None
    return {"win32": "windows-x64", "linux": "linux-x64"}.get(sys.platform)


def _load_wheel_lock() -> dict:
    if not _LOCK_PATH.is_file():
        raise RuntimeError(f"Warp wheel lock is missing: {_LOCK_PATH.name}")
    return json.loads(_LOCK_PATH.read_text(encoding="utf-8-sig"))


def get_wheel_info_for_platform(platform: str | None = None) -> WheelInfo | None:
    """Return the locked wheel for ``platform``, or the host platform."""
    if platform is None:
        platform = _host_platform()
    if platform is None:
        return None
    lock = _load_wheel_lock()
    for filename, meta in lock.get("files", {}).items():
        if meta.get("platform") == platform:
            return WheelInfo(
                filename=filename,
                sha256=str(meta["sha256"]),
                size=int(meta["size"]),
                url=str(meta["url"]),
                platform=platform,
            )
    return None


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_wheel(
    url: str,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel: CancelFn | None = None,
) -> None:
    """Download one locked wheel and reject it unless the checksum matches."""
    url_parts = urlparse(url)
    if url_parts.scheme != "https" or url_parts.hostname != "files.pythonhosted.org":
        raise RuntimeError("Refusing untrusted Warp download URL; expected HTTPS files.pythonhosted.org")

    temp = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        _check_cancel(cancel)
        with urlopen(url, timeout=_DOWNLOAD_TIMEOUT) as response:
            final = urlparse(response.geturl())
            if final.scheme != "https" or final.hostname != "files.pythonhosted.org":
                raise RuntimeError("Refusing redirect to an untrusted Warp download host")
            with temp.open("wb") as handle:
                while chunk := response.read(_DOWNLOAD_CHUNK_SIZE):
                    _check_cancel(cancel)
                    size += len(chunk)
                    if size > expected_size:
                        raise RuntimeError("Wheel exceeds locked byte size")
                    handle.write(chunk)
                    digest.update(chunk)
                    if progress_callback:
                        progress_callback(size, expected_size)
                handle.flush()
                os.fsync(handle.fileno())

        if size != expected_size:
            raise RuntimeError(
                f"Downloaded wheel size mismatch: expected {expected_size}, got {size}"
            )
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise RuntimeError(
                "Downloaded wheel checksum mismatch: "
                f"expected {expected_sha256}, got {actual}"
            )
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _user_site_root() -> Path | None:
    import site

    candidates: list[Path] = []
    getter = getattr(site, "getusersitepackages", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:  # noqa: BLE001 - a missing user site is not an install
            value = None
        if value:
            candidates.append(Path(value))
    user_site = getattr(site, "USER_SITE", None)
    if user_site:
        candidates.append(Path(user_site))
    for candidate in candidates:
        if candidate:
            return candidate
    return None


def _user_site_warp_path() -> Path | None:
    """Only a user-site that Python can actually import is a conflict.

    A directory existing in USER_SITE is not a loaded foreign runtime. With
    PYTHONNOUSERSITE, isolated Blender 5.2 must be able to use its own copy.
    """
    root = _user_site_root()
    if root is None:
        return None
    if sys.flags.no_user_site:
        return None
    try:
        import site
        if site.ENABLE_USER_SITE is False:
            return None
    except ImportError:
        pass
    if not any(_is_relative_to(root, Path(p)) for p in sys.path if p):
        return None
    init = root / "warp" / "__init__.py"
    if init.is_file():
        return init.resolve()
    return None


def _blender_extensions_dir() -> Path | None:
    try:
        import bpy
    except ImportError:
        return None
    resource = getattr(getattr(bpy, "utils", None), "user_resource", None)
    if not callable(resource):
        return None
    try:
        value = resource("EXTENSIONS")
    except Exception:  # noqa: BLE001 - preferences must still render
        return None
    if not value:
        return None
    return Path(value)


def _default_install_target(*, create: bool = False) -> Path:
    """Return the directory this add-on owns for Warp installation.

    Blender's extension user site for this add-on
    (``bpy.utils.extension_path_user``). It is never the directory an
    imported ``warp`` module came from, it is never
    ``site.getusersitepackages()``, and it is never the shared
    Blender ``extensions/.local`` directory. ``create`` is false on the
    already-installed check so a second press does not write.
    """
    try:
        import bpy
    except ImportError as exc:
        raise RuntimeError(
            "Install Warp must run inside Blender so it can use this "
            "add-on's own directory."
        ) from exc

    extension_path_user = getattr(getattr(bpy, "utils", None), "extension_path_user", None)
    package = __package__ or ""
    if callable(extension_path_user) and package:
        try:
            value = extension_path_user(package, path=_OWNED_LOCAL_DIR, create=create)
            if value:
                target = Path(value)
                user_site = _user_site_root()
                if user_site is not None and (
                    target.resolve() == user_site.resolve() or _is_relative_to(target, user_site)
                ):
                    raise RuntimeError(
                        "Refusing to write Warp into Blender's user site-packages. "
                        "Remove that copy, then press Install Warp."
                    )
                extensions = _blender_extensions_dir()
                if extensions is not None:
                    shared_local = extensions / ".local"
                    if target.resolve() == shared_local.resolve() or _is_relative_to(target, shared_local):
                        raise RuntimeError(
                            "Refusing to write Warp into Blender's shared extensions directory. "
                            "WarpSPH requires its own extension directory."
                        )
                return target
        except Exception as exc:
            if "does not name an extension" not in str(exc) and "Refusing" in str(exc):
                raise
            if "does not name an extension" not in str(exc):
                raise

    raise RuntimeError(
        "Could not determine this add-on's extension directory. "
        "Install Warp must run inside Blender with extensions enabled."
    )


SOURCE_DOWNLOAD = "DOWNLOAD"
SOURCE_WHEEL = "WHEEL"
SOURCE_FOLDER = "FOLDER"
SOURCE_LABELS = {
    SOURCE_DOWNLOAD: "Download",
    SOURCE_WHEEL: "Install from local wheel file",
    SOURCE_FOLDER: "Use existing Warp folder",
}


@dataclass(frozen=True)
class WarpSource:
    """The Warp source chosen in Preferences."""

    kind: str = SOURCE_DOWNLOAD
    folder: Path | None = None
    allow_uncertified: bool = False

    @property
    def is_folder(self) -> bool:
        return self.kind == SOURCE_FOLDER


@dataclass(frozen=True)
class FolderCheck:
    """Static (nothing imported) verdict on a user-chosen Warp folder."""

    ok: bool
    message: str
    version: str = ""
    uncertified: bool = False


def _read_source_preference() -> WarpSource:
    """Read the Preferences choice; any failure means the default (Download)."""
    try:
        import bpy

        prefs = bpy.context.preferences.addons[__package__ or ""].preferences
        kind = getattr(prefs, "warp_source", SOURCE_DOWNLOAD)
        if kind not in SOURCE_LABELS:
            return WarpSource()
        raw = getattr(prefs, "warp_folder_path", "")
        folder = None
        if isinstance(raw, str) and raw.strip():
            folder = Path(os.path.expanduser(bpy.path.abspath(raw.strip())))
        return WarpSource(kind, folder, getattr(prefs, "warp_allow_uncertified", False) is True)
    except Exception:  # noqa: BLE001 - no Blender, or preferences not registered yet
        return WarpSource()


def active_source() -> WarpSource:
    return _read_source_preference()


def _folder_config_version(folder: Path) -> str | None:
    """Version of the Warp package in ``folder`` without importing it."""
    for entry in sorted(folder.glob("warp_lang-*.dist-info")):
        stem = entry.name[: -len(".dist-info")]
        version = stem.split("-", 1)[1] if "-" in stem else ""
        if version:
            return version
    try:
        text = (folder / "warp" / "config.py").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    found = re.search(r"^version\s*(?::\s*str\s*)?=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
    return found.group(1) if found else None


def check_existing_folder(folder: Path | None, allow_uncertified: bool = False) -> FolderCheck:
    """Read-only checks on a folder that should contain a ``warp`` package.

    The CUDA driver and known-answer kernel checks (P0.7) need Warp imported, so
    they run at Start in ``warp_runtime.require_cuda_runtime``.
    """
    if folder is None:
        return FolderCheck(False, "Choose the folder that contains the 'warp' package")
    if not folder.is_dir():
        return FolderCheck(False, f"Existing Warp folder not found: {folder}")
    if not (folder / "warp" / "__init__.py").is_file():
        if folder.name == "warp" and (folder / "__init__.py").is_file():
            return FolderCheck(
                False, f"{folder} is the warp package itself; choose its parent folder (the one that contains 'warp')"
            )
        return FolderCheck(False, f"No 'warp' package in {folder} (expected {folder / 'warp' / '__init__.py'})")
    version = _folder_config_version(folder)
    if not version:
        return FolderCheck(
            False, f"Cannot determine the Warp version in {folder} (no warp_lang dist-info or warp/config.py version)"
        )
    support, reason = folder_version_support(version, allow_uncertified)
    if support == "unsupported":
        return FolderCheck(False, reason, version=version)
    return FolderCheck(True, reason or f"Warp {version}", version=version, uncertified=support == "uncertified")


def runtime_root() -> Path:
    """The one directory Warp may be imported from under the chosen source."""
    source = active_source()
    if source.is_folder:
        if source.folder is None:
            raise RuntimeError("Choose the folder that contains the 'warp' package")
        return source.folder
    return _default_install_target(create=False)


def describe_source() -> str:
    """One line for diagnostics: which source is in use and in what state."""
    source = active_source()
    label = SOURCE_LABELS[source.kind]
    if source.is_folder:
        check = check_existing_folder(source.folder, source.allow_uncertified)
        if not check.ok:
            return f"{label}: {source.folder or 'no folder chosen'} - {check.message}"
        flag = "UNCERTIFIED, unsupported" if check.uncertified else "certified series"
        return f"{label}: {source.folder} (Warp {check.version}, {flag}, not modified)"
    blocked = foreign_warp_block_reason()
    if blocked:
        return f"{label}: foreign: {blocked}"
    owned = "owned (WarpSPH-managed install)" if owned_install_status() else "not installed"
    return f"{label}: {owned}"


def download_block_reason() -> str | None:
    """Why the Download source cannot run (Blender 'Allow Online Access' off), else ``None``."""
    return _online_access_block_reason()


_ADDED_FOLDERS: set[str] = set()  # existing-folder paths this process put on sys.path


def _drop_unselected_paths(source: WarpSource) -> None:
    """Take the other source's directory off sys.path so it cannot shadow the chosen one."""
    if "warp" in sys.modules:
        return
    if source.is_folder:
        stale = [str(owned) for owned in _owned_candidates()]
    else:
        stale = list(_ADDED_FOLDERS)
        _ADDED_FOLDERS.clear()
    for value in stale:
        while value in sys.path:
            sys.path.remove(value)


def _select_import_path(source: WarpSource) -> None:
    """Make sys.path match the chosen source (before Warp is imported)."""
    _drop_unselected_paths(source)
    if source.is_folder and source.folder is not None:
        value = str(source.folder)
        if value not in sys.path:
            sys.path.append(value)  # last: a site-packages folder must not shadow Blender's own numpy etc.
        _ADDED_FOLDERS.add(value)


def apply_pending_repair_on_startup() -> None:
    if "warp" in sys.modules:
        return
    for owned in _owned_candidates():
        if owned.is_dir():
            _recover_interrupted_promotion(owned)
            _apply_pending_repair(owned)

def ensure_installed_warp_on_path() -> None:
    if install_running():
        raise RuntimeError("A Warp install is still running; wait for it to finish (or cancel it), then press Start")
    block_reason = foreign_warp_block_reason()
    if block_reason:
        raise RuntimeError(block_reason)
    source = active_source()
    if source.is_folder:
        check = check_existing_folder(source.folder, source.allow_uncertified)
        if not check.ok:
            raise RuntimeError(check.message)
        _select_import_path(source)
        return
    _select_import_path(source)
    for owned in _owned_candidates():
        if owned.is_dir():
            _recover_interrupted_promotion(owned)
            _apply_pending_repair(owned)
            if not _is_managed(owned) or _owned_distribution_version(owned) != CERTIFIED_WARP_VERSION:
                raise RuntimeError("Warp installation is unverified or unmanaged")
            value = str(owned)
            if value not in sys.path:
                sys.path.append(value)

def _importable_warp_file() -> Path | None:
    """Return the Warp file Python would import, without executing it."""
    module = sys.modules.get("warp")
    if module is not None:
        file_name = getattr(module, "__file__", None)
        if file_name:
            return Path(file_name).resolve()
    spec = importlib.util.find_spec("warp")
    origin = getattr(spec, "origin", None) if spec is not None else None
    if not origin or origin == "namespace":
        return None
    return Path(origin).resolve()


def _owned_distribution_version(site_packages: Path) -> str | None:
    """Return the Warp version extracted into ``site_packages``, if complete."""
    if not site_packages.is_dir():
        return None
    if not (site_packages / "warp" / "__init__.py").is_file():
        return None
    versions = []
    for entry in site_packages.glob("warp_lang-*.dist-info"):
        stem = entry.name[: -len(".dist-info")]
        version = stem.split("-", 1)[1] if "-" in stem else ""
        if version:
            versions.append(version.split("+", 1)[0])
    if not versions:
        return None
    return versions[-1]


def _owned_candidates() -> list[Path]:
    candidates: list[Path] = []
    try:
        primary = _default_install_target(create=False)
        candidates.append(primary)
    except RuntimeError:
        pass
    return candidates


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


def _foreign_warp_version(init_file: Path) -> str:
    """Best-effort version of a Warp we do not own; never imports or executes it."""
    module = sys.modules.get("warp")
    module_file = getattr(module, "__file__", None)
    if module_file and _same_path(Path(module_file), init_file):
        version = getattr(module, "__version__", None)
        if version:
            return str(version)
    for entry in sorted(init_file.parent.parent.glob("warp_lang-*.dist-info")):
        stem = entry.name[: -len(".dist-info")]
        if "-" in stem:
            return stem.split("-", 1)[1]
    return "unknown version"


def foreign_warp_block_reason() -> str | None:
    """Return why WarpSPH will not write, or ``None`` when no foreign Warp is present."""
    source = active_source()
    chosen = source.folder if source.is_folder else None
    added_before = set(_ADDED_FOLDERS)
    _drop_unselected_paths(source)
    user_site = _user_site_warp_path()
    if user_site is not None and not (chosen is not None and _is_relative_to(user_site, chosen)):
        return (
            f"Warp {_foreign_warp_version(user_site)} is already installed in Blender's "
            f"user site-packages ({user_site.parent}). Remove that copy, then press Install Warp. "
            "WarpSPH will not change it."
        )
    if source.is_folder:
        if chosen is None:
            return None  # nothing chosen yet: check_existing_folder says so
        accepted = [chosen]
    else:
        accepted = _owned_candidates()
        if not accepted:
            try:
                _default_install_target(create=False)
            except RuntimeError as exc:
                return str(exc)
    loaded = _importable_warp_file()
    if loaded is not None and not any(_is_relative_to(loaded, root) for root in accepted):
        version = _foreign_warp_version(loaded)
        if any(_is_relative_to(loaded, Path(value)) for value in added_before):
            return (
                f"Warp {version} is already loaded from the existing folder {loaded.parent.parent}. "
                "Restart Blender to switch the Warp source."
            )
        if source.is_folder:
            return (
                f"Warp {version} is already loaded from {loaded}, which is not the Warp folder you chose "
                f"({chosen}). Restart Blender to switch the Warp source; WarpSPH will not change that copy."
            )
        return (
            f"Warp {version} is already loaded from {loaded}, which WarpSPH does not own. "
            "Remove that copy, then press Install Warp. WarpSPH will not change it."
        )
    return None


def owned_install_status() -> str | None:
    """Return ``certified``, ``untested``, ``unsupported``, or ``None``."""
    for owned in _owned_candidates():
        version = _owned_distribution_version(owned)
        if version is not None:
            status, _message = warp_version_support(version)
            return status
    return None


def _online_access_block_reason() -> str | None:
    """Return Blender's online-access refusal, or ``None`` when downloads are allowed.

    This reads ``bpy.app.online_access``. ``--offline-mode`` sets
    ``bpy.app.online_access_override`` and forces that flag off. Outside
    Blender there is no flag to read.
    """
    try:
        import bpy
    except ImportError:
        return None
    app = getattr(bpy, "app", None)
    if app is None or not hasattr(app, "online_access"):
        return None
    if app.online_access:
        return None
    if getattr(app, "online_access_override", False):
        return (
            "Online access required to install Warp. "
            "Launch Blender without --offline-mode"
        )
    return (
        "Online access required to install Warp. "
        "Enable online access in System preferences"
    )


def _filesystem_path(path: Path) -> Path:
    """Allow long Windows paths even when Blender's executable is not opted in."""
    if os.name != "nt":
        return path
    absolute = str(path.resolve())
    if absolute.startswith("\\\\?\\"):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)



def _receipt_path(dest: Path) -> Path:
    return dest / _RECEIPT

def _is_managed(dest: Path) -> bool:
    """Do not overwrite an unrecognized installation, even in our storage."""
    try:
        wheel = get_wheel_info_for_platform()
        return bool(wheel and json.loads(_receipt_path(dest).read_text("utf8")) ==
                    {"version": CERTIFIED_WARP_VERSION, "platform": wheel.platform,
                     "sha256": wheel.sha256})
    except (OSError, ValueError, TypeError, KeyError):
        return False

def _has_warp(dest: Path) -> bool:
    return ((dest / "warp").exists() or (dest / "warp.libs").exists()
            or any(dest.glob("warp_lang-*.dist-info")))

def _is_warp_entry(name: str) -> bool:
    """A top-level name a Warp install owns: the package, its libs, its dist-info and our receipt."""
    return name in {"warp", "warp.libs", _RECEIPT} or (name.startswith("warp_lang-") and name.endswith(".dist-info"))

def _delete_warp_entries(dest: Path) -> None:
    dest = _filesystem_path(dest)
    if not dest.is_dir():
        return
    for child in dest.iterdir():
        if not _is_warp_entry(child.name):
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()

def _receipt_write(dest: Path, wheel: WheelInfo) -> None:
    data = {"version": CERTIFIED_WARP_VERSION, "platform": wheel.platform, "sha256": wheel.sha256}
    partial = dest / (_RECEIPT + ".part")
    partial.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    os.replace(_filesystem_path(partial), _filesystem_path(_receipt_path(dest)))

def _clear_owned_warp(site_packages: Path) -> None:
    """Only remove WarpSPH-receipted files, never foreign or unrelated files."""
    if not _is_managed(site_packages):
        raise RuntimeError("Refusing cleanup: WarpSPH ownership receipt missing or invalid")
    _delete_warp_entries(site_packages)

def _pending_repair_path(site_packages: Path) -> Path:
    return site_packages / _PENDING_REPAIR_WHEEL


def _pending_repair_pid_path(site_packages: Path) -> Path:
    return site_packages / _PENDING_REPAIR_PID


def _owned_warp_is_loaded(site_packages: Path) -> bool:
    module = sys.modules.get("warp")
    file_name = getattr(module, "__file__", None) if module is not None else None
    if not file_name:
        return False
    try:
        return _is_relative_to(Path(file_name).resolve(), site_packages)
    except OSError:
        return False


def _wheel_matches_lock(wheel: Path, wheel_info: WheelInfo) -> bool:
    return (
        wheel.is_file()
        and wheel.stat().st_size == wheel_info.size
        and _sha256_path(wheel) == wheel_info.sha256
    )


def _stage_pending_repair(wheel: Path, destination: Path, wheel_info: WheelInfo) -> None:
    """Copy a verified wheel into owned storage without touching loaded files."""
    destination.mkdir(parents=True, exist_ok=True)
    pending = _pending_repair_path(destination)
    partial = pending.with_name(pending.name + ".part")
    pid_marker = _pending_repair_pid_path(destination)
    pid_partial = pid_marker.with_name(pid_marker.name + ".part")
    try:
        shutil.copyfile(wheel, partial)
        if not _wheel_matches_lock(partial, wheel_info):
            raise RuntimeError("Staged Warp repair failed checksum or size verification")
        pid_partial.write_text(str(os.getpid()), encoding="ascii")
        pid_partial.replace(pid_marker)
        partial.replace(pending)
    finally:
        partial.unlink(missing_ok=True)
        pid_partial.unlink(missing_ok=True)


def _apply_pending_repair(destination: Path) -> bool:
    pending = _pending_repair_path(destination)
    if not pending.is_file():
        return False
    try:
        pid = int(_pending_repair_pid_path(destination).read_text(encoding="ascii"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("Pending repair marker invalid; use Repair Warp again") from exc
    if pid == os.getpid() or _owned_warp_is_loaded(destination):
        raise RuntimeError("Warp repair staged: restart Blender before using the solver")
    info = get_wheel_info_for_platform()
    if info is None or not _wheel_matches_lock(pending, info):
        raise RuntimeError("Pending Warp repair failed SHA-256 or size verification")
    _promote_verified_wheel(pending, destination, info)
    pending.unlink()
    _pending_repair_pid_path(destination).unlink(missing_ok=True)
    return True

def _promote_verified_wheel(
    wheel: Path,
    destination: Path,
    info: WheelInfo,
    *,
    cancel: CancelFn | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Promote only a locked wheel; any failure rolls back to the previous install.

    The wheel is extracted into a journal directory inside ``destination``
    (``.warpsph-promote``), so every move is an atomic same-volume rename: no
    copy+delete fallback, never a half-moved package. The previous install is
    renamed into ``.warpsph-promote/backup`` and is deleted only after the new
    install and its receipt are in place. If a rollback step fails (a DLL held
    open by a scanner) or the process dies mid-promotion, the journal stays on
    disk and ``_recover_interrupted_promotion`` restores the previous install on
    the next Install/Repair press or Blender start.

    Cancel is honored while extracting into staging. Once files start moving
    into ``destination`` the promotion runs to the end (or rolls back), so a
    cancel never leaves half an install.
    """
    if not _wheel_matches_lock(wheel, info):
        raise RuntimeError("Wheel differs from immutable SHA-256/size lock")
    _recover_interrupted_promotion(destination)
    if _has_warp(destination) and not _is_managed(destination):
        raise RuntimeError("Unmanaged Warp already occupies the target; refusing overwrite")
    if _owned_warp_is_loaded(destination):
        raise RuntimeError("Native Warp library is still loaded; restart to repair")
    created = not destination.exists()
    destination.mkdir(parents=True, exist_ok=True)
    root = _filesystem_path(destination)
    work = root / _PROMOTE_DIR
    staging = work / "staging"
    backup = work / "backup"
    if work.exists():
        raise RuntimeError(f"Files left by an earlier Warp install are still locked ({work}); restart Blender, then retry")
    work.mkdir()
    try:
        _extract_wheel(wheel, staging, cancel=cancel, progress=progress)
        if _owned_distribution_version(staging) != CERTIFIED_WARP_VERSION:
            raise RuntimeError("Wheel metadata does not match certified version")
        _check_cancel(cancel)
        backup.mkdir()
    except BaseException:
        # Nothing in ``destination`` was touched yet: drop the staging copy.
        with contextlib.suppress(OSError):
            _discard(work)
        if created:
            with contextlib.suppress(OSError):
                root.rmdir()
        raise
    try:
        for entry in list(root.iterdir()):
            if _is_warp_entry(entry.name):
                os.rename(entry, backup / entry.name)
        (work / _BACKED_UP).touch()
        for entry in list(staging.iterdir()):
            os.rename(entry, root / entry.name)
        _receipt_write(destination, info)
        (work / _COMMITTED).touch()
    except BaseException as exc:
        try:
            _recover_interrupted_promotion(destination)
        except Exception as rollback_error:  # noqa: BLE001 - reported together with the original failure
            raise RuntimeError(
                f"Warp install failed ({exc}) and the rollback did not finish ({rollback_error}). "
                f"The previous Warp is kept in {backup} and is restored automatically the next "
                "time you press Install/Repair Warp or restart Blender."
            ) from exc
        raise
    with contextlib.suppress(OSError):
        _discard(work)  # the new install is verified in place; a locked leftover is retried later


def _discard(path: Path) -> None:
    """Rename ``path`` aside atomically, then delete it best-effort (a locked file just lingers)."""
    aside = path.with_name(f"{_DISCARD_PREFIX}{uuid.uuid4().hex}")
    os.rename(path, aside)
    shutil.rmtree(aside, ignore_errors=True)


def _recover_interrupted_promotion(destination: Path) -> bool:
    """Finish or undo a promotion left half-done by a crash or a failed rollback.

    Idempotent and safe to retry: every step is a same-volume rename, and the
    ``BACKED_UP`` marker is removed only once the new entries are out of the way,
    so a retry never mistakes a restored install for a new one. Raises
    ``OSError`` (journal kept) while a file is still locked. Returns whether a
    journal was found.
    """
    root = _filesystem_path(destination)
    if not root.is_dir():
        return False
    for leftover in root.glob(_DISCARD_PREFIX + "*"):
        shutil.rmtree(leftover, ignore_errors=True)
    work = root / _PROMOTE_DIR
    if not work.is_dir():
        return False
    if not (work / _COMMITTED).exists():
        if (work / _BACKED_UP).exists():
            # Every Warp entry now in ``destination`` came from staging: move it aside.
            trash = work / "trash"
            trash.mkdir(exist_ok=True)
            for entry in list(root.iterdir()):
                if _is_warp_entry(entry.name):
                    os.rename(entry, trash / f"{entry.name}-{uuid.uuid4().hex}")
            (work / _BACKED_UP).unlink()
        (root / (_RECEIPT + ".part")).unlink(missing_ok=True)
        backup = work / "backup"
        if backup.is_dir():
            for entry in list(backup.iterdir()):
                os.rename(entry, root / entry.name)
    # The install is consistent now; what is left in the journal is garbage.
    with contextlib.suppress(OSError):
        _discard(work)
    shutil.rmtree(work, ignore_errors=True)  # the rename fails while a leftover is locked
    return True

def _extract_wheel(
    wheel: Path,
    destination: Path,
    *,
    cancel: CancelFn | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Only expected wheel roots: no symlink, traversal or duplicate archive names."""
    allowed = {"warp", "warp.libs", f"warp_lang-{CERTIFIED_WARP_VERSION}.dist-info"}
    with zipfile.ZipFile(wheel) as archive:
        seen = set()
        for entry in archive.infolist():
            name = entry.filename
            parts = name.rstrip("/").split("/")
            if (name.startswith(("/", ".")) or "\\" in name or ":" in name
                    or any(part in {"", ".", ".."} for part in parts)
                    or parts[0] not in allowed):
                raise RuntimeError(f"Unsafe wheel member: {name}")
            if name in seen:
                raise RuntimeError(f"Duplicate wheel member: {name}")
            seen.add(name)
            if stat.S_IFMT(entry.external_attr >> 16) == stat.S_IFLNK:
                raise RuntimeError(f"Symlink wheel member forbidden: {name}")
        meta = f"warp_lang-{CERTIFIED_WARP_VERSION}.dist-info/METADATA"
        if meta not in seen or "warp/__init__.py" not in seen:
            raise RuntimeError("Wheel missing Warp package or distribution metadata")
        content = archive.read(meta).decode("utf-8", "replace")
        if f"Version: {CERTIFIED_WARP_VERSION}" not in content or "Name: warp-lang" not in content:
            raise RuntimeError("Wheel metadata version/name mismatch")
        destination = _filesystem_path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        members = archive.infolist()
        for index, entry in enumerate(members, 1):
            _check_cancel(cancel)
            archive.extract(entry, destination)
            if progress:
                progress(index, len(members))

def is_warp_installed() -> bool:
    """Return whether some Warp build can be imported."""
    return _importable_warp_file() is not None or owned_install_status() is not None


def is_certified_warp_installed() -> bool:
    """Only a version-matched, WarpSPH-receipted install is certified."""
    return any(_is_managed(p) and _owned_distribution_version(p) == CERTIFIED_WARP_VERSION
               for p in _owned_candidates())


def get_installed_warp_info() -> tuple[bool, str]:
    """Return ``(is_installed, version_or_message)`` for the preferences panel."""
    blocked = foreign_warp_block_reason()
    if blocked:
        return True, blocked
    for owned in _owned_candidates():
        version = _owned_distribution_version(owned)
        if version is not None:
            status, _message = warp_version_support(version)
            if status == "certified":
                return True, (f"Warp {version} (certified)" if _is_managed(owned) else "Unmanaged Warp copy; preserved but untrusted")
            if status == "untested":
                return True, f"Warp {version} (newer than certified {CERTIFIED_WARP_VERSION})"
            return True, f"Warp {version} (incompatible; certified is {CERTIFIED_WARP_VERSION})"
    return False, "Warp is not installed"


def _copy_verified_wheel(
    source: Path, destination: Path, info: WheelInfo, *, progress: Callable[[int, int], None], cancel: CancelFn | None
) -> None:
    """Copy a user-supplied wheel into private temp storage, accepting it only if it matches the lock.

    Hashing the private copy (not the original) closes the window in which the
    chosen file could be swapped between verification and extraction.
    """
    if source.stat().st_size != info.size:
        raise RuntimeError(
            f"{source.name} is not the certified Warp wheel: size {source.stat().st_size} bytes, "
            f"expected {info.size} ({info.filename})"
        )
    temp = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    copied = 0
    try:
        with source.open("rb") as reader, temp.open("wb") as writer:
            while chunk := reader.read(_CHUNK_SIZE):
                _check_cancel(cancel)
                copied += len(chunk)
                if copied > info.size:
                    raise RuntimeError(f"{source.name} grew while being read; refusing it")
                writer.write(chunk)
                digest.update(chunk)
                progress(copied, info.size)
        if digest.hexdigest() != info.sha256:
            raise RuntimeError(
                f"{source.name} does not match the certified SHA-256 lock for {info.filename} "
                f"(expected {info.sha256}, got {digest.hexdigest()})"
            )
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def plan_install(
    *, force_reinstall: bool = False, local_wheel: Path | None = None
) -> InstallPlan | InstallResult:
    """Main-thread half of an install: every check that reads Blender or the live process.

    Returns an ``InstallResult`` when the install must not or need not run
    (blocked, already installed, offline, wrong source) and an ``InstallPlan``
    for ``run_plan`` otherwise. ``local_wheel`` selects the offline path.
    """
    source = active_source()
    if source.is_folder:
        return InstallResult(
            success=False,
            message="The Warp source is 'Use existing Warp folder'; switch it to Download to install Warp.",
        )
    blocked = foreign_warp_block_reason()
    if blocked:
        return InstallResult(success=False, message=blocked)

    for owned in _owned_candidates():
        try:
            _recover_interrupted_promotion(owned)  # a crashed promotion otherwise looks unmanaged forever
        except OSError as exc:
            return InstallResult(
                success=False,
                message=f"An interrupted Warp install could not be rolled back yet ({exc}). Restart Blender, then retry.",
            )

    repairing_certified = False
    owned_version = None
    for owned in _owned_candidates():
        owned_version = _owned_distribution_version(owned)
        if owned_version:
            status, _message = warp_version_support(owned_version)
            if status == "certified":
                if not _is_managed(owned):
                    return InstallResult(False, "Existing Warp is unmanaged; preserved, not overwritten")
                if not force_reinstall:
                    return InstallResult(
                        success=True,
                        message=f"Warp {owned_version} is already installed",
                        version=owned_version,
                        already_installed=True,
                    )
                repairing_certified = True
                break
            if status != "certified":
                return InstallResult(
                    False,
                    f"Existing Warp {owned_version} is unsupported and preserved. "
                    "Do not overwrite it before determining ownership.",
                )

    if local_wheel is None:
        online = _online_access_block_reason()
        if online:
            return InstallResult(success=False, message=online)
    elif not local_wheel.is_file():
        return InstallResult(success=False, message=f"Wheel file not found: {local_wheel}")

    wheel_info = get_wheel_info_for_platform()
    if wheel_info is None:
        host = _host_platform() or "unknown"
        return InstallResult(
            success=False,
            message=f"No certified Warp wheel is published for this platform ({host})",
        )

    try:
        destination = _default_install_target(create=False)
    except RuntimeError as exc:
        return InstallResult(success=False, message=str(exc))
    user_site = _user_site_root()
    if user_site is not None and (
        _is_relative_to(destination, user_site) or destination.resolve() == user_site.resolve()
    ):
        return InstallResult(
            success=False,
            message=(
                "Refusing to write Warp into Blender's user site-packages. "
                "Remove any Warp there, then press Install Warp."
            ),
        )
    extensions = _blender_extensions_dir()
    if extensions is not None:
        shared_local = extensions / ".local"
        if destination.resolve() == shared_local.resolve() or _is_relative_to(destination, shared_local):
            return InstallResult(
                success=False,
                message=(
                    "Refusing to write Warp into Blender's shared extensions directory. "
                    "WarpSPH requires its own extension directory."
                ),
            )
    loaded = _importable_warp_file()
    if loaded is not None and not any(_is_relative_to(loaded, cand) for cand in _owned_candidates()):
        return InstallResult(
            success=False,
            message=(
                f"Warp {_foreign_warp_version(loaded)} is already loaded from {loaded}, which WarpSPH does not own. "
                "Remove that copy, then press Install Warp. WarpSPH will not change it."
            ),
        )
    return InstallPlan(
        wheel_info=wheel_info,
        destination=destination,
        stage_repair=repairing_certified or _owned_warp_is_loaded(destination),
        local_wheel=local_wheel,
    )


def run_plan(
    plan: InstallPlan, *, progress: ProgressFn | None = None, cancel: CancelFn | None = None
) -> InstallResult:
    """Download (or copy) the wheel, verify it, then extract and promote it atomically.

    Safe to run on a worker thread: it never touches ``bpy``. The download
    happens in a temporary directory and extraction in the owned directory's
    promotion journal, so a cancel or failure before promotion leaves the
    installed Warp untouched.
    """

    def report(message: str, fraction: float) -> None:
        if progress:
            progress(message, min(max(fraction, 0.0), 1.0))

    info = plan.wheel_info
    size_mb = info.size / 1024 / 1024
    transfer = (
        f"Verifying local wheel ({size_mb:.1f} MB)..."
        if plan.local_wheel is not None
        else f"Downloading Warp {CERTIFIED_WARP_VERSION} ({size_mb:.1f} MB)..."
    )
    report(transfer, 0.0)
    try:
        with tempfile.TemporaryDirectory(prefix="warpsph-install-") as tmpdir:
            wheel_path = Path(tmpdir) / info.filename

            def on_bytes(done: int, total: int) -> None:
                report(transfer, 0.85 * done / max(total, 1))

            try:
                if plan.local_wheel is not None:
                    _copy_verified_wheel(plan.local_wheel, wheel_path, info, progress=on_bytes, cancel=cancel)
                else:
                    _download_wheel(
                        info.url,
                        wheel_path,
                        expected_sha256=info.sha256,
                        expected_size=info.size,
                        progress_callback=on_bytes,
                        cancel=cancel,
                    )
            except InstallCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - the UI reports the failure
                _check_cancel(cancel)  # a read cut short by Cancel is a cancel, not a failure
                prefix = "Wheel refused" if plan.local_wheel is not None else "Download failed"
                return InstallResult(success=False, message=f"{prefix}: {exc}")
            report("Verifying checksum...", 0.86)
            if _sha256_path(wheel_path) != info.sha256:
                return InstallResult(
                    success=False,
                    message="Refusing to install a wheel that failed checksum verification",
                )
            _check_cancel(cancel)

            report("Installing Warp...", 0.9)

            def on_extract(done: int, total: int) -> None:
                report("Installing Warp...", 0.9 + 0.09 * done / max(total, 1))

            try:
                if plan.stage_repair:
                    _stage_pending_repair(wheel_path, plan.destination, info)
                    return InstallResult(
                        success=True,
                        message=(
                            f"Staged Warp {CERTIFIED_WARP_VERSION} for repair. "
                            "Restart Blender to apply it before running the solver."
                        ),
                        version=CERTIFIED_WARP_VERSION,
                    )
                _promote_verified_wheel(wheel_path, plan.destination, info, cancel=cancel, progress=on_extract)
            except InstallCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - the UI reports the failure
                return InstallResult(success=False, message=f"Installation failed: {exc}")
    except InstallCancelled:
        return InstallResult(success=False, message="Install cancelled; nothing was installed.", cancelled=True)

    installed = _owned_distribution_version(plan.destination)
    if installed is None:
        return InstallResult(
            success=False,
            message=(
                "Installation finished, but the certified Warp files are missing "
                "from this extension's directory."
            ),
        )
    report("Done", 1.0)
    if "warp" in sys.modules:
        return InstallResult(
            success=True,
            message=(
                f"Installed Warp {installed} into this extension's directory. "
                "Restart Blender before running the solver."
            ),
            version=installed,
        )
    return InstallResult(
        success=True,
        message=f"Successfully installed Warp {installed}",
        version=installed,
    )


def install_warp(
    *,
    force_reinstall: bool = False,
    progress_callback: Callable[[str], None] | None = None,
    local_wheel: Path | None = None,
) -> InstallResult:
    """Install the locked Warp wheel into this extension's own directory (blocking).

    The add-on UI uses ``plan_install`` + ``install_job.InstallJob`` instead so
    Blender stays responsive; this is the synchronous equivalent.

    A certified and managed copy is idempotent and never downloaded twice.
    An explicit ``force_reinstall`` stages a locked
    repair for an owned certified copy, applied before import on the next
    process. Incompatible copies are preserved unless provenance is verified. Warp loaded from
    any other path, including another add-on or Blender's user site-packages,
    is left untouched.
    """
    plan = plan_install(force_reinstall=force_reinstall, local_wheel=local_wheel)
    if isinstance(plan, InstallResult):
        return plan
    last: list[str] = []

    def relay(message: str, _fraction: float) -> None:
        if progress_callback and (not last or last[-1] != message):
            last.append(message)
            progress_callback(message)

    begin_install_activity()
    try:
        return run_plan(plan, progress=relay)
    finally:
        end_install_activity()
