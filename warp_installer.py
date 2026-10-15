"""Install the certified NVIDIA Warp wheel into this extension's own directory.

The pin is ``warp_wheels.lock.json`` next to this module. It is a copy of
``tools/release/wheels.lock.json`` and a test keeps the two files equal.
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
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform as _platform
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
import stat
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlparse

try:
    from .warp_runtime import warp_version_support, CERTIFIED_WARP_VERSION
except ImportError:
    from warp_runtime import warp_version_support, CERTIFIED_WARP_VERSION


_CHUNK_SIZE = 1024 * 1024
_LOCK_PATH = Path(__file__).with_name("warp_wheels.lock.json")
_OWNED_LOCAL_DIR = ".local"
_RECEIPT = ".warpsph-owned.json"
_PENDING_REPAIR_WHEEL = ".warpsph-warp-repair.whl"
_PENDING_REPAIR_PID = ".warpsph-warp-repair.pid"


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
) -> None:
    """Download one locked wheel and reject it unless the checksum matches."""
    url_parts = urlparse(url)
    if url_parts.scheme != "https" or url_parts.hostname != "files.pythonhosted.org":
        raise RuntimeError("Refusing untrusted Warp download URL; expected HTTPS files.pythonhosted.org")

    temp = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with urlopen(url, timeout=300) as response:
            final = urlparse(response.geturl())
            if final.scheme != "https" or final.hostname != "files.pythonhosted.org":
                raise RuntimeError("Refusing redirect to an untrusted Warp download host")
            with temp.open("wb") as handle:
                while chunk := response.read(_CHUNK_SIZE):
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


def apply_pending_repair_on_startup() -> None:
    if "warp" in sys.modules:
        return
    for owned in _owned_candidates():
        if owned.is_dir():
            _apply_pending_repair(owned)

def ensure_installed_warp_on_path() -> None:
    block_reason = foreign_warp_block_reason()
    if block_reason:
        raise RuntimeError(block_reason)
    for owned in _owned_candidates():
        if owned.is_dir():
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


def foreign_warp_block_reason() -> str | None:
    """Return why WarpSPH will not write, or ``None`` when no foreign Warp is present."""
    user_site = _user_site_warp_path()
    if user_site is not None:
        return (
            "Warp is already installed in Blender's user site-packages "
            f"({user_site.parent}). Remove that copy, then press Install Warp. "
            "WarpSPH will not change it."
        )
    owned_list = _owned_candidates()
    if not owned_list:
        try:
            _default_install_target(create=False)
        except RuntimeError as exc:
            return str(exc)
    loaded = _importable_warp_file()
    if loaded is not None and not any(_is_relative_to(loaded, owned) for owned in owned_list):
        return (
            f"Warp is already loaded from {loaded}, which WarpSPH does not own. "
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

def _delete_warp_entries(dest: Path) -> None:
    dest = _filesystem_path(dest)
    if not dest.is_dir():
        return
    for child in dest.iterdir():
        name = child.name
        if name not in {"warp", "warp.libs", _RECEIPT} and not (name.startswith("warp_lang-") and name.endswith(".dist-info")):
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

def _promote_verified_wheel(wheel: Path, destination: Path, info: WheelInfo) -> None:
    """Promote only a locked wheel; rollback any failed filesystem operation."""
    if not _wheel_matches_lock(wheel, info):
        raise RuntimeError("Wheel differs from immutable SHA-256/size lock")
    destination.mkdir(parents=True, exist_ok=True)
    if _has_warp(destination) and not _is_managed(destination):
        raise RuntimeError("Unmanaged Warp already occupies the target; refusing overwrite")
    if _owned_warp_is_loaded(destination):
        raise RuntimeError("Native Warp library is still loaded; restart to repair")
    with tempfile.TemporaryDirectory(prefix="warpsph-verified-") as temporary:
        staging = Path(temporary) / "staging"
        backup = Path(temporary) / "backup"
        backup.mkdir()
        _extract_wheel(wheel, staging)
        if _owned_distribution_version(staging) != CERTIFIED_WARP_VERSION:
            raise RuntimeError("Wheel metadata does not match certified version")
        old = []
        added = []
        receipt_write_started = False
        try:
            for e in list(destination.iterdir()):
                if e.name in {"warp", "warp.libs", _RECEIPT} or (e.name.startswith("warp_lang-") and e.name.endswith(".dist-info")):
                    shutil.move(str(_filesystem_path(e)), str(_filesystem_path(backup / e.name)))
                    old.append(e.name)
            for e in staging.iterdir():
                shutil.move(str(_filesystem_path(e)), str(_filesystem_path(destination / e.name)))
                added.append(e.name)
            receipt_write_started = True
            _receipt_write(destination, info)
        except BaseException:
            for name in added:
                current = _filesystem_path(destination / name)
                if current.is_dir() and not current.is_symlink():
                    shutil.rmtree(current)
                elif current.exists():
                    current.unlink()
            # If the original receipt was not moved yet, leave it intact.
            if receipt_write_started or _RECEIPT in old:
                _receipt_path(destination).unlink(missing_ok=True)
            (destination / (_RECEIPT + ".part")).unlink(missing_ok=True)
            for name in reversed(old):
                source = _filesystem_path(backup / name)
                if source.exists():
                    shutil.move(str(source), str(_filesystem_path(destination / name)))
            raise

def _extract_wheel(wheel: Path, destination: Path) -> None:
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
        archive.extractall(destination)

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


def install_warp(
    *,
    force_reinstall: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> InstallResult:
    """Install the locked Warp wheel into this extension's own directory.

    A certified and managed copy is idempotent and never downloaded twice.
    An explicit ``force_reinstall`` stages a locked
    repair for an owned certified copy, applied before import on the next
    process. Incompatible copies are preserved unless provenance is verified. Warp loaded from
    any other path, including another add-on or Blender's user site-packages,
    is left untouched.
    """

    def report(message: str) -> None:
        if progress_callback:
            progress_callback(message)

    blocked = foreign_warp_block_reason()
    if blocked:
        return InstallResult(success=False, message=blocked)

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

    online = _online_access_block_reason()
    if online:
        return InstallResult(success=False, message=online)

    wheel_info = get_wheel_info_for_platform()
    if wheel_info is None:
        host = _host_platform() or "unknown"
        return InstallResult(
            success=False,
            message=f"No certified Warp wheel is published for this platform ({host})",
        )

    report(
        f"Downloading Warp {CERTIFIED_WARP_VERSION} "
        f"({wheel_info.size / 1024 / 1024:.1f} MB)..."
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        wheel_path = Path(tmpdir) / wheel_info.filename
        try:
            _download_wheel(
                wheel_info.url,
                wheel_path,
                expected_sha256=wheel_info.sha256,
                expected_size=wheel_info.size,
            )
        except Exception as exc:  # noqa: BLE001 - the button reports the failure
            return InstallResult(success=False, message=f"Download failed: {exc}")
        if _sha256_path(wheel_path) != wheel_info.sha256:
            return InstallResult(
                success=False,
                message="Refusing to install a wheel that failed checksum verification",
            )

        try:
            destination = _default_install_target(create=True)
        except RuntimeError as exc:
            return InstallResult(success=False, message=str(exc))
        if _user_site_root() is not None and (
            _is_relative_to(destination, _user_site_root())
            or destination.resolve() == _user_site_root().resolve()
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
        if loaded is not None and not any(
            _is_relative_to(loaded, cand) for cand in _owned_candidates()
        ):
            return InstallResult(
                success=False,
                message=(
                    f"Warp is already loaded from {loaded}, which WarpSPH does not own. "
                    "Remove that copy, then press Install Warp. WarpSPH will not change it."
                ),
            )

        report("Installing Warp...")
        try:
            if repairing_certified or _owned_warp_is_loaded(destination):
                _stage_pending_repair(wheel_path, destination, wheel_info)
                return InstallResult(
                    success=True,
                    message=(
                        f"Staged Warp {CERTIFIED_WARP_VERSION} for repair. "
                        "Restart Blender to apply it before running the solver."
                    ),
                    version=CERTIFIED_WARP_VERSION,
                )
            _promote_verified_wheel(wheel_path, destination, wheel_info)
        except Exception as exc:  # noqa: BLE001 - the button reports the failure
            return InstallResult(success=False, message=f"Installation failed: {exc}")

    installed = _owned_distribution_version(destination)
    if installed is None:
        return InstallResult(
            success=False,
            message=(
                "Installation finished, but the certified Warp files are missing "
                "from this extension's directory."
            ),
        )
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
