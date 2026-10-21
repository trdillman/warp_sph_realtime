"""Preferences UI for the Warp runtime: source choice, background install, hardware check.

Blender state is touched only here, on the main thread. The download runs on an
``install_job.InstallJob`` worker; a ``bpy.app.timers`` timer polls it, redraws
the Preferences window and records the result. The hardware probe
(``nvidia-smi``) also runs on a thread and publishes a plain dict.

``addon.register()`` passes the few values this module needs from the simulation
state (running flag, startup error, activated Warp) as callables, so it never
imports ``addon``.
"""
from __future__ import annotations

import logging
import textwrap
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty

from . import diagnostics, install_job, warp_installer, warp_runtime
from .warp_installer import InstallResult

_LOG = logging.getLogger(__name__)
_POLL_SECONDS = 0.2
_BACKEND_NAMES = {"VULKAN": "Vulkan", "OPENGL": "OpenGL", "METAL": "Metal"}


@dataclass
class _Hooks:
    is_running: Callable[[], bool] = lambda: False
    startup_error: Callable[[], str] = lambda: ""
    active_warp: Callable[[], Any] = lambda: None


_hooks = _Hooks()
_job: install_job.InstallJob | None = None
_last_result: InstallResult | None = None
_smi: dict[str, str] | None = None            # nvidia-smi facts, once probed
_smi_thread: threading.Thread | None = None


def job_active() -> bool:
    return _job is not None


def job_status_text() -> str:
    snap = _job.snapshot() if _job is not None else None
    if snap is None:
        return ""
    return f"{'Cancelling...' if snap.cancelling else snap.message} {snap.fraction:.0%}"


def _prefs(context=None):
    context = context or bpy.context
    return context.preferences.addons[__package__ or ""].preferences


def _tag_redraw() -> None:
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
    except Exception:  # noqa: BLE001 - a redraw hint must never break the timer
        pass


# ---- background install -----------------------------------------------------------------------

def _poll_job():
    global _job, _last_result
    job = _job
    if job is None:
        return None
    snap = job.snapshot()
    _tag_redraw()
    if not snap.finished or snap.result is None:
        return _POLL_SECONDS
    _last_result = snap.result
    _job = None
    log = _LOG.info if snap.result.success or snap.result.cancelled else _LOG.error
    log("Warp install finished: %s", snap.result.message)
    return None


def start_install(*, force: bool = False, wheel: Path | None = None) -> tuple[bool, str]:
    """Plan on the main thread, then run the download/verify/extract on a worker thread."""
    global _job, _last_result
    if _job is not None:
        return False, "A Warp install is already running"
    plan = warp_installer.plan_install(force_reinstall=force, local_wheel=wheel)
    if isinstance(plan, InstallResult):
        _last_result = plan
        return plan.success, plan.message
    _last_result = None
    _job = install_job.InstallJob(plan)
    _job.start()
    bpy.app.timers.register(_poll_job, first_interval=_POLL_SECONDS, persistent=True)
    _tag_redraw()
    return True, "Installing Warp in the background: progress is shown in the add-on Preferences"


def cancel_install() -> None:
    if _job is not None:
        _job.cancel()
        _tag_redraw()


def shutdown() -> None:
    """Add-on unregister: stop the worker so no thread outlives the module."""
    global _job
    job, _job = _job, None
    if job is not None:
        job.cancel()
        job.join(10.0)
    timers = getattr(getattr(bpy, "app", None), "timers", None)
    for callback in (_poll_job, _poll_hardware):
        if timers is not None and timers.is_registered(callback):
            timers.unregister(callback)


# ---- hardware check ----------------------------------------------------------------------------

def _probe_hardware() -> None:
    global _smi
    _smi = diagnostics.nvidia_smi_info()


def _poll_hardware():
    thread = _smi_thread
    if thread is not None and thread.is_alive():
        return _POLL_SECONDS
    _tag_redraw()
    return None


def request_hardware_probe(force: bool = False) -> None:
    """Start the probe thread if there is no result yet (or ``force``); never blocks."""
    global _smi, _smi_thread
    if _smi_thread is not None and _smi_thread.is_alive():
        return
    if _smi is not None and not force:
        return
    _smi = None
    _smi_thread = threading.Thread(target=_probe_hardware, name="warpsph-hardware", daemon=True)
    _smi_thread.start()
    bpy.app.timers.register(_poll_hardware, first_interval=_POLL_SECONDS, persistent=True)


def _backend_name() -> str:
    try:
        import gpu
        kind = gpu.platform.backend_type_get()
        return _BACKEND_NAMES.get(kind, str(kind).title())
    except Exception as exc:  # noqa: BLE001 - background mode has no GPU module backend
        return diagnostics.unavailable(exc)


# ---- operators ---------------------------------------------------------------------------------

class WM_OT_warp_sph_install(bpy.types.Operator):
    bl_idname = "wm.warp_sph_install"
    bl_label = "Install NVIDIA Warp"
    bl_description = (
        "Install the checksum-verified Warp 1.17.0 CUDA 12 wheel into this extension's isolated user directory, "
        "in the background"
    )

    repair: BoolProperty(name="Repair Warp", default=False)

    def execute(self, context):
        if _hooks.is_running():
            self.report({'ERROR'}, "Stop the simulation before changing the Warp runtime")
            return {'CANCELLED'}
        wheel = None
        prefs = _prefs(context)
        if getattr(prefs, "warp_source", "") == warp_installer.SOURCE_WHEEL:
            raw = str(getattr(prefs, "warp_wheel_path", "")).strip()
            if not raw:
                self.report({'ERROR'}, "Choose a Warp wheel (.whl) file first")
                return {'CANCELLED'}
            wheel = Path(bpy.path.abspath(raw))
        ok, message = start_install(force=self.repair, wheel=wheel)
        self.report({'INFO'} if ok else {'ERROR'}, message)
        return {'FINISHED'} if ok else {'CANCELLED'}


class WM_OT_warp_sph_install_cancel(bpy.types.Operator):
    bl_idname = "wm.warp_sph_install_cancel"
    bl_label = "Cancel"
    bl_description = "Stop the Warp install. Nothing is installed and no partial files are kept"

    def execute(self, context):
        cancel_install()
        return {'FINISHED'}


class WM_OT_warp_sph_check_hardware(bpy.types.Operator):
    bl_idname = "wm.warp_sph_check_hardware"
    bl_label = "Check Again"
    bl_description = "Re-read the GPU, NVIDIA driver and VRAM"

    def execute(self, context):
        request_hardware_probe(force=True)
        return {'FINISHED'}


# ---- preferences -------------------------------------------------------------------------------

def _wrapped(layout, text: str, icon: str = 'NONE', width: int = 78) -> None:
    lines = textwrap.wrap(text, width) or [""]
    col = layout.column(align=True)
    for index, line in enumerate(lines):
        col.label(text=line, icon=icon if index == 0 else 'BLANK1')


class WARPSPH_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__ or "warp_sph_realtime"

    warp_source: EnumProperty(
        name="Warp Source",
        description="Where WarpSPH gets NVIDIA Warp",
        items=[
            (warp_installer.SOURCE_DOWNLOAD, "Download",
             "Download the checksum-verified Warp 1.17.0 wheel (needs Allow Online Access)"),
            (warp_installer.SOURCE_WHEEL, "Install from local wheel file",
             "Install a Warp 1.17.0 wheel you already have; works offline. It must match the certified SHA-256"),
            (warp_installer.SOURCE_FOLDER, "Use existing Warp folder",
             "Import Warp from a folder you manage (for example a studio site-packages). Never modified"),
        ],
        default=warp_installer.SOURCE_DOWNLOAD,
    )
    warp_wheel_path: StringProperty(
        name="Wheel File", description="Path to warp_lang-1.17.0 .whl for this platform", subtype='FILE_PATH',
    )
    warp_folder_path: StringProperty(
        name="Warp Folder", description="Folder that contains the 'warp' package (not the 'warp' folder itself)",
        subtype='DIR_PATH',
    )
    warp_allow_uncertified: BoolProperty(
        name="Allow uncertified version (unsupported)",
        description="Use a Warp that is not 1.17.x. Unsupported: the solver may fail or give wrong results",
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="NVIDIA Warp Runtime (CUDA 12.9)", icon='PREFERENCES')
        state = warp_runtime.status()
        _wrapped(box, state.message, 'CHECKMARK' if state.available else 'ERROR')
        startup_error = _hooks.startup_error()
        if startup_error:
            _wrapped(box, startup_error, 'ERROR')
        box.prop(self, "warp_source")
        if _job is not None:
            self._draw_progress(box)
        elif self.warp_source == warp_installer.SOURCE_FOLDER:
            self._draw_folder(box)
        elif self.warp_source == warp_installer.SOURCE_WHEEL:
            self._draw_wheel(box, state)
        else:
            self._draw_download(box, state)
        if _last_result is not None and _job is None:
            icon = 'CHECKMARK' if _last_result.success else 'ERROR'
            _wrapped(box, _last_result.message, icon)
        box.label(text="WarpSPH never modifies Blender's Python or other extensions.")
        self._draw_hardware(layout)
        layout.operator("wm.warp_sph_copy_diagnostics", icon='COPYDOWN')

    @staticmethod
    def _draw_progress(box) -> None:
        snap = _job.snapshot() if _job is not None else None
        if snap is None:
            return
        box.progress(factor=snap.fraction, type='BAR', text=job_status_text())
        row = box.row()
        row.enabled = not snap.cancelling
        row.operator("wm.warp_sph_install_cancel", icon='CANCEL')

    @staticmethod
    def _draw_download(box, state) -> None:
        offline = warp_installer.download_block_reason()
        if offline:
            _wrapped(box, offline, 'ERROR')
        row = box.row()
        row.enabled = not offline
        if not state.available:
            row.operator("wm.warp_sph_install", text="Download and Install Warp 1.17.0 (~150 MB)", icon='IMPORT')
        op = row.operator("wm.warp_sph_install", text="Repair Warp (restart required)", icon='FILE_REFRESH')
        op.repair = True
        box.label(text="Switching back here uses the install WarpSPH already owns; nothing is downloaded again.")

    def _draw_wheel(self, box, state) -> None:
        box.prop(self, "warp_wheel_path")
        row = box.row()
        row.enabled = bool(self.warp_wheel_path.strip())
        row.operator("wm.warp_sph_install", text="Install from This Wheel (offline)", icon='IMPORT')
        box.label(text="The file must match the certified SHA-256; no network access is used.")
        if state.available:
            box.label(text="Warp is already installed; this replaces it only through Repair.")

    def _draw_folder(self, box) -> None:
        box.prop(self, "warp_folder_path")
        box.prop(self, "warp_allow_uncertified")
        check = warp_installer.check_existing_folder(
            warp_installer.active_source().folder, self.warp_allow_uncertified)
        _wrapped(box, check.message if not check.ok else f"Folder OK: Warp {check.version}",
                 'CHECKMARK' if check.ok else 'ERROR')
        if self.warp_allow_uncertified:
            warn = box.box()
            _wrapped(warn, "UNSUPPORTED: uncertified Warp versions are not tested with WarpSPH and may "
                           "crash, fail or simulate incorrectly.", 'ERROR')
        box.label(text="The folder is only read: WarpSPH never modifies or deletes it.")
        box.label(text="CUDA driver and a GPU self-test run when you press Start.")

    @staticmethod
    def _draw_hardware(layout) -> None:
        request_hardware_probe()
        summary = diagnostics.hardware_summary(_smi, _backend_name(), _hooks.active_warp())
        box = layout.box()
        box.label(text="Hardware Check", icon='SYSTEM')
        col = box.column(align=True)
        for label, value in summary.rows:
            col.label(text=f"{label}: {value}")
        icon = 'TIME' if summary.ready is None else ('CHECKMARK' if summary.ready else 'ERROR')
        _wrapped(box, summary.verdict, icon)
        box.operator("wm.warp_sph_check_hardware", icon='FILE_REFRESH')


CLASSES = (
    WM_OT_warp_sph_install,
    WM_OT_warp_sph_install_cancel,
    WM_OT_warp_sph_check_hardware,
    WARPSPH_Preferences,
)


def register(*, is_running: Callable[[], bool], startup_error: Callable[[], str],
             active_warp: Callable[[], Any]) -> None:
    _hooks.is_running = is_running
    _hooks.startup_error = startup_error
    _hooks.active_warp = active_warp
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    shutdown()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
