"""Blender/CUDA begin_bake regression; launched only by scripts/verify_cache_gravity.py."""
import json
import os
import sys
import time
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(os.environ["GRAVITY_WARP_PARENT"])
import warp as wp  # noqa: E402

import warp_sph_realtime as extension  # noqa: E402
from warp_sph_realtime import sim_session  # noqa: E402
from warp_sph_realtime.cache.disk import DiskCache  # noqa: E402
from warp_sph_realtime.sim.sim_core import RealtimeWCSPHSolver  # noqa: E402

extension.register()
wp.init()
assert wp.__version__ == "1.17.0"
assert wp.get_device("cuda:0").is_cuda
sim_session.runtime.wp = wp
sim_session.runtime.RealtimeWCSPHSolver = RealtimeWCSPHSolver
scene = bpy.context.scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
for name, location, dimensions, role in [
    ("Domain", (0, 0, 0.5), (1, 1, 1), "DOMAIN"),
    ("Fluid", (0, 0, 0.42), (0.24, 0.24, 0.24), "FLUID"),
]:
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name, obj.dimensions, obj.warp_sph.role = name, dimensions, role
scene.unit_settings.scale_length = 1.0
scene.use_gravity = True
scene.gravity = (1.0, 0.5, -3.0)
settings = scene.warp_sph
settings.scene_source, settings.particle_spacing, settings.bake_steps = "CUSTOM", 0.04, 3
out = Path(os.environ["GRAVITY_PROBE_OUT"])
messages = []


def fresh(folder):
    settings.cache_dir = str(out / folder)
    session = sim_session.SimSession(scene)
    session.solver = session._custom_solver(scene, bpy.context.evaluated_depsgraph_get())
    session.active_demo, session.running = sim_session.CUSTOM_SCENE, True
    return session


def begin(session, resume):
    return session.begin_bake(bpy.context, lambda level, msg: messages.append(msg),
                              first=1, last=8, resume=resume)


def drain(session, stop_after=None):
    current = session.bake
    deadline = time.monotonic() + 30
    while current.active and time.monotonic() < deadline:
        current.run_chunk(0.0)
        if stop_after is not None and current.captured >= stop_after:
            session.cancel_bake()
        time.sleep(0.001)
    assert not current.active, "bake timed out"
    session._finish_bake(lambda level, msg: messages.append(msg))


whole = fresh("whole")
assert begin(whole, False), messages
drain(whole)
part = fresh("part")
assert begin(part, False), messages
drain(part, 3)
disk = DiskCache(out / "part")
assert disk.cached_count() == 3
before = {p.name: p.read_bytes() for p in disk.directory.iterdir()}
scene.gravity.z = -1.0
changed = fresh("part")  # actual Blender gravity -> adapter -> new CUDA solver
assert not begin(changed, True)
assert "gravity changed" in messages[-1] and "Bake Cache" in messages[-1], messages
assert changed.bake is None
assert before == {p.name: p.read_bytes() for p in disk.directory.iterdir()}
scene.gravity.z = -3.0
resumed = fresh("part")
assert begin(resumed, True), messages
drain(resumed)
reference = DiskCache(out / "whole")
for frame in range(1, 9):
    a, b = reference.read_frame(frame), disk.read_frame(frame)
    for name in ("position", "velocity", "id"):
        assert getattr(a, name).tobytes() == getattr(b, name).tobytes(), (frame, name)
manifest = disk.manifest()
assert manifest is not None
del manifest["signature"]["gravity"]
disk.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
legacy = fresh("part")
assert not begin(legacy, True)
assert "gravity is unknown" in messages[-1] and "Bake Cache" in messages[-1]
assert legacy.bake is None
receipt = {"ok": True, "blender": bpy.app.version_string, "warp": wp.__version__,
           "device": str(wp.get_device("cuda:0")), "messages": messages,
           "checks": ["actual begin_bake", "changed gravity rejection", "unchanged restart bitwise",
                      "legacy rejection", "cancel prefix", "cache unchanged on rejection"],
           "scope": "Background Blender/CUDA session and real authored cubes; no UI or viewport acceptance"}
(out / "summary.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print("GRAVITY_PROBE", json.dumps(receipt))
