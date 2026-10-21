"""Live Blender check of the P1.3 settings: old flat properties migrate once and the group round-trips.

Run three background Blender processes, one at a time, from the repository root:

    blender -b --factory-startup --python tests/runtime_settings_migration.py -- save-legacy  <dir> <old addon.py>
    blender -b --factory-startup --python tests/runtime_settings_migration.py -- migrate      <dir>
    blender -b --factory-startup --python tests/runtime_settings_migration.py -- reopen       <dir>

``<old addon.py>`` is the pre-P1.3 module (``git show 2543fe1:warp_sph_realtime/addon.py``); it registers the
flat ``scene.warp_sph_*`` properties and saves ``legacy.blend`` with non-default values. ``migrate`` opens it
with this version (the load_post handler copies the values), checks them and saves ``migrated.blend``;
``reopen`` checks they survived the save and that nothing is migrated twice. Each phase writes
``<dir>/<phase>.json``; any failed check exits non-zero.
"""
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
args = sys.argv[sys.argv.index("--") + 1:]
PHASE, OUT = args[0], Path(args[1])
OUT.mkdir(parents=True, exist_ok=True)
R = {"phase": PHASE, "blender": bpy.app.version_string, "checks": [], "status": "RUNNING"}

# Non-default values in the old layout, and what the new group must hold afterwards.
LEGACY = {"scene_source": "CUSTOM", "demo_scene": "BLOCK_FLOP_SPLASH", "render_mode": "POINTS",
          "particle_count": 2_500_000, "substeps": 4, "filter_iterations": 3, "show_particles": False,
          "depth_test": False, "use_cuda_graph": False, "absorption": (1.0, 0.5, 0.25),
          "particle_color": (0.5, 0.25, 0.125, 0.75), "particle_spacing": 0.015, "time_step_scale": 0.3,
          "stiffness": 123000.0, "point_size": 4.5, "particle_radius": 0.03, "filter_sigma": 6.0}
EXPECTED = {**LEGACY, "stiffness": 123.0}  # now in thousands


def check(name, ok, detail=""):
    R["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)})


def close(value, want):
    if isinstance(want, tuple):
        return len(value) == len(want) and all(abs(a - b) < 1e-5 for a, b in zip(value, want, strict=True))
    if isinstance(want, float):
        return abs(value - want) < 1e-5
    return value == want


def settings_snapshot(scene):
    s = scene.warp_sph
    return {k: (tuple(getattr(s, k)) if isinstance(EXPECTED[k], tuple) else getattr(s, k)) for k in EXPECTED}


def legacy_keys(scene):
    """Debug-only view of Blender's add-on property storage, used here as evidence only."""
    store = scene.bl_system_properties_get()
    return sorted(k for k in (store.keys() if store is not None else ()) if k.startswith("warp_sph_"))


def run():
    if PHASE == "save-legacy":
        import warp_sph_realtime  # noqa: F401  (parent package, so the old module's relative imports resolve)
        spec =importlib.util.spec_from_file_location("warp_sph_realtime.addon_legacy", args[2])
        old = sys.modules[spec.name] = importlib.util.module_from_spec(spec)  # annotations resolve via sys.modules
        spec.loader.exec_module(old)
        old.register()
        scene = bpy.context.scene
        for name, value in LEGACY.items():
            setattr(scene, "warp_sph_" + name, value)
        R["stored"] = legacy_keys(scene)
        check("old layout stores all 17 flat values", len(R["stored"]) == 17, R["stored"])
        bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "legacy.blend"))
        return

    import warp_sph_realtime
    from warp_sph_realtime import addon, props
    warp_sph_realtime.register()
    settings_rna = bpy.types.Scene.bl_rna.properties["warp_sph"].fixed_type.properties
    R["stiffness_rna"] = {k: getattr(settings_rna["stiffness"], k)
                          for k in ("step", "hard_min", "hard_max", "default", "precision")}
    check("stiffness step is a legal 1..100", 1 <= settings_rna["stiffness"].step <= 100, R["stiffness_rna"])
    check("advanced sub-panel registered", hasattr(bpy.types, "VIEW3D_PT_warp_sph_advanced"))
    check("load_post handler installed", addon._on_load_post in bpy.app.handlers.load_post)
    check("no flat scene properties registered",
          not [p.identifier for p in bpy.types.Scene.bl_rna.properties if p.identifier.startswith("warp_sph_")])

    if PHASE == "migrate":
        bpy.ops.wm.open_mainfile(filepath=str(OUT / "legacy.blend"))
        scene = bpy.context.scene
        got = settings_snapshot(scene)
        R["migrated"] = got
        for name, want in EXPECTED.items():
            check(f"migrated {name}", close(got[name], want), f"got {got[name]!r}, want {want!r}")
        check("legacy values unset", legacy_keys(scene) == [], legacy_keys(scene))
        check("legacy definitions removed again",
              not [p.identifier for p in bpy.types.Scene.bl_rna.properties if p.identifier.startswith("warp_sph_")])
        scene.warp_sph.substeps = 6  # an edit after the upgrade must survive the next load
        bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "migrated.blend"))
        bpy.ops.wm.read_homefile(use_empty=True)  # an unrelated file without WarpSPH settings
        fresh = bpy.context.scene.warp_sph
        check("file without settings opens with defaults",
              fresh.stiffness == 90.0 and not fresh.is_property_set("stiffness"))
    elif PHASE == "reopen":
        bpy.ops.wm.open_mainfile(filepath=str(OUT / "migrated.blend"))
        scene = bpy.context.scene
        got = settings_snapshot(scene)
        R["reopened"] = got
        for name, want in {**EXPECTED, "substeps": 6}.items():
            check(f"saved {name}", close(got[name], want), f"got {got[name]!r}, want {want!r}")
        check("nothing migrated twice", props.migrate_scenes(list(bpy.data.scenes)) == {})
        check("still no legacy values", legacy_keys(scene) == [], legacy_keys(scene))
        # A stale old value next to a set new one (a file saved before it was migrated): the new one wins.
        from bpy.props import FloatProperty
        bpy.types.Scene.warp_sph_stiffness = FloatProperty()
        scene.warp_sph_stiffness = 999000.0
        del bpy.types.Scene.warp_sph_stiffness
        stale = (legacy_keys(scene), props.migrate_scenes(list(bpy.data.scenes)), scene.warp_sph.stiffness,
                 legacy_keys(scene))
        check("stale old value dropped, newer setting kept", stale == (["warp_sph_stiffness"], {}, 123.0, []), stale)
    else:
        raise SystemExit(f"unknown phase {PHASE}")
    warp_sph_realtime.unregister()
    check("unregister removes the group", not hasattr(bpy.types.Scene, "warp_sph"))


try:
    run()
    R["status"] = "PASS" if all(c["ok"] for c in R["checks"]) else "FAIL"
except Exception:
    R["status"] = "ERROR"
    R["error"] = traceback.format_exc()
(OUT / f"{PHASE}.json").write_text(json.dumps(R, indent=2, default=str))
print(f"SETTINGS_MIGRATION {PHASE} {R['status']}")
for c in R["checks"]:
    if not c["ok"]:
        print("FAILED", c["name"], c["detail"])
if R.get("error"):
    print(R["error"])
sys.stdout.flush()
if R["status"] != "PASS":
    sys.exit(1)
