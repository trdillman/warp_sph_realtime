"""Shared helpers for the showcase demos in this folder (not a demo itself).

Each demo builds its scene with the add-on's own properties and operators, then runs or bakes it. When
``SHOWCASE_OUT`` is set (``scripts/render_showcase.py`` sets it) the demo also renders its media into that folder,
writes ``summary.json`` (or ``error.txt``) and quits Blender. Without it the demo stops after the run or bake and
leaves Blender open on the result.

The layout and argument helpers at the top need no Blender, so they are unit-tested on their own. Run as a script,
``blender -b --python _showcase.py -- contact-sheet spec.json`` composes the runner's contact sheet.
"""
import importlib
import json
import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

try:
    import bpy
except ImportError:  # the pure helpers below are unit-tested without Blender
    bpy = None

OUT_ENV = "SHOWCASE_OUT"
WARP_FOLDER_ENV = "WARPSPH_SHOWCASE_WARP_FOLDER"
PACKAGE = "warp_sph_realtime"
PREFIX = "Showcase "
BOX_QUADS = ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5))
# The legacy OpenGL route (SSFR and spheres on OpenGL) has no transport session object; the panel shows no
# Transport line for it, so the overlay names it here (docs/manual/display.md).
GL_VBO_LABEL = "OpenGL VBO interop (GPU copy, no CPU readback)"


# -- pure helpers (no Blender) ----------------------------------------------------------------------------------

def grid_layout(count: int, columns: int, cell_w: int, cell_h: int, pad: int = 12, label_h: int = 34,
                title_h: int = 0) -> dict:
    """Canvas size and cell rectangles (x, y, w, h; top-left origin) of a labelled grid. Each cell has a label
    strip ``label_h`` high under it; ``title_h`` is reserved across the top."""
    if count < 0 or columns < 1 or cell_w < 1 or cell_h < 1:
        raise ValueError("count must be >= 0, columns and cell sizes >= 1")
    columns = min(columns, count) if count else 1
    rows = math.ceil(count / columns) if count else 0
    cells = []
    for index in range(count):
        row, column = divmod(index, columns)
        cells.append((pad + column * (cell_w + pad), title_h + pad + row * (cell_h + label_h + pad), cell_w, cell_h))
    return {"width": pad + columns * (cell_w + pad), "height": title_h + pad + rows * (cell_h + label_h + pad),
            "columns": columns, "rows": rows, "cells": cells, "label_h": label_h, "title_h": title_h}


def fit(src_w: int, src_h: int, box_w: int, box_h: int) -> tuple[int, int, int, int]:
    """Size (w, h) of an image scaled to fit a box with its aspect ratio kept, and its offset (x, y) in the box."""
    if src_w < 1 or src_h < 1:
        raise ValueError("image size must be positive")
    scale = min(box_w / src_w, box_h / src_h)
    w, h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    return w, h, (box_w - w) // 2, (box_h - h) // 2


def script_args(argv: list[str]) -> list[str]:
    """The arguments after Blender's ``--`` separator."""
    return argv[argv.index("--") + 1:] if "--" in argv else []


def reference_front(T: float, ref_T, ref_Z) -> float:
    """Martin & Moyce front Z at non-dimensional time T: linear between the table points, Z = 1 at T = 0."""
    ts, zs = [0.0, *map(float, ref_T)], [1.0, *map(float, ref_Z)]
    if T <= ts[0]:
        return zs[0]
    for i in range(1, len(ts)):
        if T <= ts[i]:
            return zs[i - 1] + (zs[i] - zs[i - 1]) * (T - ts[i - 1]) / (ts[i] - ts[i - 1])
    return zs[-1]  # held after the last measurement


# -- run driver ------------------------------------------------------------------------------------------------

def out_dir() -> Path | None:
    """The media folder when the runner launched this demo, else None."""
    value = os.environ.get(OUT_ENV, "").strip()
    return Path(value).resolve() if value else None


def recording() -> bool:
    return out_dir() is not None


def work_dir(demo: str) -> Path:
    """Where a demo keeps its caches and exports: the media folder, else a temporary folder."""
    folder = out_dir() or Path(tempfile.gettempdir()) / "warpsph_showcase" / demo
    folder.mkdir(parents=True, exist_ok=True)
    return folder


SUMMARY: dict = {"demo": None, "ok": False, "checks": {}, "timings": {}, "media": []}


def check(name: str, ok, detail=None) -> bool:
    SUMMARY["checks"][name] = {"ok": bool(ok), "detail": detail}
    print(f"[showcase] {'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)
    return bool(ok)


class timed:
    """``with timed("bake"): ...`` records the seconds in the summary."""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        SUMMARY["timings"][self.name] = round(time.perf_counter() - self.t0, 2)
        return False


def add_media(path) -> None:
    SUMMARY["media"].append(str(path))


def wait_until(condition, timeout: float, label: str, interval: float = 0.05):
    """Generator for ``run``: yields timer intervals until ``condition()`` is true; TimeoutError after ``timeout``."""
    t0 = time.perf_counter()
    while not condition():
        if time.perf_counter() - t0 > timeout:
            raise TimeoutError(f"{label}: not done after {timeout:.0f} s")
        yield interval


def pause(seconds: float):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        yield 0.05


def _finish(error: str | None) -> None:
    folder = out_dir()
    SUMMARY["ok"] = error is None and bool(SUMMARY["checks"]) and all(c["ok"] for c in SUMMARY["checks"].values())
    if error is not None:
        SUMMARY["error"] = error.strip().splitlines()[-1]
    print(f"[showcase] {SUMMARY['demo']}: {'PASS' if SUMMARY['ok'] else 'FAIL'}", flush=True)
    if folder is None:
        return  # interactive: leave Blender open on the result
    folder.mkdir(parents=True, exist_ok=True)
    if error is not None:
        (folder / "error.txt").write_text(error, encoding="utf-8")
    (folder / "summary.json").write_text(json.dumps(SUMMARY, indent=2, default=str), encoding="utf-8")
    bpy.ops.wm.quit_blender()


def run(demo: str, main) -> None:
    """Drive the generator function ``main`` from a Blender timer (it yields seconds to wait), so the 3D viewport
    and the simulation's own modal timer keep running between steps."""
    SUMMARY["demo"] = demo
    SUMMARY["blender"] = bpy.app.version_string
    state: dict = {}

    def driver():
        try:
            if "gen" not in state:
                state["t0"] = time.perf_counter()
                state["gen"] = main()
            return next(state["gen"])
        except StopIteration:
            error = None
        except Exception:
            error = traceback.format_exc()
            print(error, flush=True)
        SUMMARY["timings"]["total"] = round(time.perf_counter() - state.get("t0", time.perf_counter()), 2)
        _finish(error)
        return None

    if bpy.app.background:
        print("[showcase] the simulation needs a window: run without -b", flush=True)
    bpy.app.timers.register(driver, first_interval=1.0)


# -- add-on access ------------------------------------------------------------------------------------------------

_package: list[str] = []


def package() -> str:
    """Module name of the installed add-on (``bl_ext.<repo>.warp_sph_realtime`` for an extension)."""
    if _package:
        return _package[0]
    import addon_utils
    names = [m.__name__ for m in addon_utils.modules()]
    found = [n for n in names if n == PACKAGE or n.endswith("." + PACKAGE)]
    if not found:
        raise RuntimeError("the WarpSPH add-on is not installed (see docs/manual/install.md)")
    _package.append(next((n for n in found if n in bpy.context.preferences.addons), found[0]))
    return _package[0]


def ensure_addon() -> str:
    """Enable the add-on if needed and, when WARPSPH_SHOWCASE_WARP_FOLDER is set, point it at that Warp folder."""
    import addon_utils
    name = package()
    if name not in bpy.context.preferences.addons:
        addon_utils.enable(name, default_set=True, handle_error=None)
    folder = os.environ.get(WARP_FOLDER_ENV, "").strip()
    if folder:
        prefs = bpy.context.preferences.addons[name].preferences
        prefs.warp_source = "FOLDER"
        prefs.warp_folder_path = folder
    return name


def module(sub: str = ""):
    return importlib.import_module(package() + sub)


def session():
    """The current scene's simulation session (state of the run, the solver and the transport)."""
    return module(".sim_session").get(bpy.context.scene)


def transport(sim) -> dict:
    """What carries the particles to the viewport: label, backend and whether it is the Readback fallback."""
    backend = module(".sim_session").gpu_backend()
    if sim.session is not None:
        readback = not sim.session.transport.capabilities().zero_copy
        return {"label": sim.session.label, "backend": backend, "readback": readback, "reason": sim.session.reason}
    if sim.gl_buf is not None:
        return {"label": GL_VBO_LABEL, "backend": backend, "readback": False, "reason": ""}
    return {"label": "none", "backend": backend, "readback": False, "reason": ""}


# -- scene building -------------------------------------------------------------------------------------------------

def clear_scene() -> None:
    """Delete what an earlier run of a showcase demo left in this file."""
    scene = bpy.context.scene
    try:
        module(".cache.blender_output").remove_mesh_object(scene)
        module(".cache.blender_adapter").remove_playback_object(scene)
    except Exception:
        pass
    for obj in [o for o in bpy.data.objects if o.name.startswith(PREFIX)]:
        bpy.data.objects.remove(obj, do_unlink=True)


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def add_box(name: str, size, location, role: str = "NONE", color=None):
    """A box mesh (closed, as every WarpSPH volume must be) centred at ``location``, with a WarpSPH role."""
    hx, hy, hz = (0.5 * v for v in size)
    corners = [(hx if i & 1 else -hx, hy if i & 2 else -hy, hz if i & 4 else -hz) for i in range(8)]
    mesh = bpy.data.meshes.new(PREFIX + name)
    mesh.from_pydata(corners, [], BOX_QUADS)
    mesh.update()
    obj = link(bpy.data.objects.new(PREFIX + name, mesh))
    obj.location = location
    obj.warp_sph.role = role
    if role in ("DOMAIN", "FLUID"):
        obj.display_type = "WIRE"
        obj.hide_render = True
    if color is not None:
        obj.color = (*color, 1.0)
    return obj


def add_plane(name: str, size, location, rotation=(0.0, 0.0, 0.0), role: str = "NONE", color=None):
    """A rectangle ``size`` = (x, y) facing +z before ``rotation`` (an emitter emits along its normal)."""
    hx, hy = 0.5 * size[0], 0.5 * size[1]
    mesh = bpy.data.meshes.new(PREFIX + name)
    mesh.from_pydata([(-hx, -hy, 0.0), (hx, -hy, 0.0), (hx, hy, 0.0), (-hx, hy, 0.0)], [], [(0, 1, 2, 3)])
    mesh.update()
    obj = link(bpy.data.objects.new(PREFIX + name, mesh))
    obj.location, obj.rotation_euler = location, rotation
    obj.warp_sph.role = role
    if role == "EMITTER":
        obj.hide_render = True
    if color is not None:
        obj.color = (*color, 1.0)
    return obj


def look_at(obj, target) -> None:
    from mathutils import Vector
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_camera(location, target, lens: float = 35.0, clip_end: float = 1000.0):
    cam = link(bpy.data.objects.new(PREFIX + "Camera", bpy.data.cameras.new(PREFIX + "Camera")))
    cam.location = location
    look_at(cam, target)
    cam.data.lens, cam.data.clip_start, cam.data.clip_end = lens, clip_end * 1e-4, clip_end
    bpy.context.scene.camera = cam
    return cam


def add_sun(strength: float = 3.0, angles=(50.0, 10.0, 30.0)):
    sun = link(bpy.data.objects.new(PREFIX + "Sun", bpy.data.lights.new(PREFIX + "Sun", "SUN")))
    sun.data.energy = strength
    sun.rotation_euler = tuple(math.radians(a) for a in angles)
    return sun


def set_world(color=(0.05, 0.06, 0.08), strength: float = 1.0) -> None:
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new(PREFIX + "World")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (*color, 1.0)
        background.inputs["Strength"].default_value = strength
    world.color = color


# -- viewport ------------------------------------------------------------------------------------------------------

def view3d():
    window = bpy.context.window_manager.windows[0]
    area = next(a for a in window.screen.areas if a.type == "VIEW_3D")
    return window, area, next(r for r in area.regions if r.type == "WINDOW")


def override():
    """Context for operators called from a timer (the simulation needs a window and a 3D viewport)."""
    window, area, region = view3d()
    return bpy.context.temp_override(window=window, area=area, region=region, screen=window.screen)


def frame_view(target, eye, clean: bool = True) -> None:
    """Point the 3D viewport from ``eye`` at ``target`` (perspective), optionally without sidebars and gizmos."""
    from mathutils import Vector
    _, area, _ = view3d()
    space = area.spaces.active
    r3d = space.region_3d
    offset = Vector(target) - Vector(eye)
    r3d.view_perspective = "PERSP"
    r3d.view_location = target
    r3d.view_distance = offset.length
    r3d.view_rotation = offset.to_track_quat("-Z", "Y")
    if clean:
        space.show_region_ui = False
        space.show_region_toolbar = False
        space.show_gizmo = False
        space.overlay.show_extras = False
        space.shading.type = "SOLID"
    redraw()


def redraw() -> None:
    _, area, _ = view3d()
    for region in area.regions:
        region.tag_redraw()


_overlay: dict = {"lines": [], "handle": None}


def set_overlay(lines) -> None:
    """Text drawn over the bottom-left of every 3D viewport (screenshots include it)."""
    import blf
    _overlay["lines"] = list(lines)
    if _overlay["handle"] is None:
        def draw():
            font = 0
            blf.size(font, 20)
            blf.enable(font, blf.SHADOW)
            blf.shadow(font, 5, 0.0, 0.0, 0.0, 1.0)
            # Bottom left: the top left holds Blender's own text and the add-on's status lines.
            y = 24 + 28 * (len(_overlay["lines"]) - 1)
            for line in _overlay["lines"]:
                blf.position(font, 18, y, 0)
                blf.color(font, 1.0, 1.0, 1.0, 1.0)
                blf.draw(font, line)
                y -= 28
            blf.disable(font, blf.SHADOW)
        _overlay["handle"] = bpy.types.SpaceView3D.draw_handler_add(draw, (), "WINDOW", "POST_PIXEL")
    redraw()


def clear_overlay() -> None:
    if _overlay["handle"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_overlay["handle"], "WINDOW")
        _overlay["handle"] = None
    redraw()


def screenshot_area(path) -> Path:
    """Save the 3D viewport area as a PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with override():
        bpy.ops.screen.screenshot_area(filepath=str(path), check_existing=False)
    add_media(path)
    return path


# -- rendering -----------------------------------------------------------------------------------------------------

def _render_size(scene, size) -> None:
    scene.render.resolution_x, scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False


def use_workbench(size=(960, 540)) -> None:
    """Workbench render (fast; colours from each object's Viewport Display colour)."""
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light, shading.color_type = "STUDIO", "OBJECT"
    shading.show_shadows = True
    shading.show_cavity = False
    shading.background_type = "WORLD"
    _render_size(scene, size)


def use_cycles(size=(960, 540), samples: int = 16) -> str:
    """Cycles render on the GPU (OptiX, else CUDA, else the CPU) with denoising. Returns the device used."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    device = "CPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = kind
            except TypeError:
                continue
            prefs.get_devices()
            if any(d.type == kind for d in prefs.devices):
                for d in prefs.devices:
                    d.use = d.type == kind
                device = kind
                break
    except Exception as exc:  # no Cycles preferences: render on the CPU
        print(f"[showcase] Cycles GPU unavailable: {exc}", flush=True)
    scene.cycles.device = "CPU" if device == "CPU" else "GPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.max_bounces = 8
    scene.cycles.transmission_bounces = 8
    _render_size(scene, size)
    return device


def render_still(path) -> Path:
    scene = bpy.context.scene
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(path)
    with override():
        bpy.ops.render.render(write_still=True)
    return path


def render_frames(frames, folder, prefix: str = "frame") -> list[Path]:
    """Render each timeline frame of ``frames`` to ``folder/<prefix>_0001.png``, ... (numbered from 1 so the
    sequence encodes directly). Each frame_set runs the cache playback handler."""
    scene = bpy.context.scene
    paths = []
    for number, frame in enumerate(frames, 1):
        scene.frame_set(frame)
        paths.append(render_still(Path(folder) / f"{prefix}_{number:04d}.png"))
    return paths


def label_image(path, lines, size: int = 22) -> None:
    """Write text lines onto the top-left of a PNG in place."""
    import blf
    import imbuf
    image = imbuf.load(str(path))
    height = image.size[1]
    font = 0
    with blf.bind_imbuf(font, image, display_name="sRGB"):
        blf.size(font, size)
        for row, line in enumerate(lines):
            for dx, dy, shade in ((2, -2, 0.0), (0, 0, 1.0)):  # a drop shadow, then the text
                blf.position(font, 16 + dx, height - (row + 1) * (size + 10) + dy, 0)
                blf.color(font, shade, shade, shade, 1.0)
                blf.draw_buffer(font, line)
    imbuf.write(image, filepath=str(path))


def compose_grid(cells, path, columns: int, cell_w: int = 480, title: str = "") -> Path:
    """Compose images [(path, label), ...] into one labelled PNG grid (missing images stay empty)."""
    import numpy as np
    loaded = []
    for source, label in cells:
        image = None
        if source and Path(source).is_file():
            image = bpy.data.images.load(str(source), check_existing=False)
        loaded.append((image, label))
    first = next((im for im, _ in loaded if im is not None and im.size[0] > 0), None)
    aspect = first.size[1] / first.size[0] if first is not None else 9 / 16
    cell_h = max(1, round(cell_w * aspect))
    title_h = 44 if title else 0
    layout = grid_layout(len(cells), columns, cell_w, cell_h, title_h=title_h)
    width, height = layout["width"], layout["height"]
    canvas = np.empty((height, width, 4), np.float32)
    canvas[:] = (0.07, 0.07, 0.08, 1.0)
    for (image, _label), (x, y, w, h) in zip(loaded, layout["cells"], strict=True):
        if image is None or image.size[0] == 0:
            continue
        fw, fh, ox, oy = fit(image.size[0], image.size[1], w, h)
        image.scale(fw, fh)
        pixels = np.empty(fw * fh * 4, np.float32)
        image.pixels.foreach_get(pixels)
        bottom = height - (y + oy + fh)  # Blender images start at the bottom row
        canvas[bottom:bottom + fh, x + ox:x + ox + fw] = pixels.reshape(fh, fw, 4)
        bpy.data.images.remove(image)
    canvas[..., 3] = 1.0
    out = bpy.data.images.new("showcase_grid", width, height, alpha=False)
    out.pixels.foreach_set(canvas.ravel())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.filepath_raw = str(path)
    out.file_format = "PNG"
    out.save()
    bpy.data.images.remove(out)

    import blf
    import imbuf
    image = imbuf.load(str(path))
    font = 0
    with blf.bind_imbuf(font, image, display_name="sRGB"):
        blf.color(font, 1.0, 1.0, 1.0, 1.0)
        if title:
            blf.size(font, 26)
            blf.position(font, 14, height - 34, 0)
            blf.draw_buffer(font, title)
        blf.size(font, 20)
        for (_image, label), (x, y, _w, h) in zip(loaded, layout["cells"], strict=True):
            blf.position(font, x + 6, height - (y + h + 25), 0)
            blf.draw_buffer(font, label)
    imbuf.write(image, filepath=str(path))
    return path


# -- materials -----------------------------------------------------------------------------------------------------

def water_material():
    """Clear water: Principled BSDF with full transmission, IOR 1.33, low roughness and a faint blue tint."""
    material = bpy.data.materials.get(PREFIX + "Water") or bpy.data.materials.new(PREFIX + "Water")
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.75, 0.9, 1.0, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.03
    bsdf.inputs["IOR"].default_value = 1.33
    bsdf.inputs["Transmission Weight"].default_value = 1.0
    material.diffuse_color = (0.3, 0.55, 0.9, 1.0)
    return material


def age_material(max_age: float):
    """Colour a point cloud by its ``age`` attribute (seconds since emission): young yellow, old deep blue."""
    material = bpy.data.materials.get(PREFIX + "Age") or bpy.data.materials.new(PREFIX + "Age")
    material.use_nodes = True
    tree = material.node_tree
    nodes, links = tree.nodes, tree.links
    bsdf = nodes["Principled BSDF"]
    attribute = nodes.new("ShaderNodeAttribute")
    attribute.attribute_type, attribute.attribute_name = "GEOMETRY", "age"
    scale = nodes.new("ShaderNodeMapRange")
    scale.inputs["From Max"].default_value = max(max_age, 1e-3)
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (1.0, 0.85, 0.15, 1.0)
    ramp.color_ramp.elements[1].color = (0.02, 0.08, 0.45, 1.0)
    links.new(attribute.outputs["Fac"], scale.inputs["Value"])
    links.new(scale.outputs["Result"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.4
    return material


def set_mesh_input(obj, name: str, value) -> None:
    """Set an input of the Add Mesh Preset modifier (Radius, Voxel Size, Material, ...)."""
    socket = module(".cache.blender_output").input_socket(obj, name)
    if socket is None:
        raise RuntimeError(f"the mesh preset has no input {name!r}")
    socket.value = value
    obj.update_tag()


# -- contact sheet (runner) ---------------------------------------------------------------------------------------

def _contact_sheet(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    compose_grid([(c["image"], c["label"]) for c in spec["cells"]], spec["output"], spec.get("columns", 3),
                 spec.get("cell_width", 480), spec.get("title", ""))


if __name__ == "__main__" and bpy is not None:
    args = script_args(sys.argv)
    if args[:1] == ["contact-sheet"] and len(args) == 2:
        try:
            _contact_sheet(args[1])
        except Exception:
            traceback.print_exc()
            sys.exit(1)
