"""description_from_blender against a small fake of the bpy object model (no Blender needed)."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from tests.scene_fixtures import box_mesh
from warp_sph_realtime.scene.blender_scene import (
    ROLE_PROP,
    description_from_blender,
    object_role,
    scene_warnings,
    sync_collider_keyframes,
    unit_scale,
)
from warp_sph_realtime.scene.description import SceneValidationError


class FakeCollection:
    def __init__(self, items, dtype):
        self._items = items
        self._dtype = dtype

    def __len__(self):
        return len(self._items)

    def foreach_get(self, attr, out):
        flat = np.asarray(self._items, dtype=self._dtype).reshape(-1)
        assert attr in ("co", "vertices")
        out[:] = flat


class FakeMesh:
    def __init__(self, vertices, triangles):
        self.vertices = FakeCollection(vertices, np.float32)
        self.loop_triangles = FakeCollection(triangles, np.int32)


class FakeObject:
    """Stands in for both the original and evaluated object, like a Blender object without modifiers."""

    def __init__(self, name, mesh_data=None, role=None, matrix=None, type="MESH", evaluated_vertices=None):
        self.name = name
        self.type = type
        self._props = {} if role is None else {ROLE_PROP: role}
        self._mesh_data = mesh_data
        self._evaluated_vertices = evaluated_vertices
        self.matrix_world = np.eye(4) if matrix is None else np.asarray(matrix, dtype=float)
        self.cleared = 0
        self.depsgraphs = []

    def get(self, key, default=None):
        return self._props.get(key, default)

    def __getitem__(self, key):
        return self._props[key]

    def evaluated_get(self, depsgraph):
        self.depsgraphs.append(depsgraph)
        return self

    def to_mesh(self):
        mesh = self._mesh_data
        if mesh is None:
            return None
        verts = mesh.vertices if self._evaluated_vertices is None else self._evaluated_vertices
        return FakeMesh(verts, mesh.triangles)

    def to_mesh_clear(self):
        self.cleared += 1


class FakeScene:
    def __init__(self, objects, gravity=(0.0, 0.0, -9.81), use_gravity=True):
        self.objects = objects
        self.gravity = gravity
        self.use_gravity = use_gravity


def translation(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def domain_obj():
    return FakeObject("Domain", box_mesh((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)), "DOMAIN", translation(0, 0, 1.0))


def test_builds_description_with_world_transforms_and_frees_meshes():
    fluid = FakeObject("Water", box_mesh((-0.2, -0.2, -0.2), (0.2, 0.2, 0.2)), "FLUID", translation(0, 0, 1.2))
    plate = FakeObject("Plate", box_mesh((-0.3, -0.3, -0.05), (0.3, 0.3, 0.05)), "COLLIDER", translation(0, 0, 0.3))
    ignored = FakeObject("Camera", None, None, type="CAMERA")
    objects = [plate, ignored, fluid, domain_obj()]
    graph = object()
    desc = description_from_blender(FakeScene(objects, gravity=(0.0, 0.0, -1.62)), graph)

    assert desc.domain_min == (-0.5, -0.5, 0.5) and desc.domain_max == (0.5, 0.5, 1.5)
    assert desc.gravity == pytest.approx((0.0, 0.0, -1.62))
    assert [m.name for m in desc.fluid_volumes] == ["Water"] and [m.name for m in desc.colliders] == ["Plate"]
    np.testing.assert_allclose(desc.fluid_volumes[0].vertices.min(axis=0), (-0.2, -0.2, 1.0), atol=1e-6)
    np.testing.assert_allclose(desc.colliders[0].vertices.max(axis=0), (0.3, 0.3, 0.35), atol=1e-6)
    assert all(o.cleared == 1 and o.depsgraphs == [graph] for o in (plate, fluid, objects[3]))
    assert ignored.cleared == 0


def test_uses_evaluated_geometry_and_scale():
    scale = np.diag([0.8, 0.8, 0.8, 1.0])
    evaluated = np.array(box_mesh((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)).vertices) * [1.0, 1.0, 0.5]  # a modifier
    fluid = FakeObject("Water", box_mesh((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)), "FLUID",
                       translation(0, 0, 1.0) @ scale, evaluated_vertices=evaluated)
    desc = description_from_blender(FakeScene([domain_obj(), fluid]), None)
    lo, hi = desc.fluid_volumes[0].bounds()
    np.testing.assert_allclose(lo, (-0.4, -0.4, 0.8), atol=1e-6)  # modifier halves z, then scale 0.8
    np.testing.assert_allclose(hi, (0.4, 0.4, 1.2), atol=1e-6)


def test_mirrored_object_keeps_outward_orientation():
    mirror = np.diag([-1.0, 1.0, 1.0, 1.0]) @ translation(0, 0, 1.0)
    fluid = FakeObject("Water", box_mesh((-0.2, -0.2, -0.2), (0.2, 0.2, 0.2)), "FLUID", mirror)
    description_from_blender(FakeScene([domain_obj(), fluid]), None)  # validation rejects flipped faces


def test_gravity_off_gives_zero_gravity():
    fluid = FakeObject("Water", box_mesh((-0.2, -0.2, -0.2), (0.2, 0.2, 0.2)), "FLUID", translation(0, 0, 1.0))
    desc = description_from_blender(FakeScene([domain_obj(), fluid], use_gravity=False), None)
    assert desc.gravity == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("build, message", [
    (lambda: [], "exactly one"),
    (lambda: [domain_obj(), domain_obj()], r"found 2 \(Domain, Domain\)"),
    (lambda: [domain_obj(), FakeObject("Odd", None, "WATER")], "expected one of DOMAIN, FLUID, COLLIDER"),
    (lambda: [domain_obj(), FakeObject("Light", None, "FLUID", type="LIGHT")], "is a LIGHT, not a MESH"),
    (lambda: [domain_obj()], "no fluid volumes"),
])
def test_authoring_errors_are_specific(build, message):
    with pytest.raises(SceneValidationError, match=message):
        description_from_blender(FakeScene(build()), None)


def test_open_collider_is_reported_by_name():
    plate = box_mesh((-0.3, -0.3, -0.05), (0.3, 0.3, 0.05))
    open_plate = FakeObject("OpenPlate", type("M", (), {"vertices": plate.vertices, "triangles": plate.triangles[:-1]}),
                            "COLLIDER", translation(0, 0, 0.3))
    fluid = FakeObject("Water", box_mesh((-0.2, -0.2, -0.2), (0.2, 0.2, 0.2)), "FLUID", translation(0, 0, 1.0))
    with pytest.raises(SceneValidationError, match="collider 'OpenPlate' is not closed"):
        description_from_blender(FakeScene([domain_obj(), fluid, open_plate]), None)


# --- Object settings PropertyGroup (obj.warp_sph.role) -------------------------------------------

class Settings:
    def __init__(self, role="NONE"):
        self.role = role


def with_settings(obj, role):
    obj.warp_sph = Settings(role)
    return obj


def box_corners(lo, hi):
    return [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]


def test_role_is_read_from_object_settings():
    fluid = with_settings(FakeObject("Water", box_mesh((-0.2, -0.2, -0.2), (0.2, 0.2, 0.2)),
                                     matrix=translation(0, 0, 1.0)), "FLUID")
    domain = with_settings(FakeObject("Box", box_mesh((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
                                      matrix=translation(0, 0, 1.0)), "DOMAIN")
    collider = with_settings(FakeObject("Rock", box_mesh((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)),
                                        matrix=translation(0.3, 0, 0.6)), "COLLIDER")
    untagged = with_settings(FakeObject("Floor", box_mesh((-3, -3, -1), (3, 3, 0))), "NONE")
    desc = description_from_blender(FakeScene([collider, untagged, fluid, domain]), None)
    assert desc.domain_min == (-0.5, -0.5, 0.5) and desc.domain_max == (0.5, 0.5, 1.5)
    assert [m.name for m in desc.fluid_volumes] == ["Water"] and [m.name for m in desc.colliders] == ["Rock"]
    assert untagged.cleared == 0  # untagged objects are never evaluated


def test_settings_role_wins_and_none_falls_back_to_legacy_property():
    assert object_role(with_settings(FakeObject("A", role="FLUID"), "COLLIDER")) == "COLLIDER"
    assert object_role(with_settings(FakeObject("B", role="FLUID"), "NONE")) == "FLUID"
    assert object_role(with_settings(FakeObject("C"), "NONE")) is None
    assert object_role(FakeObject("D")) is None


def test_scene_warnings_cover_missing_roles_and_fluid_outside_domain():
    assert scene_warnings(FakeScene([])) == ["No domain: mark a box as Domain", "No fluid: mark a closed mesh as Fluid"]
    domain = with_settings(FakeObject("Domain", matrix=translation(0, 0, 1.0)), "DOMAIN")
    domain.bound_box = box_corners((-1, -1, -1), (1, 1, 1))
    inside = with_settings(FakeObject("Inside", matrix=translation(0, 0, 0.5)), "FLUID")
    inside.bound_box = box_corners((-0.2, -0.2, -0.2), (0.2, 0.2, 0.2))
    assert scene_warnings(FakeScene([domain, inside])) == []
    outside = with_settings(FakeObject("Spill", matrix=translation(0, 0, 2.5)), "FLUID")
    outside.bound_box = box_corners((-0.2, -0.2, -0.2), (0.2, 0.2, 0.2))
    lamp = with_settings(FakeObject("Lamp", type="LIGHT"), "COLLIDER")
    second = with_settings(FakeObject("Domain2"), "DOMAIN")
    assert scene_warnings(FakeScene([domain, inside, outside])) == ["Fluid 'Spill' extends outside the domain"]
    assert scene_warnings(FakeScene([domain, second, inside, lamp])) == [
        "'Lamp' is a LIGHT, not a mesh", "2 domains; exactly one is allowed"]


# --- P2.1: emitters, unit scale, frame clock, animated colliders ----------------------------------

def plane_mesh(half, z):
    verts = [(-half, -half, z), (half, -half, z), (half, half, z), (-half, half, z)]
    return type("M", (), {"vertices": verts, "triangles": [(0, 2, 1), (0, 3, 2)]})  # facing down


def emitter_obj(name="Tap", **settings):
    obj = FakeObject(name, plane_mesh(0.1, 0.0), matrix=translation(0, 0, 1.5))
    obj.warp_sph = NS(role="EMITTER", **settings)
    return obj


def test_emitter_settings_unit_scale_and_frame_clock():
    tap = emitter_obj(emit_velocity_mode="VECTOR", emit_speed=1.0, emit_vector=(0.0, 0.0, -300.0),
                      emit_rate=500.0, emit_frame_start=10, emit_frame_end=40)
    scene = FakeScene([tap, domain_obj()], gravity=(0.0, 0.0, -9.81))
    scene.unit_settings = NS(scale_length=0.01)                     # centimetres
    scene.frame_current = 7
    scene.render = NS(fps=30, fps_base=1.001)
    scene.warp_sph = NS(max_particles=123_456)
    desc = description_from_blender(scene, None)
    assert desc.unit_scale == 0.01 and desc.start_frame == 7.0 and desc.fps == pytest.approx(30 / 1.001)
    assert desc.particle_budget == 123_456 and desc.fluid_volumes == ()
    assert desc.domain_min == pytest.approx((-0.005, -0.005, 0.005))  # geometry in metres
    assert desc.gravity == pytest.approx((0.0, 0.0, -9.81))           # Blender gravity is already m/s^2
    (e,) = desc.emitters
    assert (e.velocity_mode, e.rate, e.frame_start, e.frame_end) == ("VECTOR", 500.0, 10.0, 40.0)
    assert e.vector == pytest.approx((0.0, 0.0, -3.0))                # 300 cm/s
    np.testing.assert_allclose(e.mesh.vertices[:, 2], 0.015, atol=1e-7)


def test_emitter_defaults_and_unit_default():
    desc = description_from_blender(FakeScene([emitter_obj(), domain_obj()]), None)
    (e,) = desc.emitters
    assert (e.velocity_mode, e.speed, e.rate) == ("NORMAL", 2.0, 0.0)
    assert desc.unit_scale == 1.0 and desc.collider_motion == ()
    assert unit_scale(NS(unit_settings=NS(scale_length=0.0))) == 1.0
    assert unit_scale(NS(unit_settings=NS(system='METRIC', scale_length=0.01))) == pytest.approx(0.01)
    assert unit_scale(NS(unit_settings=NS(system='NONE', scale_length=0.01))) == 1.0  # Blender ignores it too


def animated(obj, kind):
    if kind == "action":
        obj.animation_data = NS(action=object(), drivers=[])
    elif kind == "driver":
        obj.animation_data = NS(action=None, drivers=[object()])
    elif kind == "nla":  # action pushed down to an NLA strip: no active action
        obj.animation_data = NS(action=None, drivers=[], nla_tracks=[NS(mute=True, strips=[object()]),
                                                                     NS(mute=False, strips=[object()])])
    elif kind == "constraint":
        obj.constraints = [object()]
    elif kind == "parent":
        obj.parent = NS(animation_data=NS(action=object(), drivers=[]), parent=None)
    return obj


@pytest.mark.parametrize("kind", ["action", "driver", "nla", "constraint", "parent"])
def test_animated_colliders_get_a_motion(kind):
    still = FakeObject("Still", box_mesh((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)), "COLLIDER", translation(-0.3, 0, 1.0))
    paddle = animated(FakeObject("Paddle", box_mesh((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)), "COLLIDER",
                                 translation(0.3, 0, 1.0)), kind)
    fluid = FakeObject("Water", box_mesh((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)), "FLUID", translation(0, 0, 1.3))
    scene = FakeScene([still, paddle, fluid, domain_obj()])
    scene.unit_settings = NS(scale_length=2.0)
    desc = description_from_blender(scene, None)
    (m,) = desc.collider_motion
    assert (m.collider, m.name) == ([c.name for c in desc.colliders].index("Paddle"), "Paddle")
    assert m.pivot == pytest.approx((0.6, 0.0, 2.0))  # metres
    np.testing.assert_allclose(m.start_linear, np.eye(3))


def test_muted_or_empty_nla_tracks_are_still():
    from warp_sph_realtime.scene.blender_scene import _is_animated
    obj = NS(animation_data=NS(action=None, drivers=[], nla_tracks=[NS(mute=True, strips=[object()]),
                                                                    NS(mute=False, strips=[])]),
             constraints=[], parent=None)
    assert not _is_animated(obj)


def test_scene_warnings_accept_an_emitter_instead_of_fluid():
    domain = with_settings(FakeObject("Domain", matrix=translation(0, 0, 1.0)), "DOMAIN")
    domain.bound_box = box_corners((-1, -1, -1), (1, 1, 1))
    assert scene_warnings(FakeScene([domain, emitter_obj()])) == []
    assert scene_warnings(FakeScene([domain])) == ["No fluid: mark a closed mesh as Fluid"]


class KeyedSolver:
    def __init__(self, frames, current):
        self.frames, self.current_frame, self.keys = frames, current, []
        self.animated_collider_names = ["Paddle"]

    def collider_frames_needed(self):
        return self.frames

    def set_collider_keyframe(self, frame, matrices):
        self.keys.append((frame, [np.asarray(m) for m in matrices]))


class FrameScene:
    """frame_set moves the paddle to x = frame, like an animated object."""

    def __init__(self, frame):
        self.paddle = NS(matrix_world=translation(frame, 0, 0))
        self.objects = {"Paddle": self.paddle}
        self.frame_current, self.calls = frame, []

    def frame_set(self, frame):
        self.calls.append(frame)
        self.frame_current = frame
        self.paddle.matrix_world = translation(frame, 0, 0)


def test_sync_collider_keyframes_reads_each_frame_once_and_returns_to_the_sim_frame():
    scene = FrameScene(1)
    solver = KeyedSolver([2, 3], current=2)
    sync_collider_keyframes(scene, solver)
    assert [f for f, _ in solver.keys] == [2, 3]
    assert [m[0][0, 3] for _, m in solver.keys] == [2.0, 3.0]
    assert scene.calls == [2, 3, 2] and scene.frame_current == 2
    scene.calls.clear()
    sync_collider_keyframes(scene, KeyedSolver([], current=5))  # nothing needed: the timeline is untouched
    assert scene.calls == []


class Deleted:
    @property
    def matrix_world(self):
        raise ReferenceError("StructRNA of type Object has been removed")


def test_sync_holds_the_last_pose_of_a_deleted_or_missing_collider():
    """P2.9: deleting an animated collider mid-run never stops the run; it holds its last pose."""
    scene, held = FrameScene(1), {}
    sync_collider_keyframes(scene, KeyedSolver([2], current=2), objects=[scene.paddle], held=held)
    for gone in (Deleted(), None):
        solver = KeyedSolver([3, 4], current=3)
        sync_collider_keyframes(scene, solver, objects=[gone], held=held)
        assert [m[0][0, 3] for _, m in solver.keys] == [2.0, 2.0]
    with pytest.raises(SceneValidationError, match="missing"):
        sync_collider_keyframes(scene, KeyedSolver([5], current=5), objects=[None], held={})
