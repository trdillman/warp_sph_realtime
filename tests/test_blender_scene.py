"""description_from_blender against a small fake of the bpy object model (no Blender needed)."""

import numpy as np
import pytest

from tests.scene_fixtures import box_mesh
from warp_sph_realtime.scene.blender_scene import ROLE_PROP, description_from_blender
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
