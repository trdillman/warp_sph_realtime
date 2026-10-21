"""P2.1 scene seam, CPU side: emitter and collider-motion data, rigid keys, nozzles, demo descriptions."""

import math

import numpy as np
import pytest

from tests.scene_fixtures import box_mesh
from warp_sph_realtime.scene.demo_description import (
    DOMAIN_MAX,
    DOMAIN_MIN,
    PILLAR_CENTERS,
    PILLAR_RADIUS,
    SPHERE_CENTER,
    SPHERE_RADIUS,
    demo_description,
)
from warp_sph_realtime.scene.description import (
    ColliderMotion,
    EmitterData,
    MeshData,
    SceneDescription,
    SceneFillError,
    SceneValidationError,
    quat_from_rotation,
    rigid_key,
    validate_closed_mesh,
)
from warp_sph_realtime.scene.emitters import MIN_NOZZLE_GAP, emitter_nozzles

DOMAIN = ((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0))


def plane(lo, hi, z, down=True, divisions=1, name="tap") -> MeshData:
    """A horizontal rectangle at height z, split into divisions x divisions quads; normal -z if `down`."""
    xs = np.linspace(lo[0], hi[0], divisions + 1)
    ys = np.linspace(lo[1], hi[1], divisions + 1)
    verts = np.array([(x, y, z) for y in ys for x in xs], dtype=np.float32)
    tris = []
    for j in range(divisions):
        for i in range(divisions):
            a = j * (divisions + 1) + i
            b, c, d = a + 1, a + divisions + 2, a + divisions + 1
            tris += [(a, b, c), (a, c, d)]  # counter-clockwise from above: normal +z
    tris = np.array(tris, dtype=np.int32)
    return MeshData(verts, tris[:, ::-1].copy() if down else tris, name)


def rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def matrix(linear, origin):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = linear, origin
    return m


# -- EmitterData and SceneDescription validation ----------------------------------------------------

def test_emitter_defaults_and_validation():
    e = EmitterData(plane((-0.1, -0.1), (0.1, 0.1), 1.0))
    assert (e.velocity_mode, e.speed, e.rate, e.frame_start, e.frame_end) == ("NORMAL", 2.0, 0.0, 1.0, 250.0)
    mesh = plane((-0.1, -0.1), (0.1, 0.1), 1.0)
    for kwargs, message in [
        ({"velocity_mode": "SIDEWAYS"}, "velocity mode"),
        ({"speed": 0.0}, "speed must be above 0"),
        ({"velocity_mode": "VECTOR", "vector": (0, 0, 0)}, "vector is zero"),
        ({"rate": -1.0}, "rate must be 0"),
        ({"frame_start": 10, "frame_end": 10}, "must be after frame start"),
        ({"speed": float("inf")}, "finite"),
    ]:
        with pytest.raises(SceneValidationError, match=message):
            EmitterData(mesh, **kwargs)
    with pytest.raises(SceneValidationError, match="has no faces"):
        EmitterData(MeshData(np.zeros((3, 3), np.float32), np.zeros((0, 3), np.int32), "empty"))


def test_scene_needs_fluid_or_emitter_and_valid_settings():
    tap = EmitterData(plane((-0.1, -0.1), (0.1, 0.1), 1.0))
    SceneDescription(*DOMAIN, (0, 0, -9.81), (), (), emitters=(tap,))  # emitter-only scene is valid
    with pytest.raises(SceneValidationError, match="no fluid volumes and no emitters"):
        SceneDescription(*DOMAIN, (0, 0, -9.81), (), ())
    for kwargs, message in [({"unit_scale": 0.0}, "unit scale"), ({"fps": 0.0}, "frame rate"),
                            ({"particle_budget": 0}, "particle budget"), ({"particle_budget": 2.5}, "particle budget")]:
        with pytest.raises(SceneValidationError, match=message):
            SceneDescription(*DOMAIN, (0, 0, -9.81), (), (), emitters=(tap,), **kwargs)


def test_collider_motion_must_name_distinct_colliders():
    rock = box_mesh((-0.1, -0.1, 0.4), (0.1, 0.1, 0.6), "rock")
    fluid = box_mesh((-0.5, -0.5, 1.0), (0.5, 0.5, 1.5))
    m = ColliderMotion(0, "rock", np.eye(3), (0, 0, 0.5))
    SceneDescription(*DOMAIN, (0, 0, -9.81), (fluid,), (rock,), collider_motion=(m,))
    with pytest.raises(SceneValidationError, match="distinct collider index"):
        SceneDescription(*DOMAIN, (0, 0, -9.81), (fluid,), (rock,), collider_motion=(m, m))
    with pytest.raises(SceneValidationError, match="distinct collider index"):
        SceneDescription(*DOMAIN, (0, 0, -9.81), (fluid,), (rock,),
                         collider_motion=(ColliderMotion(1, "x", np.eye(3), (0, 0, 0)),))
    with pytest.raises(SceneValidationError, match="invertible"):
        ColliderMotion(0, "rock", np.zeros((3, 3)), (0, 0, 0))


# -- rigid keys -------------------------------------------------------------------------------------

def test_quat_from_rotation_matches_known_rotations():
    np.testing.assert_allclose(quat_from_rotation(np.eye(3)), (0, 0, 0, 1), atol=1e-12)
    half = math.sqrt(0.5)
    np.testing.assert_allclose(quat_from_rotation(rot_z(math.pi / 2)), (0, 0, half, half), atol=1e-12)
    q = quat_from_rotation(rot_z(math.pi))  # trace -1: the non-trace branches
    assert abs(abs(q[2]) - 1.0) < 1e-12
    rng = np.random.default_rng(3)
    for _ in range(20):
        u, _, vt = np.linalg.svd(rng.normal(size=(3, 3)))
        r = u @ vt * np.sign(np.linalg.det(u @ vt))
        x, y, z, w = quat_from_rotation(r)
        back = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        np.testing.assert_allclose(back, r, atol=1e-9)


def test_rigid_key_is_relative_to_the_start_frame_and_rejects_scale():
    start = rot_z(0.3) * 2.0  # a scaled, rotated object at the start frame
    motion = ColliderMotion(0, "paddle", start, (0.0, 0.0, 0.0))
    origin, q = rigid_key(motion, matrix(start, (1.0, 2.0, 3.0)), unit_scale=0.01)
    np.testing.assert_allclose(origin, (0.01, 0.02, 0.03))
    np.testing.assert_allclose(q, (0, 0, 0, 1), atol=1e-12)          # same orientation: identity
    _, q = rigid_key(motion, matrix(rot_z(0.3 + math.pi / 2) * 2.0, (0, 0, 0)))
    np.testing.assert_allclose(q, quat_from_rotation(rot_z(math.pi / 2)), atol=1e-9)
    with pytest.raises(SceneValidationError, match="changes scale or shear while animated"):
        rigid_key(motion, matrix(start * 1.5, (0, 0, 0)))
    with pytest.raises(SceneValidationError, match="not finite"):
        rigid_key(motion, np.full((4, 4), np.nan))


# -- emitter nozzles --------------------------------------------------------------------------------

def test_nozzles_cover_the_surface_at_spacing_and_point_along_the_normal():
    s = 0.02
    pos, vel, flux = emitter_nozzles(EmitterData(plane((-0.1, -0.1), (0.1, 0.1), 1.0), speed=3.0), s)
    assert 0.8 * (0.2 / s) ** 2 <= len(pos) <= 1.2 * (0.2 / s) ** 2
    np.testing.assert_allclose(vel, np.broadcast_to((0, 0, -3.0), vel.shape), atol=1e-12)  # down-facing
    np.testing.assert_allclose(flux, 3.0)
    d = np.linalg.norm(pos[:, None] - pos[None], axis=2) + np.eye(len(pos))
    assert d.min() >= MIN_NOZZLE_GAP * s - 1e-9
    assert (np.abs(pos[:, :2]) <= 0.1 + 1e-9).all() and np.allclose(pos[:, 2], 1.0)


def test_subdivided_emitter_matches_a_single_quad():
    s = 0.02
    one = emitter_nozzles(EmitterData(plane((-0.1, -0.1), (0.1, 0.1), 1.0)), s)[0]
    many = emitter_nozzles(EmitterData(plane((-0.1, -0.1), (0.1, 0.1), 1.0, divisions=7)), s)[0]
    assert abs(len(many) - len(one)) <= 0.15 * len(one)
    d = np.linalg.norm(many[:, None] - many[None], axis=2) + np.eye(len(many))
    assert d.min() >= MIN_NOZZLE_GAP * s - 1e-9  # shared edges never double the nozzles


def test_vector_mode_flux_is_the_normal_component():
    pos, vel, flux = emitter_nozzles(
        EmitterData(plane((-0.1, -0.1), (0.1, 0.1), 1.0), velocity_mode="VECTOR", vector=(1.0, 0.0, -2.0)), 0.02)
    np.testing.assert_allclose(vel[0], (1.0, 0.0, -2.0))
    np.testing.assert_allclose(flux, 2.0)


def test_nozzles_are_bounded():
    with pytest.raises(SceneFillError, match="too large"):
        emitter_nozzles(EmitterData(plane((-50, -50), (50, 50), 1.0)), 0.01)
    with pytest.raises(SceneFillError, match="positive"):
        emitter_nozzles(EmitterData(plane((-0.1, -0.1), (0.1, 0.1), 1.0)), 0.0)


# -- demo descriptions ------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4096, 100_000])
def test_demo_descriptions_are_valid_and_count_exactly(n):
    dam = demo_description("DOUBLE_DAM_PILLARS", n)
    assert (dam.domain_min, dam.domain_max) == (DOMAIN_MIN, DOMAIN_MAX)
    assert [m.name for m in dam.fluid_volumes] == ["Dam left", "Dam right"] and len(dam.colliders) == 1
    flop = demo_description("BLOCK_FLOP_SPLASH", n)
    assert [m.name for m in flop.fluid_volumes] == ["Puddle", "Block"] and flop.colliders == ()


def test_demo_obstacle_mesh_is_closed_and_matches_the_analytic_shapes():
    mesh = demo_description("DOUBLE_DAM_PILLARS", 4096).colliders[0]
    validate_closed_mesh(mesh, "collider")
    v = mesh.vertices.astype(np.float64)
    on_sphere = np.abs(np.linalg.norm(v - SPHERE_CENTER, axis=1) - SPHERE_RADIUS) < 0.02
    on_pillar = np.zeros(len(v), bool)
    for cx, cy in PILLAR_CENTERS:
        on_pillar |= np.abs(np.hypot(v[:, 0] - cx, v[:, 1] - cy) - PILLAR_RADIUS) < 0.02
    on_axis = np.zeros(len(v), bool)  # cap centres
    for cx, cy in PILLAR_CENTERS:
        on_axis |= np.hypot(v[:, 0] - cx, v[:, 1] - cy) < 1e-6
    assert (on_sphere | on_pillar | on_axis).all()
