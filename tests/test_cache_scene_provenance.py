"""Scene provenance controls using real description, signature, disk and bake planning."""
import json
from dataclasses import replace

import numpy as np
import pytest

from tests.scene_fixtures import box_mesh
from warp_sph_realtime.cache import bake
from warp_sph_realtime.cache.disk import CacheMismatchError, DiskCache


def scene():
    from warp_sph_realtime.scene.description import ColliderMotion, EmitterData, SceneDescription
    collider = box_mesh((0.2, 0.2, 0.2), (0.3, 0.3, 0.3), "collider")
    return SceneDescription(
        (-1, -1, -1), (1, 1, 1), (0, 0, -9.81),
        (box_mesh((-0.4, -0.4, 0), (0, 0, 0.4), "fluid"),), (collider,),
        emitters=(EmitterData(box_mesh((-0.1, -0.1, 0.5), (0.1, 0.1, 0.6), "emitter")),),
        collider_motion=(ColliderMotion(0, "collider", np.eye(3), (0, 0, 0)),),
    )


def provenance(desc=None, **kwargs):
    from warp_sph_realtime.cache.provenance import scene_provenance
    return scene_provenance(desc or scene(), spacing=0.04, **kwargs)


def signature(prov):
    from types import SimpleNamespace
    solver = SimpleNamespace(num_particles=10, dt=0.001, h=0.08,
                             stiffness=90000, viscosity=0.15, gravity=(0, 0, -9.81))
    return bake.solver_signature(solver, 3, 24.0, "Scene", "CUSTOM", provenance=prov)


def changed_scenes():
    desc = scene()
    mesh = desc.fluid_volumes[0]
    moved = replace(mesh, vertices=mesh.vertices + (0.02, 0, 0))
    yield replace(desc, fluid_volumes=(moved,))
    yield replace(desc, domain_max=(1.1, 1, 1))
    yield replace(desc, colliders=(replace(desc.colliders[0], vertices=desc.colliders[0].vertices + 0.01),))
    yield replace(desc, colliders=(), collider_motion=())
    yield replace(desc, collider_motion=())
    yield replace(desc, collider_motion=(replace(desc.collider_motion[0], pivot=(0.01, 0, 0)),))
    emitter = desc.emitters[0]
    for values in ({"speed": 3.0}, {"rate": 20.0}, {"frame_end": 200.0},
                   {"velocity_mode": "VECTOR"}, {"vector": (1, 0, -2)},
                   {"mesh": replace(emitter.mesh, vertices=emitter.mesh.vertices + 0.01)}):
        yield replace(desc, emitters=(replace(emitter, **values),))
    yield replace(desc, emitters=())
    yield replace(desc, particle_budget=2000)


def test_each_changed_input_rejects_resume_without_modifying_crash_prefix(tmp_path):
    disk = DiskCache(tmp_path)
    sig = signature(provenance())
    disk.begin(1, sig)
    disk.write_frame(1, np.zeros((10, 3), np.float32), np.zeros((10, 3), np.float32), np.arange(10))
    (tmp_path / "frame_000002.npz.tmp").write_bytes(b"interrupted")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    for changed in changed_scenes():
        with pytest.raises(CacheMismatchError, match="scene provenance.*Bake Cache"):
            bake.plan_bake(disk, signature(provenance(changed)), 1, 8, resume=True)
        assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_json_reopen_and_unchanged_scene_are_stable(tmp_path):
    sig = signature(provenance())
    disk = DiskCache(tmp_path)
    disk.begin(1, sig)
    disk.write_frame(1, np.zeros((10, 3), np.float32), np.zeros((10, 3), np.float32), np.arange(10))
    reopened = DiskCache(tmp_path)
    plan = bake.plan_bake(reopened, json.loads(json.dumps(sig)), 1, 8, resume=True)
    assert plan.next_frame == 2 and plan.restore is not None
    assert provenance() == provenance(scene())


def test_legacy_is_readable_but_resume_cannot_invent_provenance(tmp_path):
    disk = DiskCache(tmp_path)
    old = signature(provenance())
    del old["scene_provenance"]
    disk.begin(1, old)
    disk.write_frame(1, np.zeros((10, 3)), np.zeros((10, 3)), np.arange(10))
    assert disk.read_frame(1).count == 10
    before = disk.manifest_path.read_bytes()
    with pytest.raises(CacheMismatchError, match="scene provenance.*unknown.*Bake Cache"):
        bake.plan_bake(disk, signature(provenance()), 1, 8, resume=True)
    assert disk.manifest_path.read_bytes() == before
    bake.plan_bake(disk, signature(provenance()), 1, 8, resume=False)
    assert disk.cached_count() == 0


def test_motion_keys_and_schema_are_part_of_provenance():
    matrix = np.eye(4)
    keys = {1: [matrix.copy()], 2: [matrix.copy()]}
    a = provenance(motion_keys=keys)
    matrix[0, 3] = 0.25
    keys[2] = [matrix]
    assert a != provenance(motion_keys=keys)
    assert a["version"] == 1 and a["algorithm"] == "sha256"
    assert len(a["digest"]) == 64


def test_numeric_layout_and_endianness_do_not_change_identity():
    desc = scene()
    mesh = desc.fluid_volumes[0]
    other = replace(mesh, vertices=np.asfortranarray(mesh.vertices.astype(">f4")))
    assert provenance(desc) == provenance(replace(desc, fluid_volumes=(other,)))


def test_provenance_is_a_snapshot_not_a_reference():
    desc = scene()
    saved = provenance(desc)
    desc.fluid_volumes[0].vertices[0, 0] += 0.01
    assert saved != provenance(desc)

def adapter_scene():
    from types import SimpleNamespace as NS

    from tests.test_blender_scene import FakeObject, domain_obj, translation

    class Objects(list):
        def __getitem__(self, key):
            return next(o for o in self if o.name == key) if isinstance(key, str) else super().__getitem__(key)

    fluid = FakeObject("Water", box_mesh((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)),
                       "FLUID", translation(0, 0, 1.0))
    collider = FakeObject("Plate", box_mesh((-0.2, -0.2, -0.02), (0.2, 0.2, 0.02)),
                          "COLLIDER", translation(0, 0, 0.7))
    collider.animation_data = NS(action=object(), drivers=())
    s = NS(objects=Objects([domain_obj(), fluid, collider]), gravity=(0, 0, -9.81),
           frame_current=7, frame_subframe=0.25, render=NS(fps=24, fps_base=1),
           slope=0.01, fail_at=None)

    def frame_set(frame, subframe=0.0):
        s.frame_current, s.frame_subframe = frame, subframe
        collider.matrix_world[0, 3] = (frame - 1) * s.slope
        if frame == s.fail_at:
            raise ValueError("evaluation failed")
    s.frame_set = frame_set
    frame_set(7, subframe=0.25)
    return s


def test_capture_effective_motion_restores_frame_and_hashes_geometry_once():
    from warp_sph_realtime.cache.provenance import capture_scene, feed_motion
    s = adapter_scene()
    args = dict(first=1, last=10, steps=5, dt=0.001, spacing=0.04)
    initial, saved, keys = capture_scene(s, None, **args)
    assert (s.frame_current, s.frame_subframe) == (7, 0.25)
    assert all(o.cleared == 1 for o in s.objects)
    s.slope = 0.03  # same start pose, different constraint/driver/parent result later
    initial2, changed, _ = capture_scene(s, None, **args)
    assert initial == initial2 and saved != changed
    from types import SimpleNamespace as NS
    received = []
    solver = NS(collider_frames_needed=lambda: [2], current_frame=1,
                set_collider_keyframe=lambda f, m: received.append((f, m)))
    feed_motion(s, solver, keys)
    assert received[0][1][0][0, 3] == pytest.approx(0.01)
    assert all(o.cleared == 2 for o in s.objects)  # feeding never reads meshes
    with pytest.raises(ValueError, match="read-only"):
        keys[2][0][0, 3] = 1
    solver.collider_frames_needed = lambda: [100]
    with pytest.raises(ValueError, match="horizon"):
        feed_motion(s, solver, keys)


def test_capture_failure_restores_frame_without_returning_partial_authority():
    from warp_sph_realtime.cache.provenance import capture_scene
    s = adapter_scene()
    s.fail_at = 2
    with pytest.raises(ValueError, match="evaluation failed"):
        capture_scene(s, None, first=1, last=10, steps=5, dt=0.001, spacing=0.04)
    assert (s.frame_current, s.frame_subframe) == (7, 0.25)


@pytest.mark.parametrize("version", [0, 2, None])
def test_unknown_provenance_version_refuses_resume(tmp_path, version):
    disk = DiskCache(tmp_path)
    sig = signature(provenance())
    sig["scene_provenance"]["version"] = version
    disk.begin(1, sig)
    with pytest.raises(CacheMismatchError, match="scene provenance.*Bake Cache"):
        bake.plan_bake(disk, signature(provenance()), 1, 8, resume=True)


@pytest.mark.parametrize("field,value", [
    ("domain_min", (-2, -1, -1)), ("gravity", (0, 0, -10)),
    ("unit_scale", 0.5), ("start_frame", 2), ("fps", 30),
])
def test_scalar_scene_inputs_change_identity(field, value):
    desc = scene()
    assert provenance(desc) != provenance(replace(desc, **{field: value}))


def test_spacing_and_initializer_change_identity():
    from warp_sph_realtime.cache.provenance import scene_provenance
    desc = scene()
    assert provenance(desc) != scene_provenance(desc, spacing=0.05)
    assert provenance(desc) != provenance(desc, initializer={"version": 2})
