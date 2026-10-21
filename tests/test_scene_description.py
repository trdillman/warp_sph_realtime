import numpy as np
import pytest
import warp as wp

from tests.scene_fixtures import box_mesh, box_signed_distance
from warp_sph_realtime.scene.description import MeshData, SceneDescription, SceneValidationError
from warp_sph_realtime.scene.fill import SceneFillError, fill_particles

DOMAIN = ((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0))


def scene(fluids, colliders=(), domain=DOMAIN, gravity=(0.0, 0.0, -9.81)):
    return SceneDescription(domain[0], domain[1], gravity, tuple(fluids), tuple(colliders))


def test_valid_scene_normalizes_and_accepts_boxes():
    desc = scene([box_mesh((-0.5, -0.5, 0.1), (0.5, 0.5, 0.9))], [box_mesh((-0.2, -0.2, 0.0), (0.2, 0.2, 0.05))])
    assert desc.gravity == (0.0, 0.0, -9.81) and len(desc.colliders) == 1
    assert desc.fluid_volumes[0].vertices.dtype == np.float32 and desc.fluid_volumes[0].triangles.dtype == np.int32


@pytest.mark.parametrize("domain, message", [
    (((0, 0, 0), (1, 1, 1)), None),
    (((0, 0, 0), (1, 0, 1)), "min < max"),
    (((0, 0, 0), (1, 1, float("nan"))), "finite"),
])
def test_domain_validation(domain, message):
    fluid = box_mesh((0.1, 0.1, 0.1), (0.5, 0.5, 0.5))
    if message is None:
        scene([fluid], domain=domain)
    else:
        with pytest.raises(SceneValidationError, match=message):
            scene([fluid], domain=domain)


def test_requires_fluid():
    with pytest.raises(SceneValidationError, match="no fluid volumes"):
        scene([])


def test_fluid_must_be_inside_domain():
    with pytest.raises(SceneValidationError, match=r"fluid volume 'out' extends outside the domain"):
        scene([box_mesh((0.5, 0.5, 0.5), (1.5, 0.9, 0.9), "out")])


def test_open_mesh_is_rejected_with_boundary_count():
    box = box_mesh((-0.5, -0.5, 0.5), (0.5, 0.5, 1.0), "holey")
    open_box = MeshData(box.vertices, box.triangles[:-2], "holey")  # drop the +x face
    with pytest.raises(SceneValidationError, match=r"collider 'holey' is not closed .* 4 boundary edge"):
        scene([box_mesh((0, 0, 0.1), (0.2, 0.2, 0.3))], [open_box])


def test_inverted_normals_are_rejected():
    box = box_mesh((-0.5, -0.5, 0.5), (0.5, 0.5, 1.0), "inside_out")
    flipped = MeshData(box.vertices, box.triangles[:, ::-1], "inside_out")
    with pytest.raises(SceneValidationError, match="inward-facing normals"):
        scene([flipped])


def test_one_flipped_face_is_rejected_as_inconsistent():
    box = box_mesh((-0.5, -0.5, 0.5), (0.5, 0.5, 1.0), "bad")
    tri = box.triangles.copy()
    tri[0] = tri[0][::-1]
    with pytest.raises(SceneValidationError, match="not a consistently oriented manifold"):
        scene([MeshData(box.vertices, tri, "bad")])


def test_empty_degenerate_and_bad_index_meshes_are_rejected():
    with pytest.raises(SceneValidationError, match="has no triangles"):
        scene([MeshData(np.zeros((3, 3)), np.zeros((0, 3)), "empty")])
    with pytest.raises(SceneValidationError, match="repeat a vertex index"):
        scene([MeshData(np.zeros((3, 3)), np.array([[0, 1, 1]]), "sliver")])
    with pytest.raises(SceneValidationError, match="indices must be in"):
        MeshData(np.zeros((3, 3)), np.array([[0, 1, 5]]), "oob")
    with pytest.raises(SceneValidationError, match="shape"):
        MeshData(np.zeros((3, 2)), np.zeros((1, 3), dtype=np.int32), "flat")
    with pytest.raises(SceneValidationError, match="NaN"):
        MeshData(np.full((3, 3), np.nan), np.array([[0, 1, 2]]), "nan")


def test_gravity_must_be_finite_vector():
    with pytest.raises(SceneValidationError, match="gravity"):
        scene([box_mesh((0, 0, 0.1), (0.2, 0.2, 0.3))], gravity=(0.0, float("inf"), 0.0))


@pytest.mark.skipif(not wp.is_cuda_available(), reason="fill tests use the CUDA device like the solver")
class TestFill:
    def test_cube_fill_count_and_bounds(self):
        spacing = 0.05
        desc = scene([box_mesh((-0.5, -0.5, 0.1), (0.5, 0.5, 0.9))])
        pts = fill_particles(desc, spacing)
        assert pts.dtype == np.float32 and pts.shape[1] == 3
        assert len(pts) == 20 * 20 * 16  # extent / spacing, cell-centred lattice
        assert (pts >= [-0.5, -0.5, 0.1]).all() and (pts <= [0.5, 0.5, 0.9]).all()
        assert len(np.unique(np.round(pts / spacing * 8), axis=0)) == len(pts)  # no duplicates

    def test_fill_excludes_collider_interior(self):
        spacing = 0.05
        collider = box_mesh((-0.2, -0.2, 0.3), (0.2, 0.2, 0.6))
        desc = scene([box_mesh((-0.5, -0.5, 0.1), (0.5, 0.5, 0.9))], [collider])
        pts = fill_particles(desc, spacing)
        # One spacing of clearance from the collider surface (the solver's soft layer), not just outside it.
        assert (box_signed_distance(pts, (-0.2, -0.2, 0.3), (0.2, 0.2, 0.6)) >= spacing - 1e-6).all()
        assert len(pts) == 20 * 20 * 16 - 10 * 10 * 8
        assert len(fill_particles(desc, spacing, clearance=0.0)) == 20 * 20 * 16 - 8 * 8 * 6

    def test_overlapping_fluid_volumes_do_not_double_count(self):
        a = box_mesh((-0.5, -0.5, 0.1), (0.1, 0.5, 0.9), "a")
        b = box_mesh((-0.1, -0.5, 0.1), (0.5, 0.5, 0.9), "b")
        assert len(fill_particles(scene([a, b]), 0.05)) == 20 * 20 * 16

    def test_fill_is_clipped_to_domain_and_fails_loudly_when_empty(self):
        spacing = 0.05
        pts = fill_particles(scene([box_mesh((-1.0, -1.0, 0.0), (1.0, 1.0, 0.2))]), spacing)
        assert (pts >= [-1, -1, 0]).all() and (pts <= [1, 1, 0.2]).all()
        with pytest.raises(SceneFillError, match="no lattice point"):
            fill_particles(scene([box_mesh((0.0, 0.0, 0.1), (0.01, 0.01, 0.11))]), spacing)
        fully_covered = scene([box_mesh((-0.2, -0.2, 0.3), (0.2, 0.2, 0.6))],
                              [box_mesh((-0.5, -0.5, 0.2), (0.5, 0.5, 0.7))])
        with pytest.raises(SceneFillError, match="no particles"):
            fill_particles(fully_covered, spacing)
        with pytest.raises(SceneFillError, match="positive"):
            fill_particles(fully_covered, 0.0)

    def test_overly_fine_spacing_is_rejected_before_allocating(self):
        with pytest.raises(SceneFillError, match="increase the particle spacing"):
            fill_particles(scene([box_mesh((-1, -1, 0), (1, 1, 2))]), 0.001)
