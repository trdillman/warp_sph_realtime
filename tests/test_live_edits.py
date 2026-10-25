"""Interactive mode (P2.9): every scene change is classified by live_edits.EDITS, and only real edits count."""
import unittest
from types import SimpleNamespace as NS

from warp_sph_realtime import live_edits
from warp_sph_realtime.live_edits import EDITS, LIVE, RESET, ChangeTracker, diff, outcome, snapshot

IDENTITY = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
_uids = iter(range(1000, 10**6))


def translated(x):
    return tuple(tuple(row[:3]) + ((x,) if i == 0 else (row[3],)) for i, row in enumerate(IDENTITY))


def make_object(name, role, animated=False, **emit):
    settings = NS(role=role, emit_velocity_mode="NORMAL", emit_speed=2.0, emit_vector=(0.0, 0.0, -2.0),
                  emit_rate=0.0, emit_frame_start=1, emit_frame_end=250)
    for key, value in emit.items():
        setattr(settings, f"emit_{key}", value)
    mesh = NS(session_uid=next(_uids), vertices=[0] * 8, polygons=[0] * 6)
    return NS(name=name, type="MESH", warp_sph=settings, session_uid=next(_uids), matrix_world=IDENTITY, data=mesh,
              bound_box=[(-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)], constraints=[], parent=None,
              animation_data=NS(action=object(), drivers=[]) if animated else None, get=lambda _key, _default=None: None)


def make_scene(*objects, source="CUSTOM"):
    return NS(objects=list(objects), gravity=(0.0, 0.0, -9.81), use_gravity=True, frame_current=1,
              unit_settings=NS(scale_length=1.0), render=NS(fps=24, fps_base=1.0),
              warp_sph=NS(scene_source=source, particle_spacing=0.02, max_particles=100_000))


def starter():
    return make_scene(make_object("Domain", "DOMAIN"), make_object("Fluid", "FLUID"),
                      make_object("Block", "COLLIDER"), make_object("Paddle", "COLLIDER", animated=True),
                      make_object("Tap", "EMITTER"))


def find(scene, name):
    return next(o for o in scene.objects if o.name == name)


def changes(edit, setup=starter):
    scene = setup()
    base = snapshot(scene, custom=True)
    edit(scene)
    return diff(base, snapshot(scene, custom=True))


def setattrs(obj, **values):
    for key, value in values.items():
        setattr(obj, key, value)


# One scene edit per EDITS row (and the panel text it produces). "unreadable" comes from the tracker.
EDIT_CASES = {
    "gravity": lambda s: setattr(s, "gravity", (0.0, 0.0, -3.0)),
    "spacing": lambda s: setattr(s.warp_sph, "particle_spacing", 0.03),
    "max_particles": lambda s: setattr(s.warp_sph, "max_particles", 50_000),
    "unit_scale": lambda s: setattr(s.unit_settings, "scale_length", 0.01),
    "fps": lambda s: setattr(s.render, "fps", 60),
    "scene_source": lambda s: setattr(s.warp_sph, "scene_source", "DEMO"),
    "objects": lambda s: s.objects.append(make_object("Cup", "COLLIDER")),
    "DOMAIN": lambda s: setattr(find(s, "Domain"), "matrix_world", translated(0.5)),
    "FLUID": lambda s: setattr(find(s, "Fluid").data, "vertices", [0] * 9),
    "COLLIDER": lambda s: setattr(find(s, "Block"), "matrix_world", translated(0.5)),
    "EMITTER": lambda s: setattr(find(s, "Tap").warp_sph, "emit_rate", 500.0),
}


class EditTableTest(unittest.TestCase):
    def test_every_row_is_produced_by_its_edit_and_nothing_else(self):
        self.assertEqual(set(EDIT_CASES) | {"unreadable"}, set(EDITS))
        for kind, edit in EDIT_CASES.items():
            with self.subTest(kind=kind):
                found = changes(edit)
                self.assertEqual(list(found), [kind])
                self.assertIn(outcome(kind), (LIVE, RESET))

    def test_outcomes(self):
        self.assertEqual([k for k in EDITS if outcome(k) == LIVE], ["gravity"])
        self.assertTrue(all(len(row) == 2 and row[1] for row in EDITS.values()))

    def test_an_unclassified_change_fails(self):
        with self.assertRaises(KeyError):
            outcome("viscosity_texture")
        scene = starter()
        scene.objects.append(make_object("Puff", "SMOKE"))   # a role the table does not know
        base = snapshot(scene, custom=True)
        find(scene, "Puff").matrix_world = translated(1.0)
        with self.assertRaises(KeyError):
            diff(base, snapshot(scene, custom=True))

    def test_panel_texts_name_the_object(self):
        self.assertEqual(changes(EDIT_CASES["COLLIDER"]), {"COLLIDER": "collider 'Block' moved"})
        self.assertEqual(changes(EDIT_CASES["FLUID"]), {"FLUID": "fluid 'Fluid' reshaped"})
        self.assertEqual(changes(EDIT_CASES["EMITTER"]), {"EMITTER": "emitter 'Tap' settings changed"})
        self.assertEqual(changes(EDIT_CASES["objects"]), {"objects": "'Cup' joined as Collider"})
        self.assertEqual(changes(lambda s: s.objects.remove(find(s, "Block"))),
                         {"objects": "'Block' left the simulation"})
        self.assertEqual(changes(lambda s: setattr(find(s, "Block").warp_sph, "role", "FLUID")),
                         {"objects": "'Block' is now Fluid"})

    def test_animated_collider_motion_is_not_an_edit(self):
        self.assertEqual(changes(lambda s: setattr(find(s, "Paddle"), "matrix_world", translated(3.0))), {})
        self.assertEqual(changes(lambda s: setattr(find(s, "Paddle"), "animation_data", None)),
                         {"COLLIDER": "collider 'Paddle' animation added or removed"})

    def test_identity_survives_a_rename(self):
        self.assertEqual(changes(lambda s: setattr(find(s, "Block"), "name", "Rock")), {})

    def test_gravity_off_is_zero_gravity(self):
        self.assertEqual(list(changes(lambda s: setattr(s, "use_gravity", False))), ["gravity"])

    def test_max_particles_matters_only_with_an_emitter(self):
        def no_emitter():
            scene = starter()
            scene.objects.remove(find(scene, "Tap"))
            return scene
        self.assertEqual(changes(EDIT_CASES["max_particles"], no_emitter), {})

    def test_demo_runs_track_only_the_scene_source(self):
        scene = make_scene(source="DEMO")
        base = snapshot(scene, custom=False)
        setattrs(scene, gravity=(0.0, 0.0, 0.0))
        scene.warp_sph.particle_spacing = 0.05
        self.assertEqual(diff(base, snapshot(scene, custom=False)), {})
        scene.warp_sph.scene_source = "CUSTOM"
        self.assertEqual(list(diff(base, snapshot(scene, custom=False))), ["scene_source"])


class ChangeTrackerTest(unittest.TestCase):
    def setUp(self):
        self.scene = starter()
        self.tracker = ChangeTracker(self.scene, custom=True)

    def test_refresh_reads_the_scene_only_after_an_update(self):
        EDIT_CASES["COLLIDER"](self.scene)
        self.assertEqual(self.tracker.refresh(self.scene), {})          # no depsgraph update yet
        self.tracker.note_updates(())
        self.assertEqual(self.tracker.refresh(self.scene), {"COLLIDER": "collider 'Block' moved"})
        self.assertFalse(self.tracker.dirty)
        self.assertEqual(self.tracker.reset_reasons(), ["collider 'Block' moved"])

    def test_muted_updates_are_not_edits(self):
        with self.tracker.muted():
            self.tracker.note_updates([NS(is_updated_geometry=True, id=find(self.scene, "Fluid"))])
        self.assertFalse(self.tracker.dirty)
        self.assertEqual(self.tracker.reshaped, set())

    def test_geometry_flag_marks_a_reshape_that_the_token_cannot_see(self):
        fluid = find(self.scene, "Fluid")
        mesh_update = NS(is_updated_geometry=True, id=NS(original=fluid.data))   # meshes have no matrix_world
        self.tracker.note_updates([NS(is_updated_geometry=True, id=NS(original=fluid)), mesh_update])
        self.assertEqual(self.tracker.refresh(self.scene), {"FLUID": "fluid 'Fluid' reshaped"})

    def test_live_gravity_is_accepted_once(self):
        EDIT_CASES["gravity"](self.scene)
        self.tracker.note_updates()
        self.assertEqual(self.tracker.refresh(self.scene), {"gravity": "gravity"})
        self.assertEqual(self.tracker.reset_reasons(), [])                 # live: nothing to Reset
        self.tracker.accept("gravity")
        self.assertEqual(self.tracker.pending, {})
        self.tracker.note_updates()
        self.assertEqual(self.tracker.refresh(self.scene), {})             # now part of the run
        with self.assertRaises(ValueError):
            self.tracker.accept("spacing")

    def test_a_scene_that_cannot_be_read_never_raises(self):
        self.tracker.note_updates()
        del self.scene.gravity
        self.assertEqual(list(self.tracker.refresh(self.scene)), ["unreadable"])
        self.assertEqual(outcome("unreadable"), RESET)
        broken = ChangeTracker(NS(warp_sph=None), custom=True)
        self.assertEqual(list(broken.pending), ["unreadable"])

    def test_rebase_clears_everything(self):
        EDIT_CASES["spacing"](self.scene)
        self.tracker.note_updates([NS(is_updated_geometry=True, id=find(self.scene, "Fluid"))])
        self.tracker.refresh(self.scene)
        self.tracker.rebase(self.scene)
        self.assertEqual((self.tracker.pending, self.tracker.reshaped, self.tracker.dirty), ({}, set(), False))

    def full_reads(self):
        calls = []
        real = live_edits.snapshot
        def counted(scene, custom):
            calls.append(scene)
            return real(scene, custom)
        live_edits.snapshot = counted
        self.addCleanup(setattr, live_edits, "snapshot", real)
        return calls

    def test_an_object_update_re_reads_only_that_object(self):
        calls = self.full_reads()
        block = find(self.scene, "Block")
        for x in (0.1, 0.2, 0.5):                                         # a drag: one object per tick
            block.matrix_world = translated(x)
            self.tracker.note_updates([NS(is_updated_geometry=False, id=NS(original=block)), NS(id=self.scene)])
            self.assertEqual(self.tracker.refresh(self.scene), {"COLLIDER": "collider 'Block' moved"})
        self.assertEqual(calls, [])
        block.matrix_world = IDENTITY                                     # dragged back: nothing pending
        self.tracker.note_updates([NS(id=block)])
        self.assertEqual(self.tracker.refresh(self.scene), {})
        self.assertEqual(calls, [])

    def test_an_object_update_sees_role_and_name_changes(self):
        calls = self.full_reads()
        cup = make_object("Cup", "NONE")
        self.scene.objects.append(cup)
        self.tracker.note_updates([NS(id=NS(name="Collection"))])         # membership changed: read everything
        self.assertEqual(self.tracker.refresh(self.scene), {})
        self.assertEqual(len(calls), 1)
        cup.warp_sph.role = "COLLIDER"
        self.tracker.note_updates([NS(id=cup)])
        self.assertEqual(self.tracker.refresh(self.scene), {"objects": "'Cup' joined as Collider"})
        cup.warp_sph.role = "NONE"
        find(self.scene, "Block").name = "Rock"
        self.tracker.note_updates([NS(id=cup), NS(id=find(self.scene, "Rock"))])
        self.assertEqual(self.tracker.refresh(self.scene), {})
        self.assertEqual(len(calls), 1)

    def test_a_removed_object_is_found_without_its_own_update(self):
        self.full_reads()
        self.scene.objects.remove(find(self.scene, "Block"))
        self.tracker.note_updates([NS(id=self.scene)])
        self.assertEqual(self.tracker.refresh(self.scene), {"objects": "'Block' left the simulation"})

    def test_undo_preserves_pending_geometry_that_tokens_cannot_compare(self):
        fluid = find(self.scene, "Fluid")
        # Interior vertex edits can leave counts and bounds identical. The geometry flag is the only evidence.
        self.tracker.note_updates([NS(is_updated_geometry=True, id=fluid)])
        self.assertIn("FLUID", self.tracker.refresh(self.scene))
        # Undo a later unrelated edit: the fluid edit and its Reset requirement still exist.
        self.tracker.after_undo()
        self.assertIn("FLUID", self.tracker.refresh(self.scene))

    def test_redo_geometry_flags_are_not_discarded(self):
        self.tracker.after_undo()
        self.tracker.refresh(self.scene)  # redo_post may precede depsgraph evaluation
        self.tracker.note_updates([NS(is_updated_geometry=True, id=find(self.scene, "Fluid"))])
        self.assertIn("FLUID", self.tracker.refresh(self.scene))

    def test_undo_geometry_uncertainty_is_cleared_by_reset(self):
        self.tracker.after_undo()
        self.tracker.note_updates([NS(is_updated_geometry=True, id=find(self.scene, "Fluid"))])
        self.assertIn("FLUID", self.tracker.refresh(self.scene))
        self.tracker.rebase(self.scene)
        self.assertEqual(self.tracker.refresh(self.scene), {})

    def test_uid_falls_back_to_the_name(self):
        self.assertEqual(live_edits.uid(NS(name="Cube")), ("name", "Cube"))
        self.assertEqual(live_edits.uid(NS(name="Cube", session_uid=5)), 5)


if __name__ == "__main__":
    unittest.main()
