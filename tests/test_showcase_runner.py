"""The showcase runner's (scripts/render_showcase.py) and helper's (examples/showcase/_showcase.py) pure logic,
without Blender: argument parsing, the run plan, tasklist parsing, the summary, the contact-sheet layout, and the
reference front of the validation demo."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("render_showcase", ROOT / "scripts" / "render_showcase.py")
helper = _load("showcase_helper", ROOT / "examples" / "showcase" / "_showcase.py")


class ArgumentsTest(unittest.TestCase):
    def test_defaults_run_everything_on_both_backends(self):
        args = runner.parse_args([])
        self.assertEqual(args.demos, list(runner.DEMOS))
        self.assertEqual(args.backends, ["vulkan", "opengl"])
        self.assertEqual(args.timeout, runner.DEFAULT_TIMEOUT)
        self.assertIsNone(args.out)

    def test_subsets_are_normalised_and_deduplicated(self):
        args = runner.parse_args(["--demos", " Presets,display,presets ", "--backends", "OPENGL"])
        self.assertEqual(args.demos, ["presets", "display"])
        self.assertEqual(args.backends, ["opengl"])

    def test_unknown_names_and_bad_timeout_are_rejected(self):
        for argv in (["--demos", "fireworks"], ["--backends", "metal"], ["--demos", ","], ["--timeout", "0"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                runner.parse_args(argv)


class PlanTest(unittest.TestCase):
    def test_display_runs_once_per_backend_others_once(self):
        runs = runner.plan(["display", "validation", "presets"], ["vulkan", "opengl"])
        self.assertEqual([(r["demo"], r["backend"], r["folder"]) for r in runs], [
            ("display", "vulkan", "display/vulkan"), ("display", "opengl", "display/opengl"),
            ("validation", "vulkan", "validation"), ("presets", "vulkan", "presets")])

    def test_bakes_use_the_first_backend(self):
        runs = runner.plan(["emitter_mixer"], ["opengl", "vulkan"])
        self.assertEqual(runs, [{"demo": "emitter_mixer", "backend": "opengl", "folder": "emitter_mixer"}])

    def test_blender_command(self):
        command = runner.blender_command("blender", "opengl", Path("start.blend"), Path("display.py"))
        self.assertEqual(command[command.index("--gpu-backend") + 1], "opengl")
        self.assertEqual(command[-2:], ["--python", "display.py"])
        self.assertIn("--factory-startup", command)
        self.assertLess(command.index("start.blend"), command.index("--python"))

    def test_profile_env(self):
        env = runner.profile_env(Path("p"))
        self.assertEqual(env["BLENDER_USER_EXTENSIONS"], str(Path("p") / "extensions"))
        self.assertEqual(len(env), 5)


class TasklistTest(unittest.TestCase):
    def test_detects_blender(self):
        listing = '"explorer.exe","100","Console","1","90,000 K"\r\n"blender.exe","86800","Console","1","1,629,304 K"\r\n'
        self.assertTrue(runner.blender_running(listing))
        self.assertTrue(runner.blender_running('"Blender.EXE","1","Console","1","1 K"'))

    def test_no_blender(self):
        self.assertFalse(runner.blender_running(""))
        self.assertFalse(runner.blender_running("INFO: No tasks are running which match the specified criteria.\r\n"))
        self.assertFalse(runner.blender_running('"blender-launcher.exe","5","Console","1","1 K"'))


def _summary(ok=True, **extra):
    checks = {"bake_complete": {"ok": True, "detail": None}, "rendered": {"ok": ok, "detail": 3}}
    return {"demo": "presets", "ok": ok, "checks": checks, "timings": {"bake": 12.5}, "particles": 21000,
            "transport": "Vulkan interop (GPU copy, no CPU readback)", "key_frame": "k.png",
            "sequences": ["seq"], **extra}


class SummaryTest(unittest.TestCase):
    RUN = {"demo": "presets", "backend": "vulkan", "folder": "presets"}

    def test_passing_run(self):
        row = runner.run_result(self.RUN, _summary(video_fps=12), 0, 61.23)
        self.assertTrue(row["ok"])
        self.assertEqual((row["particles"], row["seconds"], row["video_fps"], row["checks"]), (21000, 61.2, 12, 2))
        self.assertEqual(row["failed_checks"], [])
        self.assertIsNone(row["error"])
        self.assertEqual(row["transport"], "Vulkan interop (GPU copy, no CPU readback)")

    def test_failed_check_exit_code_timeout_or_missing_summary_fail(self):
        self.assertEqual(runner.run_result(self.RUN, _summary(ok=False), 0, 1.0)["failed_checks"], ["rendered"])
        self.assertFalse(runner.run_result(self.RUN, _summary(), 3, 1.0)["ok"])
        timed_out = runner.run_result(self.RUN, None, None, 5.0, "timed out after 5 s")
        self.assertFalse(timed_out["ok"])
        self.assertEqual(timed_out["error"], "timed out after 5 s")
        missing = runner.run_result(self.RUN, None, 0, 1.0)
        self.assertEqual((missing["ok"], missing["error"], missing["video_fps"]), (False, "no summary.json", 24))

    def test_run_summary(self):
        rows = [runner.run_result(self.RUN, _summary(), 0, 1.0), runner.run_result(self.RUN, None, 1, 1.0)]
        summary = runner.run_summary(rows, "20261021-120000", "blender", Path("out"))
        self.assertEqual((summary["ok"], summary["passed"], len(summary["runs"])), (False, 1, 2))
        self.assertTrue(runner.run_summary(rows[:1], "t", "b", Path("o"))["ok"])
        self.assertFalse(runner.run_summary([], "t", "b", Path("o"))["ok"])

    def test_contact_sheet_spec(self):
        rows = [runner.run_result({"demo": "display", "backend": b, "folder": f"display/{b}"}, _summary(), 0, 1.0)
                for b in ("vulkan", "opengl")]
        rows.append(runner.run_result(self.RUN, None, 1, 1.0))
        spec = runner.contact_sheet_spec(rows, Path("sheet.png"), "Showcase")
        self.assertEqual([c["label"] for c in spec["cells"]],
                         ["display (vulkan): PASS", "display (opengl): PASS", "presets: FAIL"])
        self.assertEqual([c["image"] for c in spec["cells"]], ["k.png", "k.png", ""])  # a failed run keeps its cell
        self.assertEqual((spec["columns"], spec["output"], spec["title"]), (3, "sheet.png", "Showcase"))
        self.assertEqual(runner.contact_sheet_spec(rows[:1], Path("s.png"), "")["columns"], 1)

    def test_ffmpeg_command(self):
        command = runner.ffmpeg_command("ffmpeg", Path("seq"), 12, Path("out.mp4"))
        self.assertEqual(command[command.index("-framerate") + 1], "12")
        self.assertEqual(command[command.index("-i") + 1], str(Path("seq") / "frame_%04d.png"))
        self.assertEqual(command[-1], "out.mp4")
        self.assertIn("yuv420p", command)


class LayoutTest(unittest.TestCase):
    def test_grid_cells_do_not_overlap_and_fit_the_canvas(self):
        layout = helper.grid_layout(5, 3, 100, 50, pad=10, label_h=20, title_h=30)
        self.assertEqual((layout["columns"], layout["rows"]), (3, 2))
        self.assertEqual((layout["width"], layout["height"]), (10 + 3 * 110, 30 + 10 + 2 * 80))
        cells = layout["cells"]
        self.assertEqual(cells[0], (10, 40, 100, 50))
        self.assertEqual(cells[4], (120, 120, 100, 50))         # second row, second column
        for x, y, w, h in cells:
            self.assertLessEqual(x + w, layout["width"])
            self.assertLessEqual(y + h + layout["label_h"], layout["height"])
        for i, a in enumerate(cells):
            for b in cells[i + 1:]:
                self.assertTrue(a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or a[1] + a[3] <= b[1] or
                                b[1] + b[3] <= a[1], (a, b))

    def test_grid_narrower_than_its_columns_and_empty(self):
        self.assertEqual(helper.grid_layout(2, 3, 100, 50)["columns"], 2)
        empty = helper.grid_layout(0, 3, 100, 50, pad=12, title_h=0)
        self.assertEqual((empty["rows"], empty["cells"], empty["height"]), (0, [], 12))
        for bad in ((-1, 3, 10, 10), (2, 0, 10, 10), (2, 2, 0, 10)):
            with self.assertRaises(ValueError):
                helper.grid_layout(*bad)

    def test_fit_keeps_aspect_and_centres(self):
        self.assertEqual(helper.fit(1920, 1080, 480, 270), (480, 270, 0, 0))
        self.assertEqual(helper.fit(1000, 1000, 480, 270), (270, 270, 105, 0))
        self.assertEqual(helper.fit(200, 100, 480, 480), (480, 240, 0, 120))
        with self.assertRaises(ValueError):
            helper.fit(0, 10, 10, 10)

    def test_script_args(self):
        self.assertEqual(helper.script_args(["blender", "-b", "--python", "x.py", "--", "contact-sheet", "s.json"]),
                         ["contact-sheet", "s.json"])
        self.assertEqual(helper.script_args(["blender", "--python", "x.py"]), [])


class ReferenceFrontTest(unittest.TestCase):
    T = np.array([0.41, 0.84, 1.19, 3.33])
    Z = np.array([1.11, 1.22, 1.44, 4.11])

    def test_matches_linear_interpolation_from_the_initial_column(self):
        for t in (0.0, 0.2, 0.41, 0.6, 1.0, 2.5, 3.33):
            expected = float(np.interp(t, [0.0, *self.T], [1.0, *self.Z]))
            self.assertAlmostEqual(helper.reference_front(t, self.T, self.Z), expected, msg=t)

    def test_holds_the_ends(self):
        self.assertEqual(helper.reference_front(-1.0, self.T, self.Z), 1.0)
        self.assertAlmostEqual(helper.reference_front(9.0, self.T, self.Z), 4.11)


if __name__ == "__main__":
    unittest.main()
