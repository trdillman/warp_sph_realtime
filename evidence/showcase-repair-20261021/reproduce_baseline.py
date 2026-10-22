"""Run with this checkout's Python. Source-extracted controls; no Blender/GPU launch."""
import ast
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = "e55dc1dec3de6d4ab92e011fa6f105a281c12536"


def source(path):
    return subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT, text=True)


display = ast.parse(source("examples/showcase/display.py"))
paddle = ast.parse(source("examples/showcase/emitter_mixer.py"))
runner_source = source("scripts/render_showcase.py")
tree = ast.parse(runner_source)
report = {"base": BASE, "lane": "source expressions and injected host process results; no GPU"}

for name, module, check, context in (
    ("wrong_transport_accepted", display, "_transport_labelled", {"info": {"label": "WRONG GPU ROUTE"}}),
    ("gravity_only_paddle_accepted", paddle, "paddle_stirs", {"speed": np.full(40, 3.0)}),
):
    call = next(n for n in ast.walk(module) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "check" and n.args and
                (isinstance(n.args[0], ast.Constant) and n.args[0].value == check or
                 isinstance(n.args[0], ast.JoinedStr) and any(isinstance(v, ast.Constant) and v.value == check
                                                           for v in n.args[0].values)))
    report[name] = bool(eval(compile(ast.Expression(call.args[1]), "baseline", "eval"), context))

report["build_outside_driver"] = any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                                       and isinstance(n.value.func, ast.Name) and n.value.func.id == "build"
                                       for n in display.body)
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
install = next(n.lineno for n in ast.walk(main) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Name) and n.func.id == "install_addon")
wait = min(n.lineno for n in ast.walk(main) if isinstance(n, ast.Call)
           and isinstance(n.func, ast.Name) and n.func.id == "wait_for_other_blender")
report["profile_mutated_before_busy_check"] = install < wait
encoder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "encode_videos")
report["encoder_has_no_timeout"] = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                                      and n.func.attr == "run" and not any(k.arg == "timeout" for k in n.keywords)
                                      for n in ast.walk(encoder))

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    path = root / "runner.py"
    path.write_text(runner_source)
    spec = importlib.util.spec_from_file_location("baseline_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def child(*args, **kwargs):
        (root / "presets/summary.json").write_text('{"ok":')
        return SimpleNamespace(returncode=0)
    try:
        with patch.object(module, "wait_for_other_blender", return_value=True), \
                patch.object(module.subprocess, "run", side_effect=child):
            module.run_demo("b", {"demo": "presets", "backend": "opengl", "folder": "presets"},
                            root, root / "start", {}, 1, 0)
    except json.JSONDecodeError:
        report["malformed_summary_escapes"] = True

print(json.dumps(report, indent=2))
(Path(__file__).parent / "baseline-reproduction.json").write_text(json.dumps(report, indent=2))
