"""The example scene scripts (examples/*.py) only use properties, operators and enum values the add-on defines.

The scripts need a live Blender to run (see examples/README.md); this test compiles each one and checks every
``scene.warp_sph.*`` / ``obj.warp_sph.*`` name, every ``bpy.ops.<module>.<operator>`` call with its keyword
arguments, and every constant enum value against the add-on source (parsed, not imported, so it needs no bpy).
"""
import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "warp_sph_realtime"
EXAMPLES = sorted(p for p in (ROOT / "examples").glob("*.py") if not p.name.startswith("_"))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _constants(tree: ast.Module) -> dict[str, object]:
    """Module-level ``NAME = <literal>`` assignments."""
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return found


def _enum_values(call: ast.Call, module_constants: dict[str, object]) -> set[str] | None:
    """Item identifiers of an ``EnumProperty(items=...)`` call, or None when it is not an enum / not resolvable."""
    if not (isinstance(call.func, ast.Name) and call.func.id == "EnumProperty"):
        return None
    for keyword in call.keywords:
        if keyword.arg == "items":
            node = keyword.value
            try:
                items = module_constants[node.id] if isinstance(node, ast.Name) else ast.literal_eval(node)
            except (KeyError, ValueError):
                return None
            return {item[0] for item in items}  # pyright: ignore[reportGeneralTypeIssues]
    return None


class _AddonFacts:
    """Property names (with enum values) and operator ids the add-on registers."""

    def __init__(self):
        self.constants = {}
        trees = {path: _tree(path) for path in PKG.glob("*.py")}
        for tree in trees.values():
            self.constants.update(_constants(tree))
        presets = _tree(PKG / "physics_presets.py")
        self.constants["physics_presets.PRESET_ITEMS"] = ast.literal_eval(
            next(n.value for n in presets.body if isinstance(n, ast.Assign) and n.targets[0].id == "PRESET_ITEMS"))
        self.scene_props = self._class_props(trees[PKG / "props.py"], "WARPSPH_SceneSettings")
        self.object_props = self._class_props(trees[PKG / "addon.py"], "WARPSPH_ObjectSettings")
        self.operators = {}  # bl_idname -> {kwarg name: enum values or None}
        for tree in trees.values():
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    idname = self._literal_attr(node, "bl_idname")
                    if idname and "." in idname:
                        self.operators[idname] = self._class_props_node(node)

    @staticmethod
    def _literal_attr(node: ast.ClassDef, name: str):
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in stmt.targets):
                try:
                    return ast.literal_eval(stmt.value)
                except ValueError:
                    return None
        return None

    def _class_props(self, tree: ast.Module, name: str) -> dict[str, set[str] | None]:
        return self._class_props_node(next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name))

    def _class_props_node(self, cls: ast.ClassDef) -> dict[str, set[str] | None]:
        props = {}
        for stmt in cls.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                values = None
                if isinstance(stmt.annotation, ast.Call):  # Blender reads the definition from the annotation
                    values = _enum_values(stmt.annotation, self.constants)
                    if values is None and self._is_preset_enum(stmt.annotation):
                        values = {item[0] for item in self.constants["physics_presets.PRESET_ITEMS"]}
                props[stmt.target.id] = values
        return props

    @staticmethod
    def _is_preset_enum(call: ast.Call) -> bool:
        return any(k.arg == "items" and ast.unparse(k.value) == "physics_presets.PRESET_ITEMS" for k in call.keywords)


FACTS = _AddonFacts()


def _is_scene(node: ast.AST) -> bool:
    return ast.unparse(node).split(".")[-1] == "scene"


class ExampleScriptsTest(unittest.TestCase):
    def test_examples_exist(self):
        self.assertGreaterEqual(len(EXAMPLES), 3)

    def test_each_example_compiles_and_uses_real_names(self):
        for path in EXAMPLES:
            with self.subTest(example=path.name):
                tree = _tree(path)
                compile(tree, str(path), "exec")
                constants = _constants(tree)
                self._check_properties(path.name, tree, constants)
                self._check_operators(path.name, tree, constants)

    def _resolve(self, node: ast.AST, constants):
        if isinstance(node, ast.Name) and node.id in constants:
            return constants[node.id]
        try:
            return ast.literal_eval(node)
        except ValueError:
            return None

    def _check_properties(self, name, tree, constants):
        group_vars = {}  # variable -> scene-settings (True) or object-settings (False)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Attribute) and node.value.attr == "warp_sph"):
                group_vars[node.targets[0].id] = _is_scene(node.value.value)
        used = 0
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    targets.extend(target.elts if isinstance(target, ast.Tuple) else [target])
            elif isinstance(node, ast.Attribute):
                targets.append(node)
            for target in targets:
                if not isinstance(target, ast.Attribute):
                    continue
                owner = target.value
                if isinstance(owner, ast.Attribute) and owner.attr == "warp_sph":
                    scene_group = _is_scene(owner.value)
                elif isinstance(owner, ast.Name) and owner.id in group_vars:
                    scene_group = group_vars[owner.id]
                else:
                    continue
                known = FACTS.scene_props if scene_group else FACTS.object_props
                self.assertIn(target.attr, known, f"{name}:{target.lineno}: no such "
                                                  f"{'scene' if scene_group else 'object'} setting {target.attr!r}")
                used += 1
                if isinstance(node, ast.Assign) and known[target.attr] is not None:
                    value = self._resolve(node.value, constants)
                    if isinstance(value, str):
                        self.assertIn(value, known[target.attr], f"{name}:{target.lineno}: {target.attr}={value!r}")
        self.assertGreater(used, 0, f"{name} sets no WarpSPH settings")

    def _check_operators(self, name, tree, constants):
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                match = re.fullmatch(r"bpy\.ops\.(\w+)\.(\w+)", ast.unparse(node.func))
                if match:
                    calls.append((node, f"{match.group(1)}.{match.group(2)}"))
        self.assertTrue(any(i.startswith("wm.warp_sph_") for _, i in calls), f"{name} calls no WarpSPH operator")
        for node, idname in calls:
            self.assertIn(idname, FACTS.operators, f"{name}:{node.lineno}: unknown operator {idname}")
            props = FACTS.operators[idname]
            for keyword in node.keywords:
                self.assertIn(keyword.arg, props, f"{name}:{node.lineno}: {idname} has no property {keyword.arg!r}")
                value = self._resolve(keyword.value, constants)
                if props[keyword.arg] is not None and isinstance(value, str):
                    self.assertIn(value, props[keyword.arg], f"{name}:{node.lineno}: {idname} {keyword.arg}={value!r}")

    def test_readme_lists_every_example(self):
        readme = (ROOT / "examples" / "README.md").read_text(encoding="utf-8")
        for path in EXAMPLES:
            self.assertIn(path.name, readme)

    def test_docs_manual_links_resolve(self):
        manual = ROOT / "docs" / "manual"
        for page in sorted(manual.glob("*.md")):
            for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", page.read_text(encoding="utf-8")):
                if "://" in target:
                    continue
                self.assertTrue((page.parent / target).exists(), f"{page.name}: broken link {target}")


if __name__ == "__main__":
    unittest.main()
