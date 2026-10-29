"""The core does not know the concrete asset types: it asks the plugin registry."""

import ast
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[2] / "src" / "mst" / "engine"
CONCRETE = {"common_stock", "real_estate", "bond", "loan", "deposit"}
MODEL_PACKAGES = {"equity", "bond", "credit", "housing"}
MODEL_IDS = {
    "equity.paths.v1",
    "bond.paths.v1",
    "credit.paths.v1",
    "housing.paths.v1",
}


def imports(path: Path):
    """Every module a file imports, with the names it takes from a ``from`` import."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, ()
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = "." * node.level + module
            yield module, tuple(alias.name for alias in node.names)


def boundary_problems(paths):
    """Imports of a concrete plugin package or an instrument type in the given files."""
    problems = []
    for path in paths:
        for module, names in imports(path):
            normalized = module.lstrip(".")
            if normalized.startswith("mst.plugins.") or normalized.startswith("plugins."):
                problems.append(f"{path.name}: imports {module}")
            lower = {name.lower() for name in names}
            if normalized in {"mst.plugins", "plugins"} and MODEL_PACKAGES & lower:
                problems.append(f"{path.name}: imports {sorted(MODEL_PACKAGES & lower)}")
    return problems


def test_the_engine_does_not_import_a_concrete_asset_type():
    assert boundary_problems(ENGINE.rglob("*.py")) == []


def test_a_relative_import_of_a_concrete_type_is_flagged(tmp_path):
    file = tmp_path / "engine.py"
    file.write_text("from ..plugins.equity import MODEL\n", encoding="utf-8")
    assert boundary_problems([file]) == ["engine.py: imports ..plugins.equity"]


def test_the_engine_does_not_name_an_instrument_type():
    names = CONCRETE | MODEL_PACKAGES | MODEL_IDS
    problems = []
    for path in ENGINE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value in names:
                problems.append(f"{path.name}: {node.value!r}")
    assert problems == []
