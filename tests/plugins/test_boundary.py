"""The core does not know the concrete asset types: it asks the plugin registry."""

import ast
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[2] / "src" / "mst" / "engine"
CONCRETE = {"common_stock", "real_estate", "bond", "loan", "deposit"}
MODEL_PACKAGES = {"equity", "bond", "credit", "housing"}


def imports(path: Path):
    """Every module a file imports, with the names it takes from a ``from`` import."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, ()
        elif isinstance(node, ast.ImportFrom):
            yield node.module or "", tuple(alias.name for alias in node.names)


def test_the_engine_does_not_import_a_concrete_asset_type():
    problems = []
    for path in ENGINE.glob("*.py"):
        for module, names in imports(path):
            if module.startswith("mst.plugins.") or module.startswith("plugins."):
                problems.append(f"{path.name}: imports {module}")
            if module in {"mst.plugins", "plugins"} and MODEL_PACKAGES & set(names):
                problems.append(f"{path.name}: imports {sorted(MODEL_PACKAGES & set(names))}")
    assert problems == []


def test_the_engine_does_not_name_an_instrument_type():
    problems = []
    for path in ENGINE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value in CONCRETE:
                problems.append(f"{path.name}: {node.value!r}")
    assert problems == []
