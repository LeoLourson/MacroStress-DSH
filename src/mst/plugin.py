"""``python -m mst.plugin``: scaffold, check and try out an asset plugin.

::

  python -m mst.plugin new <plugin_id> [--type-id ID] [--label TEXT] [--dest DIR]
  python -m mst.plugin check [<plugin_id> | --path DIR]
  python -m mst.plugin demo-run <plugin_id> [--path DIR]

``new`` writes a package that already passes ``check``: its declaration, one model, a demo
case, ``tests/`` and ``README.md``. By default the package goes to
``src/mst/plugins/<plugin_id>``, where the platform discovers it on the next start; with
``--dest DIR`` it goes to ``DIR/<plugin_id>`` and is checked there with ``--path``.

``check`` runs the conformance check (``mst.plugin_sdk.testing``): the declaration, the demo
case through the platform's dispatcher, determinism and the output contract. It exits with 1
on any finding. ``demo-run`` calculates the demo cases and prints P5/P50/P95 per position.
"""

import argparse
import importlib.util
import json
import keyword
import re
import sys
from pathlib import Path

from . import plugins
from .plugin_sdk.plugin import Plugin
from .plugin_sdk.testing import calculate_demo, check_plugin

PLUGINS = Path(__file__).resolve().parent / "plugins"
TEMPLATES = Path(__file__).resolve().parent / "templates" / "plugin"
#: Template file -> the file it becomes, for a new asset plugin.
FILES = {
    "__init__.py.tmpl": "__init__.py",
    "plugin.py.tmpl": "plugin.py",
    "model.py.tmpl": "model.py",
    "tests/__init__.py.tmpl": "tests/__init__.py",
    "tests/test_plugin.py.tmpl": "tests/test_{plugin_id}.py",
    "README.md.tmpl": "README.md",
}
_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_PLACEHOLDER = re.compile(r"\{\{([a-z_]+)\}\}")


def _template(name: str) -> str:
    return (TEMPLATES / name).read_text(encoding="utf-8")


def _literal(value: str) -> str:
    """``value`` as the inside of a double-quoted Python string literal."""
    return json.dumps(value, ensure_ascii=False)[1:-1]


def _render(text: str, values: dict) -> str:
    """Replace the template placeholders in one pass: user values are never re-scanned."""
    missing = []

    def replace(match):
        key = match.group(1)
        if key not in values:
            missing.append(key)
            return match.group(0)
        return values[key]

    rendered = _PLACEHOLDER.sub(replace, text)
    if missing:
        raise ValueError(f"unknown placeholder {missing[0]!r} in the template")
    return rendered


def new(plugin_id: str, *, type_id: str | None = None, label: str | None = None, dest=None) -> Path:
    """Write a new package and return its directory."""
    if not _ID.match(plugin_id) or keyword.iskeyword(plugin_id):
        raise SystemExit(f"plugin id must be a lowercase Python identifier, got {plugin_id!r}")
    target = (Path(dest) if dest else PLUGINS) / plugin_id
    if target.exists():
        raise SystemExit(f"{target} already exists")
    type_id = type_id or f"{plugin_id}_position"
    if not _ID.match(type_id):
        raise SystemExit(f"type id must be lowercase letters, digits and _, got {type_id!r}")
    label = label or plugin_id
    values = {
        "plugin_id": plugin_id,
        "type_id": type_id,
        "label": _literal(label),
        "label_text": label,
        "class": "".join(part.title() for part in plugin_id.split("_")) + "Model",
        "description": _literal("Позиция, стоимость которой следует пути одного драйвера."),
    }
    # Render everything before creating the directory: a bad template leaves no half package.
    rendered = {
        name.format(plugin_id=plugin_id): _render(_template(template), values)
        for template, name in FILES.items()
    }
    target.mkdir(parents=True)
    for name, text in rendered.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return target


def load(path) -> Plugin:
    """The ``plugin`` of a package outside ``mst.plugins`` (a plugin being developed)."""
    path = Path(path).resolve()
    init = path / "__init__.py"
    if not init.is_file():
        raise SystemExit(f"{path} is not a Python package: no __init__.py")
    name = f"mst_plugin_{path.name}"
    spec = importlib.util.spec_from_file_location(
        name, init, submodule_search_locations=[str(path)]
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"{path} is not a Python package")
    module = importlib.util.module_from_spec(spec)
    # A package edited between two checks must be read afresh: relative imports would
    # otherwise take the model and the declaration of the previous check from sys.modules.
    for key in [key for key in sys.modules if key == name or key.startswith(name + ".")]:
        del sys.modules[key]
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001 - a broken package must name itself
        raise SystemExit(f"{path} cannot be imported: {type(error).__name__}: {error}") from None
    declared = getattr(module, "plugin", None)
    if not isinstance(declared, Plugin):
        raise SystemExit(f"{path} exports no `plugin = Plugin(...)` declaration")
    return declared


def _installed() -> list[Plugin]:
    return plugins.discover()


def _problems(plugin: Plugin, registry=None) -> list[str]:
    return [str(finding) for finding in check_plugin(plugin, registry)]


def check(plugin_id: str | None = None, *, path=None) -> list[str]:
    """Every conformance problem of one plugin, of the named one, or of every installed one."""
    if plugin_id is not None and path is not None:
        raise SystemExit("check: give a plugin id or --path, not both")
    if path is not None:
        try:
            plugin = load(path)
        except SystemExit as error:
            return [str(error)]
        problems = _problems(plugin)
        if problems:
            return problems
        try:
            installed = _installed()
        except plugins.PluginError as error:
            return error.problems
        try:
            plugins.build([*installed, plugin])
        except plugins.PluginError as error:
            return error.problems
        return []
    if plugin_id is not None:
        try:
            registry = plugins.registry()
        except plugins.PluginError as error:
            return error.problems
        if plugin_id not in registry.plugins:
            raise SystemExit(
                f"no installed plugin {plugin_id!r}; known: {sorted(registry.plugins)}"
            )
        return _problems(registry.plugins[plugin_id], registry)
    problems: list[str] = []
    try:
        installed = _installed()
    except plugins.PluginError as error:
        return error.problems
    try:
        plugins.build(installed)
    except plugins.PluginError as error:
        problems += error.problems
    for plugin in installed:
        problems += _problems(plugin)
    return list(dict.fromkeys(problems))


def _chosen(plugin_id: str | None, path) -> tuple[Plugin, plugins.Registry]:
    """The plugin to calculate and the registry it is judged in."""
    if plugin_id is not None and path is not None:
        raise SystemExit("demo-run: give a plugin id or --path, not both")
    if path is not None:
        plugin = load(path)
        try:
            registry = plugins.build([plugin])
        except plugins.PluginError as error:
            raise SystemExit("\n".join(error.problems)) from None
        return plugin, registry
    if plugin_id is None:
        raise SystemExit("demo-run needs a plugin id or --path")
    try:
        registry = plugins.registry()
    except plugins.PluginError as error:
        raise SystemExit("\n".join(error.problems)) from None
    if plugin_id not in registry.plugins:
        raise SystemExit(
            f"no installed plugin {plugin_id!r}; known: {sorted(registry.plugins)}"
        )
    return registry.plugins[plugin_id], registry


def demo_run(plugin_id: str | None = None, *, path=None) -> str:
    """The demo cases of one plugin calculated on their own paths, as printable lines."""
    plugin, registry = _chosen(plugin_id, path)
    if not plugin.demo:
        raise SystemExit(f"{plugin.plugin_id} has no demo case")
    lines = []
    for case in plugin.demo:
        batch = calculate_demo(case, registry)
        lines.append(
            f"# {case.name}: позиций {len(batch.results)}, розыгрышей {case.paths.draws}, "
            f"горизонт {case.paths.horizon} мес."
        )
        for result in batch.results:
            lines.append(f"{result.position_id}  {result.model_id}  статус {result.status}")
            for metric_id, value in sorted(result.metrics.items()):
                if value.quantiles is not None:
                    quantiles = value.quantiles
                    text = (
                        f"P5 / P50 / P95 = {quantiles.p05:.4g} / "
                        f"{quantiles.p50:.4g} / {quantiles.p95:.4g}"
                    )
                    if quantiles.es95 is not None:
                        text += f"; ES95 {quantiles.es95:.4g}"
                elif value.value is not None:
                    text = f"{value.value:.4g}"
                else:
                    continue
                lines.append(f"    {metric_id}: {text} {value.unit}")
            for flag in result.flags:
                lines.append(f"    ! {flag}")
            for diagnostic in result.diagnostics:
                lines.append(f"    {diagnostic.severity}: {diagnostic.code} {diagnostic.message}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mst.plugin", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("new", help="scaffold a plugin package")
    create.add_argument("plugin_id")
    create.add_argument("--type-id")
    create.add_argument("--label")
    create.add_argument("--dest")

    checking = commands.add_parser("check", help="conformance of one plugin or all")
    checking.add_argument("plugin_id", nargs="?")
    checking.add_argument("--path")

    demo = commands.add_parser("demo-run", help="calculate a plugin's demo cases")
    demo.add_argument("plugin_id", nargs="?")
    demo.add_argument("--path")

    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        # Russian labels survive a Windows pipe.
        sys.stdout.reconfigure(encoding="utf-8")

    if args.command == "new":
        target = new(
            args.plugin_id, type_id=args.type_id, label=args.label, dest=args.dest
        )
        print(f"created {target}")
        where = f' --path "{target}"' if args.dest else f" {args.plugin_id}"
        print(f"next: python -m mst.plugin check{where}")
        return 0
    if args.command == "check":
        problems = check(args.plugin_id, path=args.path)
        for problem in problems:
            print(problem)
        print("ok" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    print(demo_run(args.plugin_id, path=args.path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
