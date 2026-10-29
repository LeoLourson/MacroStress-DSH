"""Discovery and validation of plugins, and the models everything else reads.

At start-up every package under ``mst.plugins`` is imported and its ``plugin = Plugin(...)``
declaration checked. The checks are all-or-nothing: the process does not start with a partial
set of plugins, and the error lists every problem with the plugin that caused it.

Nothing outside ``mst.plugins`` branches on an instrument type: the engine asks this registry
for the model of a position. A new asset type is a package here, not an edit of the core.
"""

import importlib
import pkgutil
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from threading import RLock

from ..plugin_sdk.model import CalculationModel, RunOptions
from ..plugin_sdk.plugin import DemoCase, Plugin
from ..plugin_sdk.version import SDK_VERSION, compatible, parse

PACKAGE = "mst.plugins"

PLUGIN_ID = re.compile(r"^[a-z][a-z0-9_]*$")
DOTTED_ID = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
INSTRUMENT_TYPE = re.compile(r"^[a-z][a-z0-9_]*$")


class PluginError(RuntimeError):
    """The declared plugins are inconsistent; the message lists every problem."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("Plugins cannot be loaded:\n- " + "\n- ".join(problems))


@dataclass
class Registry:
    """The installed plugins, their models by id and which plugin owns each model."""

    plugins: dict[str, Plugin] = field(default_factory=dict)
    models: dict[str, CalculationModel] = field(default_factory=dict)
    owner: dict[str, str] = field(default_factory=dict)


def declaration_problems(plugin: Plugin) -> list[str]:
    """Everything wrong with the plugin's own declaration; an empty list means it is consistent."""
    problems: list[str] = []

    def fail(message: str) -> None:
        problems.append(f"{plugin.plugin_id}: {message}")

    if not PLUGIN_ID.match(plugin.plugin_id):
        fail(f"plugin id {plugin.plugin_id!r} must be lower_snake_case")
    try:
        parse(plugin.version)
    except ValueError:
        fail(f"version {plugin.version!r} is not x.y.z")
    try:
        if not compatible(plugin.sdk):
            fail(f"requires SDK {plugin.sdk}, platform has {SDK_VERSION}")
    except ValueError as error:
        fail(f"sdk range {plugin.sdk!r}: {error}")
    if not plugin.label.strip():
        fail("the label is empty")
    if not plugin.models:
        fail("no calculation model is declared")
    if not plugin.demo:
        fail("an asset plugin must ship a demo case")

    for model in plugin.models:
        where = f"model {model.model_id}"
        if not DOTTED_ID.match(model.model_id):
            problems.append(
                f"{plugin.plugin_id}: {where} must be a dotted lower_snake_case identifier"
            )
        elif not plugin.namespaced(model.model_id):
            problems.append(
                f"{plugin.plugin_id}: {where} must start with {plugin.plugin_id + '.'!r}"
            )
        try:
            parse(model.version)
        except ValueError:
            problems.append(f"{plugin.plugin_id}: {where} version {model.version!r} is not x.y.z")
        if not model.instrument_types:
            problems.append(f"{plugin.plugin_id}: {where} serves no instrument type")
        for type_id in model.instrument_types:
            if not INSTRUMENT_TYPE.match(type_id):
                problems.append(
                    f"{plugin.plugin_id}: {where} instrument type {type_id!r} is not "
                    "lower_snake_case"
                )
        if len(set(model.instrument_types)) != len(model.instrument_types):
            problems.append(f"{plugin.plugin_id}: {where} names an instrument type twice")
        if len(set(model.metrics)) != len(model.metrics):
            problems.append(f"{plugin.plugin_id}: {where} declares a metric twice")
        for metric in model.metrics:
            if not DOTTED_ID.match(metric):
                problems.append(
                    f"{plugin.plugin_id}: {where} metric {metric!r} must be a dotted "
                    "lower_snake_case identifier"
                )
            elif not plugin.namespaced(metric):
                problems.append(
                    f"{plugin.plugin_id}: {where} metric {metric!r} must start with "
                    f"{plugin.plugin_id + '.'!r}"
                )
        missing = sorted(set(model.required_metrics) - set(model.metrics))
        if missing:
            problems.append(
                f"{plugin.plugin_id}: {where}: required metric "
                f"{', '.join(missing)} is not among its metrics"
            )
        if model.terms_schema is not None:
            try:
                model.terms_schema.model_json_schema()
            except Exception as error:  # noqa: BLE001 - a broken schema is a finding
                problems.append(
                    f"{plugin.plugin_id}: {where} terms schema has no JSON Schema: "
                    f"{type(error).__name__}: {error}"
                )

    names = [case.name for case in plugin.demo]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        problems.append(f"{plugin.plugin_id}: demo case {', '.join(repeated)} is declared twice")
    for case in plugin.demo:
        problems += _demo_problems(plugin, case)
    return list(dict.fromkeys(problems))


def _demo_problems(plugin: Plugin, case: DemoCase) -> list[str]:
    """The demo case against the models it names and the paths it carries."""
    problems: list[str] = []
    where = f"demo case {case.name!r}"
    models = [model for model in plugin.models if case.position.type in model.instrument_types]
    if not models:
        served = sorted({type_id for model in plugin.models for type_id in model.instrument_types})
        return [
            f"{plugin.plugin_id}: {where} has position type {case.position.type!r}, "
            f"which no model serves (served: {', '.join(served) or 'none'})"
        ]
    options = RunOptions(as_of=case.paths.months[0], horizon=case.paths.horizon)
    for model in models:
        try:
            needs = model.requirements(case.position, options)
        except Exception as error:  # noqa: BLE001 - a broken model is a finding
            problems.append(
                f"{plugin.plugin_id}: {where} model {model.model_id} cannot state its "
                f"requirements: {type(error).__name__}: {error}"
            )
            continue
        for variable in needs.macro:
            if variable not in case.paths.variables:
                problems.append(
                    f"{plugin.plugin_id}: {where} paths carry no macro variable {variable!r}"
                )
        for requirement in needs.drivers:
            if requirement.required and requirement.quantity not in case.drivers:
                problems.append(
                    f"{plugin.plugin_id}: {where} maps no driver for quantity "
                    f"{requirement.quantity!r}"
                )
    return problems


def build(plugins: Iterable[Plugin]) -> Registry:
    """Check the declarations together and index them; raise ``PluginError`` on any problem."""
    registry = Registry()
    problems: list[str] = []
    for plugin in plugins:
        problems += declaration_problems(plugin)
        if plugin.plugin_id in registry.plugins:
            problems.append(f"plugin {plugin.plugin_id!r} is declared twice")
            continue
        registry.plugins[plugin.plugin_id] = plugin
        for model in plugin.models:
            if model.model_id in registry.models:
                problems.append(
                    f"model {model.model_id!r} is declared by "
                    f"{registry.owner[model.model_id]} and {plugin.plugin_id}"
                )
                continue
            registry.models[model.model_id] = model
            registry.owner[model.model_id] = plugin.plugin_id
    served: dict[str, list[str]] = {}
    for model_id, model in registry.models.items():
        for type_id in model.instrument_types:
            served.setdefault(type_id, []).append(model_id)
    for type_id, model_ids in sorted(served.items()):
        if len(model_ids) > 1:
            problems.append(
                f"instrument type {type_id!r} is served by {', '.join(sorted(model_ids))}"
            )
    problems = list(dict.fromkeys(problems))
    if problems:
        raise PluginError(problems)
    return registry


def discover(package: str = PACKAGE) -> list[Plugin]:
    """Every plugin package under ``package``, in the alphabetical order of its name."""
    root = importlib.import_module(package)
    names = [
        f"{package}.{info.name}"
        for info in sorted(pkgutil.iter_modules(root.__path__), key=lambda item: item.name)
        if not info.name.startswith("_")
    ]
    plugins, problems = [], []
    for name in names:
        try:
            module = importlib.import_module(name)
        except Exception as error:  # noqa: BLE001 - a broken plugin must name itself
            problems.append(f"{name} cannot be imported: {type(error).__name__}: {error}")
            continue
        declared = getattr(module, "plugin", None)
        if not isinstance(declared, Plugin):
            problems.append(f"{name} exports no `plugin = Plugin(...)` declaration")
            continue
        plugins.append(declared)
    if problems:
        raise PluginError(problems)
    return plugins


_LOCK = RLock()
_CURRENT: Registry | None = None


def registry() -> Registry:
    """The process registry, discovered on first use and kept for the process lifetime."""
    global _CURRENT
    with _LOCK:
        if _CURRENT is None:
            _CURRENT = build(discover())
        return _CURRENT
