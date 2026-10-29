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


def _full_version(value) -> bool:
    """A declared version is ``x.y.z``; the range bounds may omit the patch."""
    if not isinstance(value, str) or len(value.split(".")) != 3:
        return False
    try:
        parse(value)
    except ValueError:
        return False
    return True


def declaration_problems(plugin: Plugin) -> list[str]:
    """Everything wrong with the plugin's own declaration; an empty list means it is consistent."""
    try:
        return _declaration_problems(plugin)
    except Exception as error:  # noqa: BLE001 - a malformed declaration is a finding
        plugin_id = getattr(plugin, "plugin_id", "?")
        return [f"{plugin_id}: declaration: {type(error).__name__}: {error}"]


def _declaration_problems(plugin: Plugin) -> list[str]:
    problems: list[str] = []

    def fail(message: str) -> None:
        problems.append(f"{plugin.plugin_id}: {message}")

    if not isinstance(plugin.plugin_id, str) or not PLUGIN_ID.match(plugin.plugin_id):
        fail(f"plugin id {plugin.plugin_id!r} must be lower_snake_case")
    if not _full_version(plugin.version):
        fail(f"version {plugin.version!r} is not x.y.z")
    if not isinstance(plugin.sdk, str):
        fail("sdk must be a version range string")
    else:
        try:
            if not compatible(plugin.sdk):
                fail(f"requires SDK {plugin.sdk}, platform has {SDK_VERSION}")
        except ValueError as error:
            fail(f"sdk range {plugin.sdk!r}: {error}")
    if not isinstance(plugin.label, str) or not plugin.label.strip():
        fail("the label is empty")
    if not isinstance(plugin.models, tuple | list):
        fail(f"models must be a tuple of models, got {type(plugin.models).__name__}")
        models = ()
    else:
        models = plugin.models
    if not models:
        fail("no calculation model is declared")
    if not isinstance(plugin.demo, tuple | list):
        fail(f"demo must be a tuple of cases, got {type(plugin.demo).__name__}")
        cases = ()
    else:
        cases = plugin.demo
    if not cases:
        fail("an asset plugin must ship a demo case")

    model_ids = [
        model.model_id for model in models if isinstance(getattr(model, "model_id", None), str)
    ]
    repeated = sorted({model_id for model_id in model_ids if model_ids.count(model_id) > 1})
    if repeated:
        problems.append(f"{plugin.plugin_id}: model {', '.join(repeated)} is declared twice")
    for model in models:
        try:
            problems += _model_problems(plugin, model)
        except Exception as error:  # noqa: BLE001 - a malformed model is a finding
            problems.append(
                f"{plugin.plugin_id}: model declaration: {type(error).__name__}: {error}"
            )

    names = [case.name for case in cases if isinstance(getattr(case, "name", None), str)]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        problems.append(f"{plugin.plugin_id}: demo case {', '.join(repeated)} is declared twice")
    for case in cases:
        if not isinstance(case, DemoCase):
            problems.append(f"{plugin.plugin_id}: demo case {case!r} is not a DemoCase")
            continue
        try:
            problems += _demo_problems(plugin, case)
        except Exception as error:  # noqa: BLE001 - a malformed demo case is a finding
            problems.append(
                f"{plugin.plugin_id}: demo case {case.name!r}: {type(error).__name__}: {error}"
            )
    return list(dict.fromkeys(problems))


def _model_problems(plugin: Plugin, model) -> list[str]:
    """One model's declaration against the plugin that owns it."""
    problems: list[str] = []
    model_id = getattr(model, "model_id", None)
    where = f"model {model_id!r}"
    if not isinstance(model_id, str) or not DOTTED_ID.match(model_id):
        problems.append(
            f"{plugin.plugin_id}: {where} must be a dotted lower_snake_case identifier"
        )
    elif not plugin.namespaced(model_id):
        problems.append(f"{plugin.plugin_id}: {where} must start with {plugin.plugin_id + '.'!r}")
    if not _full_version(getattr(model, "version", None)):
        problems.append(
            f"{plugin.plugin_id}: {where} version {getattr(model, 'version', None)!r} is not x.y.z"
        )
    types = getattr(model, "instrument_types", ())
    if not isinstance(types, tuple | list):
        problems.append(f"{plugin.plugin_id}: {where} instrument_types must be a tuple")
        types = ()
    if not types:
        problems.append(f"{plugin.plugin_id}: {where} serves no instrument type")
    for type_id in types:
        if not isinstance(type_id, str) or not INSTRUMENT_TYPE.match(type_id):
            problems.append(
                f"{plugin.plugin_id}: {where} instrument type {type_id!r} "
                "is not lower_snake_case"
            )
    if len(set(types)) != len(types):
        problems.append(f"{plugin.plugin_id}: {where} names an instrument type twice")
    metrics = getattr(model, "metrics", ())
    if not isinstance(metrics, tuple | list):
        problems.append(f"{plugin.plugin_id}: {where} metrics must be a tuple")
        metrics = ()
    if len(set(metrics)) != len(metrics):
        problems.append(f"{plugin.plugin_id}: {where} declares a metric twice")
    for metric in metrics:
        if not isinstance(metric, str) or not DOTTED_ID.match(metric):
            problems.append(
                f"{plugin.plugin_id}: {where} metric {metric!r} must be a dotted "
                "lower_snake_case identifier"
            )
        elif not plugin.namespaced(metric):
            problems.append(
                f"{plugin.plugin_id}: {where} metric {metric!r} must start with "
                f"{plugin.plugin_id + '.'!r}"
            )
    required = getattr(model, "required_metrics", ())
    if not isinstance(required, tuple | list):
        problems.append(f"{plugin.plugin_id}: {where} required_metrics must be a tuple")
        required = ()
    missing = sorted(set(required) - set(metrics))
    if missing:
        problems.append(
            f"{plugin.plugin_id}: {where}: required metric {', '.join(missing)} "
            "is not among its metrics"
        )
    terms = getattr(model, "terms_schema", None)
    if terms is not None:
        try:
            terms.model_json_schema()
        except Exception as error:  # noqa: BLE001 - a broken schema is a finding
            problems.append(
                f"{plugin.plugin_id}: {where} terms schema has no JSON Schema: "
                f"{type(error).__name__}: {error}"
            )
    return problems


def _demo_problems(plugin: Plugin, case: DemoCase) -> list[str]:
    """The demo case against the models it names and the paths it carries."""
    problems: list[str] = []
    where = f"demo case {case.name!r}"
    models = [
        model
        for model in plugin.models
        if case.position.type in getattr(model, "instrument_types", ())
    ]
    if not models:
        served = sorted(
            {
                type_id
                for model in plugin.models
                for type_id in getattr(model, "instrument_types", ())
                if isinstance(type_id, str)
            }
        )
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
            if not requirement.required:
                continue
            driver = case.drivers.get(requirement.quantity)
            if driver is None:
                problems.append(
                    f"{plugin.plugin_id}: {where} maps no driver for quantity "
                    f"{requirement.quantity!r}"
                )
                continue
            try:
                case.paths.driver(driver)
            except (KeyError, ValueError):
                problems.append(
                    f"{plugin.plugin_id}: {where} names driver {driver!r}, "
                    "which the demo paths do not carry"
                )
    return problems


def build(plugins: Iterable[Plugin]) -> Registry:
    """Check the declarations together and index them; raise ``PluginError`` on any problem."""
    registry = Registry()
    problems: list[str] = []
    for plugin in plugins:
        problems += declaration_problems(plugin)
        plugin_id = getattr(plugin, "plugin_id", "?")
        if plugin_id in registry.plugins:
            problems.append(f"plugin {plugin_id!r} is declared twice")
            continue
        registry.plugins[plugin_id] = plugin
        models = plugin.models if isinstance(getattr(plugin, "models", None), tuple | list) else ()
        for model in models:
            model_id = getattr(model, "model_id", None)
            if not isinstance(model_id, str):
                continue
            if model_id in registry.models:
                owner = registry.owner[model_id]
                if owner == plugin_id:
                    problems.append(f"model {model_id!r} is declared twice by {plugin_id}")
                else:
                    problems.append(
                        f"model {model_id!r} is declared by {owner} and {plugin_id}"
                    )
                continue
            registry.models[model_id] = model
            registry.owner[model_id] = plugin_id
    served: dict[str, list[str]] = {}
    for model_id, model in registry.models.items():
        types = getattr(model, "instrument_types", ())
        if not isinstance(types, tuple | list):
            continue
        for type_id in types:
            if isinstance(type_id, str):
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
        if package == PACKAGE:
            expected = name.rsplit(".", 1)[-1]
            if declared.plugin_id != expected:
                problems.append(
                    f"{name} declares plugin_id {declared.plugin_id!r}, expected {expected!r}"
                )
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
