"""Discovery and validation of plugins, and the models everything else reads.

At start-up every package under ``mst.plugins`` is imported and its ``plugin = Plugin(...)``
declaration checked, together with the plugins of the installed distributions (the entry points
of the ``mst.plugins`` group) and the packages of the directories named in ``MST_PLUGINS_PATH``
— a model developed outside the platform is connected by any of the three, without touching
this package. The checks are all-or-nothing: the process does not start with a partial set of
plugins, and the error lists every problem with the plugin that caused it.

A plugin may carry position models, macro models and satellite models. Nothing outside this
package branches on an instrument type or a driver quantity: the engine asks the registry for
the model of a position, the run asks it for the macro model of a version and the satellite
model of a quantity. A new asset type or a new satellite of a quantity is a package, not an
edit of the core, and a quantity claimed by two satellite models is an error, not a choice.
"""

import importlib
import importlib.metadata
import importlib.util
import os
import pkgutil
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

from ..model_sdk.contracts import MacroModel, SatelliteModel
from ..model_sdk.version import MODEL_SDK_VERSION, model_compatible
from ..plugin_sdk.model import CalculationModel, RunOptions
from ..plugin_sdk.plugin import DemoCase, Plugin
from ..plugin_sdk.version import SDK_VERSION, compatible, parse

PACKAGE = "mst.plugins"
#: The entry-point group of the plugins an installed distribution declares.
ENTRY_POINTS = "mst.plugins"
#: The environment variable with the directories plugin packages live in, ``os.pathsep``-separated.
PATH_ENV = "MST_PLUGINS_PATH"

PLUGIN_ID = re.compile(r"^[a-z][a-z0-9_]*$")
DOTTED_ID = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
INSTRUMENT_TYPE = re.compile(r"^[a-z][a-z0-9_]*$")
QUANTITY_ID = re.compile(r"^[a-z][a-z0-9_]*$")


class PluginError(RuntimeError):
    """The declared plugins are inconsistent; the message lists every problem."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("Plugins cannot be loaded:\n- " + "\n- ".join(problems))


class PluginLoadError(RuntimeError):
    """A plugin package outside ``mst.plugins`` cannot be read; the message names it."""


@dataclass
class Registry:
    """The installed plugins and their models by kind: positions, macro, satellites."""

    plugins: dict[str, Plugin] = field(default_factory=dict)
    models: dict[str, CalculationModel] = field(default_factory=dict)
    owner: dict[str, str] = field(default_factory=dict)
    macro: dict[str, MacroModel] = field(default_factory=dict)
    """The macro models ``G_0`` by model id."""
    macro_owner: dict[str, str] = field(default_factory=dict)
    satellites: dict[str, SatelliteModel] = field(default_factory=dict)
    """The satellite models ``G_l`` by model id."""
    satellite_owner: dict[str, str] = field(default_factory=dict)
    quantity_model: dict[str, str] = field(default_factory=dict)
    """The satellite model id every driver quantity is assigned to."""


def load_package(path) -> Plugin:
    """The ``plugin`` declaration of a package outside ``mst.plugins``.

    The package is read afresh on every call, so a plugin being developed is checked as it
    stands, not as it was the last time.
    """
    path = Path(path).resolve()
    init = path / "__init__.py"
    if not init.is_file():
        raise PluginLoadError(f"{path} is not a Python package: no __init__.py")
    name = f"mst_plugin_{path.name}"
    spec = importlib.util.spec_from_file_location(
        name, init, submodule_search_locations=[str(path)]
    )
    if spec is None or spec.loader is None:
        raise PluginLoadError(f"{path} is not a Python package")
    module = importlib.util.module_from_spec(spec)
    # A package edited between two checks must be read afresh: relative imports would
    # otherwise take the model and the declaration of the previous check from sys.modules.
    for key in [key for key in sys.modules if key == name or key.startswith(name + ".")]:
        del sys.modules[key]
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # noqa: BLE001 - a broken package must name itself
        raise PluginLoadError(
            f"{path} cannot be imported: {type(error).__name__}: {error}"
        ) from None
    declared = getattr(module, "plugin", None)
    if not isinstance(declared, Plugin):
        raise PluginLoadError(f"{path} exports no `plugin = Plugin(...)` declaration")
    return declared


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
    if not isinstance(plugin.label, str) or not plugin.label.strip():
        fail("the label is empty")
    if not isinstance(plugin.macro, tuple | list):
        fail(f"macro must be a tuple of models, got {type(plugin.macro).__name__}")
        macro = ()
    else:
        macro = plugin.macro
    if not isinstance(plugin.satellites, tuple | list):
        fail(f"satellites must be a tuple of models, got {type(plugin.satellites).__name__}")
        satellites = ()
    else:
        satellites = plugin.satellites
    if not isinstance(plugin.models, tuple | list):
        fail(f"models must be a tuple of models, got {type(plugin.models).__name__}")
        models = ()
    else:
        models = plugin.models
    if not models and not macro and not satellites:
        fail("no model is declared: positions, macro or satellites")
    if models:
        if not isinstance(plugin.sdk, str) or not plugin.sdk:
            fail("position models need the position SDK range in sdk")
        else:
            try:
                if not compatible(plugin.sdk):
                    fail(f"requires SDK {plugin.sdk}, platform has {SDK_VERSION}")
            except ValueError as error:
                fail(f"sdk range {plugin.sdk!r}: {error}")
    if macro or satellites:
        if not isinstance(plugin.model_sdk, str) or not plugin.model_sdk:
            fail("macro and satellite models need the model SDK range in model_sdk")
        else:
            try:
                if not model_compatible(plugin.model_sdk):
                    fail(
                        f"requires model SDK {plugin.model_sdk}, platform has {MODEL_SDK_VERSION}"
                    )
            except ValueError as error:
                fail(f"model_sdk range {plugin.model_sdk!r}: {error}")
    if not isinstance(plugin.demo, tuple | list):
        fail(f"demo must be a tuple of cases, got {type(plugin.demo).__name__}")
        cases = ()
    else:
        cases = plugin.demo
    if models and not cases:
        fail("a plugin with position models must ship a demo case")

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
    for model in macro:
        try:
            problems += _macro_problems(plugin, model)
        except Exception as error:  # noqa: BLE001 - a malformed model is a finding
            problems.append(
                f"{plugin.plugin_id}: macro model declaration: {type(error).__name__}: {error}"
            )
    for model in satellites:
        try:
            problems += _satellite_problems(plugin, model)
        except Exception as error:  # noqa: BLE001 - a malformed model is a finding
            problems.append(
                f"{plugin.plugin_id}: satellite model declaration: {type(error).__name__}: {error}"
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
    """One position model's declaration against the plugin that owns it."""
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


def _macro_problems(plugin: Plugin, model) -> list[str]:
    """One macro model's declaration: its id, version and the variables it names."""
    problems: list[str] = []
    model_id = getattr(model, "model_id", None)
    where = f"macro model {model_id!r}"
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
    if not str(getattr(model, "label", "")).strip():
        problems.append(f"{plugin.plugin_id}: {where} has an empty label")
    variables = getattr(model, "variables", ())
    if not isinstance(variables, tuple | list):
        problems.append(f"{plugin.plugin_id}: {where} variables must be a tuple")
        variables = ()
    unknown = _unknown_macro_variables(variables)
    if unknown:
        problems.append(
            f"{plugin.plugin_id}: {where} names unknown macro variables: {', '.join(unknown)}"
        )
    return problems


def _satellite_problems(plugin: Plugin, model) -> list[str]:
    """One satellite model's declaration: its id, version and the quantities it serves."""
    problems: list[str] = []
    model_id = getattr(model, "model_id", None)
    where = f"satellite model {model_id!r}"
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
    if not str(getattr(model, "label", "")).strip():
        problems.append(f"{plugin.plugin_id}: {where} has an empty label")
    quantities = getattr(model, "quantities", ())
    if not isinstance(quantities, tuple | list):
        problems.append(f"{plugin.plugin_id}: {where} quantities must be a tuple")
        quantities = ()
    if not quantities:
        problems.append(f"{plugin.plugin_id}: {where} serves no quantity")
    unknown = _unknown_quantities(quantities)
    if unknown:
        problems.append(
            f"{plugin.plugin_id}: {where} names unknown quantities: {', '.join(unknown)}"
        )
    return problems


def _unknown_macro_variables(variables) -> list[str]:
    """The named variables the macro registry does not hold."""
    from ..macro.registry import load_registry

    known = {variable.id for variable in load_registry().variables}
    return [name for name in variables if name not in known]


def _unknown_quantities(quantities) -> list[str]:
    """The named quantities the driver registry does not hold."""
    from ..drivers.registry import load_registry

    known = {quantity.id for quantity in load_registry().quantities}
    return [name for name in quantities if name not in known]


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
        macro = plugin.macro if isinstance(getattr(plugin, "macro", None), tuple | list) else ()
        for model in macro:
            model_id = getattr(model, "model_id", None)
            if not isinstance(model_id, str):
                continue
            if model_id in registry.macro:
                owner = registry.macro_owner[model_id]
                problems.append(f"macro model {model_id!r} is declared by {owner} and {plugin_id}")
                continue
            registry.macro[model_id] = model
            registry.macro_owner[model_id] = plugin_id
        satellites = (
            plugin.satellites
            if isinstance(getattr(plugin, "satellites", None), tuple | list)
            else ()
        )
        for model in satellites:
            model_id = getattr(model, "model_id", None)
            if not isinstance(model_id, str):
                continue
            if model_id in registry.satellites:
                owner = registry.satellite_owner[model_id]
                problems.append(
                    f"satellite model {model_id!r} is declared by {owner} and {plugin_id}"
                )
                continue
            registry.satellites[model_id] = model
            registry.satellite_owner[model_id] = plugin_id
            for quantity in getattr(model, "quantities", ()) or ():
                if not isinstance(quantity, str):
                    continue
                if quantity in registry.quantity_model:
                    other = registry.quantity_model[quantity]
                    problems.append(
                        f"quantity {quantity!r} is served by satellite models "
                        f"{other} and {model_id}"
                    )
                    continue
                registry.quantity_model[quantity] = model_id
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


def _from_namespace(package: str, problems: list[str]) -> list[Plugin]:
    """Every plugin package under ``package``, in the alphabetical order of its name."""
    root = importlib.import_module(package)
    names = [
        f"{package}.{info.name}"
        for info in sorted(pkgutil.iter_modules(root.__path__), key=lambda item: item.name)
        if not info.name.startswith("_")
    ]
    found = []
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
        found.append(declared)
    return found


def _from_entry_points(problems: list[str]) -> list[Plugin]:
    """The plugins of the installed distributions, by the entry point of this group."""
    try:
        points = list(importlib.metadata.entry_points(group=ENTRY_POINTS))
    except Exception:  # noqa: BLE001 - no distribution metadata at all is no plugins
        return []
    found = []
    for point in sorted(points, key=lambda item: item.name):
        try:
            declared = point.load()
        except Exception as error:  # noqa: BLE001 - a broken entry point must name itself
            problems.append(
                f"entry point {point.name!r} cannot be loaded: {type(error).__name__}: {error}"
            )
            continue
        if not isinstance(declared, Plugin):
            problems.append(f"entry point {point.name!r} exports no Plugin declaration")
            continue
        if declared.plugin_id != point.name:
            problems.append(
                f"entry point {point.name!r} declares plugin_id {declared.plugin_id!r}"
            )
            continue
        found.append(declared)
    return found


def _from_paths(problems: list[str]) -> list[Plugin]:
    """The plugin packages of the directories of ``MST_PLUGINS_PATH``."""
    roots = [part for part in os.environ.get(PATH_ENV, "").split(os.pathsep) if part]
    found = []
    for root in roots:
        for folder in sorted(Path(root).iterdir()) if Path(root).is_dir() else []:
            if not (folder / "__init__.py").is_file():
                continue
            try:
                declared = load_package(folder)
            except PluginLoadError as error:
                problems.append(str(error))
                continue
            if declared.plugin_id != folder.name:
                problems.append(
                    f"{folder} declares plugin_id {declared.plugin_id!r}, expected {folder.name!r}"
                )
                continue
            found.append(declared)
    return found


def discover(package: str = PACKAGE) -> list[Plugin]:
    """Every plugin of the platform: the namespace, the entry points and the extra paths.

    A plugin id declared by two of them is a problem, not a preference.
    """
    problems: list[str] = []
    found = _from_namespace(package, problems)
    if package == PACKAGE:
        found += _from_entry_points(problems)
        found += _from_paths(problems)
        seen: dict[str, str] = {}
        for declared in found:
            where = seen.get(declared.plugin_id)
            if where is not None:
                problems.append(
                    f"plugin {declared.plugin_id!r} is declared twice: {where} and again later"
                )
            seen[declared.plugin_id] = declared.plugin_id
    if problems:
        raise PluginError(problems)
    return found


_LOCK = RLock()
_CURRENT: Registry | None = None


def registry() -> Registry:
    """The process registry, discovered on first use and kept for the process lifetime."""
    global _CURRENT
    with _LOCK:
        if _CURRENT is None:
            _CURRENT = build(discover())
        return _CURRENT


def reset() -> None:
    """Forget the process registry; the next ``registry()`` discovers the plugins afresh."""
    global _CURRENT
    with _LOCK:
        _CURRENT = None
