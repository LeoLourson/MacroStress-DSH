"""The operations of the ``portfolio`` tool: list, get, save, delete, check and exposure.

A portfolio is ``portfolios/<id>.json`` of the workspace. ``save`` checks the contract and writes
the file atomically; ``check`` loads the scenario the portfolio would run under and names every
position its macro and satellite versions cannot serve — an unknown instrument type, terms its
model cannot read, a macro variable the version lacks, an object no driver covers and a driver
the out-of-sample check rejected, with that check's reason — and warns what an impact on a macro
variable reaches through the positions' drivers, with the long-run loading of each one: a
loading below a tenth in absolute value is named as barely reacting, which is how a shock the
scenario does aim at a position still misses it. ``exposure`` values the positions at the
calculation month of the scenario's macro version, before any path is drawn.
"""

import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .. import store
from ..drivers import bundle as satellite_bundle
from ..engine.model_catalog import Catalog
from ..engine.model_catalog import catalog as default_catalog
from ..macro import registry as macro_registry
from ..macro import version as macro_version
from ..plugin_sdk import RunOptions, fx
from ..plugin_sdk.model import Position
from ..scenario import build as scenario_build
from ..scenario import tool as scenario_tool
from .spec import Portfolio

ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
"""The id of a portfolio is also the name of its file."""
BARELY_REACTS = 0.1
"""A long-run loading below this in absolute value reaches a driver in name only."""


def catalogue(root: Path) -> dict:
    """Every portfolio file with the base currency and the positions it names."""
    found, broken = [], []
    for file in sorted(root.glob("*.json")):
        try:
            item = Portfolio.model_validate(store.read_json(file))
        except Exception as error:  # noqa: BLE001 - one broken file does not hide the rest
            broken.append({"id": file.stem, "error": f"{type(error).__name__}: {error}"})
            continue
        found.append(
            {
                "id": file.stem,
                "name": item.name,
                "base_currency": item.base_currency,
                "positions": len(item.positions),
            }
        )
    return {"portfolios": found, "broken": broken}


def get(root: Path, id: str) -> dict:
    """One portfolio file as it is stored."""
    return {"id": id, "portfolio": load(root, id).model_dump(mode="json")}


def save(root: Path, id: str, raw: dict[str, Any]) -> dict:
    """Check the contract of ``raw`` and write it as ``<id>.json``."""
    portfolio = Portfolio.model_validate(raw)
    store.write_json(path(root, id), portfolio.model_dump(mode="json"))
    return {"id": id, "portfolio": portfolio.model_dump(mode="json")}


def delete(root: Path, id: str) -> dict:
    """Remove the portfolio file; a portfolio that is not there is not deleted silently."""
    file = path(root, id)
    if not file.is_file():
        raise FileNotFoundError(f"no portfolio {id!r} in {root}")
    file.unlink()
    return {"id": id, "deleted": True}


def load(root: Path, id: str) -> Portfolio:
    """The portfolio ``id`` of ``root``."""
    data = store.read_json(path(root, id))
    if data is None:
        raise FileNotFoundError(f"no portfolio {id!r} in {root}")
    return Portfolio.model_validate(data)


def path(root: Path, id: str) -> Path:
    """The file of portfolio ``id``; an id that is not a file name is refused."""
    if not ID.fullmatch(id or ""):
        raise ValueError(f"{id!r} is not a portfolio id: use lowercase letters, digits and dashes")
    return root / f"{id}.json"


def check(
    ws: store.Workspace, id: str, scenario: str, catalog: Catalog = default_catalog
) -> dict:
    """Every problem that keeps the portfolio from running under ``scenario``, each one named."""
    from mst import plugins

    portfolio = load(ws.portfolios, id)
    spec = scenario_tool.load(ws.scenarios, scenario)
    installed = plugins.registry()
    macro = macro_version.load(ws.models, spec.macro, installed.macro)
    satellites = satellite_bundle.load(ws.models, spec.satellites, installed.satellites)
    as_of = macro.as_of
    problems = []
    if satellites.macro != spec.macro:
        problems.append(
            f"the satellites {spec.satellites} were estimated on the macro version "
            f"{satellites.macro}, the scenario names {spec.macro}"
        )
    corpus = ws.corpus if (ws.corpus / "index" / "chunks.jsonl").is_file() else None
    problems += scenario_build.validate(
        spec, macro, satellites, as_of=as_of, corpus=corpus
    )["problems"]
    options = RunOptions(
        as_of=as_of, horizon=spec.horizon, base_currency=portfolio.base_currency
    )
    warnings = []
    for position in portfolio.positions:
        problems += [
            f"{position.id}: {problem}"
            for problem in _position_problems(position, options, macro, satellites, catalog)
        ]
        warnings += _reach_warnings(position, options, satellites, spec, catalog)
    return {
        "id": id,
        "scenario": scenario,
        "ok": not problems,
        "problems": problems,
        "warnings": warnings,
    }


def exposure(ws: store.Workspace, id: str, scenario: str) -> dict:
    """The value of every position at the calculation month of the scenario's macro version."""
    portfolio = load(ws.portfolios, id)
    spec = scenario_tool.load(ws.scenarios, scenario)
    panel, as_of = macro_version.panel_of(ws.models, spec.macro)
    macro = macro_registry.load_registry()
    index = {name: position for position, name in enumerate(panel.ids)}
    rows, problems = [], []
    for position in portfolio.positions:
        try:
            variable = fx.variable(position.currency, portfolio.base_currency)
        except ValueError as error:
            problems.append(f"{position.id}: {error}")
            continue
        if variable is not None and variable not in index:
            problems.append(f"{position.id}: the macro version has no {variable}")
            continue
        rate = (
            1.0
            if variable is None
            else float(macro.variable(variable).from_model(panel.y[-1, index[variable]]))
        )
        rows.append(
            {
                "position": position.id,
                "currency": position.currency,
                "notional": position.notional,
                "fx": rate,
                "value": position.notional * rate,
            }
        )
    return {
        "id": id,
        "scenario": scenario,
        "as_of": as_of.isoformat(),
        "base_currency": portfolio.base_currency,
        "exposures": rows,
        "problems": problems,
    }


def _position_problems(
    position: Position, options: RunOptions, macro, satellites, catalog: Catalog
) -> list[str]:
    """Why the portfolio's scenario cannot serve this position; empty when it can."""
    try:
        chosen = catalog.model_for(position)
    except (KeyError, ValueError) as error:
        return [str(error)]
    if chosen.terms_schema is not None:
        try:
            chosen.terms_schema.model_validate(position.terms)
        except ValidationError as error:
            return [
                f"terms /{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
                for item in error.errors()
            ]
    try:
        fx.variable(position.currency, options.base_currency)
    except ValueError as error:
        return [str(error)]
    try:
        needs = chosen.requirements(position, options)
    except (ValidationError, ValueError) as error:
        return [f"requirements: {error}"]
    problems = [
        f"the macro version has no {variable}"
        for variable in needs.macro
        if variable not in macro.ids
    ]
    served = set(satellites.served())
    for requirement in needs.drivers:
        driver = satellites.registry.resolve(
            requirement.quantity,
            object_=requirement.object,
            codes=requirement.codes or position.codes,
            among=served,
        )
        if driver is not None:
            continue
        known = satellites.registry.resolve(
            requirement.quantity,
            object_=requirement.object,
            codes=requirement.codes or position.codes,
        )
        if known is None:
            codes = requirement.codes or position.codes
            named = ", ".join(f"{kind} {codes[kind]}" for kind in sorted(codes)) or "no codes"
            problems.append(f"no object of {requirement.quantity} covers {named}")
            continue
        problems.append(f"{known} is not served: {_reason(satellites, known)}")
    return problems


def _reason(satellites, driver: str) -> str:
    """The reason the out-of-sample check gave for rejecting ``driver``."""
    try:
        quantity, node = satellites.node(driver)
        return satellites.checks[quantity][node].reason
    except (KeyError, AttributeError):
        return "it has no history of its own"


def _reach_warnings(
    position: Position, options: RunOptions, satellites, spec, catalog: Catalog
) -> list[str]:
    """What the scenario's impacts on macro variables reach through this position's drivers.

    A driver whose quantity reads an impacted variable is warned with its long-run loading —
    how much of a permanent unit of the variable the driver's level carries — and the inputs
    it loads on most, so a shock that reaches a position in name only is seen before the run.
    Positions the check already names as problems are left to the problems.
    """
    impacted = [impact.target for impact in spec.impacts if not impact.on_driver]
    if not impacted:
        return []
    try:
        chosen = catalog.model_for(position)
        needs = chosen.requirements(position, options)
    except (KeyError, ValueError, ValidationError):
        return []
    served = set(satellites.served())
    warnings = []
    for requirement in needs.drivers:
        driver = satellites.registry.resolve(
            requirement.quantity,
            object_=requirement.object,
            codes=requirement.codes or position.codes,
            among=served,
        )
        if driver is None:
            continue
        declared = satellites.registry.quantity(requirement.quantity)
        for variable in impacted:
            reads = variable in declared.regressors or any(
                feature.variable == variable for feature in declared.features
            )
            if reads:
                warnings.append(_loading_warning(position.id, driver, variable, satellites))
    return warnings


def _loading_warning(position_id: str, driver: str, variable: str, satellites) -> str:
    """The long-run loading of a driver to an impacted variable, with its largest inputs."""
    import numpy as np

    from mst.drivers.bundle import HIER, long_run_loadings

    quantity, node = satellites.node(driver)
    part = satellites.parts[quantity]
    gamma = satellites.arrays.get(f"{quantity}.gamma")
    if part.get("model") != HIER or gamma is None:
        return (
            f"{position_id}: an impact on {variable} reaches {driver}; the model that "
            "estimated it records no long-run loading"
        )
    loadings = {
        name: float(np.median(values))
        for name, values in long_run_loadings(part["quantity"], gamma, node).items()
    }
    of_variable = [feature["variable"] for feature in part["quantity"].get("features") or []]
    reading = dict.fromkeys(part["quantity"]["regressors"] + of_variable, 0.0)
    for name, source in zip(
        loadings,
        part["quantity"]["regressors"] + [f["id"] for f in part["quantity"].get("features") or []],
        strict=True,
    ):
        reading[source] += loadings[name]
    value = reading[variable]
    barely = " — it barely reacts" if abs(value) < BARELY_REACTS else ""
    top = sorted(loadings, key=lambda name: -abs(loadings[name]))[:3]
    named = ", ".join(f"{name} ({loadings[name]:+.2f})" for name in top)
    return (
        f"{position_id}: an impact on {variable} reaches {driver} with long-run loading "
        f"{value:+.2f}{barely}; it loads most on {named}"
    )
