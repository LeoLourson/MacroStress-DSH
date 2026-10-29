"""The operations of the ``portfolio`` tool: list, get, save, delete, check and exposure.

A portfolio is ``portfolios/<id>.json`` of the workspace. ``save`` checks the contract and writes
the file atomically; ``check`` loads the scenario the portfolio would run under and names every
position its macro and satellite versions cannot serve — an unknown instrument type, terms its
model cannot read, a macro variable the version lacks, an object no driver covers and a driver
the out-of-sample check rejected, with that check's reason. ``exposure`` values the positions at
the calculation month of the scenario's macro version, before any path is drawn.
"""

import re
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .. import store
from ..drivers import version as satellite_version
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
    portfolio = load(ws.portfolios, id)
    spec = scenario_tool.load(ws.scenarios, scenario)
    posterior, meta = macro_version.load(ws.models, spec.macro)
    satellites = satellite_version.load(ws.models, spec.satellites)
    as_of = date.fromisoformat(meta["as_of"])
    problems = []
    if satellites.macro != spec.macro:
        problems.append(
            f"the satellites {spec.satellites} were estimated on the macro version "
            f"{satellites.macro}, the scenario names {spec.macro}"
        )
    corpus = ws.corpus if (ws.corpus / "index" / "chunks.jsonl").is_file() else None
    problems += scenario_build.validate(
        spec, posterior, satellites, as_of=as_of, corpus=corpus
    )["problems"]
    options = RunOptions(
        as_of=as_of, horizon=spec.horizon, base_currency=portfolio.base_currency
    )
    for position in portfolio.positions:
        problems += [
            f"{position.id}: {problem}"
            for problem in _position_problems(position, options, posterior, satellites, catalog)
        ]
    return {"id": id, "scenario": scenario, "ok": not problems, "problems": problems}


def exposure(ws: store.Workspace, id: str, scenario: str) -> dict:
    """The value of every position at the calculation month of the scenario's macro version."""
    portfolio = load(ws.portfolios, id)
    spec = scenario_tool.load(ws.scenarios, scenario)
    panel, as_of = satellite_version.macro_panel(ws.models, spec.macro)
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
    position: Position, options: RunOptions, posterior, satellites, catalog: Catalog
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
        if variable not in posterior.ids
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
