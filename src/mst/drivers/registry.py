"""Drivers: quantities of objects tied to the macro variables (scenario-formalization.md, 6).

``registry.toml`` declares the quantities: unit, transform, the macro variables and features
of ``x_t``, the deepest lag, how often the source observes it and the catalog that lists its
objects (``mst.drivers.catalogs``). Objects and drivers come from the catalogs, through the
catalog file (``<workspace>/drivers/catalog.json``) that ``python -m mst.drivers update``
writes, and from ``[[objects]]`` and ``[[drivers]]`` tables of the registry for anything no
catalog covers. A driver's id is
``<quantity>:<object>``. Transforms are those of the macro variables plus ``logit``, ``ln(d /
(1 - d))``, for shares and probabilities (section 6.2).
"""

import json
import tomllib
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from mst.contracts import Contract, Identifier, Number
from mst.macro.registry import BlockId
from mst.macro.registry import Registry as MacroRegistry

from .catalogs import CatalogSource
from .objects import Driver, Name, Object, series_id

REGISTRY_PATH = Path(__file__).with_name("registry.toml")

Transform = Literal["identity", "log100", "logit"]


def to_model(transform: Transform, x):
    """``g(d)``: the value in the scale the satellite models use."""
    x = np.asarray(x, dtype=float)
    if transform == "identity":
        return x
    if transform == "log100":
        if np.any(x <= 0):
            raise ValueError("100 ln d needs positive values")
        return 100 * np.log(x)
    if np.any((x <= 0) | (x >= 1)):
        raise ValueError("the logit needs values strictly between 0 and 1")
    return np.log(x / (1 - x))


def from_model(transform: Transform, y):
    """``d = g^-1(y)``: the value in the driver's own unit."""
    y = np.asarray(y, dtype=float)
    if transform == "identity":
        return y
    if transform == "log100":
        return np.exp(y / 100)
    return 1 / (1 + np.exp(-y))


class Feature(Contract):
    """``max(0, y - above)`` of a macro variable in the model scale."""

    id: Name
    variable: BlockId
    above: Number


class Quantity(Contract):
    id: Name
    label: Identifier
    unit: Identifier
    transform: Transform
    regressors: list[BlockId] = Field(min_length=1)
    features: list[Feature] = []
    lags: int = Field(strict=True, ge=0, le=3)
    span: int = Field(default=1, strict=True, ge=1, le=12)
    """Months one stored value covers: 1 for a monthly value, 3 for a quarter's average."""
    catalog: CatalogSource | None = None

    @property
    def inputs(self) -> list[str]:
        """The names of the elements of ``x_t``: regressors, then features."""
        return self.regressors + [feature.id for feature in self.features]


class Registry(Contract):
    quantities: list[Quantity] = Field(min_length=1)
    objects: list[Object] = []
    drivers: list[Driver] = []

    @model_validator(mode="after")
    def consistent(self):
        for kind, items in (("quantity", self.quantities), ("object", self.objects)):
            seen = set()
            for item in items:
                if item.id in seen:
                    raise ValueError(f"{kind} {item.id} is registered twice")
                seen.add(item.id)
        objects = {o.id: o for o in self.objects}
        for item in self.objects:
            chain, current = [item.id], item
            while current.parent is not None:
                if current.parent not in objects:
                    raise ValueError(f"{current.id} names unknown parent {current.parent}")
                current = objects[current.parent]
                if current.id in chain:
                    raise ValueError(f"objects {' > '.join(chain)} form a cycle")
                chain.append(current.id)
        quantities = {q.id for q in self.quantities}
        seen = set()
        for driver in self.drivers:
            if driver.quantity not in quantities:
                raise ValueError(f"{driver.id} names unknown quantity {driver.quantity}")
            if driver.object not in objects:
                raise ValueError(f"{driver.id} names unknown object {driver.object}")
            if driver.id in seen:
                raise ValueError(f"driver {driver.id} is registered twice")
            seen.add(driver.id)
        return self

    def quantity(self, id_: str) -> Quantity:
        for item in self.quantities:
            if item.id == id_:
                return item
        raise KeyError(id_)

    def object(self, id_: str) -> Object:
        for item in self.objects:
            if item.id == id_:
                return item
        raise KeyError(id_)

    def ancestors(self, id_: str) -> list[str]:
        """The objects above ``id_``, nearest first."""
        chain, current = [], self.object(id_)
        while current.parent is not None:
            chain.append(current.parent)
            current = self.object(current.parent)
        return chain

    def match(self, quantity: str, kind: str, code: str, among=None) -> str | None:
        """The driver of ``quantity`` whose object covers ``code`` of ``kind``, or None.

        An object covers a code when one of its codes of that kind is a prefix of it; the
        deepest object wins, then the longest code. ``among`` limits the drivers, e.g. to those
        a satellite version serves.
        """
        best, rank = None, None
        for driver in self.drivers:
            if driver.quantity != quantity or (among is not None and driver.id not in among):
                continue
            item = self.object(driver.object)
            for prefix in item.codes.get(kind, []):
                if code.startswith(prefix):
                    key = (len(self.ancestors(item.id)), len(prefix))
                    if rank is None or key > rank:
                        best, rank = driver.id, key
        return best

    def resolve(
        self, quantity: str, *, object_: str | None = None, codes=None, among=None
    ) -> str | None:
        """The driver a requirement (a quantity, an object or codes) resolves to, or None.

        A named object gives ``<quantity>:<object>`` when that driver exists (and is among
        ``among``); otherwise the first of ``codes`` that ``match`` resolves decides.
        """
        if object_ is not None:
            driver = f"{quantity}:{object_}"
            known = any(d.id == driver for d in self.drivers)
            return driver if known and (among is None or driver in among) else None
        for kind, code in (codes or {}).items():
            driver = self.match(quantity, kind, code, among)
            if driver is not None:
                return driver
        return None

    def check(self, macro: MacroRegistry) -> None:
        """Raises ``ValueError`` naming every macro variable the macro registry does not know."""
        known = {v.id for v in macro.variables}
        unknown = [
            name
            for q in self.quantities
            for name in q.regressors + [f.variable for f in q.features]
            if name not in known
        ]
        if unknown:
            raise ValueError("unknown macro variables: " + ", ".join(dict.fromkeys(unknown)))


def merge(raw: dict, catalog: dict) -> dict:
    """The registry tables with the objects and drivers of a catalog file added."""
    objects = {o["id"]: o for o in raw.get("objects", [])}
    drivers = list(raw.get("drivers", []))
    for quantity, part in catalog.get("quantities", {}).items():
        for item in part["objects"]:
            known = objects.setdefault(item["id"], item)
            if known != item:
                raise ValueError(f"object {item['id']} differs between catalogs")
        drivers += [
            {"quantity": quantity, "object": o, "series": series_id(quantity, o)}
            for o in part["drivers"]
        ]
    return {**raw, "objects": list(objects.values()), "drivers": drivers}


def load_registry(path: Path = REGISTRY_PATH, catalog: Path | None = None) -> Registry:
    with path.open("rb") as file:
        raw = tomllib.load(file)
    if catalog is not None and catalog.exists():
        raw = merge(raw, json.loads(catalog.read_text(encoding="utf-8")))
    return Registry.model_validate(raw)
