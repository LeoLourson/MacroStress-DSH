"""The calculation model interface and what flows through it.

::

    requirements(position, options) -> DataRequirements
    validate(prepared_inputs)       -> Diagnostics
    calculate(prepared_inputs, calculation_context) -> CalculationResult

``requirements`` says what the model needs from the run's paths. The platform resolves the
inputs (``PreparedInputs``) — a model never reads a file, the network or an LLM. ``validate``
judges the prepared inputs and the position's terms; an ``error`` stops the calculation and
the position stays in the run with its reason. ``calculate`` is a pure function of the
prepared inputs and the context: the run's paths, dates, seed and versions. It reads joint
realisations from the context and never draws random numbers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Literal

from pydantic import BaseModel, JsonValue

from ..contracts import CodeKind, Contract, Currency, Horizon, Identifier, NonNegative, Positive
from .paths import Paths


class Diagnostic(Contract):
    """One thing ``validate`` or the platform found about a position."""

    code: Identifier
    severity: Literal["warning", "error"] = "error"
    message: str = ""


class Position(Contract):
    """One position of a portfolio: what it is, what it is worth and by which object it matches.

    ``type`` is the instrument type of the position (``common_stock``, ``bond``, ``loan``,
    ``real_estate``); the catalog of models picks the model by it. ``notional`` is the
    position's current value in millions of ``currency``. ``codes`` are the object's codes the
    position knows (``secid``, ``isin``, ``inn``, ``okved``, ``okato``); the platform matches
    them against the objects the satellites serve. ``terms`` are the model's own inputs, checked
    against the schema the model declares.
    """

    id: Identifier
    name: str = ""
    type: Identifier
    currency: Currency = "RUB"
    notional: Positive
    codes: dict[CodeKind, Identifier] = {}
    terms: dict[str, JsonValue] = {}


class DriverRequirement(Contract):
    """A driver a model reads: a quantity of a named object, or of the position's codes."""

    key: Identifier
    """The name the model reads the resolved driver by in ``PreparedInputs.drivers``."""
    quantity: Identifier
    object: Identifier | None = None
    """The object the quantity is measured on; without it the position's codes decide."""
    codes: dict[CodeKind, Identifier] = {}
    """Codes to match instead of the position's own, in the order the model names them."""
    required: bool = True
    purpose: str = ""


class DataRequirements(Contract):
    """What a model reads from the run: drivers and macro variables."""

    drivers: list[DriverRequirement] = []
    #: Macro variables the model reads from ``Paths.macro``.
    macro: list[Identifier] = []


class RunOptions(Contract):
    """What a run tells a model about itself before any path is read."""

    as_of: date
    horizon: Horizon
    base_currency: Currency = "RUB"


class PreparedInputs(Contract):
    """A position with the paths its model asked for, resolved by the platform."""

    position: Position
    model_id: Identifier
    as_of: date
    horizon: Horizon
    base_currency: Currency
    #: The position's value at ``t = 0`` in millions of the base currency.
    value_current: NonNegative
    #: Requirement key -> the driver ``<quantity>:<object>`` the platform resolved it to.
    drivers: dict[Identifier, Identifier] = {}
    #: Unresolved driver requirement keys and why no served driver matched them. A declared
    #: macro variable the paths lack is a diagnostic of its own, not a driver miss.
    missing: dict[Identifier, Identifier] = {}
    requirements: DataRequirements = DataRequirements()

    def driver(self, key: str) -> str:
        """The driver of requirement ``key``; a missing one is a question to the platform."""
        if key not in self.drivers:
            raise KeyError(f"no driver {key!r}: {self.missing.get(key, 'not required')}")
        return self.drivers[key]


@dataclass(frozen=True)
class CalculationContext:
    """The run a model calculates in: its paths, dates, seed and base currency."""

    as_of: date
    horizon: int
    seed: int
    base_currency: str
    paths: Paths


class CalculationModel(ABC):
    """One calculation model. An instrument type may have several, chosen explicitly."""

    model_id: str
    version: str
    label: str
    instrument_types: tuple[str, ...]
    #: The schema of ``Position.terms``; None means the model takes no terms.
    terms_schema: type[BaseModel] | None = None
    #: Metrics the model may report and those a completed result must carry.
    metrics: tuple[str, ...] = ()
    required_metrics: tuple[str, ...] = ()
    description: str = ""

    @abstractmethod
    def requirements(self, position: Position, options: RunOptions) -> DataRequirements:
        """Drivers and macro variables the position needs; no path has been read yet."""

    @abstractmethod
    def validate(self, prepared: PreparedInputs) -> list[Diagnostic]:
        """Judge the prepared inputs and the terms: an ``error`` stops the calculation."""

    @abstractmethod
    def calculate(self, prepared: PreparedInputs, context: CalculationContext):
        """The ``CalculationResult`` of the position under the run's paths."""

    def describe(self) -> dict:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "label": self.label,
            "instrument_types": list(self.instrument_types),
            "metrics": list(self.metrics),
            "required_metrics": list(self.required_metrics),
            "description": self.description,
        }
