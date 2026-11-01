"""The contracts of the model plugins: the macro model ``G_0`` and the satellites ``G_l``.

The platform keeps the scheme of docs/concept/scenario-formalization.md: the macro model turns the
shocks into the paths of the macro variables (level 0), and every satellite model turns the
levels above it into the quantities of its objects (levels 1 and further). A model plugin is a
package that declares ``plugin = Plugin(...)`` with ``macro=...`` and/or ``satellites=...``; it
is estimated on the history of the workspace, published as an immutable version and loaded back
by the same plugin that estimated it.

The interfaces below are deliberately narrow. ``estimate`` is pure computation from a context
(the series store, the registries and the calculation date) and returns plain metadata and
arrays; the platform saves them as a version, so a version never depends on the code that reads
it. ``load`` rebuilds the model of a saved version. ``simulate`` is a pure function of the
loaded model, the plan (constraints and driving shocks) or the macro simulation, the impacts of
the scenario and the seed: the same version and seed give the same numbers bit for bit.

Nothing here imports ``mst.scenario``, ``mst.runs`` or ``mst.plugins``: the neutral structures
(``ScenarioPlan``, ``AdditiveImpact``) are converted from the scenario contracts by the caller,
so a model plugin never sees the platform's orchestration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

import numpy as np

#: What a macro version must record for the platform to work with it: the model that estimated
#: it, the calculation date, the variable ids of its panel and simulation, the panel months and
#: the shock names of its ``J`` set.
MACRO_META_REQUIRED = ("model", "as_of", "ids", "months", "shocks")
#: A macro version must carry the panel it was estimated on: the satellites of that version and
#: the metrics read it.
MACRO_ARRAYS_REQUIRED = ("panel_y", "panel_approximate")
#: What a satellite bundle must record: the macro version, the calculation date, the driver
#: registry, one part per quantity and the order of the correlation matrix.
BUNDLE_META_REQUIRED = ("macro", "as_of", "registry", "quantities", "correlated")
#: What every quantity part of a bundle must record: the model that estimated it (the platform
#: writes the id of the model that returned the part), its months, the node tree, the fitted-row
#: count of every node and the check of every driver.
PART_META_REQUIRED = ("model", "months", "nodes", "driver", "checks", "rows")

#: How many candidates of a hidden coordinate — a regime path, a volatility path: anything the
#: constraints reweight but no gaussian draw reaches — the platform and the models draw per
#: value of ``R``. The stage-4 imposition selects one candidate by its box probability, and the
#: price of a scenario averages the candidates of a group into ``P(S | R)``.
LATENT_CANDIDATES = 64


@dataclass(frozen=True)
class LinearTerm:
    """One summand of a linear constraint: ``weight * y[variable, month]``."""

    variable: str
    month: int
    weight: float


@dataclass(frozen=True)
class LinearConstraint:
    """``low <= sum(weight * y[variable, month]) <= high`` with the point mark of its origin.

    The scenario contract fixes a value either by an interval ``low < high`` or by a point
    ``value``; the platform expands a point to ``value ± tolerance`` (half-width ``tau``), so a
    plan always carries ``low < high``. ``point`` keeps the origin of such an interval — the
    1 % gate of the metrics does not apply to it, because its probability measures ``tau``
    rather than the scenario — and ``tolerance`` is the ``tau`` the point was expanded to (the
    same scale the platform would use for a narrow interval of the same terms).
    """

    terms: tuple[LinearTerm, ...]
    low: float
    high: float
    point: bool = False
    tolerance: float = 0.0


@dataclass(frozen=True)
class ScenarioPlan:
    """What a macro model simulates under: the horizon, the constraints and the causes.

    ``constraints`` are in the model scale ``y = g(X)`` of each variable; ``causes`` names the
    coordinates of ``omega`` beyond the parameters that the scenario holds responsible for the
    event — shocks of the model's own ``J`` set and the hidden coordinates the model declares
    in ``latents``. An empty tuple means every coordinate except the parameters, so the law of
    ``R`` the price conditions on is the parameters alone. ``regime_event`` names the regime
    element the event itself asserts, fixed over the horizon — a cause named ``regime_path``
    explains the event through the regime instead, without fixing it.
    """

    horizon: int
    constraints: tuple[LinearConstraint, ...] = ()
    causes: tuple[str, ...] = ()
    regime_event: str | None = None
    inner: int = 1
    """How many values of the causes the model draws per value of ``R``: the conditions on
    means and probabilities reweight them inside their group, so a plan that carries such
    conditions asks for more than one."""


@dataclass(frozen=True)
class AdditiveImpact:
    """A direct impact ``o_l``: ``delta`` added to ``target`` in months ``start .. end``.

    ``target`` is a macro variable or a driver ``<quantity>:<object>``, in the model scale of
    the level it belongs to. The platform applies the impacts of the level before the levels
    below it are simulated, so an impact on the macro variables reaches every satellite.
    """

    target: str
    start: int
    end: int
    delta: float


@dataclass(frozen=True)
class MacroSimulation:
    """``R`` paths of the macro variables of a macro version.

    ``paths`` is ``(R, H + 1, n)`` in the model scale ``y = g(X)``; month ``t = 0`` is the last
    month of history. ``history`` carries the months ``1 - p .. 0`` of every repetition, drawn
    where the panel did not know them, for the satellites that need the recent past.
    ``diagnostics`` reports what the imposition did beyond drawing: the spread of the
    effective sizes of the hidden-coordinate candidates, so a scenario riding a single
    candidate is seen as such.
    """

    ids: list[str]
    months: list[date]
    paths: np.ndarray
    history: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)
    group: np.ndarray | None = None
    """``(R,)`` which value of ``R`` every path belongs to: the paths of one group share
    their parameters and everything else the causes do not name, and the conditions on
    means reweight them inside it. ``None`` when every path is its own group."""


@dataclass(frozen=True)
class ConstraintLaw:
    """The law of ``W y`` given ``R``, row by row: what the price of a scenario mixes.

    One row is one gaussian law of the ``m`` constraint functionals: ``means`` is ``(N, m)``
    and ``covs`` is ``(N, m, m)``. Rows share a group — ``group`` is ``(N,)`` — when they share
    the value of ``R``; within a group the rows are the candidates of everything ``R`` does not
    fix (the hidden coordinates and the free shocks) with importance weights ``e**log_weight``
    relative to a plain draw. A group of one row with ``log_weight = log pi`` is a forced
    hidden coordinate — the event that a regime path holds — whose prior weight the row
    carries. The platform turns the rows of a group into ``P(S | R)`` and averages
    ``-log P(S | R)`` over the groups (``mst.model_sdk.dist``).
    """

    means: np.ndarray
    covs: np.ndarray
    log_weight: np.ndarray
    group: np.ndarray


@dataclass(frozen=True)
class Dist:
    """The price of a scenario: ``dist(s) = KL(P_s || P) = E_P[-log P(S | R)]``.

    With no moment conditions the imposition of the canon is conditioning on the event ``S``,
    and the Kullback—Leibler divergence from the base measure is the mean of ``-log P(S | R)``
    over the law of ``R`` — not ``-log P(S)``, which the older code measured and which ignores
    the causes. ``log_probability`` is ``log P(S)`` beside it: the difference
    ``distance + log_probability`` is the parameter uncertainty the causes cannot explain.
    ``standard_error`` is the spread of ``-log P(S | R)`` over the groups; ``causes`` names the
    coordinates the scenario holds responsible, and ``tolerances`` carries the half-width
    ``tau`` of every point-fixed constraint, in the order of the plan. ``message`` is set when
    a group has no probability at all: the scenario contradicts the models, the case lemma 2
    of the canon names.
    """

    distance: float
    log_probability: float
    standard_error: float
    causes: tuple[str, ...] = ()
    tolerances: tuple[float, ...] = ()
    by_level: dict[int, float] = field(default_factory=dict)
    """The price split by the level of the conditioned quantity, at ``R = theta``: the level
    the scenario moves furthest from its base model. Empty when there is nothing to split —
    no conditions below the macro level, or a price no law measured."""
    message: str | None = None


@dataclass(frozen=True)
class DriverPaths:
    """The paths of drivers on a macro simulation, in the model scale of each quantity."""

    ids: list[str]
    paths: np.ndarray
    """``(R, H + 1, K)``; ``t = 0`` is the calculation month."""
    flags: dict[str, frozenset[str]]
    """The disclosures of every driver: parameters from above, approximate data, an impact."""


@dataclass(frozen=True)
class UpperPaths:
    """Everything a satellite of level ``l`` simulates on: the macro paths and the paths of
    every level above it, with its impacts already applied.

    ``macro`` is the level-0 simulation the run starts from; ``levels`` maps a level number
    to the paths of that level's quantities, in the model scale, with every impact of those
    levels applied — the recursion of the canon draws the levels in ascending order, so a
    model of level ``l`` reads the levels below its own and nothing else.
    """

    macro: MacroSimulation
    levels: dict[int, DriverPaths]


@dataclass(frozen=True)
class UpperHistory:
    """The history of one quantity of a level above, as a level-``l`` estimate reads it.

    ``values`` is ``(T, N)`` in the model scale of the quantity on its ``months``, NaN where
    unknown; ``drivers`` names the object of every column. A model of a level above builds
    its own inputs from these rows when its quantities read quantities instead of macro
    variables.
    """

    months: list[date]
    drivers: list[str]
    values: np.ndarray


@dataclass(frozen=True)
class Check:
    """The out-of-sample check of one driver, as the platform reads it from a version."""

    satellite: float | None
    """Root mean squared error of the satellite; None when no month could be forecast."""
    baseline: float | None
    """The same for the model without macro variables."""
    forecasts: int
    served: bool
    reason: str


class SeriesStoreLike(Protocol):
    """The series store a context carries; ``mst.data.store.SeriesStore`` implements it."""

    def as_of(self, series_id: str, day: date): ...


class MacroRegistryLike(Protocol):
    """The macro variables a context carries; ``mst.macro.registry.Registry`` implements it."""

    variables: list

    def variable(self, id_: str): ...


class DriverRegistryLike(Protocol):
    """The drivers a context carries; ``mst.drivers.registry.Registry`` implements it."""

    quantities: list
    objects: list
    drivers: list

    def quantity(self, id_: str): ...


@dataclass(frozen=True)
class MacroEstimateContext:
    """What a macro model estimates on: the series, the variables and the calculation date.

    The model builds its own panel (``mst.macro.panel.build_panel``) with its own spec, so two
    macro models may read the same history differently; nothing but the date limits what the
    model is allowed to see.
    """

    store: SeriesStoreLike
    registry: MacroRegistryLike
    as_of: date


@dataclass(frozen=True)
class MacroEstimate:
    """The estimate of a macro model: metadata and arrays of the version to be.

    ``meta`` must hold ``as_of``, ``ids``, ``months`` and ``shocks``; ``arrays`` must hold
    ``panel_y`` and ``panel_approximate`` — the panel the model was estimated on, which the
    satellites and the metrics read. Everything else is the model's own, read back only by its
    own ``load``.
    """

    meta: dict
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class SatelliteEstimateContext:
    """What a satellite model estimates on: the history of the objects and the macro panel.

    ``macro`` is the hash of the macro version whose panel this is; ``quantities`` are the
    quantities the platform assigned to this model — the ones it declares, plus any quantity an
    explicit ``quantity=model_id`` assignment hands to it, even one it does not declare. The
    driver registry carries the quantities, the object tree and the drivers with their series;
    ``mst.drivers.history`` builds the standard inputs of a quantity from it and the panel.
    ``upper_history`` carries, for every quantity of the levels above the model's own, its
    observed rows in the model scale — the inputs of a quantity that reads quantities instead
    of macro variables are built from these, not from the panel.
    """

    store: SeriesStoreLike
    registry: DriverRegistryLike
    panel: object
    """``mst.macro.panel.Panel``: the panel of the macro version, in the model scale."""
    as_of: date
    macro: str
    quantities: tuple[str, ...]
    upper_history: dict[str, UpperHistory] = field(default_factory=dict)


@dataclass(frozen=True)
class SatellitePart:
    """One quantity of a satellite estimate: the metadata and arrays of its part of a version.

    ``meta`` must hold ``months``, ``nodes``, ``driver``, ``rows`` and ``checks`` — the months of
    the panel part, the object tree, the driver id of every node, the number of months every node
    was fitted on and the ``Check`` of every driver with one. The platform adds ``model`` — the id
    of the model that returned the part; a part that names another model is refused. A model that
    stores the drivers' own history conventionally names its array ``d`` inside the part (the
    platform saves it as ``<quantity>.d``). It records the bounds a node was trained on as
    ``train_low`` and ``train_high`` (a list per node, ``None`` where a node has no range):
    the platform marks the paths that leave those bounds, and a part without the bounds is
    refused — the outside-range check cannot run on it.
    """

    meta: dict
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class SatelliteEstimate:
    """The estimate of a satellite model over its assigned quantities.

    ``residuals`` maps every driver with a history of its own to its residual series on the
    panel months, aligned with ``part.meta["months"]``; the platform correlates them across
    every model of the bundle.
    """

    parts: dict[str, SatellitePart] = field(default_factory=dict)
    residuals: dict[str, np.ndarray] = field(default_factory=dict)


class MacroModel(ABC):
    """The level-0 model ``G_0``: the distribution of the shocks and the transmission to ``x^0``.

    ``model_id`` starts with the plugin's id; ``variables`` names the macro variables the model
    works with, empty meaning all of the registry. ``shocks`` — the ``J`` set a scenario's
    driving shocks are checked against — is a property of a version: the model records it in
    its estimate.
    """

    model_id: str
    version: str
    label: str
    variables: tuple[str, ...] = ()
    latents: tuple[str, ...] = ()
    """The hidden coordinates of ``omega`` beyond the parameters and the shocks: a regime
    path, a volatility path — anything the constraints reweight but no gaussian draw
    reaches. A scenario names them among its causes to keep them out of ``R``, and the
    model draws their candidates (``LATENT_CANDIDATES``) when the price and the imposition
    need them. A model without such coordinates declares none."""

    @abstractmethod
    def estimate(self, context: MacroEstimateContext) -> MacroEstimate:
        """Estimate on the history known on ``context.as_of``; the version the platform saves."""

    @abstractmethod
    def load(self, meta: dict, arrays: dict):
        """The model of a saved version; only the model that estimated it can read it back."""

    @abstractmethod
    def simulate(
        self, model, plan: ScenarioPlan, *, draws: int, seed: int
    ) -> MacroSimulation:
        """``draws`` paths of the macro variables under the constraints and driving shocks.

        Raises ``InfeasibleScenario`` when the driving shocks cannot meet the constraints.
        """

    def constraint_law(
        self, model, plan: ScenarioPlan, *, draws: int, seed: int
    ) -> ConstraintLaw | None:
        """The law of the constraint functionals given ``R``, as ``ConstraintLaw`` rows.

        One group of rows belongs to one draw of ``R`` — the parameters and every coordinate
        the plan's driving shocks do not name — and holds the candidates of the hidden
        coordinates the model carries (``LATENT_CANDIDATES`` of them) with their importance
        weights. A model whose representation has no hidden structure returns one row per
        draw. ``None`` — the default — says the model does not measure the law, and the
        platform reports no price for its scenarios.
        """
        return None

    def describe(self) -> dict:
        return {
            "model_id": self.model_id,
            "kind": "macro",
            "version": self.version,
            "label": self.label,
            "variables": list(self.variables),
            "description": getattr(self, "description", ""),
        }


class SatelliteModel(ABC):
    """A model of the levels above 0: turns them into the quantities of the objects it serves.

    ``quantities`` are driver quantities of the core registry; several satellite models may
    declare one quantity — a plugin under development next to the installed one — and the
    platform refuses an *ambiguous assignment*, not the declaration itself: ``--model`` (an
    explicit ``quantity=model_id``) names the model that estimates it. The model of one version
    is loaded once for all its quantities: ``load`` receives the parts of every quantity
    assigned to it.

    ``simulate`` receives the paths of every level above the quantities it serves
    (``UpperPaths``): the macro simulation and the driver paths of the levels below its own,
    impacts applied, so a model of any level reads its inputs the same way.
    """

    model_id: str
    version: str
    label: str
    quantities: tuple[str, ...] = ()

    @abstractmethod
    def estimate(self, context: SatelliteEstimateContext) -> SatelliteEstimate:
        """Estimate every assigned quantity on the panel of the macro version."""

    @abstractmethod
    def load(self, parts: dict[str, SatellitePart]):
        """The model of the saved parts of its quantities, keyed by quantity id."""

    @abstractmethod
    def simulate(
        self,
        model,
        upper: UpperPaths,
        drivers: list[str],
        impacts: list[AdditiveImpact],
        *,
        correlation: tuple[list[str], np.ndarray] | None = None,
        shocks: np.ndarray,
        seed: int,
        stream: int,
    ) -> DriverPaths:
        """The paths of ``drivers`` on the levels above them, with the impacts added.

        ``upper`` carries the macro simulation and the driver paths of every level above the
        quantities the model serves, impacts of those levels applied; a model of level 1 reads
        ``upper.macro`` alone. ``correlation`` is the bundle's residual correlation: the driver
        ids it is measured on and their matrix. ``shocks`` holds the standardized residuals the
        platform drew for this model's correlated drivers — ``(R, steps, k)`` columns in the
        order of ``drivers``, counting only the drivers with a history of their own; a driver
        without one stays on the model's own independent noise. The model must use these
        columns as the residuals of those drivers and read the first rows it needs, not all
        ``steps``: the platform draws the joint shocks of all its drivers once per level, so
        the measured cross-model correlation reaches the joint paths, and the levels draw
        independently of one another as the product measure of the canon asks.
        ``stream`` separates the random streams of the models of one bundle — the posterior
        draw and the independent noise come from it, while the platform's shocks come from its
        own stream.
        """

    def describe(self) -> dict:
        return {
            "model_id": self.model_id,
            "kind": "satellite",
            "version": self.version,
            "label": self.label,
            "quantities": list(self.quantities),
            "description": getattr(self, "description", ""),
        }
