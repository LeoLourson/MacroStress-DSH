"""Every position on one set of paths: a run's calculation, without any input or output.

The run draws the paths once (``mst.scenario.build``) and hands the same object to every model.
A position whose model or driver is missing stays in the run with its reason; the report reads
only the positions that were calculated, and ``invalid`` results never enter portfolio
arithmetic.
"""

from dataclasses import dataclass

from ..drivers.registry import Registry
from ..plugin_sdk import fx
from ..plugin_sdk.model import CalculationContext, Diagnostic, Position, RunOptions
from ..plugin_sdk.paths import Paths
from ..plugin_sdk.results import CalculationResult
from .dispatch import calculate, prepare
from .model_catalog import Catalog
from .model_catalog import catalog as default_catalog
from .status import worst


@dataclass(frozen=True)
class Batch:
    """The results of a run, in the order of the positions that were given."""

    results: list[CalculationResult]
    status: str
    """The worst status of the positions: ``ok``, ``needs_review`` or ``invalid``."""

    @property
    def calculated(self) -> list[CalculationResult]:
        """The results that carry numbers; ``invalid`` ones carry their reason instead."""
        return [result for result in self.results if result.status != "invalid"]


def run(
    positions: list[Position],
    paths: Paths,
    *,
    drivers: Registry,
    served=None,
    catalog: Catalog = default_catalog,
    base_currency: str = "RUB",
) -> Batch:
    """The results of ``positions`` on ``paths``.

    ``drivers`` is the driver registry of the run's satellite version and ``served`` the drivers
    it serves; without ``served`` every registered driver resolves.
    """
    ids = [position.id for position in positions]
    if len(set(ids)) != len(ids):
        raise ValueError("two positions share one id")
    options = RunOptions(
        as_of=paths.months[0], horizon=paths.horizon, base_currency=base_currency
    )
    context = CalculationContext(
        as_of=options.as_of,
        horizon=paths.horizon,
        seed=paths.seed,
        base_currency=base_currency,
        paths=paths,
    )
    results = []
    for position in positions:
        prepared, diagnostics = prepare(position, options, paths, drivers, served, catalog)
        results.append(
            calculate(prepared, diagnostics, context, catalog)
            if prepared is not None
            else _unprepared(position, diagnostics, paths, context)
        )
    return Batch(results=results, status=worst(*(result.status for result in results)))


def _unprepared(
    position: Position,
    diagnostics: list[Diagnostic],
    paths: Paths,
    context: CalculationContext,
) -> CalculationResult:
    """A position that has no prepared inputs — no model, or terms its model cannot read."""
    try:
        value_current = fx.exposure(position, paths, context.base_currency)
    except ValueError:
        value_current = 0.0
    return CalculationResult(
        position_id=position.id,
        name=position.name,
        instrument_type=position.type,
        model_id="none",
        model_version="none",
        status="invalid",
        base_currency=context.base_currency,
        value_current=value_current,
        flags=[item.code for item in diagnostics if item.severity == "error"],
        diagnostics=diagnostics,
    )
