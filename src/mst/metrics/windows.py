"""The forecast window: what every check of a model is made of, whatever the level.

A window is a calculation date, the months of the horizon after the history known that day, a
scenario and the facts of those months. The rolling grid opens a window on every date it names,
with the empty scenario; an episode opens one on the day before the event starts and fills its
scenario from what happened. Everything else is the same for the two: the models of every level
are estimated on the data known on the calculation date, the paths are simulated under the
window's scenario, and the cells of every level are read from the window
(``mst.metrics.levels``).

The macro version of a date is cached under ``<models>/backtest/`` by the state of the data
and the model's configuration: the estimate is deterministic but expensive. A model that finds
the history of some variables too short says so with ``ShortHistory``; the version is estimated
again without them and records them in its meta as ``left_out``, so a short series leaves out
its own cells, not the date. The satellite bundle of a version is cached the same way, by the
inputs that shape it.
"""

import hashlib
import inspect
from dataclasses import dataclass, field, replace
from datetime import date
from functools import cache
from pathlib import Path

import numpy as np

from mst import inputs as answers
from mst.data.store import SeriesStore
from mst.macro import version as macro_version
from mst.macro.panel import seasonal_profile, shift
from mst.macro.registry import REGISTRY_PATH as MACRO_REGISTRY_PATH
from mst.macro.registry import Registry
from mst.macro.version import MacroVersion
from mst.model_sdk.bounds import observed
from mst.model_sdk.contracts import (
    MacroEstimate,
    MacroEstimateContext,
    ScenarioPlan,
    ShortHistory,
)
from mst.model_sdk.interventions import apply_default
from mst.model_sdk.version import MODEL_SDK_VERSION
from mst.plugin_sdk.version import SDK_VERSION

from .episodes import Episodes, calculation_date


@dataclass
class Window:
    """One forecast window: its versions, its scenario, its paths and its facts."""

    id: str
    """The episode id, or the grid date in ISO form."""
    as_of: date
    """The calculation date: the models know only what was published by then."""
    months: list[date]
    """The months of the horizon, t = 1 .. H, after the last month of the macro panel."""
    macro: MacroVersion
    scenario: ScenarioPlan
    realized: np.ndarray
    """(H, n) the macro variables in the model scale as ``known_on`` knows them; NaN gaps."""
    store: SeriesStore
    known_on: date
    """The date the facts are read on."""
    half: str
    """tuning or validation: the models are fitted on the first, judged on the second."""
    observations: list[str] = field(default_factory=list)
    """The variables the scenario fixes as outcomes: they follow the realized path."""
    interventions: list[str] = field(default_factory=list)
    """The variables the scenario fixes as decisions: they take the realized level."""
    source: str | None = None
    """Who formalised the scenario, manual or llm; None for the empty scenario."""
    bundle: object = None
    """The satellite bundle of the window's macro version, when the drivers are checked."""
    bundle_name: str | None = None
    macro_paths: object = None
    """The forecast of the macro variables under the scenario, read through the floors."""
    base_shadow: object = None
    """The shadow paths under the empty scenario, before the floors: the base run of the
    scenario, which the probability of its conditions also reads. None for the empty
    scenario itself, whose forecast is its own base."""
    driver_paths: object = None
    """The driver paths on ``macro_paths``."""
    base_driver_paths: object = None
    """The driver paths on the base run; None when the scenario is empty."""
    portfolio_id: str | None = None
    portfolio: object = None
    """The portfolio whose positions are judged, when the check names one."""
    past: object = None
    """The portfolio held over the past windows the calculation date knows
    (``mst.portfolio.history.Replay``): the historical simulation of its returns."""
    excluded: dict[str, str] = field(default_factory=dict)
    """The positions the engine could not value on the window's paths, with the reason."""

    @property
    def horizon(self) -> int:
        return len(self.months)

    @property
    def panel(self):
        """The panel of the macro version, in the model scale."""
        return self.macro.panel

    @property
    def constrained(self) -> list[str]:
        """The variables whose realized paths fix the scenario, both kinds together."""
        return [*self.observations, *self.interventions]

    @property
    def left_out(self) -> dict[str, str]:
        """The macro variables and the drivers the models left out as too short, with the
        reason."""
        found = dict(self.macro.meta.get("left_out", {}))
        if self.bundle is not None:
            found.update(self.bundle.left_out)
        return found


def open_window(
    plugin,
    data: SeriesStore,
    registry: Registry,
    plan: Episodes,
    *,
    id_: str,
    start: date,
    half: str,
    root: Path,
    known_on: date,
    inputs: Path | None = None,
) -> Window:
    """The window of a forecast made the day before ``start``, to the end of the plan's span.

    The horizon runs from the last month of the macro panel to ``after_months`` after
    ``start``, so a series published with a lag adds its missing months to the horizon.
    """
    macro = estimate_macro(plugin, data, registry, calculation_date(start), root, inputs)
    panel_months = list(macro.panel.months)
    end = shift(start, plan.after_months)
    horizon = (end.year - panel_months[-1].year) * 12 + end.month - panel_months[-1].month
    if horizon < 1:
        raise ValueError(f"the history ends at {panel_months[-1]:%Y-%m}, after the window")
    months = [shift(panel_months[-1], k) for k in range(1, horizon + 1)]
    as_of = calculation_date(start)
    return Window(
        id=id_,
        as_of=as_of,
        months=months,
        macro=macro,
        scenario=ScenarioPlan(horizon=horizon),
        realized=facts(data, registry, macro.panel, months, as_of, known_on, plan),
        store=data,
        known_on=known_on,
        half=half,
    )


def simulate(window: Window, registry: Registry, plan: Episodes) -> None:
    """The paths of the window: under its scenario, under the empty one, and of the drivers.

    A recursive model executes its interventions itself; a default-form one returns the paths
    before them, as the platform's ``build`` expects, so the same substitution is made here.
    The shadow paths are read through the registry's floors, as a run reports them.
    """
    macro, scenario = window.macro, window.scenario
    found = macro.plugin.simulate(macro.model, scenario, draws=plan.draws, seed=plan.seed)
    if macro.plugin.structural_form(macro.model) is None:
        paths = apply_default(found.paths, found.ids, scenario.interventions, inner=scenario.inner)
        found = replace(found, paths=paths)
    window.macro_paths = observed(found, registry)
    if scenario.constraints or scenario.interventions:
        window.base_shadow = macro.plugin.simulate(
            macro.model, ScenarioPlan(horizon=window.horizon), draws=plan.draws, seed=plan.seed
        )
    if window.bundle is not None:
        served = window.bundle.served()
        window.driver_paths = window.bundle.simulate(window.macro_paths, served, (), seed=plan.seed)
        if window.base_shadow is not None:
            window.base_driver_paths = window.bundle.simulate(
                observed(window.base_shadow, registry), served, (), seed=plan.seed
            )


@dataclass(frozen=True)
class Scope:
    """The levels a check judges beyond the macro variables."""

    drivers: Path | None = None
    """The driver catalog: the satellites are estimated on every window and their drivers
    judged."""
    portfolio_id: str | None = None
    portfolio: object = None
    """The portfolio whose positions and whole are judged; it needs the drivers."""

    @classmethod
    def of(cls, store: Path, drivers: Path | None, portfolio_id: str | None) -> "Scope":
        """The scope of a check run on the workspace of ``store``, whose portfolios lie beside
        its series."""
        if portfolio_id is None:
            return cls(drivers=drivers)
        if drivers is None:
            raise ValueError("--portfolio needs --drivers: the positions read the driver paths")
        from mst.portfolio import tool as portfolio_tool

        folder = Path(store).parent / "portfolios"
        try:
            portfolio = portfolio_tool.load(folder, portfolio_id)
        except FileNotFoundError:
            raise ValueError(
                f"--portfolio {portfolio_id}: no portfolio {portfolio_id!r} in {folder}"
            ) from None
        return cls(drivers=drivers, portfolio_id=portfolio_id, portfolio=portfolio)


MACRO_ONLY = Scope()
"""The scope of a check of the macro variables alone."""


def extend(window: Window, data: SeriesStore, root: Path, scope: Scope) -> None:
    """The models of the levels the scope names, on the window: the satellites, then the
    portfolio's history. A level that cannot be estimated voids the window with the reason."""
    if scope.drivers is not None:
        try:
            attach_satellites(window, data, root, scope.drivers)
        except ValueError as error:
            raise ValueError(f"satellites: {error}") from error
    if scope.portfolio is not None:
        try:
            attach_portfolio(window, scope.portfolio_id, scope.portfolio)
        except ValueError as error:
            raise ValueError(f"positions: {error}") from error


def attach_portfolio(window: Window, portfolio_id: str, portfolio) -> None:
    """The portfolio of the window and its history as the calculation date knows it.

    The history reads the window's own macro panel and the driver series known on its
    calculation date, so the benchmark of a position never sees the months it is judged on.
    A position the models cannot value on that history voids the window with the reason.
    """
    from mst.portfolio.history import replay

    if window.bundle is None:
        raise ValueError("the positions are valued on the driver paths: the window has no bundle")
    window.portfolio_id, window.portfolio = portfolio_id, portfolio
    window.past = replay(
        window.macro,
        window.bundle,
        portfolio,
        window.store,
        horizon=window.horizon,
        as_of=window.as_of,
    )


def provenance() -> dict:
    """The versions of the code a result was computed with."""
    return {"git_commit": git_commit(), "plugin_sdk": SDK_VERSION, "model_sdk": MODEL_SDK_VERSION}


def estimate_macro(
    plugin,
    data: SeriesStore,
    registry: Registry,
    as_of: date,
    root: Path,
    inputs: Path | None = None,
):
    """The macro version estimated on ``as_of``, cached under ``root`` by the data.

    The estimate itself is deterministic but expensive; the cache key covers the model, its
    version, its configuration files, every value the registry names as known on the date and
    the hash of every answer the model declares for the date, so a rerun with the same data
    loads the version instead of estimating again, and an edited spec, sign table, model
    package or answer never reuses a stale one.
    """
    name = macro_name(plugin, data, registry, as_of, root, inputs)
    return macro_version.load(root, name, {plugin.model_id: plugin})


def macro_name(
    plugin,
    data: SeriesStore,
    registry: Registry,
    as_of: date,
    root: Path,
    inputs: Path | None = None,
) -> str:
    """The name of the macro version of ``as_of`` in the cache, estimated when it is not there.

    A refusal of the model is cached too (``cached``): the estimate is seeded, so the same
    code on the same data refuses again, and a model that refuses every date — the MS-VAR
    whose stability filter leaves no draw — is not estimated again on every rerun.
    """
    pointer = root / _data_key(plugin, data, registry, as_of, inputs)
    found = cached(pointer, root / "macro")
    if found is not None:
        return found
    try:
        estimate, left_out = _estimate_known(plugin, data, registry, as_of, inputs)
    except ValueError as error:
        refuse(pointer, error)
        raise
    meta = {**estimate.meta, "left_out": left_out}
    name = macro_version.save(plugin.model_id, meta, estimate.arrays, root)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(name + "\n", encoding="utf-8")
    return name


REFUSED = "refused "
"""The first word of a pointer that holds a refusal: then the code digest and the reason."""


def cached(pointer: Path, folder: Path) -> str | None:
    """The version a cache pointer names, or None when it must be estimated; a cached refusal
    raises it again.

    A refusal counts only for the code it was made with (``code_digest``): the key of the
    pointer covers the model's own package and the data, while a refusal may come from the
    platform, and a fixed platform must estimate the date again.
    """
    if not pointer.exists():
        return None
    text = pointer.read_text(encoding="utf-8")
    if text.startswith(REFUSED):
        digest, _, reason = text[len(REFUSED) :].partition("\n")
        if digest == code_digest():
            raise ValueError(reason.strip())
        return None
    name = text.strip()
    return name if (folder / name).is_dir() else None


def refuse(pointer: Path, error: ValueError) -> None:
    """Keeps a refusal under the cache pointer, with the digest of the code that made it."""
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(f"{REFUSED}{code_digest()}\n{error}\n", encoding="utf-8")


@cache
def code_digest() -> str:
    """The sha256 of every Python file of the ``mst`` package: the code a refusal belongs to."""
    import mst

    digest = hashlib.sha256()
    package = Path(mst.__file__).parent
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _estimate_known(
    plugin, data: SeriesStore, registry: Registry, as_of: date, inputs: Path | None
) -> tuple[MacroEstimate, dict[str, str]]:
    """The estimate on the data known on ``as_of``, without the variables too short for it.

    A model that finds the history of some variables too short says so with ``ShortHistory``;
    it is estimated again on the registry without them, and they come back with the reason.
    Every round removes at least one variable it was given, so the rounds end. A short variable
    leaves out its own cells; only an episode that fixes it is lost with it.
    """
    left_out: dict[str, str] = {}
    while True:
        given = registry.model_copy(
            update={"variables": [v for v in registry.variables if v.id not in left_out]}
        )
        try:
            estimate = plugin.estimate(
                MacroEstimateContext(
                    store=data,
                    registry=given,
                    as_of=as_of,
                    inputs=inputs,
                    left_out=dict(left_out),
                )
            )
        except ShortHistory as short:
            known = {variable.id for variable in given.variables}
            stray = sorted(set(short.variables) - known)
            if stray:
                raise ValueError(
                    f"{plugin.model_id} calls short variables it was not given: {', '.join(stray)}"
                ) from short
            left_out.update(short.variables)
            continue
        return estimate, left_out


def _configuration_paths(plugin) -> list[Path]:
    """The files that shape a macro version beyond the data.

    The model's package is the source of its ``spec.toml`` and ``signs.toml`` and of the code
    that estimates the version; the macro registry decides the transform, seasonality and
    order of the panel. Hashing both means an edited configuration is a different cache entry.
    """
    paths = [Path(MACRO_REGISTRY_PATH)]
    try:
        source = inspect.getsourcefile(type(plugin))
    except (TypeError, OSError):
        source = None
    if source is not None:
        folder = Path(source).parent
        if folder.is_dir():
            paths.extend(sorted(folder.glob("*.toml")))
            paths.extend(sorted(folder.glob("*.py")))
    return paths


def _configuration_digest(plugin) -> str:
    """The hash of the plugin configuration and of the model SDK version."""
    digest = hashlib.sha256()
    digest.update(plugin.model_id.encode())
    digest.update(getattr(plugin, "version", "").encode())
    digest.update(SDK_VERSION.encode())
    digest.update(repr(sorted(getattr(plugin, "variables", ()))).encode())
    for path in _configuration_paths(plugin):
        digest.update(path.name.encode())
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"missing")
    return digest.hexdigest()


def _data_key(
    plugin, data: SeriesStore, registry: Registry, as_of: date, inputs: Path | None = None
) -> str:
    digest = hashlib.sha256()
    digest.update(_configuration_digest(plugin).encode())
    digest.update(as_of.isoformat().encode())
    for variable in registry.variables:
        for row in data.as_of(variable.id, as_of):
            digest.update(
                f"{variable.id}|{row.month}|{row.value}|{row.published}|{row.approximate}".encode()
            )
    keys = plugin.input_keys(as_of)
    written = answers.read(inputs, as_of) if keys and inputs is not None else {}
    for key in keys:
        found = written.get(key)
        digest.update(f"{key}|{'missing' if found is None else answers.hash_of(found)}".encode())
    return digest.hexdigest()


def _file_into(digest, path: Path) -> None:
    """One file into a digest: its name and its bytes, or a marker when it cannot be read."""
    digest.update(Path(path).name.encode())
    try:
        digest.update(Path(path).read_bytes())
    except OSError:
        digest.update(b"missing")


def _series_into(digest, data: SeriesStore, series: str | None, as_of: date) -> None:
    """One series into a digest, as the rows known on ``as_of``; the scheme of ``_data_key``."""
    digest.update((series or "").encode())
    if series is None:
        return
    for row in data.as_of(series, as_of):
        digest.update(f"{row.month}|{row.value}|{row.published}|{row.approximate}".encode())


def _satellite_key(
    macro: str, as_of: date, data: SeriesStore, drivers: Path, installed, driver_registry
) -> str:
    """The cache key of the satellite bundle of a macro version on a date: every input that
    shapes it.

    The key holds the macro version the bundle is estimated on; the bytes of the driver
    catalog; the driver series as known on the calculation date ``as_of``; the configuration
    and version of every installed satellite plugin (``_configuration_digest``); the plugin
    SDK version; and the quantity-to-model assignment. An edited catalog or ``spec.toml``, a
    revised series, a new plugin version or another assignment is another key, so a stale
    bundle is never reused for changed inputs.
    """
    from mst import models as models_pkg

    digest = hashlib.sha256()
    digest.update(macro.encode())
    _file_into(digest, drivers)
    for driver in driver_registry.drivers:
        _series_into(digest, data, driver.series, as_of)
    for model_id in sorted(installed):
        digest.update(model_id.encode())
        digest.update(_configuration_digest(installed[model_id]).encode())
    digest.update(SDK_VERSION.encode())
    quantities = [
        quantity.id
        for quantity in driver_registry.quantities
        if any(driver.quantity == quantity.id for driver in driver_registry.drivers)
    ]
    chosen, _ = models_pkg.assignment(quantities)
    for quantity in sorted(chosen):
        digest.update(f"{quantity}={chosen[quantity]}".encode())
    return digest.hexdigest()


def satellite_name(macro: str, as_of: date, data: SeriesStore, root: Path, drivers: Path) -> str:
    """The name of the satellite bundle of the macro version ``macro`` in the cache under
    ``root``, estimated when it is not there; a refusal is cached as for the macro version."""
    from mst import models as models_pkg
    from mst.drivers.registry import load_registry as load_drivers

    driver_registry = load_drivers(catalog=drivers)
    installed = models_pkg.satellites()
    key = _satellite_key(macro, as_of, data, drivers, installed, driver_registry)
    pointer = root / f"satellites-{key}"
    found = cached(pointer, root / "satellites")
    if found is not None:
        return found
    try:
        found, _ = models_pkg.estimate_satellites(
            macro,
            store=data.root,
            models=root,
            drivers=drivers,
            root=root,
            registry=driver_registry,
        )
    except ValueError as error:
        refuse(pointer, error)
        raise
    pointer.write_text(found + "\n", encoding="utf-8")
    return found


def attach_satellites(window: Window, data: SeriesStore, root: Path, drivers: Path) -> None:
    """The satellite bundle of the window's macro version, estimated or found again
    (``satellite_name``); ``simulate`` draws the driver paths."""
    from mst import models as models_pkg
    from mst.drivers import bundle as satellite_bundle

    found = satellite_name(window.macro.name, window.as_of, data, root, drivers)
    window.bundle = satellite_bundle.load(root, found, models_pkg.satellites())
    window.bundle_name = found
    if window.bundle.as_of > window.as_of:
        raise ValueError(
            f"the satellite bundle is dated {window.bundle.as_of}, "
            f"after the calculation date {window.as_of}"
        )


def facts(
    data: SeriesStore,
    registry: Registry,
    panel,
    months: list[date],
    estimated_on: date,
    known_on: date,
    plan: Episodes,
) -> np.ndarray:
    """The macro variables of the horizon in the model scale, as known on the check date.

    The level of a fact is the value known on ``known_on``, revisions included; the scale of a
    seasonal variable is the calendar profile estimated on ``estimated_on``, the calculation
    date — exactly the profile the model's own panel was deseasonalised with. A revision after
    the episode therefore changes the level but not the scale. An exclusion drops the months of
    a variable from its start on when the history ends before it — an index splice makes the
    realized level incomparable.
    """
    panel_months = list(panel.months)
    grid = panel_months + months
    y = np.full((len(grid), len(panel.ids)), np.nan)
    profile_history = np.full((len(panel_months), len(panel.ids)), np.nan)
    at = {month: t for t, month in enumerate(grid)}
    for i, id_ in enumerate(panel.ids):
        variable = registry.variable(id_)
        for row in data.as_of(id_, known_on):
            if row.month in at:
                y[at[row.month], i] = variable.to_model(row.value)
        if variable.seasonal:
            for row in data.as_of(id_, estimated_on):
                t = at.get(row.month)
                if t is not None and t < len(panel_months):
                    profile_history[t, i] = variable.to_model(row.value)
            profile = _profile(profile_history[:, i], panel_months)
            if profile is not None:
                y[:, i] -= profile[[month.month - 1 for month in grid]]
        for exclusion in plan.exclusions:
            if id_ in exclusion.variables and exclusion.start > panel_months[-1]:
                y[[month >= exclusion.start for month in grid], i] = np.nan
    return y[len(panel_months) :]


def _profile(history: np.ndarray, months: list[date]) -> np.ndarray | None:
    """The calendar profile of a seasonal variable, from its panel months; None without one."""
    if np.isfinite(history).sum() < 13:
        return None
    try:
        return seasonal_profile(history, months)
    except ValueError:
        return None


def file_digest(path: Path | str | None) -> str | None:
    """The sha256 of a file, or None when it cannot be read."""
    if path is None:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def git_commit() -> str | None:
    """The commit of the checkout, read from ``.git`` without running git; best effort."""
    here = Path(__file__).resolve()
    for folder in (here, *here.parents):
        marker = folder / ".git"
        if marker.is_file():
            try:
                found = marker.read_text(encoding="utf-8").strip()
            except OSError:
                return None
            if not found.startswith("gitdir: "):
                return None
            marker = (folder / found[len("gitdir:") :].strip()).resolve()
        if not marker.is_dir():
            continue
        try:
            head = (marker / "HEAD").read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not head.startswith("ref: "):
            return head[:40] or None
        ref = head[len("ref:") :].strip()
        # A linked worktree keeps its HEAD apart and its refs in the common directory.
        common = marker / "commondir"
        if common.is_file():
            marker = (marker / common.read_text(encoding="utf-8").strip()).resolve()
        try:
            return (marker / ref).read_text(encoding="utf-8").strip()[:40] or None
        except OSError:
            pass
        try:
            packed = (marker / "packed-refs").read_text(encoding="utf-8")
        except OSError:
            return None
        for line in packed.splitlines():
            if line.endswith(f" {ref}"):
                return line.split()[0]
        return None
    return None
