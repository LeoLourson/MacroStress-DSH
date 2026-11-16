"""The episode backtest: what fell out, the sanity and the scores of a model on historical events.

For an episode of the plan the calculation date is the day before it starts; every model is
estimated on the data known that day — the macro model as an episode version under
``<models>/backtest/``, cached by the state of the data and the model's configuration — and the
scenario is what actually happened: the realized paths of the episode's observed variables as
constraints month by month, and each decision of the episode as an intervention that sets its
realized level, whose cells are not scored. The same model also runs the base scenario (no
constraints, no interventions), and every cell — episode, month, quantity — is scored
against the fact several times: the conditional forecast, the random walk with drift computed
from the history known at that episode's calculation date, the conditional walk that also reads
the path the scenario fixes, the unconditional (base) run of the model itself, and, for targets,
the naive forecast of a year earlier with the trend. The conditional walk is the benchmark of
the conditional forecast: the plain walk does not know the future path the model is given, so a
win over it says little. The quality of the forecast itself is judged by the rolling check
(``mst.metrics.rolling``) on a grid of dates nobody selected; the named events only add what a
model does when the event is known.

The quantities are every unconstrained macro variable the episode window knows and, with
``--satellites``, every served driver with a realized series. The report reads in the order a
reader should: first what fell out — the episodes not judged and why, the variables a model
left out as too short (``mst.model_sdk.ShortHistory``), the cells without a finite draw — then
the sanity of every quantity (``mst.metrics.sanity``: admission per quantity, the band against
the history range, the direction the scenario moved the forecast, the score of the level
without the quantity), and only then the scores. A score is relative: the mean CRPS of one
forecast over another's on the same cells, pooled in logarithms with equal weights over the
quantities and the months (``mst.metrics.quality``), so a series that rarely moves weighs as
much as any other and no scale is divided by. The pairs are the forecast against the
conditional walk, against the plain walk, the base run against the walk, the forecast against
the base run — what the scenario adds, also per episode and per source of the markup — and,
for drivers, against the naive forecast. The intervals resample whole episodes; neighbouring
windows overlap, so they are too narrow. ``--mc-bootstrap`` adds the Monte-Carlo error of the
paths apart from it. Episodes are split into tuning and validation halves by the plan's marks;
the halves are reported apart and the final tables stay over all episodes. The four-step 2-SE
decision against ``--against`` (``mst.metrics.decision.on_rows``) reads the cells of the
incumbent's ``--details`` file; the saved result keeps no cells. The driver whose simulated
path leaves the range of its own training history carries an unverified tail.
``--portfolio`` adds the Acerbi-Székely section: the portfolio's predictive losses on the
episodes' conditional paths, its realized loss on the facts of the window, ``Z``, the episode
and wild intervals and the 1 % trimming. ``meta["missing"]`` names no metric the report cannot
compute.
"""

import contextlib
import hashlib
import inspect
import io
import json
from dataclasses import dataclass, replace
from datetime import date
from functools import partial
from pathlib import Path

import numpy as np

from mst import inputs as answers
from mst.data.store import SeriesStore
from mst.macro import version as macro_version
from mst.macro.panel import seasonal_profile, shift
from mst.macro.registry import REGISTRY_PATH as MACRO_REGISTRY_PATH
from mst.macro.registry import Registry, load_registry
from mst.macro.version import MacroVersion
from mst.model_sdk.bounds import lower_bounds, observed
from mst.model_sdk.contracts import (
    Intervention,
    LinearTerm,
    MacroEstimate,
    MacroEstimateContext,
    ScenarioPlan,
    ShortHistory,
)
from mst.model_sdk.interventions import apply_default
from mst.model_sdk.version import MODEL_SDK_VERSION
from mst.plugin_sdk.version import SDK_VERSION
from mst.scenario.build import point_constraint

from .baselines import (
    CONSTRAINT,
    CONSTRAINT_THRESHOLD,
    DOMAIN,
    WIDTH,
    band_problems,
    conditional_walk,
    constraint_probabilities,
    macro_domain,
    naive_annual,
    quantity_domain,
    random_walk,
)
from .cells import BUCKET_EDGES, Cell, crps, finite_draws, history_range, pit, quantiles
from .decision import on_rows
from .episodes import Episodes, calculation_date, load_episodes
from .quality import interval, relative
from .sanity import check as sanity_check
from .sanity import worst_shares
from .tables import interval_text, number, print_left_out, print_sanity

INTERVAL_REPETITIONS = 1000
"""The repetitions of the episode bootstrap behind every reported interval."""

MISSING = ()
"""What the report cannot compute; empty when every metric is computed."""


@dataclass
class Episode:
    """One episode prepared for the check: its scenario, its model and its facts."""

    id: str
    as_of: date
    months: list[date]
    """The months of the horizon, t = 1 .. H."""
    horizon: int
    macro: MacroVersion
    scenario: ScenarioPlan
    realized: np.ndarray
    """(H, n) the macro variables in the model scale as the check date knows them; NaN gaps."""
    observations: list[str]
    """The variables the episode asserts as outcomes: they follow the realized path."""
    interventions: list[str]
    """The variables the episode asserts as decisions: they take the realized level."""
    half: str
    """The plan's mark: tuning fits the models, validation checks them."""
    source: str
    """The plan's mark of who formalised the scenario: manual or llm."""
    panel: object = None
    """The panel of the episode's macro version, in the model scale."""
    store: object = None
    """The series store the facts are read from."""
    known_on: date = date(2200, 1, 1)
    """The date the realized paths are known on."""
    bundle: object = None
    """The satellite bundle of the episode's macro version, with ``--satellites``."""
    bundle_name: str | None = None
    """The hash of that satellite version."""
    driver_paths: object = None
    """The driver paths of the episode, in the model scale."""
    macro_paths: object = None
    """The conditional forecast of the episode, kept for the tail section's valuation."""

    @property
    def constrained(self) -> list[str]:
        """The variables whose realized paths fix the scenario, both kinds together."""
        return [*self.observations, *self.interventions]


def run(
    *,
    store: Path,
    models: Path,
    episodes: Path,
    as_of: date,
    macro_id: str | None = None,
    drivers: Path | None = None,
    satellites: bool = False,
    only: list[str] = (),
    details: Path | None = None,
    against: Path | None = None,
    portfolio: str | None = None,
    mc_bootstrap: bool = False,
    target_level: str = "macro",
    rolling: bool = True,
    inputs: Path | None = None,
    save: bool = False,
) -> int:
    """The backtest of every episode the plan names; prints Markdown tables, returns an exit code.

    ``episodes`` is the plan file, ``bench/model-backtest/episodes.toml`` in the repository.
    ``as_of`` is the date the facts are read on — the realized paths as they are known today;
    each episode's models are estimated on the day before it starts, whatever ``as_of`` is.
    With ``satellites`` the served drivers are scored too: their bundle is estimated on each
    episode's macro version, which takes as long as the estimation itself. ``against`` names
    the JSON report of the incumbent version, checked on the same episode plan — the
    two-standard-error decision compares the two on the same cells. ``portfolio`` names a stored
    portfolio for the Acerbi-Székely tail section; ``mc_bootstrap`` adds the Monte-Carlo
    interval of the equal-weight aggregate, separated from the episode-sampling intervals;
    ``target_level`` is the level whose aggregate the decision compares — the macro level by
    default, or a driver level (``driver/<quantity>``) with ``--satellites``. ``rolling`` off
    leaves out the rolling check of the plan, which ``python -m mst.metrics rolling`` also runs.
    ``inputs`` is the folder of the answers a model with declared inputs reads
    (``workspace/elicitation``): every episode reads the answers of its own calculation date.
    ``save`` keeps the report and its tables in ``results/`` beside the plan, as the last result
    of the model; a part of the plan named by ``only`` is not saved.
    """
    from mst import bench
    from mst import models as models_pkg

    if save and only:
        raise ValueError("--save keeps the result of the whole plan: leave out --only")

    if portfolio is not None and not satellites:
        raise ValueError(
            "--portfolio needs --satellites: the tail values the positions on the driver paths"
        )
    plan = load_episodes(episodes)
    plugin = models_pkg.selected(macro_id)
    registry = load_registry()
    data = SeriesStore(store)
    root = Path(models) / "backtest"
    problems: dict[str, str] = {}
    episodes: list[Episode] = []
    for check in plan.episodes:
        if only and check.id not in only:
            continue
        try:
            episode = _episode(plugin, data, registry, plan, check, root, as_of, inputs)
        except ValueError as error:
            problems[check.id] = str(error)
            continue
        if satellites:
            if drivers is None:
                raise ValueError("--satellites needs --drivers, the driver catalog file")
            try:
                _attach_bundle(episode, data, root, Path(drivers), plan)
            except ValueError as error:
                # A driver series that starts after the calculation date (or another estimation
                # problem) voids that episode, not the whole check: it is named and skipped.
                problems[check.id] = f"satellites: {error}"
                continue
        if episode.bundle is not None and episode.bundle.as_of > episode.as_of:
            raise ValueError(
                f"{check.id}: the satellite bundle is dated {episode.bundle.as_of}, "
                f"after the episode calculation date {episode.as_of}"
            )
        episodes.append(episode)
    report = _report(
        episodes,
        plan,
        registry,
        plugin=plugin,
        known_on=as_of,
        store=store,
        models=models,
        root=root,
        dropped=problems,
        against=against,
        portfolio_id=portfolio,
        mc_bootstrap=mc_bootstrap,
        target_level=target_level,
        run_rolling=rolling,
        inputs=inputs,
    )
    tables = io.StringIO()
    with contextlib.redirect_stdout(tables):
        _print(report)
    print(tables.getvalue(), end="")
    if details is not None:
        details.write_text(
            json.dumps(report, ensure_ascii=False, indent=1, default=_jsonable),
            encoding="utf-8",
        )
    if save:
        plain = json.loads(json.dumps(_compact(report), default=_jsonable))
        folder = bench.results(plan.path.parent)
        bench.save_result(folder, plugin.model_id, plain, tables.getvalue())
    return 1 if problems or report["admission_failed"] else 0


def _episode(
    plugin,
    data: SeriesStore,
    registry: Registry,
    plan: Episodes,
    check,
    root: Path,
    known_on: date,
    inputs: Path | None = None,
) -> Episode:
    """One episode: its macro version, its scenario from the facts, and the realized paths."""
    start = check.start
    as_of = calculation_date(start)
    macro = _episode_macro(plugin, data, registry, as_of, root, inputs)
    left_out = macro.meta.get("left_out", {})
    lost = [name for name in check.constrained if name in left_out]
    if lost:
        raise ValueError(
            "; ".join(
                f"the model left out {name}, which the episode fixes: {left_out[name]}"
                for name in lost
            )
        )
    panel, _ = macro_version.panel_of(root, macro.name)
    unknown = [name for name in check.constrained if name not in panel.ids]
    if unknown:
        raise ValueError(f"the macro panel knows no variables {', '.join(unknown)}")
    panel_months = list(panel.months)
    end = shift(start, plan.after_months)
    horizon = (end.year - panel_months[-1].year) * 12 + end.month - panel_months[-1].month
    if horizon < 1:
        raise ValueError(f"the history ends at {panel_months[-1]:%Y-%m}, after the window")
    months = [shift(panel_months[-1], k) for k in range(1, horizon + 1)]
    realized = _realized(data, registry, panel, panel_months, months, as_of, known_on, plan)
    constraints = []
    for id_ in check.observations:
        i = panel.ids.index(id_)
        for k, value in enumerate(realized[:, i], start=1):
            if not np.isnan(value):
                constraints.append(
                    point_constraint(
                        (LinearTerm(variable=id_, month=k, weight=1.0),),
                        float(value),
                        macro,
                    )
                )
    return Episode(
        id=check.id,
        as_of=as_of,
        months=months,
        horizon=horizon,
        macro=macro,
        scenario=ScenarioPlan(
            horizon=horizon,
            constraints=tuple(constraints),
            interventions=_interventions(check.interventions, realized, panel.ids, plan.draws),
        ),
        realized=realized,
        observations=list(check.observations),
        interventions=list(check.interventions),
        half=check.half,
        source=check.source,
        panel=panel,
        store=data,
        known_on=known_on,
    )


def _interventions(
    names: list[str], realized: np.ndarray, ids: list[str], draws: int
) -> tuple[Intervention, ...]:
    """The decisions of the episode as sets of the realized level, one row per draw.

    An episode asserts that the decision was taken, so the realized value is the value the
    model must take in those months; every group of paths carries the same row, because the
    episode fixes the level rather than a law of it. A month without a fact is left out, as
    the observations leave it out.
    """
    found = []
    for name in names:
        column = ids.index(name)
        months, values = [], []
        for month, value in enumerate(realized[:, column], start=1):
            if not np.isnan(value):
                months.append(month)
                values.append(float(value))
        if not months:
            continue
        found.append(
            Intervention(
                target=name,
                months=tuple(months),
                kind="set",
                values=np.tile(np.asarray(values, dtype=float), (draws, 1)),
            )
        )
    return tuple(found)


def _simulated(macro: MacroVersion, scenario: ScenarioPlan, plan: Episodes):
    """The macro paths of a plan, with the interventions of a default-form model applied.

    A recursive model executes its interventions itself; a default-form one returns the paths
    before them, as the platform's ``build`` expects, so the same substitution is made here.
    """
    simulation = macro.plugin.simulate(macro.model, scenario, draws=plan.draws, seed=plan.seed)
    if macro.plugin.structural_form(macro.model) is None:
        simulation = replace(
            simulation,
            paths=apply_default(
                simulation.paths, simulation.ids, scenario.interventions, inner=scenario.inner
            ),
        )
    return simulation


def _realized(
    data: SeriesStore,
    registry: Registry,
    panel,
    panel_months: list[date],
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


def _episode_macro(
    plugin,
    data: SeriesStore,
    registry: Registry,
    as_of: date,
    root: Path,
    inputs: Path | None = None,
):
    """The macro version of an episode, cached under ``root`` by the state of the data.

    The estimate itself is deterministic but expensive; the cache key covers the model, its
    version, its configuration files, every value the registry names as known on the date and
    the hash of every answer the model declares for the date, so a rerun with the same data
    loads the version instead of estimating again, and an edited spec, sign table, model
    package or answer never reuses a stale one.
    """
    key = _data_key(plugin, data, registry, as_of, inputs)
    pointer = root / key
    installed = {plugin.model_id: plugin}
    if pointer.exists():
        name = pointer.read_text(encoding="utf-8").strip()
        if (root / "macro" / name).is_dir():
            return macro_version.load(root, name, installed)
    estimate, left_out = _estimate_known(plugin, data, registry, as_of, inputs)
    meta = {**estimate.meta, "left_out": left_out}
    name = macro_version.save(plugin.model_id, meta, estimate.arrays, root)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(name + "\n", encoding="utf-8")
    return macro_version.load(root, name, installed)


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
    episode: Episode, data: SeriesStore, drivers: Path, installed, driver_registry
) -> str:
    """The cache key of the satellite bundle of one episode: every input that shapes it.

    The key holds the macro version the bundle is estimated on; the bytes of the driver
    catalog; the driver series as known on the episode's calculation date; the configuration
    and version of every installed satellite plugin (``_configuration_digest``); the plugin
    SDK version; and the quantity-to-model assignment. An edited catalog or ``spec.toml``, a
    revised series, a new plugin version or another assignment is another key, so a stale
    bundle is never reused for changed inputs.
    """
    from mst import models as models_pkg

    digest = hashlib.sha256()
    digest.update(episode.macro.name.encode())
    _file_into(digest, drivers)
    for driver in driver_registry.drivers:
        _series_into(digest, data, driver.series, episode.as_of)
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


def _attach_bundle(
    episode: Episode, data: SeriesStore, root: Path, drivers: Path, plan: Episodes
) -> None:
    """Estimate the satellite bundle of the episode's macro version and simulate its drivers.

    The bundle is cached beside the episode's macro version, under the same backtest root, by
    the digest of every input that shapes it (``_satellite_key``).
    """
    from mst import models as models_pkg
    from mst.drivers import bundle as satellite_bundle
    from mst.drivers.registry import load_registry as load_drivers

    driver_registry = load_drivers(catalog=drivers)
    name = episode.macro.name
    installed = models_pkg.satellites()
    pointer = root / (
        f"satellites-{_satellite_key(episode, data, drivers, installed, driver_registry)}"
    )
    if pointer.exists():
        found = pointer.read_text(encoding="utf-8").strip()
        if (root / "satellites" / found).is_dir():
            episode.bundle = satellite_bundle.load(root, found, installed)
            episode.bundle_name = found
        else:
            pointer.unlink()
    if episode.bundle is None:
        found, _ = models_pkg.estimate_satellites(
            name,
            store=data.root if hasattr(data, "root") else data,
            models=root,
            drivers=drivers,
            root=root,
            registry=driver_registry,
        )
        pointer.write_text(found + "\n", encoding="utf-8")
        episode.bundle = satellite_bundle.load(root, found, installed)
        episode.bundle_name = found
    simulation = _simulated(episode.macro, episode.scenario, plan)
    episode.driver_paths = episode.bundle.simulate(
        observed(simulation, load_registry()), episode.bundle.served(), (), seed=plan.seed
    )


def _report(
    episodes: list[Episode],
    plan: Episodes,
    registry: Registry,
    *,
    plugin,
    known_on: date,
    store: Path,
    models: Path,
    root: Path,
    dropped: dict[str, str],
    against: Path | None = None,
    portfolio_id: str | None = None,
    mc_bootstrap: bool = False,
    target_level: str = "macro",
    run_rolling: bool = True,
    inputs: Path | None = None,
) -> dict:
    """What fell out and the sanity of the quantities, then the scores, gates and metadata.

    ``dropped`` names the episodes of the plan that could not be judged, with the reason.
    """
    cells: list[Cell] = []
    levels: dict[str, str] = {}
    unverified: set[str] = set()
    constraint_shares: dict[str, dict[int, float | None]] = {}

    for episode in episodes:
        # One run of every episode: the observations as point constraints and the decisions
        # as interventions; the uniformity of u, the calibration and the CRPS read it.
        candidate = observed(_simulated(episode.macro, episode.scenario, plan), registry)
        episode.macro_paths = candidate
        shadow = episode.macro.plugin.simulate(
            episode.macro.model,
            ScenarioPlan(horizon=episode.horizon),
            draws=plan.draws,
            seed=plan.seed,
        )
        base = observed(shadow, registry)
        # The conditions stand on the shadow, and their probability under the base reads it.
        constraint_shares[episode.id] = constraint_probabilities(
            shadow.paths, shadow.ids, episode.scenario.constraints
        )
        episode_cells: list[Cell] = []
        _macro_cells(episode, candidate, base, episode_cells, levels, registry, plan)
        if episode.bundle is not None and episode.driver_paths is not None:
            drivers = episode.bundle.simulate(
                candidate, episode.bundle.served(), (), seed=plan.seed
            )
            _driver_cells(episode, drivers, episode_cells, levels, unverified, plan)
        cells.extend(episode_cells)

    rows = _cell_rows(cells, levels)
    if rows and target_level not in {row["level"] for row in rows}:
        known = ", ".join(sorted({row["level"] for row in rows}))
        raise ValueError(f"target_level {target_level!r} has no cells; levels present: {known}")
    sanity = sanity_check(rows, benchmark="walk")
    scores = _scores(rows, plan)
    halves = {episode.id: episode.half for episode in episodes}
    sources = {episode.id: episode.source for episode in episodes}
    gate = _constraint_gate(constraint_shares)
    rolling = {}
    if plan.rolling is not None and run_rolling:
        from .rolling import rolling_report

        rolling = rolling_report(
            plugin=plugin,
            store=SeriesStore(store),
            registry=registry,
            root=root,
            plan=plan,
            known_on=known_on,
            macro_of=partial(_episode_macro, inputs=inputs),
        )
    report = {
        "meta": _meta(
            episodes,
            plan,
            plugin=plugin,
            known_on=known_on,
            store=store,
            models=models,
            root=root,
            target_level=target_level,
        ),
        "dropped": {
            "episodes": dict(dropped),
            "left_out": {
                episode.id: episode.macro.meta.get("left_out", {})
                for episode in episodes
                if episode.macro.meta.get("left_out")
            },
            "non_finite": _non_finite(cells),
        },
        "sanity": sanity,
        "constraint_gate": gate,
        "admission_failed": sanity["admission"]["failed"] or gate["failed"],
        "episodes": [
            {
                "id": episode.id,
                "as_of": episode.as_of.isoformat(),
                "known_on": episode.known_on.isoformat(),
                "horizon": episode.horizon,
                "half": episode.half,
                "source": episode.source,
                "observations": episode.observations,
                "interventions": episode.interventions,
                "constrained": episode.constrained,
                "macro_model": episode.macro.plugin.model_id,
                "macro_version": episode.macro.name,
                "satellite_version": episode.bundle_name,
                "left_out": episode.macro.meta.get("left_out", {}),
                "constraint_probabilities": constraint_shares[episode.id],
            }
            for episode in episodes
        ],
        "target_level": target_level,
        "scores": scores,
        "utility": _utility(rows, sources, target_level),
        "halves": _halves_block(episodes, rows, constraint_shares, plan, target_level),
        "quantities": _quantities(cells, rows, levels, unverified, scores),
        "rolling": rolling,
        "rows": rows,
    }
    if mc_bootstrap:
        report["monte_carlo"] = _mc_interval(cells, rows, plan, target_level)
    if against is not None:
        report["decision"] = _decision_block(
            rows, against, sanity=sanity, halves=halves, plan=plan, target_level=target_level
        )
    if portfolio_id is not None:
        report["tail"] = _tail_section(episodes, plan, portfolio_id, store=store, known_on=known_on)
    return report


PAIRS = (
    ("model", "conditional_walk"),
    ("model", "walk"),
    ("base", "walk"),
    ("model", "base"),
    ("model", "naive"),
)
"""The relative scores of the episodes: the forecast against the walk that knows the scenario's
path, against the plain walk, the base run against the walk, the forecast against the base run
(what the scenario adds) and, for drivers, against the naive forecast of a year earlier."""
HALF_PAIRS = PAIRS[:4]
MC_STREAM = 13
"""The second word of the Monte-Carlo seed, apart from the episode bootstrap's stream."""


def _cell_rows(cells: list[Cell], levels: dict[str, str]) -> list[dict]:
    """The scored cells as flat rows: the CRPS of every forecast and the band of the model.

    A constrained cell says nothing about the forecast and is left out; so is a cell whose draws
    are all non-finite, which ``_non_finite`` counts. The band columns feed the sanity of the
    quantities: the width against the history range, the admission rules and, beside the base
    run, the move of the median the scenario made and the miss of the base run's median.
    """
    rows = []
    for cell in cells:
        if cell.constrained or not np.isfinite(cell.fact):
            continue
        draws = finite_draws(cell.draws)
        if not len(draws):
            continue
        band = quantiles(draws)
        broken = band_problems(
            draws, cell.fact, domain=cell.domain, history_range=cell.history_range
        )
        row = {
            "episode": cell.episode,
            "month": cell.month,
            "quantity": cell.quantity,
            "level": levels.get(cell.quantity, ""),
            "model": crps(draws, cell.fact),
            "walk": None,
            "conditional_walk": None,
            "base": None,
            "naive": None,
            "pit": pit(draws, cell.fact),
            "inside": bool(band[0] <= cell.fact <= band[2]),
            "width": float(band[2] - band[0]),
            "history_range": cell.history_range,
            DOMAIN: DOMAIN in broken,
            WIDTH: WIDTH in broken,
            "shift": None,
            "miss": None,
            "base_width": None,
        }
        for name, column in (
            ("walk", cell.baseline),
            ("conditional_walk", cell.conditional),
            ("base", cell.base),
            ("naive", cell.naive),
        ):
            known = np.array([]) if column is None else finite_draws(column)
            if len(known):
                row[name] = crps(known, cell.fact)
        base = np.array([]) if cell.base is None else finite_draws(cell.base)
        if len(base):
            base_band = quantiles(base)
            row["shift"] = float(band[1] - base_band[1])
            row["miss"] = float(cell.fact - base_band[1])
            row["base_width"] = float(base_band[2] - base_band[0])
        rows.append(row)
    return rows


def _non_finite(cells: list[Cell]) -> dict[str, int]:
    """The unconstrained cells with a fact whose draws are all non-finite, per quantity."""
    found: dict[str, int] = {}
    for cell in cells:
        if cell.constrained or not np.isfinite(cell.fact) or len(finite_draws(cell.draws)):
            continue
        found[cell.quantity] = found.get(cell.quantity, 0) + 1
    return found


def _scores(rows: list[dict], plan: Episodes) -> dict:
    """Per level, every pair of ``PAIRS`` the rows carry, with its interval over the episodes.

    The windows of neighbouring episodes overlap, so the intervals are too narrow: they are a
    lower bound of the error, and the rolling grid carries the verdict on the forecast.
    """
    found = {}
    for level in sorted({row["level"] for row in rows}):
        own = [row for row in rows if row["level"] == level]
        entries = {}
        for first, second in PAIRS:
            score = relative(own, first, second)
            if score["score"] is None:
                continue
            entries[f"{first}/{second}"] = {
                **score,
                "interval": interval(
                    own,
                    first,
                    second,
                    unit="episode",
                    repetitions=INTERVAL_REPETITIONS,
                    seed=plan.seed,
                ),
            }
        found[level] = entries
    return found


def _utility(rows: list[dict], sources: dict[str, str], target_level: str) -> dict:
    """What the scenario adds: the forecast against the base run, per episode and per source.

    An episode counts as helped when its ratio is below one. The sources are reported apart:
    a language model may have known the outcome of a historical event from its training.
    """
    chosen = [row for row in rows if row["level"] == target_level and row["base"] is not None]
    by_episode: dict[str, list[dict]] = {}
    for row in chosen:
        by_episode.setdefault(row["episode"], []).append(row)
    ratios = {
        episode: relative(own, "model", "base")["score"] for episode, own in by_episode.items()
    }
    found = {}
    for source in sorted({sources[episode] for episode in ratios}):
        named = sorted(episode for episode in ratios if sources[episode] == source)
        judged = [ratios[episode] for episode in named if ratios[episode] is not None]
        found[source] = {
            "episodes": len(named),
            "judged": len(judged),
            "share_helped": float(np.mean([ratio < 1 for ratio in judged])) if judged else None,
            "score": relative(
                [row for row in chosen if sources[row["episode"]] == source], "model", "base"
            )["score"],
        }
    return {"target": target_level, "by_episode": ratios, "by_source": found}


def _halves_block(
    episodes: list[Episode],
    rows: list[dict],
    constraint_shares: dict[str, dict[int, float | None]],
    plan: Episodes,
    target_level: str,
) -> dict:
    """The scores of the target level, the admission and the constraint gate of each half.

    The models are fitted on the tuning half and checked on the validation one, while the final
    table is built over all episodes; the block keeps the three side by side, so a reader sees
    whether the halves agree.
    """
    chosen = {"all": [episode.id for episode in episodes]}
    for half in ("tuning", "validation"):
        named = [episode.id for episode in episodes if episode.half == half]
        if named:
            chosen[half] = named
    found: dict[str, dict] = {}
    for name, ids in chosen.items():
        named = set(ids)
        own = [row for row in rows if row["episode"] in named]
        level = [row for row in own if row["level"] == target_level]
        found[name] = {
            "episodes": sorted(named),
            "cells": len(own),
            "scores": {
                f"{first}/{second}": interval(
                    level,
                    first,
                    second,
                    unit="episode",
                    repetitions=INTERVAL_REPETITIONS,
                    seed=plan.seed,
                )
                for first, second in HALF_PAIRS
            },
            "admission_failed": sanity_check(own)["admission"]["failed"] if own else None,
            "constraint_gate": _constraint_gate(
                {
                    episode: shares
                    for episode, shares in constraint_shares.items()
                    if episode in named
                }
            ),
        }
    return found


def _quantities(
    cells: list[Cell],
    rows: list[dict],
    levels: dict[str, str],
    unverified: set[str],
    scores: dict,
) -> dict:
    """Per quantity: the cells, the relative scores, the calibration and the sharpness.

    A constrained cell is counted and never scored. Calibration reads the transform of the
    facts, ``u = F(y)``; the shares of its four buckets are a target for a continuous forecast
    and, on selected episodes, for reference only. Sharpness is the median width of the band,
    meaningful only beside the calibration.
    """
    found = {}
    for quantity in sorted({cell.quantity for cell in cells}):
        own_cells = [cell for cell in cells if cell.quantity == quantity]
        own = [row for row in rows if row["quantity"] == quantity]
        level = levels.get(quantity, "")
        entry = {
            "level": level,
            "cells": len(own_cells),
            "constrained": sum(1 for cell in own_cells if cell.constrained),
            "scored": len(own),
            "unverified_tail": quantity in unverified,
            "ratios": {
                name: score["by_quantity"].get(quantity)
                for name, score in scores.get(level, {}).items()
            },
            "calibration": None,
            "coverage": None,
            "sharpness": None,
        }
        if own:
            transforms = np.array([row["pit"] for row in own])
            buckets = np.bincount(
                np.searchsorted(BUCKET_EDGES, transforms, side="right"), minlength=4
            )
            entry["calibration"] = (buckets / len(own)).tolist()
            entry["coverage"] = float(np.mean([row["inside"] for row in own]))
            entry["sharpness"] = float(np.median([row["width"] for row in own]))
        found[quantity] = entry
    return found


def _mc_interval(cells: list[Cell], rows: list[dict], plan: Episodes, target_level: str) -> dict:
    """The Monte-Carlo interval of the forecast against the walk at the target level.

    The cells of one calculation share the draws, so a repetition resamples one index vector
    per number of draws and applies it to every cell; the CRPS of the forecast and the relative
    score are recomputed. This measures the error of the finite number of paths; the episode
    interval of ``_scores`` measures the error of the sample of episodes.
    """
    keyed = {
        (row["episode"], row["month"], row["quantity"]): row
        for row in rows
        if row["level"] == target_level and row["walk"] is not None
    }
    pairs = [
        (cell, keyed[key])
        for cell in cells
        if (key := (cell.episode, cell.month, cell.quantity)) in keyed
    ]
    point = relative([row for _, row in pairs], "model", "walk")["score"]
    if point is None:
        return {"point": None, "interval": None}
    rng = np.random.default_rng([plan.seed, MC_STREAM])
    sizes = sorted({len(cell.draws) for cell, _ in pairs})
    logs = []
    for _ in range(INTERVAL_REPETITIONS):
        index = {size: rng.integers(0, size, size) for size in sizes}
        resampled = [
            {**row, "model": crps(finite_draws(cell.draws[index[len(cell.draws)]]), cell.fact)}
            for cell, row in pairs
        ]
        score = relative(resampled, "model", "walk")["score"]
        if score is not None:
            logs.append(np.log(score))
    low, high = np.exp(np.percentile(logs, [2.5, 97.5]))
    return {
        "point": point,
        "interval": (float(low), float(high)),
        "note": (
            "bootstrap over the draws (Monte-Carlo error); the episode interval measures the "
            "error of the sample of episodes"
        ),
    }


def _decision_block(
    rows: list[dict],
    against: Path,
    *,
    sanity: dict,
    halves: dict[str, str],
    plan: Episodes,
    target_level: str = "macro",
) -> dict:
    """The two-standard-error verdict against the incumbent's details file.

    The incumbent must come from the same episode plan and carry its cells (the ``--details``
    file of its check). The comparison reads the target level on the validation half; the
    benchmark is the conditional walk where the level has one, since it knows what the forecast
    knows, and the plain walk otherwise. The admission shares are the worst quantity's.
    """
    if not against.exists():
        raise ValueError(f"--against {against}: the file does not exist")
    incumbent = json.loads(against.read_text(encoding="utf-8"))
    if incumbent.get("meta", {}).get("episodes_toml") != _file_digest(plan.path):
        raise ValueError(
            "--against was produced from a different episode plan; "
            "the two versions must be compared on the same cells"
        )
    if "rows" not in incumbent:
        raise ValueError(f"--against {against}: no cells; pass the --details file of that version")
    validation = {episode for episode, half in halves.items() if half == "validation"}
    present = {row["episode"] for row in rows}
    chosen = (validation & present) or present

    def pick(found):
        return [row for row in found if row["level"] == target_level and row["episode"] in chosen]

    mine, theirs = pick(rows), pick(incumbent["rows"])
    if not mine:
        raise ValueError(f"--against {against}: level {target_level!r} has no scored cells")
    benchmark = (
        "conditional_walk" if any(row["conditional_walk"] is not None for row in mine) else "walk"
    )
    verdict = on_rows(
        mine,
        theirs,
        unit="episode",
        benchmark=benchmark,
        block=1,
        seed=plan.seed,
        admission=worst_shares(sanity["admission"]),
        incumbent_admission=worst_shares(incumbent["sanity"]["admission"]),
    )
    verdict.update(
        target_level=target_level,
        half="validation" if validation & chosen else "all episodes",
    )
    return verdict


def _compact(report: dict) -> dict:
    """The report without its cells, small enough to keep in the repository.

    The cells stay in the ``--details`` file: a decision against this version reads them there.
    """
    plain = {key: value for key, value in report.items() if key != "rows"}
    rolling = plain.get("rolling")
    if rolling:
        rolling = dict(rolling)
        if "quality" in rolling:
            rolling["quality"] = {
                key: value for key, value in rolling["quality"].items() if key != "rows"
            }
        if "price" in rolling:
            rolling["price"] = {
                key: value for key, value in rolling["price"].items() if key != "rows"
            }
        plain["rolling"] = rolling
    return plain


def _macro_cells(
    episode: Episode,
    draws,
    base,
    cells: list[Cell],
    levels: dict[str, str],
    registry: Registry,
    plan: Episodes,
) -> None:
    """The cells of the episode's macro variables; the random walk and the base run alongside."""
    panel = episode.panel
    column_of = {id_: i for i, id_ in enumerate(draws.ids)}
    pinned = _pinned(episode.scenario.constraints) | _installed(episode.scenario.interventions)
    given = [panel.ids.index(id_) for id_ in episode.constrained]
    for j, id_ in enumerate(panel.ids):
        i = column_of.get(id_)
        if i is None:
            continue
        levels[id_] = "macro"
        history = panel.y[:, j]
        domain = macro_domain(registry.variable(id_))
        spread = history_range(history)
        floors = lower_bounds(registry, [id_])
        lower = None if floors is None else float(floors[0])
        walk = random_walk(history, episode.horizon, draws=plan.draws, seed=plan.seed, lower=lower)
        knowing = (
            None
            if j in given
            else conditional_walk(
                history,
                panel.y[:, given],
                episode.realized[:, given],
                draws=plan.draws,
                seed=plan.seed,
                lower=lower,
            )
        )
        for k in range(len(episode.months)):
            fact = episode.realized[k, j]
            if np.isnan(fact):
                continue
            cells.append(
                Cell(
                    episode=episode.id,
                    month=k + 1,
                    quantity=id_,
                    fact=float(fact),
                    draws=draws.paths[:, k + 1, i],
                    baseline=walk[:, k],
                    base=base.paths[:, k + 1, i],
                    conditional=None if knowing is None else knowing[:, k],
                    constrained=(id_, k + 1) in pinned,
                    history_range=spread,
                    domain=domain,
                )
            )


def _pinned(constraints) -> set[tuple[str, int]]:
    """The ``(variable, month)`` cells a single-term constraint fixes.

    A constraint of one term pins its cell when it comes from a point, or when its band is
    narrower than the tolerance a point of the same terms would expand to: a narrow interval
    of that scale carries no scoring information either. The mark decides for a point, since
    ``value ± tau`` need not round back to a band of ``2 tau``.
    """
    pinned = set()
    for constraint in constraints:
        if len(constraint.terms) != 1:
            continue
        term = constraint.terms[0]
        if constraint.point or constraint.high - constraint.low <= 2 * constraint.tolerance:
            pinned.add((term.variable, term.month))
    return pinned


def _installed(interventions) -> set[tuple[str, int]]:
    """The ``(variable, month)`` cells an intervention sets: the episode asserts the decision,
    so they are reported and never scored, as a point observation's cell is."""
    return {(item.target, month) for item in interventions for month in item.months}


def _driver_cells(
    episode: Episode,
    paths,
    cells: list[Cell],
    levels: dict[str, str],
    unverified: set[str],
    plan: Episodes,
) -> None:
    """The cells of the episode's served drivers, with the random walk and the naive forecast."""
    bundle = episode.bundle
    driver_registry = bundle.registry
    series_of = {driver.id: driver.series for driver in driver_registry.drivers}
    own_of: dict[str, np.ndarray] = {}
    quantity_of: dict[str, str] = {}
    for driver in paths.ids:
        quantity, node = bundle.node(driver)
        quantity_of[driver] = quantity
        d = bundle.arrays.get(f"{quantity}.d")
        own_of[driver] = d[:, node] if d is not None else np.array([])
    for k, driver in enumerate(paths.ids):
        quantity = quantity_of[driver]
        declared = driver_registry.quantity(quantity)
        levels[driver] = (
            f"driver/{quantity}" if declared.level <= 1 else f"driver{declared.level}/{quantity}"
        )
        own = own_of[driver]
        domain = quantity_domain(declared)
        spread = history_range(own)
        walk = random_walk(own, episode.horizon, draws=plan.draws, seed=plan.seed)
        naive = naive_annual(own, episode.horizon, draws=plan.draws, seed=plan.seed)
        for t, month in enumerate(episode.months):
            fact = _driver_fact(episode, series_of.get(driver), month, declared)
            if fact is None:
                continue
            cells.append(
                Cell(
                    episode=episode.id,
                    month=t + 1,
                    quantity=driver,
                    fact=fact,
                    draws=paths.paths[:, t + 1, k],
                    baseline=walk[:, t] if walk.size else None,
                    naive=naive[:, t] if naive is not None else None,
                    history_range=spread,
                    domain=domain,
                )
            )
        flags = paths.flags.get(driver, frozenset())
        if "outside_training_range" in flags:
            unverified.add(driver)


def _driver_fact(episode: Episode, series, month: date, declared):
    """The realized value of a driver in the model scale, or None when it is not known."""
    from mst.drivers.registry import to_model

    if series is None:
        return None
    rows = [row for row in episode.store.as_of(series, episode.known_on) if row.month == month]
    if not rows:
        return None
    try:
        return float(to_model(declared.transform, rows[-1].value))
    except ValueError:
        return None


def _constraint_gate(shares: dict[str, dict[int, float | None]]) -> dict:
    """``pi_c > 1 %`` for every informative constraint, per episode.

    A constraint that fixes a point — the facts of an episode, expanded to ``value ± tau`` —
    would have its probability measure the tolerance rather than the scenario, so it is not
    applicable and is left out of the gate. The report says how many constraints of each kind
    the gate saw.
    """
    applicable = 0
    not_applicable = 0
    failed = []
    for episode, found in shares.items():
        for position, share in found.items():
            if share is None:
                not_applicable += 1
                continue
            applicable += 1
            if share <= CONSTRAINT_THRESHOLD:
                failed.append(f"{episode}#{position}")
    return {
        "rule": CONSTRAINT,
        "threshold": CONSTRAINT_THRESHOLD,
        "applicable": applicable,
        "not_applicable": not_applicable,
        "failed_constraints": failed,
        "failed": bool(failed),
        "note": (
            "a fact-reconstructed observation fixes its value as a point, so pi_c is not "
            "applicable there and is excluded from the gate; an intervention is a decision, "
            "not an observation, and is not in the gate at all; a range constraint keeps pi_c"
        ),
    }


def _tail_section(
    episodes: list[Episode],
    plan: Episodes,
    portfolio_id: str,
    *,
    store: Path,
    known_on: date,
) -> dict:
    """The Acerbi-Székely section: the portfolio tail backtest over the episodes.

    The conditional forecast of every episode valued through the engine gives the predictive
    loss distribution; the realized loss comes from the portfolio value on the facts at both
    ends of the window. The ``Z`` of the Acerbi-Székely tail is computed over the episodes that
    have both; an episode without the facts of its window, with a position the engine could not
    calculate, or without a satellite bundle is dropped with a named counter. The verdict of
    admission does not read ``Z``: the section is reported, not gated.
    """
    from mst.portfolio import tool as portfolio_tool

    from .es import conditional_paths, realized_paths, tail_section

    # The portfolios live beside the store, in the workspace the run was pointed at.
    portfolios_dir = Path(store).parent / "portfolios"
    try:
        portfolio = portfolio_tool.load(portfolios_dir, portfolio_id)
    except FileNotFoundError:
        raise ValueError(
            f"--portfolio {portfolio_id}: no portfolio {portfolio_id!r} in {portfolios_dir}"
        ) from None
    rows = []
    for episode in episodes:
        row = {
            "id": episode.id,
            "as_of": episode.as_of,
            "known_on": known_on,
            "horizon": episode.horizon,
            "months": episode.months,
            "paths": None,
            "realized": None,
            "drivers": None,
            "served": [],
            "realized_missing": [],
        }
        if episode.bundle is None or episode.driver_paths is None or episode.macro_paths is None:
            row["realized_missing"] = ["no satellite bundle on the episode"]
            rows.append(row)
            continue
        row["paths"] = conditional_paths(
            episode.macro_paths, episode.driver_paths, episode.bundle, seed=plan.seed
        )
        realized, missing = realized_paths(
            episode.store,
            as_of=episode.as_of,
            known_on=known_on,
            months=episode.months,
            panel=episode.panel,
            realized=episode.realized,
            bundle=episode.bundle,
        )
        row.update(
            {
                "realized": realized,
                "realized_missing": missing,
                "drivers": episode.bundle.registry,
                "served": episode.bundle.served(),
            }
        )
        rows.append(row)
    return tail_section(rows, portfolio, seed=plan.seed)


def _meta(
    episodes: list[Episode],
    plan: Episodes,
    *,
    plugin,
    known_on: date,
    store: Path,
    models: Path,
    root: Path,
    target_level: str = "macro",
) -> dict:
    """What a rerun needs to reproduce the report: versions, data vintage, seed and commit."""
    macro_models = {(getattr(plugin, "model_id", ""), getattr(plugin, "version", ""))}
    satellites = set()
    for episode in episodes:
        macro_models.add((episode.macro.plugin.model_id, episode.macro.plugin.version))
        if episode.bundle is not None:
            for model_id, model in episode.bundle.plugins.items():
                satellites.add((model_id, model.version))
    return {
        "generated": date.today().isoformat(),
        "known_on": known_on.isoformat(),
        "draws": plan.draws,
        "seed": plan.seed,
        "after_months": plan.after_months,
        "episodes_toml": _file_digest(plan.path),
        "macro_registry": _file_digest(MACRO_REGISTRY_PATH),
        "store": str(store),
        "models": str(models),
        "backtest_root": str(root),
        "git_commit": _git_commit(),
        "plugin_sdk": SDK_VERSION,
        "model_sdk": MODEL_SDK_VERSION,
        "target_level": target_level,
        "intervals": {
            "sampling": "episodes resampled with replacement, the relative score recomputed",
            "monte_carlo": "draws resampled; only with --mc-bootstrap",
        },
        "assumptions": [
            "scores are relative: the mean CRPS of a forecast over the benchmark's on the same "
            "cells, pooled in logarithms with equal weights over quantities and months",
            "admission: every quantity keeps each band rule under 5 % of its cells",
            "direction: only a move of the median above a tenth of the base band is read",
            "the target level of the decision and of the utility is --target-level",
            "utility: an episode is helped when its forecast beats the base run, model/base < 1; "
            "the split by source is manual against llm",
            "realized loss: the portfolio valued on the facts at t = 0 and t = H; an episode "
            "without the fact at t = H is dropped by name",
            "dist(s) = E_P[-log P(S | R)] on the constraint law (docs/concept/formalization.md)",
        ],
        "macro_models": [
            {"model_id": model_id, "version": version} for model_id, version in sorted(macro_models)
        ],
        "satellite_models": [
            {"model_id": model_id, "version": version} for model_id, version in sorted(satellites)
        ],
        "macro_versions": sorted({episode.macro.name for episode in episodes}),
        "satellite_versions": sorted(
            {episode.bundle_name for episode in episodes if episode.bundle_name}
        ),
        "missing": list(MISSING),
    }


def _file_digest(path: Path | str | None) -> str | None:
    """The sha256 of a file, or None when it cannot be read."""
    if path is None:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _git_commit() -> str | None:
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


def _print(report: dict) -> None:
    """The report as Markdown: what fell out and the sanity first, the scores after them."""
    _print_dropped(report)
    print("\n## Sanity of the quantities\n")
    print(
        "Read before any score: a quantity over the admission share, a band wider than the "
        "history or a scenario that moves the forecast away from the fact is named here."
    )
    print_sanity(report["sanity"], benchmark="walk")
    gate = report["constraint_gate"]
    print(
        f"\n`{gate['rule']}`: {gate['applicable']} applicable constraint(s), "
        f"{gate['not_applicable']} not applicable (exact point constraints), "
        f"threshold {gate['threshold']:.3f} — {'failed' if gate['failed'] else 'pass'}."
    )
    _print_episodes(report)
    _print_scores(report)
    _print_utility(report)
    _print_halves(report)
    _print_quantities(report)
    if "decision" in report:
        _print_decision(report)
    if report.get("rolling"):
        from .rolling import print_rolling

        print_rolling(report["rolling"])
    if "tail" in report:
        _print_tail(report)
    meta = report["meta"]
    print("\n## Reproducibility\n")
    print(
        f"- data known on {meta['known_on']}; draws {meta['draws']}; seed {meta['seed']}; "
        f"git commit {meta['git_commit'] or 'unknown'}"
    )
    print(f"- episodes plan {meta['episodes_toml']}; macro registry {meta['macro_registry']}")
    for model in meta["macro_models"]:
        print(f"- macro {model['model_id']} {model['version']} at {meta['backtest_root']}")
    for model in meta["satellite_models"]:
        print(f"- satellites {model['model_id']} {model['version']}")
    print("\nMetrics the report cannot compute:\n")
    if meta["missing"]:
        for item in meta["missing"]:
            print(f"- {item}")
    else:
        print("- none: the report computes every metric")


def _print_dropped(report: dict) -> None:
    """The episodes the plan names that were not judged, and the cells that were left out."""
    dropped = report["dropped"]
    print("## What fell out\n")
    judged = len(report["episodes"])
    print(f"{judged} of {judged + len(dropped['episodes'])} episodes judged.")
    if dropped["episodes"]:
        print("\nEpisodes not judged:\n")
        for episode, reason in dropped["episodes"].items():
            print(f"- {episode}: {reason}")
    print_left_out(dropped["left_out"], "episodes")
    if dropped["non_finite"]:
        print("\nCells whose draws are all non-finite, left out:\n")
        for quantity, count in dropped["non_finite"].items():
            print(f"- {quantity}: {count}")


def _print_episodes(report: dict) -> None:
    print("\n## Episodes\n")
    print(
        "| episode | as of | horizon | observations | interventions | min constraint probability |"
    )
    print("|---|---|---|---|---|---|")
    for row in report["episodes"]:
        shares = row["constraint_probabilities"]
        known = [share for share in shares.values() if share is not None]
        exact = sum(1 for share in shares.values() if share is None)
        worst = f"{min(known):.2f}" if known else "n/a"
        if exact:
            worst += f" ({exact} n/a)"
        observations = ", ".join(row["observations"]) or "-"
        interventions = ", ".join(row["interventions"]) or "-"
        print(
            f"| {row['id']} | {row['as_of']} | {row['horizon']} "
            f"| {observations} | {interventions} | {worst} |"
        )


def _print_scores(report: dict) -> None:
    """The relative CRPS of every level: below one the first forecast wins."""
    print("\n## Relative CRPS (below one the first forecast wins)\n")
    print(
        "Mean CRPS of the first forecast over the second on the same cells, pooled in "
        "logarithms with equal weights over quantities and horizon months. The episode "
        "interval is too narrow: neighbouring windows overlap."
    )
    print("\n| level | comparison | score | 95 % episode interval | cells |")
    print("|---|---|---:|---|---:|")
    for level, entries in report["scores"].items():
        for name, entry in entries.items():
            found = entry["interval"]
            print(
                f"| {level} | {name} | {number(entry['score'])} "
                f"| {interval_text((found['low'], found['high']))} | {entry['cells']} |"
            )
    monte_carlo = report.get("monte_carlo")
    if monte_carlo is not None:
        print(
            f"\nMonte-Carlo interval of model/walk at `{report['target_level']}` (the draws are "
            f"resampled, not the episodes): {interval_text(monte_carlo.get('interval'))}."
        )


def _print_utility(report: dict) -> None:
    """What the scenario adds: the forecast against the base run, by source."""
    utility = report["utility"]
    print(f"\n## Scenario usefulness (model/base at `{utility['target']}`)\n")
    print("| source | episodes | judged | share helped | model / base |")
    print("|---|---:|---:|---:|---:|")
    if not utility["by_source"]:
        print("| - | 0 | 0 | - | - |")
        return
    for source, entry in utility["by_source"].items():
        print(
            f"| {source} | {entry['episodes']} | {entry['judged']} "
            f"| {number(entry['share_helped'])} | {number(entry['score'])} |"
        )
    if "llm" not in utility["by_source"]:
        print(
            "\nAll scenarios are marked manual: the llm column is absent. An llm scenario counts "
            "only in calculations forward of the date it was formalised."
        )


def _print_halves(report: dict) -> None:
    """The tuning and validation halves beside the full sample."""
    target = report["target_level"]
    names = [f"{first}/{second}" for first, second in HALF_PAIRS]
    print(f"\n## Halves (target level `{target}`)\n")
    print(f"| half | episodes | cells | {' | '.join(names)} | admission | pi_c gate |")
    print(f"|---|---|---:|{'---:|' * len(names)}---|---|")
    for name, block in report["halves"].items():
        scores = " | ".join(number(block["scores"][pair]["point"]) for pair in names)
        admitted = {None: "-", True: "fails", False: "passes"}[block["admission_failed"]]
        print(
            f"| {name} | {len(block['episodes'])} | {block['cells']} | {scores} | {admitted} "
            f"| {'failed' if block['constraint_gate']['failed'] else 'pass'} |"
        )


def _print_quantities(report: dict) -> None:
    """Per quantity: the relative scores, the calibration and the sharpness."""
    names = [f"{first}/{second}" for first, second in PAIRS]
    print("\n## Quantities\n")
    print(
        f"| quantity | level | scored | constrained | {' | '.join(names)} "
        "| PIT p5/p50/p95 | coverage | sharpness |"
    )
    print(f"|---|---|---:|---:|{'---:|' * len(names)}---|---:|---:|")
    for quantity, entry in report["quantities"].items():
        ratios = " | ".join(number(entry["ratios"].get(name)) for name in names)
        shares = (
            "-"
            if entry["calibration"] is None
            else " / ".join(f"{share:.2f}" for share in entry["calibration"])
        )
        tail = ", tail unverified" if entry["unverified_tail"] else ""
        print(
            f"| {quantity}{tail} | {entry['level']} | {entry['scored']} | {entry['constrained']} "
            f"| {ratios} | {shares} | {number(entry['coverage'])} | {number(entry['sharpness'])} |"
        )


def _print_decision(report: dict) -> None:
    """The four steps of the two-standard-error rule and the numbers they read."""
    verdict = report["decision"]
    versus_benchmark, versus_incumbent = verdict["vs_benchmark"], verdict["vs_incumbent"]
    print("\n## Decision (two standard errors)\n")
    print(
        f"- target level `{verdict['target_level']}`, half {verdict['half']}: "
        f"{verdict['cells']} paired cells"
    )
    print(
        f"- model/{verdict['benchmark']} {number(versus_benchmark['point'])} "
        f"{interval_text((versus_benchmark['low'], versus_benchmark['high']))}; "
        f"candidate/incumbent {number(versus_incumbent['point'])} "
        f"{interval_text((versus_incumbent['low'], versus_incumbent['high']))}"
    )
    shares = ", ".join(
        f"{rule} {number(share)}" for rule, share in verdict["admission_shares"].items()
    )
    print(f"- worst quantity's admission shares: {shares or '-'}")
    if verdict.get("admission_grown"):
        print(f"- shares that grew against the incumbent: {', '.join(verdict['admission_grown'])}")
    print(f"- verdict: **{verdict['verdict']}** (step {verdict['step']}) — {verdict['note']}")


def _print_tail(report: dict) -> None:
    """The Acerbi-Székely ``Z``, its intervals, the trimming and the episodes left out."""
    tail = report["tail"]
    horizon = tail.get("horizon", {})
    print("\n## Tail (Acerbi-Székely)\n")
    print(f"- horizon: {horizon.get('kind', 'episode window')}, month {horizon.get('month', '-')}")
    print(
        f"- Z {number(tail.get('z'))}; episode interval {interval_text(tail.get('interval'))}; "
        f"wild {interval_text(tail.get('wild'))}; wild disagrees {tail.get('wild_disagrees')}"
    )
    print(
        f"- trimmed Z with {tail.get('trim')} of the heaviest draws dropped: "
        f"{number(tail.get('trimmed_z'))}"
    )
    print(
        f"- episodes judged {len(tail.get('episodes', []))}, dropped {len(tail.get('dropped', []))}"
    )
    for dropped in tail.get("dropped", []):
        print(f"- dropped {dropped['id']}: {dropped['reason']}")


def _jsonable(value):
    if isinstance(value, np.floating | np.integer | np.bool_):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set | frozenset):
        return sorted(value)
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(type(value).__name__)
