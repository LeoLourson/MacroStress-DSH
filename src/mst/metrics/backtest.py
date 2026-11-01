"""The episode backtest: CRPS, calibration and admission of the models on historical events.

For an episode of the corpus the calculation date is the day before it starts; every model is
estimated on the data known that day — the macro model as an episode version under
``<models>/backtest/``, cached by the state of the data and the model's configuration — and the
scenario is what actually happened: the realized paths of the episode's variables, month by
month, with the shocks the sources of the time named. The same model also runs the base
scenario (no constraints, no impacts), and every cell — episode, month, quantity — is scored
against the fact several times: the conditional forecast, the random walk with drift computed
from the history known at that episode's calculation date, the unconditional (base) run of the
model itself, and, for targets, the naive forecast of a year earlier with the trend.

The quantities are every unconstrained macro variable the episode window knows and, with
``--satellites``, every served driver with a realized series. The report aggregates the cells
per quantity: CRPS normalised by the typical movement of the quantity as known at the episode's
calculation date, the baselines alongside, the calibration buckets, the sharpness, the coverage
(for reference only), the per-rule admission counts pooled over all cells, the intervals of the
scores and of the paired model-less-random-walk difference, and the share of the base run's
paths that satisfy each informative constraint — a scenario resting on a rare tail is marked as
such. The equal-weight aggregate — inside a quantity over the episodes, then with
equal weights over the quantities, separately per level and horizon — is computed on the same
cell rows and reported with its own block, together with the four-step 2-SE decision against
``--against`` and the utility of the scenario: the share of episodes whose conditional forecast
beats the base run, split by the manual/llm mark. The wild cluster bootstrap with
Webb weights re-checks the paired intervals. Monte-Carlo error is separated from sampling
error with ``--mc-bootstrap``. Episodes are split into tuning and validation halves by the
plan's marks; the halves are reported apart and the final tables stay over all episodes. The
driver whose simulated path leaves the range of its own
training history carries an unverified tail. ``--portfolio`` adds the Acerbi-Székely section: the portfolio's
predictive losses on the episodes' conditional paths, its realized loss on the facts of the
window, ``Z``, the episode and wild intervals and the 1 % trimming. The accept/reject verdict is
spoken only in the ``decision`` block behind ``--against``: the per-quantity tables stay
descriptive, and ``meta["missing"]`` names no metric the report cannot compute.
"""

import hashlib
import inspect
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from mst.data.store import SeriesStore
from mst.macro import version as macro_version
from mst.macro.panel import seasonal_profile, shift
from mst.macro.registry import REGISTRY_PATH as MACRO_REGISTRY_PATH
from mst.macro.registry import Registry, load_registry
from mst.macro.version import MacroVersion
from mst.model_sdk.contracts import (
    LinearConstraint,
    LinearTerm,
    ScenarioPlan,
)
from mst.model_sdk.version import MODEL_SDK_VERSION
from mst.plugin_sdk.version import SDK_VERSION
from mst.scenario.build import point_constraint

from .baselines import (
    CONSTRAINT,
    CONSTRAINT_THRESHOLD,
    DOMAIN,
    WIDTH,
    band_problems,
    constraint_probabilities,
    macro_domain,
    naive_annual,
    quantity_domain,
    random_walk,
)
from .cells import (
    ADMISSION_SHARE,
    Cell,
    crps,
    finite_draws,
    history_range,
    score_cells,
    sigma_with_fallback,
)
from .episodes import EPISODES_PATH, Episodes, calculation_date, load_episodes

INTERVAL_REPETITIONS = 1000
"""The repetitions of the episode bootstrap behind every reported interval."""

WILD_PRIMARY_BELOW = 8
"""Fewer episodes than this: the ordinary cluster bootstrap understates the interval, so the
wild interval is read as the primary one."""

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
    driving: list[str]
    constrained: list[str]
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


def run(
    *,
    store: Path,
    models: Path,
    corpus: Path,
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
) -> int:
    """The backtest of every episode the plan names; prints Markdown tables, returns an exit code.

    ``as_of`` is the date the facts are read on — the realized paths as they are known today;
    each episode's models are estimated on the day before it starts, whatever ``as_of`` is.
    With ``satellites`` the served drivers are scored too: their bundle is estimated on each
    episode's macro version, which takes as long as the estimation itself. ``against`` names
    the JSON report of the incumbent version, checked on the same episode plan — the
    two-standard-error decision compares the two on the same cells. ``portfolio`` names a stored
    portfolio for the Acerbi-Székely tail section; ``mc_bootstrap`` adds the Monte-Carlo
    interval of the equal-weight aggregate, separated from the episode-sampling intervals;
    ``target_level`` is the level whose aggregate the decision compares — the macro level by
    default, or a driver level (``driver/<quantity>``) with ``--satellites``.
    """
    from mst import models as models_pkg
    from mst.corpus import manifest

    if portfolio is not None and not satellites:
        raise ValueError(
            "--portfolio needs --satellites: the tail values the positions on the driver paths"
        )
    plan = load_episodes()
    plugin = models_pkg.selected(macro_id)
    registry = load_registry()
    starts = {e.id: e.start_month for e in manifest.load(corpus).episodes}
    data = SeriesStore(store)
    root = Path(models) / "backtest"
    problems: list[str] = []
    episodes: list[Episode] = []
    for check in plan.episodes:
        if only and check.id not in only:
            continue
        if check.id not in starts:
            problems.append(f"{check.id}: not in the corpus manifest {corpus}")
            continue
        try:
            episode = _episode(plugin, data, registry, plan, check, starts[check.id], root, as_of)
        except ValueError as error:
            problems.append(f"{check.id}: {error}")
            continue
        if satellites:
            if drivers is None:
                raise ValueError("--satellites needs --drivers, the driver catalog file")
            try:
                _attach_bundle(episode, data, root, Path(drivers), plan)
            except ValueError as error:
                # A driver series that starts after the calculation date (or another estimation
                # problem) voids that episode, not the whole check: it is named and skipped.
                problems.append(f"{check.id}: satellites: {error}")
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
        against=against,
        portfolio_id=portfolio,
        mc_bootstrap=mc_bootstrap,
        target_level=target_level,
    )
    _print(report)
    for problem in problems:
        print(problem)
    report["problems"] = problems
    if details is not None:
        details.write_text(
            json.dumps(report, ensure_ascii=False, indent=1, default=_jsonable),
            encoding="utf-8",
        )
    return 1 if problems or report["admission_failed"] else 0


def _episode(
    plugin,
    data: SeriesStore,
    registry: Registry,
    plan: Episodes,
    check,
    start: date,
    root: Path,
    known_on: date,
) -> Episode:
    """One episode: its macro version, its scenario from the facts, and the realized paths."""
    as_of = calculation_date(start)
    macro = _episode_macro(plugin, data, registry, as_of, root)
    unknown = [shock for shock in check.driving if shock not in macro.shocks]
    if unknown:
        raise ValueError(f"the macro model knows no shocks {', '.join(unknown)}")
    panel, _ = macro_version.panel_of(root, macro.name)
    panel_months = list(panel.months)
    end = shift(start, plan.after_months)
    horizon = (end.year - panel_months[-1].year) * 12 + end.month - panel_months[-1].month
    if horizon < 1:
        raise ValueError(f"the history ends at {panel_months[-1]:%Y-%m}, after the window")
    months = [shift(panel_months[-1], k) for k in range(1, horizon + 1)]
    realized = _realized(data, registry, panel, panel_months, months, as_of, known_on, plan)
    constraints = []
    for id_ in check.constrained:
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
            causes=tuple(check.driving),
        ),
        realized=realized,
        driving=list(check.driving),
        constrained=list(check.constrained),
        half=check.half,
        source=check.source,
        panel=panel,
        store=data,
        known_on=known_on,
    )


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


def _episode_macro(plugin, data: SeriesStore, registry: Registry, as_of: date, root: Path):
    """The macro version of an episode, cached under ``root`` by the state of the data.

    The estimate itself is deterministic but expensive; the cache key covers the model, its
    version, its configuration files and every value the registry names as known on the date,
    so a rerun with the same data loads the version instead of estimating again, and an edited
    spec, sign table or model package never reuses a stale one.
    """
    key = _data_key(plugin, data, registry, as_of)
    pointer = root / key
    installed = {plugin.model_id: plugin}
    if pointer.exists():
        name = pointer.read_text(encoding="utf-8").strip()
        if (root / "macro" / name).is_dir():
            return macro_version.load(root, name, installed)
    from mst.model_sdk.contracts import MacroEstimateContext

    estimate = plugin.estimate(
        MacroEstimateContext(store=data, registry=registry, as_of=as_of)
    )
    name = macro_version.save(plugin.model_id, estimate.meta, estimate.arrays, root)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(name + "\n", encoding="utf-8")
    return macro_version.load(root, name, installed)


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


def _data_key(plugin, data: SeriesStore, registry: Registry, as_of: date) -> str:
    digest = hashlib.sha256()
    digest.update(_configuration_digest(plugin).encode())
    digest.update(as_of.isoformat().encode())
    for variable in registry.variables:
        for row in data.as_of(variable.id, as_of):
            digest.update(
                f"{variable.id}|{row.month}|{row.value}|{row.published}|{row.approximate}".encode()
            )
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
    simulation = episode.macro.plugin.simulate(
        episode.macro.model, episode.scenario, draws=plan.draws, seed=plan.seed
    )
    episode.driver_paths = episode.bundle.simulate(
        simulation, episode.bundle.served(), (), seed=plan.seed
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
    against: Path | None = None,
    portfolio_id: str | None = None,
    mc_bootstrap: bool = False,
    target_level: str = "macro",
) -> dict:
    """The scores of every quantity over the episodes, with the gates, intervals and metadata."""
    cells: list[Cell] = []
    levels: dict[str, str] = {}
    unverified: set[str] = set()
    constraint_shares: dict[str, dict[int, float | None]] = {}

    for episode in episodes:
        draws = episode.macro.plugin.simulate(
            episode.macro.model, episode.scenario, draws=plan.draws, seed=plan.seed
        )
        episode.macro_paths = draws
        base = episode.macro.plugin.simulate(
            episode.macro.model,
            ScenarioPlan(horizon=episode.horizon),
            draws=plan.draws,
            seed=plan.seed,
        )
        constraint_shares[episode.id] = constraint_probabilities(
            base.paths, base.ids, episode.scenario.constraints
        )
        _macro_cells(episode, draws, base, cells, levels, registry, plan)
        if episode.bundle is not None and episode.driver_paths is not None:
            _driver_cells(episode, cells, levels, unverified, plan)

    by_quantity: dict[str, list[Cell]] = {}
    for cell in cells:
        by_quantity.setdefault(cell.quantity, []).append(cell)
    quantities = {}
    for quantity in sorted(by_quantity):
        own = by_quantity[quantity]
        row = score_cells(own)
        row["level"] = levels.get(quantity, "")
        row["unverified_tail"] = quantity in unverified
        row["admission"] = _quantity_rules(own)
        row["intervals"] = {
            "crps": _interval(row["months"], "crps", plan),
            "normalized": _interval(row["months"], "normalized", plan),
            "baseline_normalized": _interval(row["months"], "baseline_normalized", plan),
            "base_normalized": _interval(row["months"], "base_normalized", plan),
            "vs_baseline": _interval(row["months"], "difference", plan),
            "vs_baseline_wild": _wild_interval(row["months"], "difference", plan),
        }
        quantities[quantity] = row

    rows = _cell_rows(cells, levels)
    halves = {episode.id: episode.half for episode in episodes}
    sources = {episode.id: episode.source for episode in episodes}
    from .aggregate import equal_weight, utility_by_episode

    if rows:
        aggregate = equal_weight(rows, target_level=target_level)
        aggregate["utility"] = utility_by_episode(rows, sources, target_level=target_level)
        aggregate["intervals"] = {
            "sampling": _aggregate_interval(
                rows, halves=halves, plan=plan, target_level=target_level
            ),
        }
        if mc_bootstrap:
            aggregate["intervals"]["monte_carlo"] = _mc_interval(
                cells, levels, plan, target_level=target_level
            )
    else:
        # Every cell was constrained by the scenario, dropped for a missing fact or had no
        # normalising scale: the aggregate says so instead of inventing a number.
        aggregate = {
            "levels": {},
            "target": target_level,
            "utility": {"by_source": {}},
            "intervals": {},
            "note": "no scored cells: every cell was constrained, dropped or had no fact",
        }
    aggregate["rows"] = rows
    admission = _admission(cells)
    gate = _constraint_gate(constraint_shares)
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
        "episodes": [
            {
                "id": episode.id,
                "as_of": episode.as_of.isoformat(),
                "known_on": episode.known_on.isoformat(),
                "horizon": episode.horizon,
                "half": episode.half,
                "source": episode.source,
                "constrained": episode.constrained,
                "driving": episode.driving,
                "macro_model": episode.macro.plugin.model_id,
                "macro_version": episode.macro.name,
                "satellite_version": episode.bundle_name,
                "constraint_probabilities": constraint_shares[episode.id],
            }
            for episode in episodes
        ],
        "quantities": quantities,
        "aggregate": aggregate,
        "halves": _halves_block(
            episodes, cells, rows, constraint_shares, target_level=target_level
        ),
        "target_level": target_level,
        "admission": admission,
        "constraint_gate": gate,
        "admission_failed": admission["failed"] or gate["failed"],
        "problems": [],
    }
    if against is not None:
        report["decision"] = _decision_block(
            rows, against, admission=admission, halves=halves, plan=plan,
            target_level=target_level,
        )
    if portfolio_id is not None:
        report["tail"] = _tail_section(
            episodes, plan, portfolio_id, store=store, known_on=known_on
        )
    return report


def _cell_rows(cells: list[Cell], levels: dict[str, str]) -> list[dict]:
    """The scored cells as flat rows, the input of the equal-weight aggregate.

    A constrained cell says nothing about the forecast and never enters the aggregate; a cell
    whose draws are all non-finite was already counted in ``dropped`` by ``score_cells`` and
    carries no normalized score, so it drops out here with the counters.
    """
    rows = []
    for cell in cells:
        if cell.constrained or not np.isfinite(cell.fact):
            continue
        draws = finite_draws(cell.draws)
        if not len(draws):
            continue
        movement = cell.sigma
        known_scale = bool(np.isfinite(movement) and movement > 0)
        value = crps(draws, cell.fact)
        row = {
            "episode": cell.episode,
            "month": cell.month,
            "quantity": cell.quantity,
            "level": levels.get(cell.quantity, ""),
            "normalized": value / movement if known_scale else None,
            "sigma_source": cell.sigma_source,
            "sigma_pairs": cell.sigma_pairs,
            "baseline_normalized": None,
            "base_normalized": None,
            "naive_normalized": None,
        }
        for name, column in (
            ("baseline", cell.baseline),
            ("base", cell.base),
            ("naive", cell.naive),
        ):
            if column is None:
                continue
            known = finite_draws(column)
            if not len(known):
                continue
            score = crps(known, cell.fact)
            row[f"{name}_normalized"] = score / movement if known_scale else None
        rows.append(row)
    return rows


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
    pinned = _pinned(episode.scenario.constraints)
    months = list(range(1, len(episode.months) + 1))
    peers = {id_: panel.y[:, j] for j, id_ in enumerate(panel.ids) if id_ in column_of}
    for j, id_ in enumerate(panel.ids):
        i = column_of.get(id_)
        if i is None:
            continue
        levels[id_] = "macro"
        history = panel.y[:, j]
        domain = macro_domain()
        spread = history_range(history)
        walk = random_walk(history, episode.horizon, draws=plan.draws, seed=plan.seed)
        scale = sigma_with_fallback(history, months, peers=peers)
        for k in range(len(episode.months)):
            fact = episode.realized[k, j]
            if np.isnan(fact):
                continue
            sigma, source, pairs = scale[k + 1]
            cells.append(
                Cell(
                    episode=episode.id,
                    month=k + 1,
                    quantity=id_,
                    fact=float(fact),
                    draws=draws.paths[:, k + 1, i],
                    baseline=walk[:, k],
                    base=base.paths[:, k + 1, i],
                    constrained=(id_, k + 1) in pinned,
                    sigma=sigma,
                    sigma_source=source,
                    sigma_pairs=pairs,
                    history_range=spread,
                    domain=domain,
                )
            )


def _pinned(constraints) -> set[tuple[str, int]]:
    """The ``(variable, month)`` cells a single-term constraint fixes.

    A constraint of one term pins its cell when its band is no wider than the tolerance a
    point of the same terms would expand to — a fact-reconstructed point arrives exactly
    that wide, and a narrow interval of the same scale carries no scoring information
    either. The weights count through the tolerance, which scales with ``|w|``.
    """
    pinned = set()
    for constraint in constraints:
        if len(constraint.terms) != 1:
            continue
        term = constraint.terms[0]
        if constraint.high - constraint.low <= 2 * constraint.tolerance:
            pinned.add((term.variable, term.month))
    return pinned


def _driver_cells(
    episode: Episode,
    cells: list[Cell],
    levels: dict[str, str],
    unverified: set[str],
    plan: Episodes,
) -> None:
    """The cells of the episode's served drivers, with the random walk and the naive forecast."""
    bundle, paths = episode.bundle, episode.driver_paths
    driver_registry = bundle.registry
    series_of = {driver.id: driver.series for driver in driver_registry.drivers}
    months = list(range(1, len(episode.months) + 1))
    own_of: dict[str, np.ndarray] = {}
    quantity_of: dict[str, str] = {}
    for driver in paths.ids:
        quantity, node = bundle.node(driver)
        quantity_of[driver] = quantity
        d = bundle.arrays.get(f"{quantity}.d")
        own_of[driver] = d[:, node] if d is not None else np.array([])
    peers: dict[str, dict[str, np.ndarray]] = {}
    for driver in paths.ids:
        peers.setdefault(quantity_of[driver], {})[driver] = own_of[driver]
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
        scale = sigma_with_fallback(own, months, peers=peers[quantity])
        for t, month in enumerate(episode.months):
            fact = _driver_fact(episode, series_of.get(driver), month, declared)
            if fact is None:
                continue
            sigma, source, pairs = scale[t + 1]
            cells.append(
                Cell(
                    episode=episode.id,
                    month=t + 1,
                    quantity=driver,
                    fact=fact,
                    draws=paths.paths[:, t + 1, k],
                    baseline=walk[:, t] if walk.size else None,
                    naive=naive[:, t] if naive is not None else None,
                    sigma=sigma,
                    sigma_source=source,
                    sigma_pairs=pairs,
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
    rows = [
        row
        for row in episode.store.as_of(series, episode.known_on)
        if row.month == month
    ]
    if not rows:
        return None
    try:
        return float(to_model(declared.transform, rows[-1].value))
    except ValueError:
        return None


def _macro_domain(variable) -> tuple[float, float]:
    """The admissible range of a macro variable in the model scale.

    Nought is the conservative lower boundary for every registered variable: rates, spreads
    and unemployment are non-negative, and a ``log100`` price, index, volume or rate above one
    maps above nought. The upper side is unbounded.
    """
    return (0.0, float("inf"))


def _model_domain(declared) -> tuple[float, float] | None:
    """The admissible range of a driver quantity in the model scale; None when unbounded.

    A declared domain is mapped through the transform, its natural endpoints becoming the
    natural boundaries instead of disabling the gate: ``log100`` keeps ``(0, +inf)`` and the
    ``logit`` maps 0 and 1 to ``-inf`` and ``+inf``. Without a declared domain ``log100`` keeps
    the positive half-line and the other transforms are unbounded.
    """
    declared_domain = declared.domain
    if declared_domain is not None and len(declared_domain) == 2:
        low = _endpoint(declared.transform, float(declared_domain[0]))
        high = _endpoint(declared.transform, float(declared_domain[1]))
        if low == -float("inf") and high == float("inf"):
            return None
        return (low, high)
    if declared.transform == "log100":
        return (0.0, float("inf"))
    return None


def _endpoint(transform: str, value: float) -> float:
    """A domain endpoint in the model scale; the natural infinity where the transform diverges."""
    if transform == "logit":
        if value <= 0.0:
            return -float("inf")
        if value >= 1.0:
            return float("inf")
        return float(np.log(value / (1.0 - value)))
    if transform == "log100":
        if value <= 0.0:
            return 0.0
        return 100.0 * float(np.log(value))
    return value


def _admission(cells: list[Cell]) -> dict:
    """The pooled admission of a version: one share per rule over every scored cell.

    The admission rule admits a version when fewer than five percent of cells break *each*
    condition.
    The shares are computed over all scored cells of all quantities — not per quantity and not
    as the union of the rules — so a version that breaks one rule on six percent of the cells
    fails, while eight percent of cells breaking different rules four percent each passes.
    """
    scored = [
        cell
        for cell in cells
        if not cell.constrained
        and np.isfinite(cell.fact)
        and len(finite_draws(cell.draws))
    ]
    broken: dict[str, list[Cell]] = {DOMAIN: [], WIDTH: []}
    for cell in scored:
        for name in band_problems(
            cell.draws, cell.fact, domain=cell.domain, history_range=cell.history_range
        ):
            broken[name].append(cell)
    total = len(scored)
    rules = {}
    for name in (DOMAIN, WIDTH):
        found = broken[name]
        share = len(found) / total if total else 0.0
        rules[name] = {
            "cells": len(found),
            "share": share,
            "failed": bool(total) and share >= ADMISSION_SHARE,
            "episodes": sorted({cell.episode for cell in found}),
        }
    return {
        "cells": total,
        "rules": rules,
        "failed": any(rule["failed"] for rule in rules.values()),
    }


def _quantity_rules(cells: list[Cell]) -> dict:
    """The per-quantity counts of the two rules; the verdict itself is pooled by ``_admission``."""
    scored = [
        cell
        for cell in cells
        if not cell.constrained
        and np.isfinite(cell.fact)
        and len(finite_draws(cell.draws))
    ]
    counts = {DOMAIN: 0, WIDTH: 0}
    for cell in scored:
        for name in band_problems(
            cell.draws, cell.fact, domain=cell.domain, history_range=cell.history_range
        ):
            counts[name] += 1
    return {"scored": len(scored), DOMAIN: counts[DOMAIN], WIDTH: counts[WIDTH]}


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
        for number, share in found.items():
            if share is None:
                not_applicable += 1
                continue
            applicable += 1
            if share <= CONSTRAINT_THRESHOLD:
                failed.append(f"{episode}#{number}")
    return {
        "rule": CONSTRAINT,
        "threshold": CONSTRAINT_THRESHOLD,
        "applicable": applicable,
        "not_applicable": not_applicable,
        "failed_constraints": failed,
        "failed": bool(failed),
        "note": (
            "a fact-reconstructed episode fixes its constrained values as points, so pi_c is "
            "not applicable there and is excluded from the gate; a range constraint keeps it"
        ),
    }


def _interval(rows: list[dict], key: str, plan: Episodes) -> tuple[float, float] | None:
    """The 95% interval of a per-cell score, resampling the episodes with ``plan.seed``."""
    from .cells import bootstrap

    samples: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(key)
        if value is None or not np.isfinite(value):
            continue
        samples.setdefault(row["episode"], []).append(float(value))
    if not samples:
        return None
    low, high = bootstrap(samples, repetitions=INTERVAL_REPETITIONS, seed=plan.seed)
    if not (np.isfinite(low) and np.isfinite(high)):
        return None
    return (low, high)


def _wild_interval(rows: list[dict], key: str, plan: Episodes) -> tuple[float, float] | None:
    """The wild cluster bootstrap interval of a per-cell score, paired to ``_interval``.

    The statistic is the same cell-weighted mean; the null is the point estimate, because the
    interval reads the spread of a mean, not a difference from zero.
    """
    samples: dict[str, list[float]] = {}
    point: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None or not np.isfinite(value):
            continue
        samples.setdefault(row["episode"], []).append(float(value))
        point.append(float(value))
    if not samples:
        return None
    from .wild import wild_interval

    low, high = wild_interval(
        samples,
        repetitions=INTERVAL_REPETITIONS,
        seed=plan.seed,
        null=float(np.mean(point)),
    )
    if not (np.isfinite(low) and np.isfinite(high)):
        return None
    return (low, high)


def _halves_block(
    episodes: list[Episode],
    cells: list[Cell],
    rows: list[dict],
    constraint_shares: dict[str, dict[int, float | None]],
    *,
    target_level: str,
) -> dict:
    """The aggregate, admission and constraint gate of each half, and of all episodes.

    The backtest fits the models on the tuning half and checks them on the validation one, while
    the final table is built over all episodes; the block keeps the three side by side so a
    reader sees whether the halves agree. The ``all`` entry repeats the top-level tables.
    """
    from .aggregate import equal_weight

    chosen = {"all": [episode.id for episode in episodes]}
    for half in ("tuning", "validation"):
        named = [episode.id for episode in episodes if episode.half == half]
        if named:
            chosen[half] = named
    found: dict[str, dict] = {}
    for name, ids in chosen.items():
        named = set(ids)
        block_rows = [row for row in rows if row["episode"] in named]
        block_cells = [cell for cell in cells if cell.episode in named]
        found[name] = {
            "episodes": sorted(named),
            "cells": len(block_cells),
            "aggregate": (
                equal_weight(block_rows, target_level=target_level) if block_rows else None
            ),
            "admission": _admission(block_cells) if block_cells else None,
            "constraint_gate": _constraint_gate(
                {
                    episode: shares
                    for episode, shares in constraint_shares.items()
                    if episode in named
                }
            ),
        }
    return found


def _aggregate_interval(
    rows: list[dict], *, halves: dict[str, str], plan: Episodes, target_level: str
) -> dict:
    """The sampling and wild intervals of the equal-weight aggregate, per half and over all.

    The statistic is the reported aggregate itself: the episodes are the resampling unit, so a
    repetition redraws episodes and recomputes the equal-weight aggregate of the target level
    (``aggregate.aggregate_bootstrap``). The wild cluster bootstrap reweights the per-episode
    contributions of the same statistic (``aggregate.episode_contributions``); a paired
    difference against a comparison is tested against zero, the mean against its point value.
    """
    from .aggregate import aggregate_bootstrap, episode_contributions, equal_weight
    from .wild import disagrees, wild_interval

    blocks = {"all": rows}
    for half in ("tuning", "validation"):
        chosen = [row for row in rows if halves.get(row["episode"]) == half]
        if chosen:
            blocks[half] = chosen
    intervals: dict[str, dict] = {}
    for name, block in blocks.items():
        level_rows = [row for row in block if row["level"] == target_level]
        if not level_rows:
            continue
        aggregate = equal_weight(block, target_level=target_level)
        level = aggregate["levels"][target_level]
        found = aggregate_bootstrap(
            level_rows,
            ("model", "diff_vs_baseline", "diff_vs_base"),
            level=target_level,
            repetitions=INTERVAL_REPETITIONS,
            seed=plan.seed,
        )
        entry: dict = {
            "episodes": sorted({row["episode"] for row in level_rows}),
            "by_horizon": {
                str(month): value for month, value in sorted(level["by_horizon"].items())
            },
            "counters": level["counters"],
        }
        entry["primary"] = "wild" if len(entry["episodes"]) < WILD_PRIMARY_BELOW else "pair"
        for column, null in (
            ("model", None),
            ("diff_vs_baseline", 0.0),
            ("diff_vs_base", 0.0),
        ):
            sample = found[column]
            if sample["point"] is None:
                entry[column] = None
                continue
            pair = (sample["low"], sample["high"])
            contributions = episode_contributions(level_rows, column, level=target_level)
            wild = (
                wild_interval(
                    {
                        episode: value * len(contributions)
                        for episode, value in contributions.items()
                    },
                    repetitions=INTERVAL_REPETITIONS,
                    seed=plan.seed,
                    null=null,
                )
                if contributions
                else None
            )
            entry[column] = {
                "point": sample["point"],
                "se": sample["se"],
                "pair": pair,
                "wild": wild,
                "wild_disagrees": disagrees(pair, wild) if wild is not None else None,
            }
        intervals[name] = entry
    return intervals


def _mc_interval(
    cells: list[Cell], levels: dict[str, str], plan: Episodes, *, target_level: str = "macro"
) -> dict:
    """The Monte-Carlo interval of the equal-weight aggregate: the bootstrap over the draws.

    The cells of one calculation share the draws, so a repetition resamples one common index
    vector and applies it to every cell; the CRPS and the equal-weight aggregate are
    recomputed on every repetition. This measures the Monte-Carlo error of the draws; the
    episode bootstrap of ``_aggregate_interval`` measures the sampling error of the episodes,
    and the report labels the two apart. Only the cells of ``target_level`` enter.
    """
    from .aggregate import MC_STREAM, equal_weight, mc_interval

    cells_mc = []
    for cell in cells:
        if cell.constrained or not np.isfinite(cell.fact):
            continue
        if levels.get(cell.quantity, "") != target_level:
            continue
        if not (np.isfinite(cell.sigma) and cell.sigma > 0):
            continue
        draws = finite_draws(cell.draws)
        if not len(draws):
            continue
        cells_mc.append(
            {
                "quantity": cell.quantity,
                "month": cell.month,
                "level": levels.get(cell.quantity, ""),
                "draws": draws,
                "fact": float(cell.fact),
            }
        )
    if not cells_mc:
        return {"point": None, "interval": None}
    rows = [
        {
            "episode": "mc",
            "month": entry["month"],
            "quantity": entry["quantity"],
            "level": entry["level"],
            "normalized": crps(entry["draws"], entry["fact"]),
            "baseline_normalized": None,
            "base_normalized": None,
            "naive_normalized": None,
        }
        for entry in cells_mc
    ]
    point = equal_weight(rows, target_level=target_level)
    low, high = mc_interval(
        cells_mc,
        repetitions=INTERVAL_REPETITIONS,
        seed=[plan.seed, MC_STREAM],
    )
    return {
        "point": point["levels"][target_level]["overall"]["model"],
        "interval": (low, high) if (np.isfinite(low) and np.isfinite(high)) else None,
        "note": (
            "bootstrap over the draws (Monte-Carlo error); the sampling block above measures "
            "the episode-sampling error"
        ),
    }


def _decision_block(
    rows: list[dict],
    against: Path,
    *,
    admission: dict,
    halves: dict[str, str],
    plan: Episodes,
    target_level: str = "macro",
) -> dict:
    """The two-standard-error verdict: the candidate report against the incumbent's JSON.

    The incumbent report is read and checked against the same episode plan: a cell the two
    reports do not share is an error naming the missing cells, not a silent comparison on the
    intersection. The comparison happens on the equal-weight aggregates of ``target_level``
    over the validation half — the half the plan checks on; the block names it. The standard
    error of every paired difference is the spread of that aggregate over an episode bootstrap
    — the same statistic the report shows — and the wild cluster bootstrap re-checks it. Cells
    where either report has no normalised score cannot be paired; they are counted and named
    instead of shrinking the comparison silently.
    """
    from .aggregate import aggregate_bootstrap, episode_contributions
    from .decision import decide
    from .wild import disagrees, wild_interval

    if not against.exists():
        raise ValueError(f"--against {against}: the file does not exist")
    incumbent = json.loads(against.read_text(encoding="utf-8"))
    if incumbent.get("meta", {}).get("episodes_toml") != _file_digest(plan.path or EPISODES_PATH):
        raise ValueError(
            "--against was produced from a different episode plan; "
            "the two versions must be compared on the same cells"
        )
    incumbent_rows = incumbent.get("aggregate", {}).get("rows", [])
    if not incumbent_rows:
        raise ValueError(
            f"--against {against}: the report carries no aggregate rows; "
            "rerun that version with this code to compare the two"
        )
    by_key_c = {(row["episode"], row["month"], row["quantity"]): row for row in rows}
    by_key_i = {(row["episode"], row["month"], row["quantity"]): row for row in incumbent_rows}
    missing = sorted(set(by_key_i) ^ set(by_key_c))
    if missing:
        sample = ", ".join(f"{e}/{q}/t{t}" for e, t, q in missing[:10])
        raise ValueError(
            f"the two reports do not share cells: {len(missing)} differ ({sample}"
            f"{'...' if len(missing) > 10 else ''})"
        )
    validation = {e for e, half in halves.items() if half == "validation"}
    chosen = {row["episode"] for row in rows if row["episode"] in validation} or {
        row["episode"] for row in rows
    }
    paired_rows: list[dict] = []
    incumbent_paired: list[dict] = []
    not_compared: list[str] = []
    for key in sorted(key for key in by_key_c if key[0] in chosen):
        candidate_row, other = by_key_c[key], by_key_i[key]
        if candidate_row.get("normalized") is None or other.get("normalized") is None:
            not_compared.append(f"{key[0]}/{key[2]}/t{key[1]}")
            continue
        paired_rows.append(
            {
                **candidate_row,
                "diff_vs_incumbent": candidate_row["normalized"] - other["normalized"],
            }
        )
        incumbent_paired.append(other)
    if not paired_rows:
        raise ValueError(
            f"--against {against}: no cell of the comparison has a normalised score in both "
            "reports"
        )
    candidate_rows = [
        {key: value for key, value in row.items() if key != "diff_vs_incumbent"}
        for row in paired_rows
    ]
    found = aggregate_bootstrap(
        candidate_rows,
        ("model", "baseline", "diff_vs_baseline"),
        level=target_level,
        repetitions=INTERVAL_REPETITIONS,
        seed=plan.seed,
    )
    incumbent_found = aggregate_bootstrap(
        incumbent_paired,
        ("model",),
        level=target_level,
        repetitions=INTERVAL_REPETITIONS,
        seed=plan.seed,
    )
    paired_found = aggregate_bootstrap(
        paired_rows,
        ("diff_vs_incumbent",),
        level=target_level,
        repetitions=INTERVAL_REPETITIONS,
        seed=[plan.seed, 19],
    )
    candidate_crps = found["model"]["point"]
    benchmark_crps = found["baseline"]["point"]
    incumbent_crps = incumbent_found["model"]["point"]
    difference = found["diff_vs_baseline"]["point"]
    if candidate_crps is None or benchmark_crps is None:
        raise ValueError(
            f"--against {against}: level {target_level!r} has no scored cells in the "
            "validation half to decide on"
        )
    notes = []
    se_benchmark = found["diff_vs_baseline"]["se"]
    se_incumbent = paired_found["diff_vs_incumbent"]["se"]
    if se_benchmark is None:
        notes.append("the benchmark SE is undefined on one episode; it is read as zero")
        se_benchmark = 0.0
    if se_incumbent is None:
        notes.append("the incumbent SE is undefined on one episode; it is read as zero")
        se_incumbent = 0.0
    pair = (found["diff_vs_baseline"]["low"], found["diff_vs_baseline"]["high"])
    contributions = episode_contributions(candidate_rows, "diff_vs_baseline", level=target_level)
    wild = (
        wild_interval(
            {episode: value * len(contributions) for episode, value in contributions.items()},
            repetitions=INTERVAL_REPETITIONS,
            seed=[plan.seed, 11],
            null=0.0,
        )
        if contributions
        else None
    )
    verdict = decide(
        candidate_crps=candidate_crps,
        benchmark_crps=benchmark_crps,
        incumbent_crps=incumbent_crps,
        se_vs_benchmark=se_benchmark,
        se_vs_incumbent=se_incumbent,
        candidate_admission_shares={
            rule: found["share"] for rule, found in admission["rules"].items()
        },
        incumbent_admission_shares={
            rule: found["share"] for rule, found in incumbent["admission"]["rules"].items()
        },
    )
    verdict.update(
        {
            "target_level": target_level,
            "half": "validation" if validation & chosen else "all episodes",
            "primary": (
                "wild"
                if len({row["episode"] for row in paired_rows}) < WILD_PRIMARY_BELOW
                else "pair"
            ),
            "cells": len(paired_rows),
            "cells_not_compared": len(not_compared),
            "not_compared": not_compared[:10],
            "diff_vs_baseline": difference,
            "se_vs_incumbent": se_incumbent,
            "pair_interval": pair,
            "wild_interval": wild,
            "wild_disagrees": (
                disagrees(pair, wild) if wild is not None and all(np.isfinite(wild)) else None
            ),
            "notes": notes,
        }
    )
    return verdict


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
        if (
            episode.bundle is None
            or episode.driver_paths is None
            or episode.macro_paths is None
        ):
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
        "episodes_toml": _file_digest(plan.path or EPISODES_PATH),
        "macro_registry": _file_digest(MACRO_REGISTRY_PATH),
        "store": str(store),
        "models": str(models),
        "backtest_root": str(root),
        "git_commit": _git_commit(),
        "plugin_sdk": SDK_VERSION,
        "model_sdk": MODEL_SDK_VERSION,
        "target_level": target_level,
        "intervals": {
            "sampling": "episodes resampled with replacement, the aggregate recomputed",
            "wild": "the same statistic, episodes reweighted by Webb weights",
            "monte_carlo": "draws resampled; only with --mc-bootstrap",
        },
        "assumptions": [
            "sigma_q(t) falls back: the nearest valid month below, then above, then the peers "
            "of the level",
            "the target level of the decision and of the utility is --target-level",
            "utility: an episode counts when the paired CRPS difference against the base run "
            "is negative; the split by source is manual against llm",
            "realized loss: the portfolio valued on the facts at t = 0 and t = H; an episode "
            "without the fact at t = H is dropped by name",
            "dist(s) = E_P[-log P(S | R)] on the constraint law (docs/concept/formalization.md)",
        ],
        "macro_models": [
            {"model_id": model_id, "version": version}
            for model_id, version in sorted(macro_models)
        ],
        "satellite_models": [
            {"model_id": model_id, "version": version}
            for model_id, version in sorted(satellites)
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
            marker = (folder / found[len("gitdir:"):].strip()).resolve()
        if not marker.is_dir():
            continue
        try:
            head = (marker / "HEAD").read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not head.startswith("ref: "):
            return head[:40] or None
        ref = head[len("ref:"):].strip()
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
    """The report as Markdown tables and the list of metrics it cannot compute."""
    print("## Episodes\n")
    print("| episode | as of | horizon | constrained | driving | min constraint probability |")
    print("|---|---|---|---|---|---|")
    for row in report["episodes"]:
        shares = row["constraint_probabilities"]
        known = [share for share in shares.values() if share is not None]
        exact = sum(1 for share in shares.values() if share is None)
        worst = f"{min(known):.2f}" if known else "n/a"
        if exact:
            worst += f" ({exact} n/a)"
        print(
            f"| {row['id']} | {row['as_of']} | {row['horizon']} "
            f"| {', '.join(row['constrained'])} | {', '.join(row['driving'])} | {worst} |"
        )
    admission = report["admission"]
    print("\n## Admission\n")
    print("| rule | broken cells | share | failing episodes |")
    print("|---|---:|---:|---|")
    for name, rule in admission["rules"].items():
        episodes = ", ".join(rule["episodes"]) or "-"
        print(
            f"| {name} | {rule['cells']} of {admission['cells']} "
            f"| {rule['share']:.3f} | {episodes} |"
        )
    gate = report["constraint_gate"]
    verdict = "failed" if gate["failed"] else "pass"
    print(
        f"\n`{gate['rule']}`: {gate['applicable']} applicable constraint(s), "
        f"{gate['not_applicable']} not applicable (exact point constraints), "
        f"threshold {gate['threshold']:.3f} — {verdict}."
    )
    _print_aggregate(report)
    _print_utility(report)
    _print_halves(report)
    print("\n## Quantities\n")
    print(
        "| quantity | level | scored | CRPS/sigma | 95% interval | random walk/sigma "
        "| base run/sigma | naive/sigma | PIT p5/p50/p95 | sharpness | rules domain/width |"
    )
    print("|---|---|---|---:|---|---:|---:|---:|---|---:|---|")
    for quantity, row in report["quantities"].items():
        if row["calibration"] is None:
            print(f"| {quantity} | {row['level']} | 0 | - | - | - | - | - | - | - | - |")
            continue
        shares = " / ".join(f"{share:.2f}" for share in row["calibration"]["shares"])
        intervals = row["intervals"]
        admission_text = f"{row['admission'][DOMAIN]}/{row['admission'][WIDTH]}"
        if row["unverified_tail"]:
            admission_text += ", tail unverified"
        print(
            f"| {quantity} | {row['level']} | {row['scored']} "
            f"| {_number(row['normalized_crps'])} | {_interval_text(intervals['normalized'])} "
            f"| {_number(row['baseline_normalized_crps'])} "
            f"| {_number(row['base_normalized_crps'])} "
            f"| {_number(row['naive_normalized_crps'])} | {shares} "
            f"| {row['sharpness']:.3f} | {admission_text} |"
        )
    hidden = [
        quantity
        for quantity, row in report["quantities"].items()
        if row["scored"] and row["normalized_cells"] < row["scored"]
    ]
    if hidden:
        print(
            "\nNormalised scores cover only the cells with a finite positive sigma at the "
            f"episode's calculation date: {', '.join(hidden)}."
        )
    if "decision" in report:
        _print_decision(report)
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


def _print_aggregate(report: dict) -> None:
    """The equal-weight table of the level × horizon aggregate, the model and its comparisons."""
    aggregate = report["aggregate"]
    target = report["target_level"]
    print(f"\n## Aggregate (equal weights across quantities; target level `{target}`)\n")
    print(
        "| level | horizon | model | random walk | base run | naive | model - random walk "
        "| model - base run |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for level, block in aggregate["levels"].items():
        for month, values in sorted(block["by_horizon"].items()):
            print(f"| {level} | {month} | {_aggregate_text(values)} |")
        print(f"| {level} | all | {_aggregate_text(block['overall'])} |")
        counters = block["counters"]
        print(
            f"\n`{level}`: {counters['model']['cells']} cells, "
            f"{counters['model']['normalized']} normalised (model), "
            f"{counters['baseline']['normalized']} scored against the random walk."
        )
    sampling = aggregate["intervals"].get("sampling", {}).get("all")
    if sampling and sampling.get("model"):
        model = sampling["model"]
        text = (
            f"\nTarget level `{target}`: CRPS/sigma {_number(model['point'])}, "
            f"95 % episode interval {_interval_text(model['pair'])}, "
            f"wild {_interval_text(model['wild'])}"
        )
        difference = model.get("diff_vs_baseline")
        if difference:
            text += (
                f"; against the random walk {_number(difference['point'])}, "
                f"interval {_interval_text(difference['pair'])}, "
                f"wild {_interval_text(difference['wild'])}"
            )
            if difference.get("wild_disagrees") is not None:
                text += f", disagreements {'yes' if difference['wild_disagrees'] else 'no'}"
        if sampling.get("primary") == "wild":
            text += (
                f"\nFewer than {WILD_PRIMARY_BELOW} episodes: the ordinary bootstrap "
                "understates the interval, so the wild one is read as primary."
            )
        print(text + ".")
    monte_carlo = aggregate["intervals"].get("monte_carlo")
    if monte_carlo is not None:
        print(
            "\nMonte-Carlo interval of the same aggregate (the draws are resampled, not the "
            f"episodes): {_interval_text(monte_carlo.get('interval'))}."
        )


def _aggregate_text(values: dict) -> str:
    """One aggregate row: the four comparisons and the two paired differences."""
    return " | ".join(
        _number(values.get(name))
        for name in ("model", "baseline", "base", "naive", "diff_vs_baseline", "diff_vs_base")
    )


def _print_utility(report: dict) -> None:
    """The usefulness of the scenario: paired differences against the base run by source."""
    utility = report["aggregate"]["utility"]["by_source"]
    print("\n## Scenario usefulness (paired difference against the base run)\n")
    print("| source | level | episodes judged | dropped | share negative | mean difference |")
    print("|---|---|---:|---:|---:|---:|")
    if not utility:
        print("| - | - | 0 | 0 | - | - |")
        print("\nNo episode could be judged: the aggregate has no scored cells.")
        return
    for source, block in utility.items():
        for level, entry in block["by_level"].items():
            print(
                f"| {source} | {level} | {entry['episodes']} | {entry['dropped']} "
                f"| {_number(entry['share_negative'])} | {_number(entry['mean_diff'])} |"
            )
    if "llm" not in utility:
        print(
            "\nAll scenarios are marked manual: the llm column is absent and the split is not "
            "yet informative. An llm scenario counts only in calculations forward of the date "
            "it was formalised."
        )


def _print_halves(report: dict) -> None:
    """The tuning and validation halves beside the full sample."""
    halves = report["halves"]
    target = report["target_level"]
    print(f"\n## Halves (target level `{target}`)\n")
    print(
        "| half | episodes | cells | CRPS/sigma | random walk | base run | domain share "
        "| width share | pi_c gate |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for name, block in halves.items():
        aggregate = block["aggregate"]
        admission = block["admission"]
        gate = block["constraint_gate"]
        values = aggregate["levels"].get(target, {}).get("overall", {}) if aggregate else {}
        rules = admission["rules"] if admission else {}
        print(
            f"| {name} | {len(block['episodes'])} | {block['cells']} "
            f"| {_number(values.get('model'))} | {_number(values.get('baseline'))} "
            f"| {_number(values.get('base'))} "
            f"| {_number(rules.get(DOMAIN, {}).get('share'))} "
            f"| {_number(rules.get(WIDTH, {}).get('share'))} "
            f"| {'failed' if gate['failed'] else 'pass'} |"
        )


def _print_decision(report: dict) -> None:
    """The four steps of the two-standard-error rule and the numbers they read."""
    verdict = report["decision"]
    print("\n## Decision (two standard errors)\n")
    print(
        f"- target level `{verdict.get('target_level')}`, half {verdict.get('half')}: "
        f"{verdict.get('cells')} paired cells, {verdict.get('cells_not_compared')} not compared"
    )
    print(
        f"- CRPS/sigma: candidate {_number(verdict.get('candidate_crps'))}, "
        f"random walk {_number(verdict.get('benchmark_crps'))}, "
        f"incumbent {_number(verdict.get('incumbent_crps'))}"
    )
    print(
        f"- SE of the difference: {_number(verdict.get('se_vs_benchmark'))} against the "
        f"benchmark, {_number(verdict.get('se_vs_incumbent'))} against the incumbent"
    )
    print(
        f"- difference against the benchmark {_number(verdict.get('diff_vs_baseline'))}; "
        f"pair interval {_interval_text(verdict.get('pair_interval'))}; "
        f"wild {_interval_text(verdict.get('wild_interval'))}; "
        f"wild disagrees {verdict.get('wild_disagrees')}"
    )
    if verdict.get("primary") == "wild":
        print(
            f"- fewer than {WILD_PRIMARY_BELOW} episodes: the wild interval is read as primary"
        )
    shares = ", ".join(
        f"{rule} {_number(share)}" for rule, share in verdict.get("admission_shares", {}).items()
    )
    print(f"- admission shares: {shares or '-'}")
    if verdict.get("admission_grown"):
        print(f"- shares that grew against the incumbent: {', '.join(verdict['admission_grown'])}")
    print(
        f"- verdict: **{verdict.get('verdict')}** (step {verdict.get('step')}) — "
        f"{verdict.get('note')}"
    )
    for note in verdict.get("notes", []):
        print(f"- {note}")


def _print_tail(report: dict) -> None:
    """The Acerbi-Székely ``Z``, its intervals, the trimming and the episodes left out."""
    tail = report["tail"]
    horizon = tail.get("horizon", {})
    print("\n## Tail (Acerbi-Székely)\n")
    print(f"- horizon: {horizon.get('kind', 'episode window')}, month {horizon.get('month', '-')}")
    print(
        f"- Z {_number(tail.get('z'))}; episode interval {_interval_text(tail.get('interval'))}; "
        f"wild {_interval_text(tail.get('wild'))}; wild disagrees {tail.get('wild_disagrees')}"
    )
    print(
        f"- trimmed Z with {tail.get('trim')} of the heaviest draws dropped: "
        f"{_number(tail.get('trimmed_z'))}"
    )
    print(
        f"- episodes judged {len(tail.get('episodes', []))}, "
        f"dropped {len(tail.get('dropped', []))}"
    )
    for dropped in tail.get("dropped", []):
        print(f"- dropped {dropped['id']}: {dropped['reason']}")


def _number(value) -> str:
    """A score as three decimals, or a dash when it is missing or non-finite."""
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.3f}"


def _interval_text(pair) -> str:
    if pair is None:
        return "-"
    low, high = pair
    return f"[{low:.3f}, {high:.3f}]"


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
