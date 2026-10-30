"""The episode backtest: CRPS, calibration and admission of the models on historical events.

For an episode of the corpus the calculation date is the day before it starts; every model is
estimated on the data known that day — the macro model as an episode version under
``<models>/backtest/``, cached by the state of the data — and the scenario is what actually
happened: the realized paths of the episode's variables, month by month, with the shocks the
sources of the time named. The same model also runs the base scenario (no constraints, no
impacts), and every cell — episode, month, quantity — is scored against the fact three times:
the conditional forecast, the baseline random walk with drift and, for drivers, the naive
forecast of a year earlier with the trend.

The quantities are every unconstrained macro variable the episode window knows and, with
``--satellites``, every served driver with a realized series. The report aggregates the cells
per quantity: CRPS normalised by the typical movement of the quantity, the baselines
alongside, the calibration buckets, the sharpness, the admission problems and the share of
the base run's paths that satisfy each constraint — a scenario resting on a rare tail is
marked as such. The driver whose simulated path leaves the range of its own training history
carries an unverified tail.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from mst.data.store import SeriesStore
from mst.macro import version as macro_version
from mst.macro.panel import seasonal_profile, shift
from mst.macro.registry import Registry, load_registry
from mst.macro.version import MacroVersion
from mst.model_sdk.contracts import (
    LinearConstraint,
    LinearTerm,
    ScenarioPlan,
)

from .baselines import constraint_probabilities, naive_annual, random_walk
from .cells import ADMISSION_SHARE, Cell, score_cells, sigma_of_month
from .episodes import Episodes, calculation_date, load_episodes


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
    panel: object = None
    """The panel of the episode's macro version, in the model scale."""
    store: object = None
    """The series store the facts are read from."""
    known_on: date = date(2200, 1, 1)
    """The date the realized paths are known on."""
    bundle: object = None
    """The satellite bundle of the episode's macro version, with ``--satellites``."""
    driver_paths: object = None
    """The driver paths of the episode, in the model scale."""


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
) -> int:
    """The backtest of every episode of the plan; prints Markdown tables, returns an exit code.

    ``as_of`` is the date the facts are read on — the realized paths as they are known today;
    each episode's models are estimated on the day before it starts, whatever ``as_of`` is.
    With ``satellites`` the served drivers are scored too: their bundle is estimated on each
    episode's macro version, which takes as long as the estimation itself.
    """
    from mst import models as models_pkg
    from mst.corpus import manifest

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
            _attach_bundle(episode, data, root, Path(drivers), plan)
        episodes.append(episode)
    report = _report(episodes, plan, registry)
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
                    LinearConstraint(
                        terms=(LinearTerm(variable=id_, month=k, weight=1.0),),
                        low=float(value),
                        high=float(value),
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
            driving_shocks=tuple(check.driving),
        ),
        realized=realized,
        driving=list(check.driving),
        constrained=list(check.constrained),
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

    Seasonal variables are compared after removing the calendar profile estimated on the
    calculation date, as the model saw them; an exclusion drops the months of a variable from
    its start on when the history ends before it — an index splice makes the level
    incomparable.
    """
    grid = panel_months + months
    y = np.full((len(grid), len(panel.ids)), np.nan)
    at = {month: t for t, month in enumerate(grid)}
    for i, id_ in enumerate(panel.ids):
        variable = registry.variable(id_)
        for row in data.as_of(id_, known_on):
            if row.month in at:
                y[at[row.month], i] = variable.to_model(row.value)
        if variable.seasonal:
            profile = _profile(y[: len(panel_months), i], panel_months)
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
    version and every value the registry names as known on the date, so a rerun with the same
    data loads the version instead of estimating again.
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


def _data_key(plugin, data: SeriesStore, registry: Registry, as_of: date) -> str:
    digest = hashlib.sha256()
    digest.update(plugin.model_id.encode())
    digest.update(getattr(plugin, "version", "").encode())
    digest.update(as_of.isoformat().encode())
    for variable in registry.variables:
        for row in data.as_of(variable.id, as_of):
            digest.update(
                f"{variable.id}|{row.month}|{row.value}|{row.published}|{row.approximate}".encode()
            )
    return digest.hexdigest()


def _attach_bundle(
    episode: Episode, data: SeriesStore, root: Path, drivers: Path, plan: Episodes
) -> None:
    """Estimate the satellite bundle of the episode's macro version and simulate its drivers.

    The bundle is cached beside the episode's macro version, under the same backtest root.
    """
    from mst import models as models_pkg
    from mst.drivers import bundle as satellite_bundle
    from mst.drivers.registry import load_registry as load_drivers

    driver_registry = load_drivers(catalog=drivers)
    name = episode.macro.name
    pointer = root / f"satellites-{name}"
    installed = models_pkg.satellites()
    if pointer.exists():
        found = pointer.read_text(encoding="utf-8").strip()
        if (root / "satellites" / found).is_dir():
            episode.bundle = satellite_bundle.load(root, found, installed)
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
    simulation = episode.macro.plugin.simulate(
        episode.macro.model, episode.scenario, draws=plan.draws, seed=plan.seed
    )
    episode.driver_paths = episode.bundle.simulate(
        simulation, episode.bundle.served(), (), seed=plan.seed
    )


def _report(episodes: list[Episode], plan: Episodes, registry: Registry) -> dict:
    """The scores of every quantity over the episodes, with the admission and the baselines."""
    cells: dict[str, list[Cell]] = {}
    histories: dict[str, np.ndarray] = {}
    domains: dict[str, tuple[float, float] | None] = {}
    levels: dict[str, str] = {}
    unverified: set[str] = set()
    constraint_shares: dict[str, dict[int, float]] = {}

    for episode in episodes:
        draws = episode.macro.plugin.simulate(
            episode.macro.model, episode.scenario, draws=plan.draws, seed=plan.seed
        )
        base = episode.macro.plugin.simulate(
            episode.macro.model,
            ScenarioPlan(horizon=episode.horizon),
            draws=plan.draws,
            seed=plan.seed,
        )
        constraint_shares[episode.id] = constraint_probabilities(
            base.paths, base.ids, episode.scenario.constraints
        )
        panel = episode.panel
        for i, id_ in enumerate(base.ids):
            levels[id_] = "macro"
            histories.setdefault(id_, panel.y[:, i])
            domains.setdefault(id_, None)
            cells.setdefault(id_, [])
            for k in range(len(episode.months)):
                fact = episode.realized[k, i]
                if np.isnan(fact):
                    continue
                cells[id_].append(
                    Cell(
                        episode=episode.id,
                        month=k + 1,
                        quantity=id_,
                        fact=float(fact),
                        draws=draws.paths[:, k + 1, i],
                        baseline=base.paths[:, k + 1, i],
                        constrained=id_ in episode.constrained,
                    )
                )

        if episode.bundle is not None and episode.driver_paths is not None:
            _driver_cells(episode, cells, histories, domains, levels, unverified, plan)

    quantities = {}
    for quantity in sorted(cells):
        history = histories.get(quantity, np.array([]))
        sigma = sigma_of_month(cells[quantity], history)
        row = score_cells(cells[quantity], sigma)
        row["level"] = levels.get(quantity, "")
        row["unverified_tail"] = quantity in unverified
        row["admission"] = _admission(cells[quantity], domains.get(quantity), history)
        row["crps_interval"] = _episode_interval(cells[quantity])
        quantities[quantity] = row

    return {
        "episodes": [
            {
                "id": episode.id,
                "as_of": episode.as_of.isoformat(),
                "horizon": episode.horizon,
                "constrained": episode.constrained,
                "driving": episode.driving,
                "constraint_probabilities": constraint_shares[episode.id],
            }
            for episode in episodes
        ],
        "quantities": quantities,
        "admission_failed": any(
            row["admission"]["share"] >= ADMISSION_SHARE for row in quantities.values()
        ),
        "problems": [],
    }


def _driver_cells(
    episode: Episode, cells, histories, domains, levels, unverified, plan: Episodes
) -> None:
    """The cells of the episode's served drivers, with the naive forecast alongside."""

    bundle, paths = episode.bundle, episode.driver_paths
    driver_registry = bundle.registry
    series_of = {driver.id: driver.series for driver in driver_registry.drivers}
    for k, driver in enumerate(paths.ids):
        quantity = bundle.quantity_of(driver)
        declared = driver_registry.quantity(quantity)
        levels[driver] = f"driver/{quantity}"
        d = bundle.arrays.get(f"{quantity}.d")
        node = bundle.node(driver)[1]
        own = d[:, node] if d is not None else np.array([])
        histories.setdefault(driver, own)
        domains.setdefault(driver, _model_domain(declared))
        cells.setdefault(driver, [])
        baseline = random_walk(own, episode.horizon, draws=plan.draws, seed=plan.seed)
        naive = naive_annual(own, episode.horizon, draws=plan.draws, seed=plan.seed)
        for t, month in enumerate(episode.months):
            fact = _driver_fact(episode, series_of.get(driver), month, declared)
            if fact is None:
                continue
            cells[driver].append(
                Cell(
                    episode=episode.id,
                    month=t + 1,
                    quantity=driver,
                    fact=fact,
                    draws=paths.paths[:, t + 1, k],
                    baseline=baseline[:, t] if baseline.size else None,
                    naive=naive[:, t] if naive is not None else None,
                )
            )
        if "outside_training_range" in paths.flags.get(driver, frozenset()):
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


def _model_domain(declared):
    """The admissible range of a quantity in the model scale; None when unbounded."""
    from mst.drivers.registry import to_model

    if not declared.domain or len(declared.domain) != 2:
        return None
    try:
        low = float(to_model(declared.transform, declared.domain[0]))
        high = float(to_model(declared.transform, declared.domain[1]))
    except ValueError:
        return None
    if not (np.isfinite(low) and np.isfinite(high)):
        return None
    return (low, high)


def _admission(cells: list[Cell], domain, history: np.ndarray) -> dict:
    """The share of scored cells that break the band rules of section 5."""
    from .baselines import band_problems

    scored = [cell for cell in cells if not cell.constrained]
    if not scored:
        return {"share": 0.0, "problems": {}}
    known = history[np.isfinite(history)] if len(history) else np.array([])
    history_range = float(known.max() - known.min()) if len(known) else float("nan")
    broken: set[str] = set()
    problems: dict[str, set[str]] = {}
    for cell in scored:
        found = band_problems(
            cell.draws, cell.fact, domain=domain, history_range=history_range
        )
        if found:
            broken.add(f"{cell.episode}/{cell.month}")
        for name in found:
            problems.setdefault(name, set()).add(cell.episode)
    return {
        "share": len(broken) / len(scored),
        "problems": {name: sorted(found) for name, found in problems.items()},
    }


def _episode_interval(cells: list[Cell]) -> tuple[float, float] | None:
    """The 95% interval of the mean CRPS, resampling the episodes."""
    from .cells import bootstrap, crps

    by_episode: dict[str, list[float]] = {}
    for cell in cells:
        if cell.constrained:
            continue
        by_episode.setdefault(cell.episode, []).append(crps(cell.draws, cell.fact))
    series = [(name, float(np.mean(values))) for name, values in by_episode.items()]
    return bootstrap(series) if series else None


def _print(report: dict) -> None:
    """The report as Markdown tables."""
    print("## Episodes\n")
    print("| episode | as of | horizon | constrained | driving | min constraint probability |")
    print("|---|---|---|---|---|---|")
    for row in report["episodes"]:
        shares = row["constraint_probabilities"]
        worst = min(shares.values()) if shares else None
        worst = "-" if worst is None else f"{worst:.2f}"
        print(
            f"| {row['id']} | {row['as_of']} | {row['horizon']} "
            f"| {', '.join(row['constrained'])} | {', '.join(row['driving'])} | {worst} |"
        )
    print("\n## Quantities\n")
    print(
        "| quantity | level | scored | CRPS/sigma | random walk | naive | "
        "calibration p5/p50/p95 | sharpness | admission |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for quantity, row in report["quantities"].items():
        if row["calibration"] is None:
            print(f"| {quantity} | {row['level']} | 0 | - | - | - | - | - | - |")
            continue
        shares = " / ".join(f"{share:.2f}" for share in row["calibration"]["shares"])
        normalized = row["normalized_crps"]
        baseline = row["baseline_normalized_crps"]
        naive = row["naive_normalized_crps"]

        def _number(value):
            return "-" if value is None else f"{value:.3f}"

        admission = f"{row['admission']['share']:.2f}"
        if row["unverified_tail"]:
            admission += ", tail unverified"
        print(
            f"| {quantity} | {row['level']} | {row['scored']} | {_number(normalized)} "
            f"| {_number(baseline)} | {_number(naive)} | {shares} "
            f"| {row['sharpness']:.3f} | {admission} |"
        )


def _jsonable(value):
    if isinstance(value, np.floating | np.integer):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set | frozenset):
        return sorted(value)
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(type(value).__name__)
