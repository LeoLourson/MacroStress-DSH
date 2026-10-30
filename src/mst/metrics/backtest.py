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
such. The driver whose simulated path leaves the range of its own training history carries an
unverified tail.

What is not done here is named in the report's ``meta["missing"]``: the equal-weight aggregate
across quantities and the four-step 2-SE decision rule of section 5, the Acerbi-Székely ``Z``
and the portfolio ES backtest, the wild cluster bootstrap, the episode halves, and the
separation of Monte-Carlo error from sampling error. The report is per quantity and gives the
intervals; it does not pronounce a verdict on a version.
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
from mst.plugin_sdk.version import SDK_VERSION

from .baselines import (
    CONSTRAINT,
    CONSTRAINT_THRESHOLD,
    DOMAIN,
    WIDTH,
    band_problems,
    constraint_probabilities,
    naive_annual,
    random_walk,
)
from .cells import (
    ADMISSION_SHARE,
    Cell,
    finite_draws,
    history_range,
    score_cells,
    typical_movement,
)
from .episodes import EPISODES_PATH, Episodes, calculation_date, load_episodes

INTERVAL_REPETITIONS = 1000
"""The repetitions of the episode bootstrap behind every reported interval."""

MISSING = (
    "equal-weight aggregation across quantities, the per-level and per-horizon split and the "
    "four-step 2-SE decision rule of section 5: the report is per quantity and gives the "
    "intervals, not a verdict",
    "Acerbi-Székely Z, the ES interval, the 1 % trimming and the portfolio ES backtest: "
    "there are no realized portfolio losses yet",
    "wild cluster bootstrap with Webb weights: too few episodes for a few-cluster check",
    "episode halves for tuning and validation: not implemented",
    "Monte-Carlo error separated from sampling error: only the episode bootstrap is computed",
)


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
    bundle_name: str | None = None
    """The hash of that satellite version."""
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
    report = _report(
        episodes,
        plan,
        registry,
        plugin=plugin,
        known_on=as_of,
        store=store,
        models=models,
        root=root,
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
        }
        quantities[quantity] = row

    admission = _admission(cells)
    gate = _constraint_gate(constraint_shares)
    return {
        "meta": _meta(
            episodes,
            plan,
            plugin=plugin,
            known_on=known_on,
            store=store,
            models=models,
            root=root,
        ),
        "episodes": [
            {
                "id": episode.id,
                "as_of": episode.as_of.isoformat(),
                "known_on": episode.known_on.isoformat(),
                "horizon": episode.horizon,
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
        "admission": admission,
        "constraint_gate": gate,
        "admission_failed": admission["failed"] or gate["failed"],
        "problems": [],
    }


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
    for j, id_ in enumerate(panel.ids):
        i = column_of.get(id_)
        if i is None:
            continue
        levels[id_] = "macro"
        history = panel.y[:, j]
        domain = _macro_domain(registry.variable(id_))
        spread = history_range(history)
        walk = random_walk(history, episode.horizon, draws=plan.draws, seed=plan.seed)
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
                    constrained=(id_, k + 1) in pinned,
                    sigma=typical_movement(history, k + 1),
                    history_range=spread,
                    domain=domain,
                )
            )


def _pinned(constraints) -> set[tuple[str, int]]:
    """The ``(variable, month)`` cells an exact single-term constraint fixes."""
    pinned = set()
    for constraint in constraints:
        if constraint.low != constraint.high or len(constraint.terms) != 1:
            continue
        term = constraint.terms[0]
        if term.weight == 1.0:
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
    for k, driver in enumerate(paths.ids):
        quantity = bundle.quantity_of(driver)
        declared = driver_registry.quantity(quantity)
        levels[driver] = f"driver/{quantity}"
        d = bundle.arrays.get(f"{quantity}.d")
        node = bundle.node(driver)[1]
        own = d[:, node] if d is not None else np.array([])
        domain = _model_domain(declared)
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
                    sigma=typical_movement(own, t + 1),
                    history_range=spread,
                    domain=domain,
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

    Section 5 admits a version when fewer than five percent of cells break *each* condition.
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

    A constraint reconstructed as an exact point (``low == high``) cannot be met by a
    continuous draw: its probability is not applicable and is left out of the gate. The report
    says how many constraints of each kind the gate saw.
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
            "a fact-reconstructed episode fixes its constrained values exactly, so pi_c is "
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


def _meta(
    episodes: list[Episode],
    plan: Episodes,
    *,
    plugin,
    known_on: date,
    store: Path,
    models: Path,
    root: Path,
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
    """The report as Markdown tables and the list of what section 5 still asks for."""
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
    print("\nMissing section 5 items:\n")
    for item in meta["missing"]:
        print(f"- {item}")


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
