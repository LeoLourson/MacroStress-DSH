"""Out of sample: on every date of the plan the models see only what was published by then.

On each date the macro model and the satellites are estimated on the data known that day,
the empty scenario is forecast, and the forecast is compared with what happened over the
next months: the share of facts inside the central band, the mean probability integral
transform (0.5 when the centre is right) and, for the portfolio, the CRPS against the
historical simulation and the tail statistic of Acerbi and Székely.

The satellites of a date are estimated on a reduced catalog: the drivers of the portfolio,
the objects above them and the drivers of those objects. A full estimate takes half an hour a
date; the reduced one keeps the hierarchy of the portfolio's own objects but pools fewer
siblings, so its coefficients are close to, not equal to, those of the working version.
The estimates are kept under ``<models>/bench-forecast/`` by the digest of their inputs.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from mst import models as models_pkg
from mst import store as files
from mst.drivers import bundle as satellite_bundle
from mst.macro import version as macro_version

from . import history
from .pipeline import Replay, forecast
from .rows import Row, info, inside

MACRO_MODEL = "rw.joint"


def reduced_catalog(catalog: dict, drivers: list[str]) -> dict:
    """The catalog of ``drivers``: their objects, every object above them, and the drivers of
    those objects; quantities with none of them are left out."""
    wanted: dict[str, set[str]] = {}
    for driver in drivers:
        quantity, object_ = driver.split(":", 1)
        wanted.setdefault(quantity, set()).add(object_)
    found = {"quantities": {}, "updated": catalog.get("updated")}
    for quantity, objects in wanted.items():
        part = catalog["quantities"][quantity]
        parent = {item["id"]: item.get("parent") for item in part["objects"]}
        kept = set()
        for object_ in objects:
            while object_ is not None and object_ not in kept:
                kept.add(object_)
                object_ = parent.get(object_)
        found["quantities"][quantity] = {
            "drivers": [d for d in part["drivers"] if d in kept],
            "objects": [item for item in part["objects"] if item["id"] in kept],
            "sources": {k: v for k, v in part["sources"].items() if k in kept},
        }
    return found


LEAST_MONTHS = 48
"""The observed months a driver needs on a date: four years. A satellite holds out its last
two years and needs more months before them than it has coefficients, so a shorter series
gives no estimate; a quantity left without drivers would stop the whole estimate."""


def known_catalog(catalog: dict, store, as_of: date) -> dict:
    """The catalog of the drivers observed in at least ``LEAST_MONTHS`` months by ``as_of``:
    a share listed later is not a driver on an earlier date, and a quantity left without
    drivers is left out."""
    from mst.drivers.objects import series_id

    found = {"quantities": {}, "updated": catalog.get("updated")}
    for quantity, part in catalog["quantities"].items():
        drivers = [
            d
            for d in part["drivers"]
            if len(store.as_of(series_id(quantity, d), as_of)) >= LEAST_MONTHS
        ]
        if drivers:
            found["quantities"][quantity] = part | {"drivers": drivers}
    return found


def _digest(*parts) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(json.dumps(part, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:16]


def _code_digest(model) -> str:
    """The digest of the source and the specification of a model's package."""
    import inspect

    folder = Path(inspect.getfile(type(model))).parent
    digest = hashlib.sha256()
    for file in sorted([*folder.glob("*.py"), *folder.glob("*.toml")]):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Origin:
    month: date
    """Month 0 of the forecast: the last month of the history it reads."""
    as_of: date
    macro: str
    satellites: str


def as_of_of(month: date) -> date:
    """The day the forecast of month 0 ``month`` is made: the 15th of the next month, when the
    market series of the month are out and the monthly statistics mostly are."""
    following = history.from_index(history.month_index(month) + 1)
    return following + timedelta(days=14)


def estimate(month: date, *, store: Path, root: Path, catalog: dict) -> Origin:
    """The macro and the reduced satellite versions of one date, estimated or found again."""
    from mst.data.store import SeriesStore

    day = as_of_of(month)
    catalog_file = root / "catalogs" / f"{day.isoformat()}.json"
    files.write_json(catalog_file, known_catalog(catalog, SeriesStore(store), day))
    macro = models_pkg.estimate_macro(MACRO_MODEL, store=store, models=root, as_of=day)
    code = sorted((m, _code_digest(model)) for m, model in models_pkg.satellites().items())
    key = _digest(macro, files.read_json(catalog_file), code)
    pointer = root / f"satellites-{key}"
    if pointer.exists() and (root / "satellites" / pointer.read_text().strip()).is_dir():
        found = pointer.read_text().strip()
    else:
        found, _ = models_pkg.estimate_satellites(
            macro, store=store, models=root, drivers=catalog_file, root=root
        )
        pointer.write_text(found + "\n", encoding="utf-8")
    return Origin(month, day, macro, found)


def _crps(sample: np.ndarray, fact: float) -> float:
    """The CRPS of an empirical distribution: ``E|X - y| - E|X - X'| / 2``."""
    x = np.sort(sample[np.isfinite(sample)])
    n = len(x)
    if not n:
        return float("nan")
    spread = 2 * np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / n**2
    return float(np.mean(np.abs(x - fact)) - spread / 2)


@dataclass
class Cells:
    """The facts of every date and horizon with their forecast quantiles, by level."""

    pit: dict
    hit: dict

    @classmethod
    def empty(cls):
        return cls({}, {})

    def add(self, key, sample: np.ndarray, fact: float, quantiles) -> None:
        sample = sample[np.isfinite(sample)]
        if not np.isfinite(fact) or not len(sample):
            return
        low, high = np.quantile(sample, quantiles)
        self.pit.setdefault(key, []).append(float(np.mean(sample <= fact)))
        self.hit.setdefault(key, []).append(bool(low <= fact <= high))


def _positions_served(portfolio, satellites, month, horizon):
    """The positions the satellites of a date can value: a share listed later is left out."""
    from .pipeline import _inputs

    kept = []
    for position in portfolio.positions:
        try:
            _inputs(position, satellites, month, horizon, portfolio.base_currency)
        except ValueError:
            continue
        kept.append(position)
    return portfolio.model_copy(update={"positions": kept})


def run(plan, *, workspace: Path, current_macro, current_satellites, portfolio, replay: Replay,
        store, progress=print) -> tuple[list[Row], list[dict]]:
    """The out-of-sample rows of every level and a record of every date."""
    from mst import plugins

    if plan.rolling is None:
        return [], []
    root = Path(workspace) / "models" / "bench-forecast"
    root.mkdir(parents=True, exist_ok=True)
    drivers = sorted(
        {d for p in portfolio.positions for d in _drivers_of(p, current_satellites, portfolio)}
    )
    catalog = reduced_catalog(
        files.read_json(Path(workspace) / "drivers" / "catalog.json"), drivers
    )
    installed = plugins.registry()
    series_root = Path(workspace) / "series"
    macro_cells, share_cells, portfolio_cells = Cells.empty(), Cells.empty(), Cells.empty()
    simulation_cells = Cells.empty()
    crps_model, crps_base, tail, dates = {}, {}, {}, []
    panel = current_macro.panel
    # The facts are read as the working version reads them: its vintage of the history.
    shares = {
        driver: history.monthly(
            store, history.driver_series(current_satellites.registry, driver), current_macro.as_of
        )
        for driver in drivers
        if driver.startswith("share_price:")
    }
    for month in plan.rolling.origins():
        origin = estimate(month, store=series_root, root=root, catalog=catalog)
        macro = macro_version.load(root, origin.macro, installed.macro)
        satellites = satellite_bundle.load(root, origin.satellites, installed.satellites)
        start = macro.panel.months[-1]
        held = _positions_served(portfolio, satellites, origin.as_of, plan.horizon)
        found = forecast(
            macro, satellites, held, draws=plan.rolling.draws, seed=plan.seed,
            horizon=plan.horizon,
        )
        realized = replay.returns([p.id for p in held.positions])
        window = replay.window(start) if start in replay.starts else None
        for h in plan.horizons:
            # Macro: the level h months after month 0, today's vintage of the history.
            fact_row = _panel_row(panel, history.from_index(history.month_index(start) + h))
            for k, variable in enumerate(macro.ids):
                if fact_row is None or variable not in panel.ids:
                    continue
                fact = fact_row[panel.ids.index(variable)]
                macro_cells.add((h, variable), found.paths.shadow[:, h, k], fact, plan.quantiles)
            for k, driver in enumerate(found.paths.drivers.ids):
                if driver not in shares:
                    continue
                end = shares[driver].get(history.from_index(history.month_index(start) + h))
                if end is None:
                    continue
                share_cells.add(
                    (h, driver), found.paths.drivers.paths[:, h, k], end, plan.quantiles
                )
            if window is None:
                continue
            fact = realized[window, h - 1]
            sample = found.returns[:, h - 1]
            portfolio_cells.add(h, sample, fact, plan.quantiles)
            if np.isfinite(fact):
                past = _known_before(replay, held, start, h)
                simulation_cells.add(h, past, fact, plan.quantiles)
                crps_model.setdefault(h, []).append(_crps(sample, fact))
                crps_base.setdefault(h, []).append(_crps(past, fact))
                if h == plan.horizon:
                    tail.setdefault(h, []).append((fact, sample))
        band = {}
        if window is not None:
            for h in plan.horizons:
                low, mid, high = np.quantile(found.returns[:, h - 1], [*plan.quantiles[:1], 0.5,
                                                                      plan.quantiles[1]])
                band[str(h)] = {
                    "low": float(low), "median": float(mid), "high": float(high),
                    "fact": _number(realized[window, h - 1]),
                }
        dates.append(
            {
                "month": start.isoformat(),
                "portfolio": band,
                "as_of": origin.as_of.isoformat(),
                "macro": origin.macro,
                "satellites": origin.satellites,
                "positions": [p.id for p in held.positions],
                "realized_12m": None if window is None else _number(realized[window, -1]),
            }
        )
        progress(f"rolling {start.isoformat()}: {len(held.positions)} positions")
    rows = _rows(plan, macro_cells, share_cells, portfolio_cells, crps_model, crps_base, tail)
    for h in plan.horizons:
        hits = simulation_cells.hit.get(h, [])
        rows.append(info("portfolio", "coverage of the historical simulation", "return", h,
                         np.mean(hits) if hits else None, plan.band, len(hits)))
    return rows, dates


def _number(value) -> float | None:
    return float(value) if np.isfinite(value) else None


def _drivers_of(position, satellites, portfolio) -> list[str]:
    from .pipeline import _inputs

    return _inputs(position, satellites, satellites.as_of, 12, portfolio.base_currency)[1]


def _panel_row(panel, month: date):
    return panel.y[panel.months.index(month)] if month in panel.months else None


def _known_before(replay: Replay, held, start: date, h: int) -> np.ndarray:
    """The historical simulation of a date: the returns of the windows that ended by then."""
    returns = replay.returns([p.id for p in held.positions])[:, h - 1]
    ended = np.array(
        [history.month_index(s) + h <= history.month_index(start) for s in replay.starts]
    )
    return returns[ended]


def _rows(plan, macro_cells, share_cells, portfolio_cells, crps_model, crps_base, tail):
    rows = []
    nominal = plan.band
    for level, cells in (("macro", macro_cells), ("shares", share_cells)):
        for h in plan.horizons:
            hits = [hit for (k, _), found in cells.hit.items() if k == h for hit in found]
            rows.append(
                inside(level, "coverage out of sample", "all series", h,
                       np.mean(hits) if hits else None, nominal, len(hits), plan.gates.coverage)
            )
        for (h, item), found in sorted(cells.hit.items()):
            if h == plan.horizon:
                rows.append(
                    info(level, "coverage out of sample", item, h, np.mean(found), nominal,
                         len(found))
                )
    for h in plan.horizons:
        hits = portfolio_cells.hit.get(h, [])
        rows.append(
            inside("portfolio", "coverage out of sample", "return", h,
                   np.mean(hits) if hits else None, nominal, len(hits), plan.gates.coverage)
        )
        pit = portfolio_cells.pit.get(h, [])
        rows.append(info("portfolio", "mean PIT", "return", h,
                         np.mean(pit) if pit else None, 0.5, len(pit)))
        model, base = crps_model.get(h, []), crps_base.get(h, [])
        rows.append(info("portfolio", "CRPS vs historical simulation", "return", h,
                         np.nanmean(model) if model else None,
                         np.nanmean(base) if base else None, len(model)))
    for h, pairs in tail.items():
        rows.append(info("portfolio", "Acerbi-Szekely Z", "loss", h, _acerbi(pairs), 0.0,
                         len(pairs)))
    return rows


def _acerbi(pairs) -> float:
    """``Z = sum(L 1{L > VaR} / ES) / (0.05 E) - 1`` on losses ``L = -return``."""
    total = 0.0
    for fact, sample in pairs:
        losses = -sample[np.isfinite(sample)]
        var = np.quantile(losses, 0.95)
        es = losses[losses >= var].mean()
        loss = -fact
        if loss > var and es > 0:
            total += loss / es
    return total / (0.05 * len(pairs)) - 1 if pairs else float("nan")
