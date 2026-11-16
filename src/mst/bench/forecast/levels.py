"""The base forecast against history at each level: its width, its links and its scale.

These checks read one forecast made on today's data and the whole history behind it, so they
say whether the forecast is as wide and as linked as the past, not whether it was right on
any date; the check out of sample, ``python -m mst.metrics rolling``, answers that.
"""

import numpy as np

from mst.portfolio.history import Replay, driver_series, monthly

from . import history
from .pipeline import Forecast
from .rows import Row, distance, info, ratio

SHARES = "share_price"


def _spliced(plan, variable: str) -> set:
    return {item.month for item in plan.splices if variable in item.variables}


def macro_width(plan, macro, forecast: Forecast) -> list[Row]:
    """The quantile width of every macro variable's h-month change, model over history.

    Both sides are in the model scale: rates in points, prices and indexes as ``100 ln X``.
    The model side is the shadow path before the registry's floors, as the model draws it.
    """
    shadow = forecast.paths.shadow
    panel = macro.panel
    rows = []
    for k, variable in enumerate(panel.ids):
        for h in plan.horizons:
            starts = history.windows(panel.months, h, plan.history_start, _spliced(plan, variable))
            past = history.changes(panel.y[:, k], h, starts)
            model = shadow[:, h, k] - shadow[:, 0, k]
            rows.append(
                ratio(
                    "macro",
                    "width",
                    variable,
                    h,
                    history.width(model, plan.quantiles),
                    history.width(past, plan.quantiles),
                    len(past),
                    plan.gates.width,
                )
            )
    return rows


def _log_driver(values: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(values > 0, 100 * np.log(values), np.nan)


def share_rows(plan, satellites, forecast: Forecast, store, as_of) -> list[Row]:
    """The width of every share's h-month return, the mean pair correlation of the year,
    and the satellite's own out-of-sample check against its AR(1) baseline."""
    drivers = forecast.paths.drivers
    shares = [d for d in drivers.ids if d.split(":", 1)[0] == SHARES]
    if not shares:
        return []
    months = history.grid(plan.history_start, forecast.paths.months[0])
    months = history.grid(history.from_index(history.month_index(months[0]) - 1), months[-1])
    past = {}
    for driver in shares:
        found = monthly(store, driver_series(satellites.registry, driver), as_of)
        past[driver] = _log_driver(np.array([found.get(m, np.nan) for m in months]))
    rows = []
    for h in plan.horizons:
        starts = history.windows(months, h, plan.history_start, set())
        model_returns, past_returns = [], []
        for driver in shares:
            k = drivers.ids.index(driver)
            model = _log_driver(drivers.paths[:, h, k]) - _log_driver(drivers.paths[:, 0, k])
            column = np.array(
                [past[driver][t + h] - past[driver][t] for t in starts], dtype=float
            )
            model_returns.append(model)
            past_returns.append(column)
            rows.append(
                ratio(
                    "shares",
                    "width",
                    driver,
                    h,
                    history.width(model, plan.quantiles),
                    history.width(column, plan.quantiles),
                    int(np.isfinite(column).sum()),
                    plan.gates.width,
                )
            )
        model_corr, _ = history.mean_pair_correlation(np.column_stack(model_returns))
        past_corr, fewest = history.mean_pair_correlation(np.column_stack(past_returns))
        make = distance if h == plan.horizon else _info_distance
        rows.append(
            make(
                "shares",
                "correlation",
                "mean pair",
                h,
                model_corr,
                past_corr,
                fewest,
                plan.gates.correlation,
            )
        )
    for driver in shares:
        quantity, node = satellites.node(driver)
        check = satellites.checks[quantity].get(node)
        if check is not None:
            rows.append(
                info("shares", "holdout RMSE vs AR(1)", driver, None,
                     check.satellite, check.baseline, check.forecasts)
            )
    return rows


def _info_distance(level, check, item, horizon, model, history_value, n, _tolerance) -> Row:
    return info(level, check, item, horizon, model, history_value, n)


def _tail_mean(sample: np.ndarray, share: float) -> float:
    """The mean of the lowest ``share`` of a sample: the expected shortfall of a return."""
    sample = np.sort(sample[np.isfinite(sample)])
    count = max(1, int(np.ceil(share * len(sample))))
    return float(sample[:count].mean()) if len(sample) else float("nan")


def portfolio_rows(plan, forecast: Forecast, replay: Replay) -> list[Row]:
    """The portfolio's return by horizon, model against the same portfolio held over history:
    the quantile width and the mean of the worst 5 % are judged, the centre is shown."""
    rows = []
    low, high = plan.quantiles
    first = history.month_index(plan.history_start) - 1
    recent = np.array([history.month_index(start) >= first for start in replay.starts])
    returns = replay.returns()[recent]
    for h in plan.horizons:
        model = forecast.returns[:, h - 1]
        past = returns[:, h - 1]
        past = past[np.isfinite(past)]
        n = len(past)
        rows.append(
            ratio("portfolio", "width", "q95 - q05, %", h,
                  history.width(model, plan.quantiles), history.width(past, plan.quantiles),
                  n, plan.gates.width)
        )
        rows.append(
            ratio("portfolio", "tail", "ES95 loss, %", h,
                  -_tail_mean(model, 0.05), -_tail_mean(past, 0.05), n, plan.gates.width)
        )
        for name, q in ((f"q{low * 100:02.0f}, %", low), ("median, %", 0.5),
                        (f"q{high * 100:02.0f}, %", high)):
            rows.append(
                info("portfolio", "quantile", name, h,
                     float(np.quantile(model, q)),
                     float(np.quantile(past, q)) if n else None, n)
            )
        rows.append(
            info("portfolio", "mean", "mean, %", h, float(model.mean()),
                 float(past.mean()) if n else None, n)
        )
    return rows
