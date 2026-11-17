"""The trial of the criteria of trust on synthetic worlds where the correct model is known.

A world repeats the shape of the grid: ``quantities`` series walk for ``history`` months, then
the grid takes ``dates`` calculation dates ``step`` months apart, the last ``validation`` of
them the validation half, and judges a forecast of ``horizon`` months at each. The monthly steps
are multivariate Student with ``nu`` degrees of freedom and a common multiplier, equicorrelated
by ``rho`` and scaled to unit variance; every series drifts by ``drift`` a month. The correct
model draws its paths from the same law; a defective one changes one thing (``DEFECTS``). The
benchmark is the random walk of ``mst.metrics.baselines`` on the history known at the date, and
the portfolio is the mean change of the series. The rows go through the same code as the rows of
the real grid (``mst.metrics.trust``), so the share of the worlds a criterion fails is its false
rejection rate for the correct model and its power against a defect.

The trial covers the criteria the grid reads: the domain, the moves rule, the tails, the
failing quantity, the accuracy and the portfolio's tail. The harm and the usefulness of a
scenario need episodes with a scenario and are not tried here.
"""

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student

from .baselines import DOMAIN, MOVES, random_walk
from .cells import crps, move_ranges, pit, quantiles
from .es import tail as es_tail
from .trust import admission, decision

DEFECTS = {
    "correct": "the law of the world",
    "narrow": "the band around the median times 0.8",
    "wide": "the band around the median times 1.3",
    "shifted": "the median moved by half the standard deviation of a year's change",
    "thin": "normal steps with the same 95th percentile, no common multiplier: a lighter tail",
    "failing": "the band of the first series around its median times 2",
}
"""The model of a world: the correct one and the defects the criteria must find."""
CRITERIA = ("domain", "moves", "tails", "failing", "accuracy", "portfolio_tail")


@dataclass(frozen=True)
class World:
    quantities: int = 23
    dates: int = 55
    validation: int = 23
    step: int = 3
    horizon: int = 14
    history: int = 240
    nu: float = 4.0
    rho: float = 0.3
    drift: float = 0.05
    draws: int = 400
    repetitions: int = 1000


def rows_of(world: World, defect: str, seed: int) -> tuple[list[dict], dict | None]:
    """The grid rows of one world under ``defect`` and the tail of its portfolio."""
    if defect not in DEFECTS:
        raise ValueError(f"unknown defect {defect!r}; known: {', '.join(DEFECTS)}")
    rng = np.random.default_rng(seed)
    length = world.history + world.step * (world.dates - 1) + world.horizon + 1
    y = np.cumsum(world.drift + _steps(world, rng, length, thin=False), axis=0)
    rows, ends = [], []
    for u in range(world.dates):
        now = world.history + world.step * u
        unit = f"{u:04d}"
        thin = defect == "thin"
        paths = y[now] + np.cumsum(
            world.drift
            + _steps(world, rng, world.horizon * world.draws, thin=thin).reshape(
                world.draws, world.horizon, world.quantities
            ),
            axis=1,
        )
        paths = _defective(paths, defect, world)
        for k in range(world.quantities):
            past = y[: now + 1, k]
            walk = random_walk(past, world.horizon, draws=world.draws, seed=seed + u)
            moves = move_ranges(past, world.horizon)
            for t in range(world.horizon):
                draws, fact = paths[:, t, k], float(y[now + 1 + t, k])
                band = quantiles(draws)
                width = float(band[2] - band[0])
                rows.append(
                    {
                        "date": unit,
                        "half": "validation" if u >= world.dates - world.validation else "tuning",
                        "month": t + 1,
                        "quantity": f"q{k:02d}",
                        "level": "macro",
                        "model": crps(draws, fact),
                        "walk": crps(walk[:, t], fact),
                        "conditional_walk": None,
                        "base": None,
                        "pit": pit(draws, fact),
                        "inside": bool(band[0] <= fact <= band[2]),
                        "width": width,
                        "move_range": float(moves[t]),
                        DOMAIN: False,
                        MOVES: None if np.isnan(moves[t]) else bool(width > moves[t]),
                    }
                )
        # The portfolio at the month whose windows do not overlap, as the grid reads it.
        change = (paths[:, world.step - 1, :] - y[now]).mean(axis=1)
        ends.append((unit, -float((y[now + world.step] - y[now]).mean()), -change))
    tail = es_tail(ends, block=1, seed=seed, repetitions=world.repetitions)
    return rows, tail


def verdicts(world: World, defect: str, seed: int) -> dict[str, bool]:
    """Which criteria fail in one world: True where the criterion rejects the model."""
    rows, tail = rows_of(world, defect, seed)
    block = _block(world)
    admitted = admission(rows, unit="date", block=block, calibration=world.step, tail=tail)
    decided = decision([], rows, target="macro", grid_block=block)
    found = {name: bool(entry["failed"]) for name, entry in admitted["criteria"].items()}
    found["accuracy"] = not (decided["accuracy"] and decided["accuracy"]["passed"])
    found["any"] = any(found[name] for name in CRITERIA)
    return found


def trial(
    world: World,
    defects: Sequence[str] = tuple(DEFECTS),
    *,
    worlds: int = 200,
    seed: int = 0,
    workers: int = 1,
) -> dict[str, dict[str, float]]:
    """Per defect, the share of the worlds each criterion and the whole check reject."""
    jobs = [(world, defect, seed + n) for defect in defects for n in range(worlds)]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            found = list(pool.map(_verdicts, jobs, chunksize=4))
    else:
        found = [_verdicts(job) for job in jobs]
    table = {}
    for d, defect in enumerate(defects):
        own = found[d * worlds : (d + 1) * worlds]
        table[defect] = {
            name: float(np.mean([entry[name] for entry in own])) for name in (*CRITERIA, "any")
        }
    return table


def _verdicts(job) -> dict[str, bool]:
    return verdicts(*job)


def _steps(world: World, rng, count: int, *, thin: bool) -> np.ndarray:
    """(count, quantities) monthly steps of unit variance, equicorrelated by ``rho``."""
    k = world.quantities
    cov = np.full((k, k), world.rho) + (1 - world.rho) * np.eye(k)
    normal = rng.multivariate_normal(np.zeros(k), cov, size=count)
    if thin:
        # The same p95 as the Student step of unit variance, so only the tail beyond it is lighter.
        return (
            normal
            * student.ppf(0.95, world.nu)
            * np.sqrt((world.nu - 2) / world.nu)
            / norm.ppf(0.95)
        )
    scale = np.sqrt(world.nu / rng.chisquare(world.nu, size=count) * (world.nu - 2) / world.nu)
    return normal * scale[:, None]


def _defective(paths: np.ndarray, defect: str, world: World) -> np.ndarray:
    """The paths of a defective model: (draws, horizon, quantities)."""
    centre = np.median(paths, axis=0, keepdims=True)
    if defect == "narrow":
        return centre + 0.8 * (paths - centre)
    if defect == "wide":
        return centre + 1.3 * (paths - centre)
    if defect == "shifted":
        return paths + 0.5 * np.sqrt(12.0)
    if defect == "failing":
        paths = paths.copy()
        paths[..., 0] = centre[..., 0] + 2.0 * (paths[..., 0] - centre[..., 0])
    return paths


def _block(world: World) -> int:
    return max(int(np.ceil(world.horizon / world.step)), 1)
