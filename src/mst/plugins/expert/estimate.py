"""The estimate of the expert model: from the answers and the history to a version.

The estimate reads the answers of the date from the elicitation folder, builds the panel
known on the calculation date and takes the starting point, the mean and the covariance of
the monthly changes from it. Every shock's tables and stream statistics are one checkpoint
unit keyed by the answer hash and the spec, so an interrupted estimate started again
resumes; the progress goes to stderr. The residual covariance is the history's change
covariance less what the shocks contribute, and a difference that is not positive
semidefinite refuses the version with the variables that made it so — the check that
stands in for a human approval of the answers.
"""

import shutil
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from mst import inputs as inputs_module
from mst import versions
from mst.macro.panel import Panel, build_panel
from mst.model_sdk.checkpoint import report
from mst.model_sdk.contracts import MacroEstimate

from .answers import HORIZONS, NODES, MarketAnswer, ShockAnswer
from .laws import level_grid, table
from .model import MARKET, RESIDUAL, load_shocks
from .spec import Spec, load_spec
from .stream import ShockTables, StreamStatistics, statistics

#: The longest horizon the model answers: the base path is tabulated for these months.
HORIZON = 12
#: The share of the width beyond which a skew-t fit miss is reported.
MISS_REPORTED = 0.01


class ResidualNotAdmitted(ValueError):
    """The expert attributed more movement to the shocks than history shows."""


@dataclass(frozen=True)
class Expert:
    """The loaded model of a saved version: everything ``simulate`` and the law need."""

    ids: list[str]
    months: list[date]
    shocks: list[str]
    x0: np.ndarray
    base_path: np.ndarray
    chol_u: np.ndarray
    tables: tuple[ShockTables, ...]
    levels: np.ndarray
    candidates: int
    subset_samples: int


def from_history(panel: Panel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(x0, mu, sigma)``: the last value, the mean change and the covariance of changes.

    A change counts only when both of its ends are known; the covariance of a pair of
    variables counts the months where both of their changes are known.
    """
    y = np.asarray(panel.y, dtype=float)
    known = np.isfinite(y)
    last = np.array(
        [
            y[known[:, k], k][-1] if known[:, k].any() else np.nan
            for k in range(y.shape[1])
        ]
    )
    changes = np.diff(y, axis=0)
    mu = np.zeros(y.shape[1])
    sigma = np.zeros((y.shape[1], y.shape[1]))
    for i in range(y.shape[1]):
        finite = np.isfinite(changes[:, i])
        if finite.any():
            mu[i] = changes[finite, i].mean()
        for j in range(i, y.shape[1]):
            rows = finite & np.isfinite(changes[:, j])
            if rows.sum() >= 3:
                sigma[i, j] = sigma[j, i] = np.cov(changes[rows, i], changes[rows, j])[0, 1]
    return last, mu, sigma


def estimate(context, *, base: str, spec: Spec | None = None) -> MacroEstimate:
    """The metadata and arrays of the version, from the answers of the date and history."""
    spec = spec or load_spec()
    answers = read_answers(context, base)
    hashes = {key: inputs_module.hash_of(normal) for key, normal in answers.items()}
    panel = build_panel(
        context.store,
        context.registry,
        start=spec.history_start,
        as_of=context.as_of,
        seasonal=[v.id for v in context.registry.variables if v.seasonal],
    )
    x0, mu, sigma_hat = from_history(panel)
    unknown = [panel.ids[k] for k in np.flatnonzero(~np.isfinite(x0))]
    if unknown:
        raise ValueError(
            f"no value of {', '.join(unknown)} is known on {context.as_of.isoformat()}"
        )
    n = len(panel.ids)
    shocks = load_shocks()

    responses = np.empty((len(shocks), n, len(HORIZONS), len(NODES), spec.levels))
    medians = np.empty((len(shocks), n, len(HORIZONS), len(NODES)))
    size_tables = np.empty((len(shocks), spec.levels))
    size_nodes = np.empty((len(shocks), len(NODES)))
    frequency = np.empty(len(shocks))
    rho = np.empty(len(shocks))
    misses = np.empty((len(shocks), n, len(HORIZONS), len(NODES)))
    sigma_shocks = np.zeros((n, n))
    mu_shocks = np.zeros(n)
    expected = np.zeros((HORIZON + 1, n))
    for index, shock in enumerate(shocks):
        report(f"expert: tables and stream of {shock.id}")
        unit = _resume(context.checkpoints, hashes[shock.id], spec)
        if unit is None:
            unit = _shock_unit(shock, answers[shock.id], panel.ids, spec, index)
            _keep(context.checkpoints, hashes[shock.id], spec, unit)
        tables, miss, stats = unit
        responses[index] = tables.responses
        medians[index] = tables.medians
        size_tables[index] = tables.size_table
        size_nodes[index] = tables.sizes
        frequency[index] = tables.frequency
        rho[index] = tables.rho
        misses[index] = miss
        sigma_shocks += stats.cov_change
        mu_shocks += stats.mean_change
        expected[1:] += np.cumsum(stats.mean_response, axis=0)

    sigma_u = sigma_hat - sigma_shocks
    smallest = np.linalg.eigvalsh(sigma_u)
    if smallest[0] < -1e-10 * max(smallest[-1], 1e-300):
        raise ResidualNotAdmitted(_residual_problem(panel.ids, sigma_u))
    chol_u = _cholesky(sigma_u)

    drift = mu - mu_shocks
    base_path = x0[None, :] + np.arange(HORIZON + 1)[:, None] * drift[None, :]
    if base == "market":
        for variable, path in MarketAnswer.model_validate(answers[MARKET]).values.items():
            column = panel.ids.index(variable)
            nodes = np.array([path.horizons[horizon] for horizon in HORIZONS])
            market = np.interp(
                np.arange(HORIZON + 1), np.array(HORIZONS), nodes, left=x0[column]
            )
            market[0] = x0[column]
            base_path[:, column] = market - expected[:, column]

    fit_misses = []
    for index, shock in enumerate(shocks):
        for k, variable in enumerate(panel.ids):
            for h, horizon in enumerate(HORIZONS):
                for s, node in enumerate(NODES):
                    if misses[index, k, h, s] > MISS_REPORTED:
                        fit_misses.append(
                            {
                                "shock": shock.id,
                                "variable": variable,
                                "horizon": horizon,
                                "size": node,
                                "miss": float(misses[index, k, h, s]),
                            }
                        )
    return MacroEstimate(
        meta={
            "as_of": context.as_of.isoformat(),
            "ids": panel.ids,
            "n_global": panel.n_global,
            "months": [month.isoformat() for month in panel.months],
            "shocks": [shock.id for shock in shocks] + [RESIDUAL],
            "base": base,
            "family": spec.family,
            "alpha": spec.alpha,
            "answers": hashes,
            "fit_misses": fit_misses,
            "spec": spec.model_dump(mode="json"),
        },
        arrays={
            "panel_y": panel.y,
            "panel_approximate": panel.approximate,
            "x0": x0,
            "base_path": base_path,
            "chol_u": chol_u,
            "levels": level_grid(spec.levels),
            "responses": responses,
            "medians": medians,
            "size_tables": size_tables,
            "size_nodes": size_nodes,
            "frequency": frequency,
            "rho": rho,
        },
    )


def load(meta: dict, arrays: dict) -> Expert:
    """The model of a saved version; only the plugin that estimated it reads it back."""
    spec = Spec.model_validate(meta["spec"])
    levels = arrays["levels"]
    tables = tuple(
        ShockTables(
            frequency=float(arrays["frequency"][index]),
            rho=float(arrays["rho"][index]),
            sizes=arrays["size_nodes"][index],
            size_table=arrays["size_tables"][index],
            responses=arrays["responses"][index],
            medians=arrays["medians"][index],
            levels=levels,
            alpha=spec.alpha,
        )
        for index in range(len(meta["shocks"]) - 1)
    )
    return Expert(
        ids=list(meta["ids"]),
        months=[date.fromisoformat(month) for month in meta["months"]],
        shocks=list(meta["shocks"]),
        x0=arrays["x0"],
        base_path=arrays["base_path"],
        chol_u=arrays["chol_u"],
        tables=tables,
        levels=levels,
        candidates=spec.candidates,
        subset_samples=spec.subset_samples,
    )


def read_answers(context, base: str) -> dict[str, dict]:
    """The answers of the date in their normal form, by key; missing keys are refused."""
    if context.inputs is None:
        raise ValueError(
            "the expert model reads the answers of the elicitation folder: "
            "pass --inputs (the server does)"
        )
    keys = tuple(shock.id for shock in load_shocks())
    if base == "market":
        keys = (*keys, MARKET)
    written = inputs_module.read(context.inputs, context.as_of)
    missing = [key for key in keys if key not in written]
    if missing:
        raise ValueError(
            f"no answer written for {', '.join(missing)} on {context.as_of.isoformat()}: "
            "answer with model(kind='inputs', action='write') first"
        )
    return {key: written[key] for key in keys}


def _shock_unit(shock, answer: dict, ids: list[str], spec: Spec, index: int):
    """The tables, the fit misses and the stream statistics of one shock."""
    parsed = ShockAnswer.model_validate(answer)
    cells = [
        [
            [parsed.responses[variable][horizon][node].q5 for node in NODES]
            for horizon in HORIZONS
        ]
        for variable in ids
    ]
    q5 = np.array(cells)
    q50 = np.array(
        [
            [
                [parsed.responses[variable][horizon][node].q50 for node in NODES]
                for horizon in HORIZONS
            ]
            for variable in ids
        ]
    )
    q95 = np.array(
        [
            [
                [parsed.responses[variable][horizon][node].q95 for node in NODES]
                for horizon in HORIZONS
            ]
            for variable in ids
        ]
    )
    levels, values, miss = table(q5.ravel(), q50.ravel(), q95.ravel(), spec)
    _, size_values, _ = table([parsed.size.q5], [parsed.size.q50], [parsed.size.q95], spec)
    tables = ShockTables(
        frequency=parsed.frequency,
        rho=parsed.rho,
        sizes=np.array([parsed.size.q5, parsed.size.q50, parsed.size.q95]),
        size_table=size_values[0],
        responses=values.reshape(len(ids), len(HORIZONS), len(NODES), len(levels)),
        medians=q50,
        levels=levels,
        alpha=spec.alpha,
    )
    stats = statistics(tables, spec.stream_draws, np.random.default_rng([spec.seed, index]))
    return tables, miss.reshape(tables.medians.shape), stats


def _resume(folder: Path | None, answer_hash: str, spec: Spec):
    """The checkpoint unit of one shock, or None when there is none to resume from."""
    if folder is None:
        return None
    path = folder / _key(answer_hash, spec)
    if not (path / "arrays.npz").is_file():
        return None
    with np.load(path / "arrays.npz") as found:
        arrays = {name: found[name] for name in found.files}
    tables = ShockTables(
        frequency=float(arrays["frequency"]),
        rho=float(arrays["rho"]),
        sizes=arrays["sizes"],
        size_table=arrays["size_table"],
        responses=arrays["responses"],
        medians=arrays["medians"],
        levels=level_grid(spec.levels),
        alpha=spec.alpha,
    )
    return (
        tables,
        arrays["misses"],
        StreamStatistics(
            mean_change=arrays["mean_change"],
            cov_change=arrays["cov_change"],
            mean_response=arrays["mean_response"],
        ),
    )


def _keep(folder: Path | None, answer_hash: str, spec: Spec, unit) -> None:
    """Keep a finished shock under its key; written whole or not at all."""
    if folder is None:
        return
    tables, miss, stats = unit
    folder.mkdir(parents=True, exist_ok=True)
    draft = Path(tempfile.mkdtemp(dir=folder, prefix=".draft-"))
    try:
        np.savez(
            draft / "arrays.npz",
            frequency=tables.frequency,
            rho=tables.rho,
            sizes=tables.sizes,
            size_table=tables.size_table,
            responses=tables.responses,
            medians=tables.medians,
            misses=miss,
            mean_change=stats.mean_change,
            cov_change=stats.cov_change,
            mean_response=stats.mean_response,
        )
        versions._publish(draft, folder / _key(answer_hash, spec))
    except BaseException:
        shutil.rmtree(draft, ignore_errors=True)
        raise


def _key(answer_hash: str, spec: Spec) -> str:
    """The name of the checkpoint unit of one shock: its answer and the spec."""
    return versions.digest(
        {"answer": answer_hash, "spec": spec.model_dump(mode="json")}, {}
    )


def _residual_problem(ids: list[str], sigma_u: np.ndarray) -> str:
    """The message of a refused residual: the variables that made it not semidefinite."""
    negative = [ids[k] for k in np.flatnonzero(np.diag(sigma_u) < 0)]
    if not negative:
        _, vectors = np.linalg.eigh(sigma_u)
        negative = [
            ids[k] for k in np.flatnonzero(np.abs(vectors[:, 0]) > 0.1)
        ]
    return (
        "the answers attribute more movement to the shocks than history shows: the "
        "residual covariance is negative on "
        f"{', '.join(negative) or 'no single variable'}"
    )


def _cholesky(sigma: np.ndarray) -> np.ndarray:
    """The Cholesky factor of the residual covariance, with a relative jitter if singular."""
    try:
        return np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        jitter = 1e-12 * max(np.trace(sigma) / len(sigma), 1e-300)
        return np.linalg.cholesky(sigma + jitter * np.eye(len(sigma)))
