"""Default structural form: replace cells without feedback within their level."""

from dataclasses import replace

import numpy as np

from .contracts import Intervention, LinearConstraint
from .dist import ContradictsModels


def inner_of_groups(simulation) -> int:
    """Derive the uniform contiguous group size; missing labels mean one path per group."""
    count = simulation.paths.shape[0]
    group = simulation.group
    if group is None:
        return 1
    labels = np.asarray(group)
    if labels.shape != (count,) or labels.dtype.kind not in "iu" or count == 0:
        raise ValueError("group labels must be one integer per path")
    if np.any(labels < 0):
        raise ValueError("group labels must be nonnegative")
    counts = np.bincount(labels)
    inner = int(counts[0])
    if inner < 1 or np.any(counts != inner):
        raise ValueError("groups must have equal nonzero counts")
    if not np.array_equal(labels, np.repeat(np.arange(len(counts)), inner)):
        raise ValueError("groups must occupy contiguous blocks numbered from zero")
    return inner


def _index(interventions, ids):
    known = set(ids)
    found = {}
    for item in interventions:
        if item.target not in known:
            raise ValueError(f"an intervention names unknown quantity {item.target}")
        for position, month in enumerate(item.months):
            key = (item.target, month)
            if key in found:
                raise ValueError(f"more than one intervention on {item.target} in month {month}")
            found[key] = (item, position)
    return found


def apply_default(paths, ids, interventions, *, inner: int = 1) -> np.ndarray:
    """Return model-scale paths after per-group set/shift, without drawing or mutating input.

    Group ``r`` occupies ``r * inner .. (r + 1) * inner - 1``. Intervention arrays carry
    exactly one row per group; no dimension is implicitly broadcast. Unmentioned cells,
    including month zero, retain their original bits.
    """
    paths = np.asarray(paths)
    if paths.ndim != 3 or paths.shape[2] != len(ids):
        raise ValueError("paths must have shape (groups * inner, horizon + 1, len(ids))")
    if type(inner) is not int or inner < 1 or paths.shape[0] % inner:
        raise ValueError("the path count must be divisible by a positive integer inner")
    interventions = tuple(interventions)
    _index(interventions, ids)
    groups = paths.shape[0] // inner
    for item in interventions:
        if item.values.shape[0] != groups:
            raise ValueError("intervention groups do not match paths and inner")
        if max(item.months) >= paths.shape[1]:
            raise ValueError("an intervention month exceeds the path horizon")
    if not interventions:
        return paths
    result = paths.copy()
    index = {target: column for column, target in enumerate(ids)}
    for item in interventions:
        column = index[item.target]
        for position, month in enumerate(item.months):
            active = (
                np.ones(groups, dtype=bool) if item.active is None else item.active[:, position]
            )
            rows = np.repeat(active, inner)
            values = np.repeat(item.values[:, position], inner)[rows]
            if item.kind == "set":
                result[rows, month, column] = values
            else:
                result[rows, month, column] += values
    return result


def affine_constraints_under_default(
    constraints, interventions, ids, *, group: int = 0
) -> tuple[tuple[LinearConstraint, ...], np.ndarray]:
    """Original functional rows with set terms removed, plus their constant offsets.

    Bounds, row order and empty rows remain intact. A model constructs the law of each
    retained functional and adds its offset to the mean, keeping the original event box.
    This is also the arithmetic used by ``constraints_under_default`` for sampling.
    """
    interventions = tuple(interventions)
    lookup = _index(interventions, ids)
    if type(group) is not int or group < 0:
        raise ValueError("group must be a nonnegative integer")
    if any(group >= item.values.shape[0] for item in interventions):
        raise ValueError("the requested group is absent from intervention values")
    rows, offsets = [], []
    for constraint in constraints:
        terms, offset = [], 0.0
        for term in constraint.terms:
            hit = lookup.get((term.variable, term.month))
            if hit is None:
                terms.append(term)
                continue
            item, position = hit
            if item.active is not None and not item.active[group, position]:
                terms.append(term)
                continue
            offset += term.weight * item.values[group, position]
            if item.kind == "shift":
                terms.append(term)
        rows.append(replace(constraint, terms=tuple(terms)))
        offsets.append(offset)
    return tuple(rows), np.asarray(offsets, dtype=float)


def constraints_under_default(
    constraints, interventions: tuple[Intervention, ...], ids, *, group: int = 0
) -> tuple[LinearConstraint, ...]:
    """The pre-intervention conditions for one group; true constant rows are removed.

    Fully determined false rows contradict every model before drawing. Error numbers refer
    to the original constraints, starting at one, even when earlier true rows disappeared.
    """
    rows, offsets = affine_constraints_under_default(constraints, interventions, ids, group=group)
    result = []
    for number, (row, offset) in enumerate(zip(rows, offsets, strict=True), start=1):
        shifted = replace(row, low=row.low - offset, high=row.high - offset)
        if shifted.terms:
            result.append(shifted)
        elif not shifted.low <= 0.0 <= shifted.high:
            raise ContradictsModels((number,), group=group)
    return tuple(result)
