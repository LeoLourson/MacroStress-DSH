"""The four-step decision rule: accept or reject a candidate version.

The rule compares two versions on the same cells, pairwise. A candidate that failed admission
is rejected regardless of its scores. A candidate whose CRPS at the target level is better
than the benchmark's and the incumbent's by more than two standard errors — with no admission
share grown — is accepted. A candidate worse by more than two standard errors is rejected.
Between the two, the difference is undetected and the incumbent stays: the default action
changes nothing.

``decide`` holds the rule and reads the numbers it is handed. ``on_rows`` feeds it from the
scored cells of two versions, the same way for the episodes and for the rolling grid: the score
is the logarithm of the relative CRPS (``mst.metrics.quality``), so the benchmark sits at zero,
and the standard errors are those of the logarithm over resampled units — episodes or blocks of
dates. The admission shares are the worst quantity's (``mst.metrics.sanity``).
"""

from collections.abc import Mapping, Sequence

import numpy as np

from .quality import interval, paired, relative
from .sanity import ADMISSION_SHARE, worst_shares


def decide(
    *,
    candidate_crps: float,
    benchmark_crps: float,
    incumbent_crps: float | None,
    se_vs_benchmark: float,
    se_vs_incumbent: float | None,
    candidate_admission_shares: dict[str, float],
    incumbent_admission_shares: dict[str, float] | None,
) -> dict:
    """The four steps of the decision rule, on the aggregates of the target level.

    ``*_admission_shares`` maps each admission rule to its share of broken cells; a candidate
    share above the 5 % threshold fails step 1 on its own, and step 2 additionally demands
    that no share grew against the incumbent — the plan's assumption: not worse, without noise
    allowance. The SEs come from ``paired_se``; for the first version there is no incumbent
    and step 2 compares against the benchmark only.
    """
    verdict: dict = {
        "candidate_crps": candidate_crps,
        "benchmark_crps": benchmark_crps,
        "incumbent_crps": incumbent_crps,
        "se_vs_benchmark": se_vs_benchmark,
        "se_vs_incumbent": se_vs_incumbent,
        "admission_shares": dict(candidate_admission_shares),
        "threshold": 2.0,
    }
    if any(share >= ADMISSION_SHARE for share in candidate_admission_shares.values()):
        verdict.update(
            verdict="rejected_admission",
            step=1,
            note="the candidate failed admission; CRPS is not read",
        )
        return verdict
    better_benchmark = candidate_crps < benchmark_crps - 2.0 * se_vs_benchmark
    if incumbent_crps is None:
        better_incumbent = True
        worse_incumbent = False
    else:
        better_incumbent = candidate_crps < incumbent_crps - 2.0 * se_vs_incumbent
        worse_incumbent = candidate_crps > incumbent_crps + 2.0 * se_vs_incumbent
    worse_benchmark = candidate_crps > benchmark_crps + 2.0 * se_vs_benchmark
    shares = dict(candidate_admission_shares)
    if incumbent_admission_shares is not None:
        grown = {
            rule: candidate
            for rule, candidate in shares.items()
            if candidate > incumbent_admission_shares.get(rule, candidate)
        }
    else:
        grown = {}
    verdict["admission_grown"] = sorted(grown)
    if better_benchmark and better_incumbent and not grown:
        verdict.update(
            verdict="accepted",
            step=2,
            note=(
                "better than the benchmark and the incumbent by more than two SE, "
                "admission not worse"
            ),
        )
        return verdict
    if worse_benchmark or worse_incumbent:
        verdict.update(
            verdict="rejected_worse",
            step=3,
            note="worse by more than two SE",
        )
        return verdict
    verdict.update(
        verdict="inconclusive",
        step=4,
        note="the difference is undetected; the incumbent stays",
    )
    return verdict


def on_rows(
    mine: Sequence[Mapping[str, object]],
    theirs: Sequence[Mapping[str, object]],
    *,
    unit: str,
    benchmark: str,
    block: int,
    seed: int,
    admission: Mapping[str, float],
    incumbent_admission: Mapping[str, float],
) -> dict:
    """The verdict on the candidate's rows against the incumbent's, cell for cell.

    ``unit`` names the key of the resampled unit (``episode`` or ``date``), ``benchmark`` the
    CRPS column both versions are measured against. Two versions judged on different cells are
    an error naming the cells: a comparison on the intersection would hide what one of them
    could not forecast.
    """

    def key(row):
        return (str(row[unit]), int(row["month"]), str(row["quantity"]))

    difference = {key(row) for row in mine} ^ {key(row) for row in theirs}
    if difference:
        sample = ", ".join(f"{u}/{q}/t{t}" for u, t, q in sorted(difference)[:10])
        more = "..." if len(difference) > 10 else ""
        raise ValueError(f"the two reports do not share {len(difference)} cells: {sample}{more}")
    versus_benchmark = interval(mine, "model", benchmark, unit=unit, block=block, seed=seed)
    if versus_benchmark["point"] is None:
        raise ValueError(f"no cell has both the model and the {benchmark} scored")
    versus_incumbent = paired(mine, theirs, "model", unit=unit, block=block, seed=seed)
    incumbent = relative(theirs, "model", benchmark)["score"]
    # A single unit leaves the error of a score unknown: an infinite SE detects no difference.
    verdict = decide(
        candidate_crps=float(np.log(versus_benchmark["point"])),
        benchmark_crps=0.0,
        incumbent_crps=None if incumbent is None else float(np.log(incumbent)),
        se_vs_benchmark=_known(versus_benchmark["log_se"]),
        se_vs_incumbent=_known(versus_incumbent["log_se"]),
        candidate_admission_shares=admission,
        incumbent_admission_shares=incumbent_admission,
    )
    verdict.update(
        benchmark=benchmark,
        cells=len(mine),
        vs_benchmark=versus_benchmark,
        vs_incumbent=versus_incumbent,
    )
    return verdict


def _known(se: float | None) -> float:
    return np.inf if se is None else float(se)


def verdict(
    rows: Sequence[Mapping[str, object]],
    admission: Mapping,
    incumbent: Mapping,
    *,
    unit: str,
    level: str,
    block: int,
    seed: int,
) -> dict:
    """The verdict of a check against the incumbent's details, at ``level``.

    The comparison reads the validation half when the check has one, and all windows
    otherwise. The benchmark is the walk that knows the scenario's path where the level has
    one, since it knows what the forecast knows, and the plain walk otherwise. ``admission``
    is the candidate's (``mst.metrics.sanity``); the incumbent's is read from its report.
    """
    if "rows" not in incumbent:
        raise ValueError("the incumbent carries no cells: pass the --details file of its check")
    half = "validation" if any(row["half"] == "validation" for row in rows) else "all"

    def pick(found):
        return [
            row for row in found if row["level"] == level and (half == "all" or row["half"] == half)
        ]

    mine, theirs = pick(rows), pick(incumbent["rows"])
    if not mine:
        raise ValueError(f"level {level!r} has no scored cells to decide on")
    benchmark = (
        "conditional_walk" if any(row["conditional_walk"] is not None for row in mine) else "walk"
    )
    found = on_rows(
        mine,
        theirs,
        unit=unit,
        benchmark=benchmark,
        block=block,
        seed=seed,
        admission=worst_shares(admission),
        incumbent_admission=worst_shares(incumbent["sanity"]["admission"]),
    )
    found.update(level=level, half=half)
    return found
