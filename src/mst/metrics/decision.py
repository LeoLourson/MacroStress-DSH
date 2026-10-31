"""The four-step decision rule: accept or reject a candidate version.

The rule compares two versions on the same cells, pairwise. A candidate that failed admission
is rejected regardless of its scores. A candidate whose CRPS at the target level is better
than the benchmark's and the incumbent's by more than two standard errors — with no admission
share grown — is accepted. A candidate worse by more than two standard errors is rejected.
Between the two, the difference is undetected and the incumbent stays: the default action
changes nothing.

This module holds the rule only: the standard errors are the ones the report's aggregate
bootstrap gives (``mst.metrics.aggregate.aggregate_bootstrap``), and the wild cluster bootstrap
re-checks the verdict. The rule reads the numbers it is handed and decides nothing else.
"""


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
    if any(share >= 0.05 for share in candidate_admission_shares.values()):
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
