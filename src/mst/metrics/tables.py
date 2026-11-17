"""The Markdown both checks print: what fell out, the sanity, the scores and the decision.

The episodes and the rolling grid print the same judgement (``mst.metrics.scores.judge``) the
same way, in the order a reader should take it: what fell out of the check first, then the
sanity of every quantity, then the scores.
"""

from collections.abc import Mapping

import numpy as np

from .baselines import DOMAIN, MOVES
from .scores import HORIZONS, PAIRS

PAIR_NAMES = [f"{first}/{second}" for first, second in PAIRS]


def number(value) -> str:
    """A score as three decimals, or a dash when it is missing or non-finite."""
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.3f}"


def interval_text(pair) -> str:
    if pair is None or pair[0] is None:
        return "-"
    low, high = pair
    return f"[{low:.3f}, {high:.3f}]"


def print_dropped(dropped: Mapping, *, judged: int, unit: str) -> None:
    """The windows not judged, with the reason, and the cells left out of the judged ones.

    ``skipped`` windows have too little history for any model; ``refused`` ones had enough,
    and the model or the scenario could not be built on them — the scores never cover them.
    """
    refused, skipped = dropped["refused"], dropped["skipped"]
    print(f"{judged} of {judged + len(refused) + len(skipped)} {unit} judged.")
    if skipped:
        print(f"\nLeft out, too little history for any model: {len(skipped)}\n")
        for name, reason in skipped.items():
            print(f"- {name}: {reason}")
    if refused:
        print(
            f"\n**Refused with enough data: {len(refused)}.** The scores below do not cover them.\n"
        )
        for name, reason in refused.items():
            print(f"- {name}: {reason}")
    if dropped["left_out"]:
        grouped: dict[tuple[str, str], list[str]] = {}
        for name, variables in dropped["left_out"].items():
            for variable, reason in variables.items():
                grouped.setdefault((variable, reason), []).append(name)
        print("\nVariables the model left out as too short, their cells unscored:\n")
        for (variable, reason), names in sorted(grouped.items()):
            print(f"- {variable} in {', '.join(sorted(names))}: {reason}")
    if dropped["excluded"]:
        grouped = {}
        for name, positions in dropped["excluded"].items():
            for position, reason in positions.items():
                grouped.setdefault((position, reason), []).append(name)
        print("\nPositions the engine could not value, their cells unscored:\n")
        for (position, reason), names in sorted(grouped.items()):
            print(f"- {position} in {', '.join(sorted(names))}: {reason}")
    if dropped["non_finite"]:
        print("\nCells whose draws are all non-finite, left out:\n")
        for quantity, count in dropped["non_finite"].items():
            print(f"- {quantity}: {count}")


def print_judgement(found: Mapping, *, title: str, reference_calibration: bool) -> None:
    """The sanity, the scores per level, the halves and the quantities of ``judge``.

    ``reference_calibration`` marks the calibration as a reference only: on selected events
    even a right model need not give the ordinary shares.
    """
    print(f"\n## {title}: sanity of the quantities\n")
    print(
        "Read before any score: a band outside the domain or wider than every move of the "
        "series, or a scenario that moves the forecast away from the fact, is named here."
    )
    _print_sanity(found["sanity"])
    _print_admission(found["admission"])
    print(f"\n## {title}: relative CRPS\n")
    print(
        "Mean CRPS of the first forecast over the second on the same cells, pooled in "
        "logarithms with equal weights over the quantities and the months; below one the first "
        "forecast wins. The interval resamples whole windows."
    )
    print("\n| level | comparison | score | 95 % interval | cells |")
    print("|---|---|---:|---|---:|")
    for level, entry in found["levels"].items():
        for name, pair in entry["pairs"].items():
            bounds = pair["interval"]
            print(
                f"| {level} | {name} | {number(pair['score'])} "
                f"| {interval_text((bounds['low'], bounds['high']))} | {pair['cells']} |"
            )
    note = " (for reference only: the windows are selected events)" if reference_calibration else ""
    print(f"\nCalibration{note}: the shares of the facts below p5, p5..p50, p50..p95, above p95")
    print("(0.05 0.45 0.45 0.05 for a calibrated forecast), and the coverage of p5..p95.\n")
    print(f"| level | all months | {' | '.join(f'month {t}' for t in HORIZONS)} | coverage |")
    print(f"|---|---|{'---|' * len(HORIZONS)}---:|")
    for level, entry in found["levels"].items():
        by_month = " | ".join(_shares(entry["calibration_by_horizon"].get(t)) for t in HORIZONS)
        print(
            f"| {level} | {_shares(entry['calibration'])} | {by_month} | {entry['coverage']:.3f} |"
        )
    print(f"\n## {title}: halves\n")
    print(f"| half | windows | cells | level | {' | '.join(PAIR_NAMES)} | admission |")
    print(f"|---|---:|---:|---|{'---:|' * len(PAIR_NAMES)}---|")
    for name, half in found["halves"].items():
        admitted = "passes" if half["admitted"] else "fails"
        for level, entry in half["levels"].items():
            scores = " | ".join(
                number(entry["pairs"].get(pair, {}).get("score")) for pair in PAIR_NAMES
            )
            print(
                f"| {name} | {len(half['units'])} | {half['cells']} | {level} | {scores} "
                f"| {admitted} |"
            )
    tail = found.get("tail")
    if tail is not None:
        print(f"\n## {title}: tail of the portfolio (Acerbi-Székely)\n")
        low, high = tail["interval"] or (None, None)
        print(
            f"Z {number(tail['z'])}, 95 % interval {number(low)} .. {number(high)}, over "
            f"{len(tail['windows'])} windows; with the heaviest {tail.get('trim', 0.01):.0%} of "
            f"the predictive draws dropped {number(tail['trimmed_z'])}. Zero for a calibrated "
            "tail, above zero when the forecast understates the losses."
        )
        for item in tail["dropped"]:
            print(f"- dropped {item['id']}: {item['reason']}")
    print(f"\n## {title}: quantities\n")
    print(
        f"| quantity | level | scored | constrained | {' | '.join(PAIR_NAMES)} "
        "| shares p5/p50/p95 | coverage | sharpness |"
    )
    print(f"|---|---|---:|---:|{'---:|' * len(PAIR_NAMES)}---|---:|---:|")
    for quantity, entry in found["quantities"].items():
        ratios = " | ".join(number(entry["ratios"].get(name)) for name in PAIR_NAMES)
        tail = ", tail unverified" if entry["unverified_tail"] else ""
        print(
            f"| {quantity}{tail} | {entry['level']} | {entry['scored']} | {entry['constrained']} "
            f"| {ratios} | {_shares(entry['calibration'])} | {number(entry['coverage'])} "
            f"| {number(entry['sharpness'])} |"
        )


def print_decision(found: Mapping) -> None:
    """The replacement rule: the paired ratios of every check and whether the candidate wins."""
    print("\n## Decision against the incumbent\n")
    print(
        f"- level `{found['level']}`; candidate trusted: {found['trusted']}, incumbent trusted: "
        f"{found['incumbent_trusted']}"
    )
    for name, entry in found["checks"].items():
        bounds = interval_text((entry.get("low"), entry.get("high")))
        marks = "better" if entry["better"] else "worse" if entry["worse"] else "no difference"
        print(f"- {name}: candidate/incumbent {number(entry['ratio'])} {bounds} — {marks}")
    verdict = "replaces the incumbent" if found["replaces"] else "the incumbent stays"
    print(f"- verdict: **{verdict}** — {found['reason']}")


def print_trust(found: Mapping) -> None:
    """The verdict of the criteria of trust and the properties of the decision."""
    decided = found["decision"]
    print(f"\n## Trust (level `{decided['target']}`)\n")
    names = {
        "accuracy": "accuracy on the grid, model/benchmark",
        "harm": "harm of the scenario, model/base",
        "usefulness": "usefulness of the scenario (additional), model/conditional_walk",
    }
    for key, label in names.items():
        entry = decided[key]
        if entry is None:
            print(f"- {label}: not checked")
            continue
        state = (
            ("proven" if entry["proven"] else "not proven")
            if key == "usefulness"
            else ("passes" if entry["passed"] else "fails")
        )
        print(
            f"- {label}: {number(entry['statistic'])}, p {number(entry['p'])} over "
            f"{entry['batches']} batches — {state}"
        )
    verdict = "trusted" if found["trusted"] else "not trusted"
    print(f"- verdict: **{verdict}**")
    for reason in found["reasons"]:
        print(f"  - {reason}")


def _print_admission(found: Mapping) -> None:
    """The admission criteria of ``mst.metrics.trust``: what each fails on."""
    print("\n| criterion | tested | untested | failed on |")
    print("|---|---:|---:|---|")
    for name, entry in found["criteria"].items():
        if name == "portfolio_tail":
            _print_portfolio_tail(entry)
            continue
        tests = entry.get("tests")
        tested = "-" if tests is None else str(len(tests) - entry["untested"])
        untested = "-" if tests is None else str(entry["untested"])
        failed = ", ".join(entry["failed"][:12]) + (" ..." if len(entry["failed"]) > 12 else "")
        print(f"| {name} | {tested} | {untested} | {failed or '-'} |")
    verdict = "fails" if found["failed"] else "passes"
    print(f"\nAdmission {verdict}.")


def _print_portfolio_tail(entry: Mapping) -> None:
    """The row of the tail of the portfolio in the admission table."""
    if not entry["checked"]:
        print("| portfolio_tail | - | - | not checked |")
        return
    shown = (
        f"Z {number(entry['z'])} (p {number(entry['p_z'])}), beyond VaR95 "
        f"{number(entry['beyond_var'])} (p {number(entry['p_beyond'])})"
    )
    failed = ", ".join(entry["failed"]) or "-"
    print(f"| portfolio_tail | {entry['windows']} windows | {shown} | {failed} |")


def _print_sanity(found: Mapping) -> None:
    """The per-quantity table of ``mst.metrics.sanity.check``."""
    print(
        "\n| level | quantity | cells | band outside domain | band wider than moves "
        "| band / moves range | toward the fact | benchmark | model / benchmark "
        "| level without it |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---|---:|---:|")
    for entry in found["quantities"]:
        direction = entry["direction"]
        toward = "-" if direction is None else number(direction["toward"])
        print(
            f"| {entry['level']} | {entry['quantity']} | {entry['cells']} "
            f"| {number(entry['shares'][DOMAIN])} | {number(entry['shares'][MOVES])} "
            f"| {number(entry['width_to_moves'])} | {toward} | {entry['benchmark'] or '-'} "
            f"| {number(entry['score'])} | {number(entry['without'])} |"
        )


def _shares(shares) -> str:
    return "-" if shares is None else " ".join(f"{share:.2f}" for share in shares)
