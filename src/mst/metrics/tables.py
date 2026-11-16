"""The Markdown both checks print: what fell out, the sanity, the scores and the decision.

The episodes and the rolling grid print the same judgement (``mst.metrics.scores.judge``) the
same way, in the order a reader should take it: what fell out of the check first, then the
sanity of every quantity, then the scores.
"""

from collections.abc import Mapping

import numpy as np

from .baselines import DOMAIN, WIDTH
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
    refused, skipped = dropped["refused"], dropped.get("skipped", {})
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
        "Read before any score: a quantity over the admission share, a band wider than the "
        "history or a scenario that moves the forecast away from the fact is named here."
    )
    _print_sanity(found["sanity"])
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
    print(
        "| half | windows | cells | level | model/walk | model/conditional_walk | model/base "
        "| admission |"
    )
    print("|---|---:|---:|---|---:|---:|---:|---|")
    for name, half in found["halves"].items():
        admitted = "fails" if half["sanity"]["failed"] else "passes"
        for level, entry in half["levels"].items():
            pairs = entry["pairs"]
            print(
                f"| {name} | {len(half['units'])} | {half['cells']} | {level} "
                f"| {number(pairs.get('model/walk', {}).get('score'))} "
                f"| {number(pairs.get('model/conditional_walk', {}).get('score'))} "
                f"| {number(pairs.get('model/base', {}).get('score'))} | {admitted} |"
            )
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


def print_decision(verdict: Mapping) -> None:
    """The four steps of the two-standard-error rule and the numbers they read."""
    versus_benchmark, versus_incumbent = verdict["vs_benchmark"], verdict["vs_incumbent"]
    print("\n## Decision (two standard errors)\n")
    print(f"- level `{verdict['level']}`, half {verdict['half']}: {verdict['cells']} paired cells")
    print(
        f"- model/{verdict['benchmark']} {number(versus_benchmark['point'])} "
        f"{interval_text((versus_benchmark['low'], versus_benchmark['high']))}; "
        f"candidate/incumbent {number(versus_incumbent['point'])} "
        f"{interval_text((versus_incumbent['low'], versus_incumbent['high']))}"
    )
    shares = ", ".join(
        f"{rule} {number(share)}" for rule, share in verdict["admission_shares"].items()
    )
    print(f"- worst quantity's admission shares: {shares or '-'}")
    if verdict.get("admission_grown"):
        print(f"- shares that grew against the incumbent: {', '.join(verdict['admission_grown'])}")
    print(f"- verdict: **{verdict['verdict']}** (step {verdict['step']}) — {verdict['note']}")
    if verdict.get("reason"):
        print(f"- {verdict['reason']}")


def _print_sanity(found: Mapping) -> None:
    """The per-quantity table of ``mst.metrics.sanity.check`` and the quantities that fail."""
    admission = found["admission"]
    print(
        "\n| level | quantity | cells | band outside domain | band wider than history "
        "| band / history range | toward the fact | model / walk | level without it |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for entry in found["quantities"]:
        direction = entry["direction"]
        toward = "-" if direction is None else number(direction["toward"])
        marks = {rule: "**" if rule in entry["failed"] else "" for rule in (DOMAIN, WIDTH)}
        print(
            f"| {entry['level']} | {entry['quantity']} | {entry['cells']} "
            f"| {marks[DOMAIN]}{entry['shares'][DOMAIN]:.3f}{marks[DOMAIN]} "
            f"| {marks[WIDTH]}{entry['shares'][WIDTH]:.3f}{marks[WIDTH]} "
            f"| {number(entry['width_to_range'])} | {toward} "
            f"| {number(entry['score'])} | {number(entry['without'])} |"
        )
    failing = [
        f"{entry['quantity']} ({', '.join(entry['failed'])})"
        for entry in found["quantities"]
        if entry["failed"]
    ]
    verdict = "fails" if admission["failed"] else "passes"
    print(
        f"\nAdmission {verdict}: every quantity must keep each band rule under "
        f"{admission['threshold']:.2f} of its cells."
        + (f" Over it: {'; '.join(failing)}." if failing else "")
    )
    pooled = ", ".join(
        f"{rule} {rule_found['pooled']:.3f}" for rule, rule_found in admission["rules"].items()
    )
    print(f"Pooled over all {admission['cells']} cells, for reference only: {pooled}.")


def _shares(shares) -> str:
    return "-" if shares is None else " ".join(f"{share:.2f}" for share in shares)
