"""The Markdown pieces both checks print: numbers, intervals and the sanity of the quantities."""

from collections.abc import Mapping

import numpy as np

from .baselines import DOMAIN, WIDTH


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


def print_left_out(left_out: Mapping[str, Mapping[str, str]], unit: str) -> None:
    """The variables each unit's version left out as too short, grouped by reason."""
    if not left_out:
        return
    grouped: dict[tuple[str, str], list[str]] = {}
    for name, variables in left_out.items():
        for variable, reason in variables.items():
            grouped.setdefault((variable, reason), []).append(name)
    print(f"\nVariables the model left out as too short, their cells unscored ({unit}):\n")
    for (variable, reason), names in sorted(grouped.items()):
        print(f"- {variable} in {', '.join(sorted(names))}: {reason}")


def print_sanity(found: Mapping, *, benchmark: str) -> None:
    """The per-quantity table of ``mst.metrics.sanity.check`` and the quantities that fail."""
    admission = found["admission"]
    threshold = admission["threshold"]
    print(
        f"\n| level | quantity | cells | band outside domain | band wider than history "
        f"| band / history range | toward the fact | model / {benchmark} | level without it |"
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
        f"{threshold:.2f} of its cells." + (f" Over it: {'; '.join(failing)}." if failing else "")
    )
    pooled = ", ".join(
        f"{rule} {rule_found['pooled']:.3f}" for rule, rule_found in admission["rules"].items()
    )
    print(f"Pooled over all {admission['cells']} cells, for reference only: {pooled}.")
