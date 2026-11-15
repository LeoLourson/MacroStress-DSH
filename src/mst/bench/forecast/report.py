"""The gates of a scorecard and its Markdown summary."""

from .rows import Row, share_gate

MARK = {"pass": "ok", "fail": "FAIL", "info": "", "short": "short"}


def gates(plan, rows: list[Row]) -> list[dict]:
    """The summary gates: one per check and horizon; a row with a rule of its own is a gate."""
    found = []
    for level in ("macro", "shares"):
        for h in plan.horizons:
            chosen = [r for r in rows if (r.level, r.check, r.horizon) == (level, "width", h)]
            if chosen:
                found.append(share_gate(chosen, plan.gates.width_share, f"{level} width, {h} m"))
    for row in rows:
        if row.level == "macro" and row.check == "width":
            continue
        if row.level == "shares" and row.check == "width":
            continue
        if row.verdict in ("pass", "fail", "short") and row.rule:
            found.append(
                {
                    "gate": f"{row.level} {row.check}, {row.item}, {row.horizon} m",
                    "rule": row.rule,
                    "model": row.model,
                    "history": row.history,
                    "judged": row.n,
                    "verdict": row.verdict,
                }
            )
    return found


def _value(x, digits=2) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _ratio(row: Row) -> str:
    if row.model is None or row.history in (None, 0):
        return "-"
    return f"{row.model / row.history:.2f}"


def markdown(card: dict) -> str:
    rows = [Row(**row) for row in card["rows"]]
    plan = card["plan"]
    versions = card["versions"]
    lines = [
        "# Forecast check",
        "",
        f"Portfolio `{card['portfolio']}`, data to {card['as_of']}, macro `{versions['macro'][:12]}`,"
        f" satellites `{versions['satellites'][:12]}`, commit `{card.get('git_commit') or '-'}`,"
        f" {plan['draws']} paths, seed {plan['seed']}. History from {plan['history_start']};"
        f" band {plan['band']:.0%}. Verdict: **{'pass' if card['passed'] else 'fail'}**.",
        "",
        "## Gates",
        "",
        "| gate | rule | verdict |",
        "|---|---|---|",
    ]
    for gate in card["gates"]:
        rule = gate["rule"]
        if "passed" in gate:
            rule = f"{rule}: {gate['passed']} pass"
        lines.append(f"| {gate['gate']} | {rule} | {MARK[gate['verdict']] or gate['verdict']} |")
    horizons = plan["horizons"]
    lines += ["", "## Macro: width of the h-month change, model / history", ""]
    lines += ["| variable | " + " | ".join(f"{h} m" for h in horizons) + " |"]
    lines += ["|---|" + "---:|" * len(horizons)]
    width = {(r.item, r.horizon): r for r in rows if (r.level, r.check) == ("macro", "width")}
    for item in dict.fromkeys(r.item for r in rows if (r.level, r.check) == ("macro", "width")):
        cells = []
        for h in horizons:
            row = width.get((item, h))
            cells.append("-" if row is None else f"{_ratio(row)} {MARK[row.verdict]}".strip())
        lines.append(f"| {item} | " + " | ".join(cells) + " |")
    for level, title in (("shares", "Shares"), ("portfolio", "Portfolio")):
        chosen = [r for r in rows if r.level == level]
        if not chosen:
            continue
        lines += ["", f"## {title}", "", "| check | item | h | model | reference | n | verdict |",
                  "|---|---|---:|---:|---:|---:|---|"]
        for r in chosen:
            lines.append(
                f"| {r.check} | {r.item} | {r.horizon or '-'} | {_value(r.model)} |"
                f" {_value(r.history)} | {r.n} | {MARK[r.verdict]} |"
            )
    shifts = [r for r in rows if r.level == "scenario" and r.check == "median return shift, %"]
    if shifts:
        horizons_of = sorted({r.horizon for r in shifts})
        lines += ["", "## Events: shift against the base, % of the portfolio", "",
                  "| scenario | admitted | " + " | ".join(f"median {h} m" for h in horizons_of)
                  + " | peak median | ES95 loss 12 m | path ES95 loss |",
                  "|---|---|" + "---:|" * (len(horizons_of) + 3)]
        by = {}
        for r in rows:
            if r.level == "scenario":
                by.setdefault(r.item, {})[(r.check, r.horizon)] = r
        for item, found in by.items():
            admitted = found.get(("admitted", None))
            cells = [_value(found[("median return shift, %", h)].model) for h in horizons_of]
            peak = next((r for (c, _), r in found.items() if c.startswith("peak")), None)
            es = next((r for (c, _), r in found.items() if c == "ES95 loss shift, %"), None)
            path = next((r for (c, _), r in found.items() if c.startswith("path")), None)
            lines.append(
                f"| {item} | {MARK[admitted.verdict] if admitted else '-'} | " + " | ".join(cells)
                + f" | {_value(peak.model) if peak else '-'} ({peak.horizon if peak else '-'} m)"
                f" | {_value(es.model) if es else '-'} | {_value(path.model) if path else '-'} |"
            )
    out = [r for r in rows if r.check.endswith("out of sample") and r.item == "all series"]
    if out:
        lines += ["", "## Macro out of sample", "", "| level | h | coverage | n | verdict |",
                  "|---|---:|---:|---:|---|"]
        for r in out:
            lines.append(f"| {r.level} | {r.horizon} | {_value(r.model)} | {r.n} | {MARK[r.verdict]} |")
    lines.append("")
    return "\n".join(lines)
