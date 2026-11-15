"""forecast: the numbers of the models against history, level by level, with fixed gates.

One run reads a macro version, a satellite version and the plan ``bench/forecast/plan.toml``.
It forecasts the empty scenario of the plan's portfolio and judges every level the same way:

- ``macro``: the quantile width of every variable's h-month change against history;
- ``shares``: the width of every share's h-month return, the mean pair correlation of the
  year, and the satellite's own check against an AR(1);
- ``portfolio``: the width and the expected shortfall of the return against the same
  portfolio held over every past window, valued by the same position models;
- out of sample, on the plan's dates, the coverage of the band at every level and, for the
  portfolio, the CRPS against the historical simulation and the Acerbi-Székely statistic.

Every number is a row (``rows.Row``) with its reference, its sample size and its verdict; the
scorecard keeps the rows, the gates, the versions and the commit, and ``save`` writes it to
``results/`` beside the plan.
"""

import subprocess
from pathlib import Path

from mst import store as files

from .. import results, save_result
from . import levels
from . import rolling as out_of_sample
from .pipeline import forecast, replay
from .plan import Plan, load_plan
from .report import gates, markdown


def _version(names: list[str], prefix: str, kind: str) -> str:
    found = [name for name in names if name.startswith(prefix)]
    if len(found) != 1:
        raise ValueError(f"{kind} {prefix!r} matches {len(found)} versions")
    return found[0]


def _commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(plan: Plan, *, workspace: Path, macro: str, satellites: str, rolling: bool = True,
        progress=print) -> dict:
    """The scorecard of the versions ``macro`` and ``satellites`` (hashes or prefixes)."""
    from mst import plugins
    from mst.data.store import SeriesStore
    from mst.drivers import bundle as satellite_bundle
    from mst.macro import version as macro_version
    from mst.portfolio import tool as portfolio_tool

    ws = files.Workspace(Path(workspace))
    installed = plugins.registry()
    macro = _version([row["version"] for row in macro_version.catalogue(ws.models)], macro, "macro")
    macro_found = macro_version.load(ws.models, macro, installed.macro)
    names = [p.name for p in (ws.models / "satellites").iterdir() if p.is_dir()]
    satellites = _version(names, satellites, "satellites")
    bundle = satellite_bundle.load(ws.models, satellites, installed.satellites)
    if bundle.macro != macro:
        raise ValueError(f"the satellites {satellites[:12]} belong to macro {bundle.macro[:12]}")
    portfolio = portfolio_tool.load(ws.portfolios, plan.portfolio)
    store = SeriesStore(ws.series)
    progress("forecast of the empty scenario")
    base = forecast(
        macro_found, bundle, portfolio, draws=plan.draws, seed=plan.seed, horizon=plan.horizon
    )
    progress("the portfolio over history")
    past = replay(macro_found, bundle, portfolio, store, horizon=plan.horizon,
                  as_of=macro_found.as_of)
    rows = [
        *levels.macro_width(plan, macro_found, base),
        *levels.share_rows(plan, bundle, base, store, macro_found.as_of),
        *levels.portfolio_rows(plan, base, past),
    ]
    dates = []
    if rolling and plan.rolling is not None:
        found, dates = out_of_sample.run(
            plan, workspace=ws.root, current_macro=macro_found, current_satellites=bundle,
            portfolio=portfolio, replay=past, store=store, progress=progress,
        )
        rows += found
    found_gates = gates(plan, rows)
    return {
        "plan": plan.model_dump(mode="json"),
        "portfolio": plan.portfolio,
        "as_of": macro_found.as_of.isoformat(),
        "versions": {"macro": macro, "satellites": satellites},
        "git_commit": _commit(),
        "value": base.value,
        "rolling_dates": dates,
        "rows": [row.as_dict() for row in rows],
        "gates": found_gates,
        "passed": all(gate["verdict"] != "fail" for gate in found_gates),
    }


def save(card: dict, folder: Path) -> list[Path]:
    """``results/<as_of>-<macro>-<satellites>.json`` and ``.md``, and ``results/latest.md``."""
    target = results(Path(folder))
    name = f"{card['as_of']}-{card['versions']['macro'][:8]}-{card['versions']['satellites'][:8]}"
    summary = markdown(card)
    written = save_result(target, name, card, summary)
    latest = target / "latest.md"
    latest.write_text(summary, encoding="utf-8", newline="\n")
    return [*written, latest]


__all__ = ["Plan", "load_plan", "markdown", "run", "save"]
