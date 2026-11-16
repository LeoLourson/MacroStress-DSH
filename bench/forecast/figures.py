"""Charts of a forecast-set result for slides.

Reads one result JSON of `python -m mst.bench forecast` and writes four PNG files:
the base fan against history, the out-of-sample check on past dates, the event shifts
and the path of one event against the base. Every number on the charts comes from the
JSON, so the same result always gives the same pictures.

    python bench/forecast/figures.py RESULT.json OUT_DIR [--event hormuz-closure-2m-random]

Needs matplotlib from the dev dependency group.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RED = "#b91c1c"  # model and loss
BLUE = "#2563eb"  # history, base and gain
INK = "#450a0a"
MUTED = "#7f6a6a"
GRID = "#eadede"
SURFACE = "#ffffff"

MONTHS = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]

# Investor-facing names of the event scenarios; the slide table uses the same words.
EVENT_NAMES = {
    "cbr-cut-5pp-random": "ЦБ снижает ставку с 14 до 9 %",
    "cbr-cut-5pp-market-reaction-random": "То же, рынок растёт как в прошлые циклы",
    "cbr-target-6-7-random": "ЦБ поднимает цель по инфляции до 6–7 %",
    "bank-admin-top5-random": "Временная администрация в банке из топ-5",
    "cn-bank-payments-stop-random": "Банки Китая 3 месяца не проводят платежи",
    "dividends-a-random": "Дивиденды: рынок рад выплатам нефтегаза",
    "dividends-b-random": "Дивиденды: рынок видит отток денег",
    "fed-zero-gold-random": "ФРС за квартал снижает ставку до нуля",
    "fuel-export-ban-random": "Запрет экспорта бензина и дизеля на полгода",
    "hormuz-closure-2m-random": "Ормузский пролив закрыт 2 месяца",
    "migrant-ban-random": "Запрет труда мигрантов в трёх отраслях",
    "sbp-outage-10d-random": "Кибератака останавливает СБП на 10 дней",
}


def month_label(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{MONTHS[d.month - 1]} {d.year % 100:02d}"


def returns_by_month(event: dict, value: float) -> list[dict]:
    """Monthly return quantiles in % of the portfolio; a loss quantile flips to a return one."""
    scale = 100.0 / value
    return [
        {
            "month": m["month"],
            "p05": -m["p95"] * scale,
            "p50": -m["p50"] * scale,
            "p95": -m["p05"] * scale,
        }
        for m in event["losses_by_month"]["by_month"]
    ]


def history_quantiles(result: dict) -> dict[int, dict[str, float]]:
    """Historical return quantiles of the portfolio by horizon, from the portfolio rows."""
    names = {"q05, %": "p05", "median, %": "p50", "q95, %": "p95"}
    out: dict[int, dict[str, float]] = {}
    for row in result["rows"]:
        if row["level"] == "portfolio" and row["check"] == "quantile" and row["item"] in names:
            out.setdefault(row["horizon"], {})[names[row["item"]]] = row["history"]
    return out


def event_shifts(result: dict) -> list[dict]:
    """Median-return shift of every event against the base, % of the portfolio, by horizon."""
    shifts: dict[str, dict[int, float]] = {}
    for row in result["rows"]:
        if row["level"] == "scenario" and row["check"] == "median return shift, %":
            shifts.setdefault(row["item"], {})[row["horizon"]] = row["model"] + 0.0
    return [
        {"scenario": s, "name": EVENT_NAMES.get(s, s), **{f"h{h}": v for h, v in hs.items()}}
        for s, hs in shifts.items()
    ]


def _axes(fig_w: float, fig_h: float):
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 17,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_fan(result: dict, path: Path) -> None:
    base = next(e for e in result["events"] if e.get("base"))
    path_q = returns_by_month(base, result["value"])
    hist = history_quantiles(result)
    xs = list(range(1, len(path_q) + 1))
    fig, ax = _axes(10.4, 6.6)
    ax.fill_between(
        xs,
        [m["p05"] for m in path_q],
        [m["p95"] for m in path_q],
        color=RED,
        alpha=0.13,
        linewidth=0,
        label="модель: от P5 до P95",
    )
    ax.plot(xs, [m["p50"] for m in path_q], color=RED, linewidth=2.5, label="модель: медиана")
    for h, q in sorted(hist.items()):
        ax.plot([h, h], [q["p05"], q["p95"]], color=BLUE, linewidth=2.5)
        ax.plot(
            [h] * 3,
            [q["p05"], q["p50"], q["p95"]],
            "o",
            color=BLUE,
            markersize=9,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
    ax.plot(
        [],
        [],
        "o-",
        color=BLUE,
        markersize=9,
        linewidth=2.5,
        label="история с 2015 г.: P5, медиана, P95",
    )
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(xs[::2], [month_label(m["month"]) for m in path_q[::2]])
    ax.set_ylabel("доходность, % стоимости портфеля")
    ax.legend(loc="upper left", frameon=False)
    _save(fig, path)


def plot_backtest(result: dict, path: Path, horizon: str = "12") -> None:
    dates = result["rolling_dates"]
    fig, ax = _axes(16.64, 5.3)
    split = next((i for i, d in enumerate(dates) if d["month"] >= "2022-10-01"), len(dates))
    for i, d in enumerate(dates):
        p = d["portfolio"][horizon]
        ax.plot(
            [i, i],
            [p["low"], p["high"]],
            color=RED,
            alpha=0.35,
            linewidth=7,
            solid_capstyle="round",
        )
        inside = p["low"] <= p["fact"] <= p["high"]
        ax.plot(
            i,
            p["fact"],
            "o" if inside else "X",
            color=BLUE if inside else INK,
            markersize=10 if inside else 13,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
    ax.plot([], [], color=RED, alpha=0.35, linewidth=7, label="90-процентная полоса модели")
    ax.plot([], [], "o", color=BLUE, markersize=10, label="факт внутри полосы")
    ax.plot([], [], "X", color=INK, markersize=13, label="факт вне полосы")
    ax.axvline(split - 0.5, color=MUTED, linewidth=1.5, linestyle="--")
    ax.axhline(0, color=MUTED, linewidth=1)
    bottom = min(
        min(d["portfolio"][horizon]["low"], d["portfolio"][horizon]["fact"]) for d in dates
    )
    for lo, hi in ((0, split), (split, len(dates))):
        part = [d["portfolio"][horizon] for d in dates[lo:hi]]
        inside = sum(p["low"] <= p["fact"] <= p["high"] for p in part)
        ax.text(
            (lo + hi - 1) / 2,
            bottom - 8,
            f"внутри полосы {inside} из {len(part)}",
            ha="center",
            va="top",
            color=INK,
            fontsize=18,
            fontweight="bold",
        )
    ticks = [i for i, d in enumerate(dates) if d["month"].endswith("-01-01")]
    ax.set_xticks(ticks, [dates[i]["month"][:4] for i in ticks])
    ax.set_xlim(-1, len(dates))
    ax.set_ylim(bottom - 18, None)
    ax.set_ylabel(f"доходность за {horizon} мес., %")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False, ncol=3)
    _save(fig, path)


def signed(v: float) -> str:
    """A shift as slides print it: one decimal, comma, true minus, a bare 0 for zero."""
    text = f"{v:+.1f}".replace(".", ",").replace("-", "−")
    return "0" if text in ("+0,0", "−0,0") else text


def plot_events(result: dict, path: Path) -> None:
    rows = sorted(event_shifts(result), key=lambda r: r["h12"])
    fig, ax = _axes(16.64, 6.0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ys = list(range(len(rows)))
    ax.barh(
        ys,
        [r["h12"] for r in rows],
        height=0.6,
        color=[BLUE if r["h12"] > 0 else RED for r in rows],
        label="через 12 мес.",
    )
    ax.plot(
        [r["h3"] for r in rows],
        ys,
        "D",
        color=INK,
        markersize=9,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        label="через 3 мес.",
    )
    for y, r in zip(ys, rows, strict=True):
        v = r["h12"]
        edge = max(v, r["h3"], 0) + 0.35 if v >= 0 else min(v, r["h3"]) - 0.35
        ax.text(
            edge,
            y,
            signed(v),
            va="center",
            ha="left" if v >= 0 else "right",
            color=INK,
            fontsize=17,
        )
    ax.set_yticks(ys, [r["name"] for r in rows])
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("сдвиг медианы против базы, % стоимости портфеля")
    lo = min(min(r["h12"], r["h3"]) for r in rows)
    hi = max(max(r["h12"], r["h3"]) for r in rows)
    ax.set_xlim(lo - 2.5, hi + 2.5)
    ax.legend(loc="lower right", frameon=False)
    _save(fig, path)


def plot_event_path(
    result: dict, path: Path, scenario: str, shaded: tuple[int, int], label: str = "пролив закрыт"
) -> None:
    events = {e["scenario"]: e for e in result["events"]}
    base = next(e for e in result["events"] if e.get("base"))
    b = returns_by_month(base, result["value"])
    e = returns_by_month(events[scenario], result["value"])
    xs = list(range(1, len(b) + 1))
    fig, ax = _axes(16.64, 5.0)
    ax.axvspan(shaded[0] - 0.5, shaded[1] + 0.5, color=RED, alpha=0.07, linewidth=0)
    ax.text(
        (shaded[0] + shaded[1]) / 2,
        0.02,
        label,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        color=MUTED,
        fontsize=17,
    )
    ax.plot(xs, [m["p50"] for m in b], color=BLUE, linewidth=2.5, label="база: медиана")
    ax.plot(xs, [m["p50"] for m in e], color=RED, linewidth=2.5, label="с событием: медиана")
    for i in (shaded[1] - 1, len(xs) - 1):
        shift = e[i]["p50"] - b[i]["p50"]
        ax.annotate(
            f"{signed(shift)} п. п.",
            (xs[i], e[i]["p50"]),
            xytext=(0, -34 if shift < 0 else 18),
            textcoords="offset points",
            ha="center",
            color=INK,
            fontsize=18,
            fontweight="bold",
        )
    ax.set_xticks(xs, [month_label(m["month"]) for m in b])
    ax.set_ylabel("доходность, %")
    ax.legend(loc="upper left", bbox_to_anchor=(0.3, 1), frameon=False)
    _save(fig, path)


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("result", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--event", default="hormuz-closure-2m-random")
    parser.add_argument("--shaded", default="2,3", help="months of the event, 1-based, from,to")
    args = parser.parse_args(argv)
    result = json.loads(args.result.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    paths = [args.out / n for n in ("fan.png", "backtest.png", "events.png", "event-path.png")]
    plot_fan(result, paths[0])
    plot_backtest(result, paths[1])
    plot_events(result, paths[2])
    lo, hi = (int(x) for x in args.shaded.split(","))
    plot_event_path(result, paths[3], args.event, (lo, hi))
    return paths


if __name__ == "__main__":
    for p in main():
        print(p)
