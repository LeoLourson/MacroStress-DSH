# mst.portfolio

The portfolio, its check against model versions and the run report: the losses $P_5$, $P_{50}$, $P_{95}$, ES95 with an interval and trimming, the scenario price $\mathrm{dist}(s)$ and the coverage. The run that writes the files is `mst.runs`; here are the portfolio contract, the tool operations and the assembly of the report.

| File | What it does |
|---|---|
| `spec.py` | `Portfolio`: name, base currency (the ruble), positions with unique ids |
| `tool.py` | Operations of the tool `portfolio`: `list`, `get`, `save`, `delete`, `check`, `exposure` |
| `report.py` | `build`, the report on position results; `dist`, the scenario price; `losses_of` and `loss_block`, the losses and their block for contributions |

`check` loads the scenario and both versions and names each reason a position will not be calculated: an unknown asset type, `terms` off the model's schema, a currency with no exchange rate path, a macro variable not in the version, an object without a served driver (with the code), a driver rejected by the satellite check, with its reason. `exposure` values positions for the calculation month of the macro version, before the paths are drawn.

Report rules:

- losses are added within a draw, and only then are quantiles taken; the sum of position P95 values is not passed off as the portfolio's P95;
- a position without a calculation is excluded with a reason, and `coverage.share` says which share of measurable value the numbers cover; unmeasurable value makes the share `null`, not 1;
- the status is `ok`, `needs_review` or `incomplete`; an uncovered share above 5 % gives `incomplete`;
- a scenario without conditions has no price: `dist` is `null`, not zero;
- `dist.by_level` splits the price by level when there are conditions on drivers, and the sum equals `dist.distance`.

Tests are in [../../../tests/portfolio](../../../tests/portfolio/README.en.md).
