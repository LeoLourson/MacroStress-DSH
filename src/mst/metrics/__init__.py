"""The check of a model of any level on history: one procedure for every plugin.

A check is a set of windows (``windows``): a calculation date, the months after it, a scenario
and the facts. The rolling grid (``rolling``) opens a window on every date of the plan with the
empty scenario; the episodes (``backtest``) open one before every named event with the scenario
of what happened. On every window the models of every level are estimated on the data known
then — the macro model, the satellites with ``--drivers``, the position models with
``--portfolio`` — and the cells of every level (``levels``) hold the fact, the forecast and its
benchmarks. The same judgement reads both checks (``scores``): what fell out first, then the
sanity of every quantity (``sanity``), then the relative CRPS per level with its interval, the
halves and the quantities; the decision between two versions (``decision``) reads their cells;
the tail of the portfolio is the Acerbi-Székely ``Z`` of its cells (``es``). The command is
``python -m mst.metrics``.
"""
