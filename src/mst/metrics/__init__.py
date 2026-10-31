"""The quality metrics of section 5 of the scheme: cells, scores, admission and episodes.

The backtest judges a version on historical episodes: every cell (episode, month, quantity)
compares the conditional forecast with the fact, and the report — CRPS normalised by the
typical movement known at the episode's calculation date, calibration, sharpness, coverage, the
admission rules pooled per rule, the random-walk and base-run comparisons, and the bootstrap
intervals — is given per quantity. On the same cell rows the report gives the equal-weight
aggregate of section 5 per level and horizon (with the tuning/validation halves beside it), the
usefulness of the scenario split by the manual/llm mark, the wild cluster bootstrap of every
few-cluster interval, the four-step 2-SE decision behind ``--against``, the Monte-Carlo interval
behind ``--mc-bootstrap`` and, behind ``--portfolio``, the Acerbi-Székely tail of the
portfolio's losses. The episode definitions live in ``episodes.toml``; the runner is
``backtest.py`` and the command is ``python -m mst.metrics``.
"""
