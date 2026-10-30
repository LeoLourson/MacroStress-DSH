"""The quality metrics of section 5 of the scheme: cells, scores, admission and episodes.

The backtest judges a version on historical episodes: every cell (episode, month, quantity)
compares the conditional forecast with the fact, and the aggregation — CRPS normalised by the
typical movement, calibration, sharpness, the admission rules and the baselines — says whether
the version is better than what a random walk gives for free. The episode definitions live in
``episodes.toml``; the runner is ``backtest.py`` and the command is ``python -m mst.metrics``.
"""
