"""The quality metrics of section 5 of the scheme: cells, scores, admission and episodes.

The backtest judges a version on historical episodes: every cell (episode, month, quantity)
compares the conditional forecast with the fact, and the report — CRPS normalised by the
typical movement known at the episode's calculation date, calibration, sharpness, coverage, the
admission rules pooled per rule, the random-walk and base-run comparisons, and the bootstrap
intervals — is given per quantity. The equal-weight aggregate across quantities, the per-level
and per-horizon split, the four-step 2-SE decision rule, the tail statistics (Acerbi-Székely,
ES) and the wild cluster bootstrap are not implemented; ``backtest.py`` names them as missing
in the report meta. The episode
definitions live in ``episodes.toml``; the runner is ``backtest.py`` and the command is
``python -m mst.metrics``.
"""
