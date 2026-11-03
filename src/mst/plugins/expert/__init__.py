"""The expert plugin: the macro models built from the agent's elicitation answers.

The questions and the answer contract (``answers``), the laws from three quantiles
(``laws``), the shock stream and the responses (``stream``), the estimate (``estimate``),
the conditional forecast (``forecast``) and the subset simulation of the named shocks
(``subset``) are here; the platform knows this package only through its ``plugin``
declaration and the model SDK.
"""

from .model import MARKET, RESIDUAL, ExpertModel, Shock, load_shocks
from .plugin import plugin

__all__ = ["MARKET", "RESIDUAL", "ExpertModel", "Shock", "load_shocks", "plugin"]
