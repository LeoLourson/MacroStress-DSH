"""The reference model of the second level: a regression on the quantities above.

The plugin serves any quantity assigned to it — it declares none, so it never competes for
one — and estimates the canon's satellite ``x^2_t = beta' g(x^{<2}_t) + sigma e_t`` with
``g = [1, x]`` over the declared inputs: the macro variables come from the panel, the
quantities of the levels above from ``upper_history``. The posterior of ``(beta, sigma)`` is
the conjugate normal-inverse-gamma of the regression, drawn once per repetition; the part
records the training bounds of every node. See ``tests/drivers/test_levels.py`` for a chain
of three levels that recovers known coefficients.
"""

from .model import MODEL, RefLevel2Model
from .plugin import plugin

__all__ = ["MODEL", "RefLevel2Model", "plugin"]
