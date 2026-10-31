"""The MS-VAR macro model plugin: a Markov-switching ``G_0`` of the platform.

The estimation (Minnesota prior per regime, Gibbs sampler with forward-filtering
backward-sampling), the sign identification within each regime and the conditional forecast
over regime paths are here; the platform knows this package only through its ``plugin``
declaration and the model SDK. Another macro model is another package with another
``model_id`` — the BVAR beside it is one.
"""

from .estimate import Posterior, estimate
from .identify import identify
from .model import MSVarModel, parts
from .plugin import MODEL, plugin
from .signs import Regime, Shock, Signs, load_signs

__all__ = [
    "MODEL",
    "MSVarModel",
    "Posterior",
    "Regime",
    "Shock",
    "Signs",
    "estimate",
    "identify",
    "load_signs",
    "parts",
    "plugin",
]
