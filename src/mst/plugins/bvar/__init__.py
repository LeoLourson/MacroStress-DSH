"""The BVAR macro model plugin: the reference ``G_0`` of the platform.

The estimation (Minnesota prior, block exogeneity, stochastic volatility, Student tails), the
sign identification and the conditional forecast are here; the platform knows this package
only through its ``plugin`` declaration and the model SDK. A different macro model is another
package with another ``model_id``.
"""

from .model import BvarModel, parts
from .plugin import MODEL, plugin
from .signs import Shock, Signs, load_signs

__all__ = ["MODEL", "BvarModel", "Shock", "Signs", "load_signs", "parts", "plugin"]
