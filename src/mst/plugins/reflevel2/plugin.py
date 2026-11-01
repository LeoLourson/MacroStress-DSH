"""The plugin declaration of the reference model of the second level."""

from mst.plugin_sdk import Plugin

from .model import MODEL

plugin = Plugin(
    plugin_id="reflevel2",
    version="0.1.0",
    model_sdk=">=2.0,<3.0",
    label="Эталонная регрессия второго уровня",
    description=(
        "Регрессия величины второго уровня на величины уровней выше: x^2 = beta' g(x^{<2}) "
        "+ sigma e с сопряжённым нормальным апостериорным законом (beta, sigma)."
    ),
    satellites=(MODEL,),
)
