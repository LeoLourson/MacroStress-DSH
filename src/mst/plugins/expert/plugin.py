"""The declaration of the expert plugin and its exports."""

from mst.plugin_sdk import Plugin

from .model import ExpertModel

plugin = Plugin(
    plugin_id="expert",
    version="0.1.0",
    model_sdk=">=3.1,<4.0",
    label="Макромодель по опросу эксперта",
    description=(
        "Макромодель уровня 0 по структурированному опросу эксперта: агент отвечает "
        "квантилями на вопросы о 18 шоках, плагин строит из ответов поток шоков, законы "
        "откликов, их t-связку и гауссов остаток как разность ковариаций."
    ),
    macro=(ExpertModel("rw"), ExpertModel("market")),
)
