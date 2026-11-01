"""The declaration of the BVAR plugin and its exports."""

from mst.plugin_sdk import Plugin

from .model import BvarModel

MODEL = BvarModel()

plugin = Plugin(
    plugin_id="bvar",
    version="0.1.0",
    model_sdk=">=2.0,<3.0",
    label="BVAR со стохастической волатильностью",
    description=(
        "Макромодель уровня 0: векторная авторегрессия с блочной экзогенностью, "
        "стохастической волатильностью, стьюдентовскими хвостами и идентификацией шоков "
        "знаками. Условный прогноз строится по Антолин-Диасу и др. (2021)."
    ),
    macro=(MODEL,),
)
