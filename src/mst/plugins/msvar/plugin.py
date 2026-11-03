"""The declaration of the MS-VAR plugin and its exports."""

from mst.plugin_sdk import Plugin

from .model import MSVarModel

MODEL = MSVarModel()

plugin = Plugin(
    plugin_id="msvar",
    version="0.1.0",
    model_sdk=">=3.0,<4.0",
    label="MS-VAR с марковскими переключениями",
    description=(
        "Макромодель уровня 0: векторная авторегрессия, у которой константа и ковариация "
        "ошибок переключаются по марковскому режиму при общих лагах, с "
        "Minnesota-приором по режимам и знаковой идентификацией шоков внутри каждого. "
        "Режимная часть события адресуется сценарием наравне с шоками: элемент режима в "
        "движущих шоках фиксирует режим на весь горизонт."
    ),
    macro=(MODEL,),
)
