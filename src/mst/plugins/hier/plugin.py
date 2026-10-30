"""The declaration of the hierarchical satellite plugin and its exports."""

from mst.plugin_sdk import Plugin

from .model import HierModel

MODEL = HierModel()

plugin = Plugin(
    plugin_id="hier",
    version="0.1.0",
    model_sdk=">=1.0,<2.0",
    label="Иерархическая сателлитная модель",
    description=(
        "Сателлитные модели уровней выше нуля: регрессия драйвера на своё прошлое и "
        "макровходы, иерархия параметров по дереву объектов, проверка вне выборки и "
        "коррелированные остатки внутри версии."
    ),
    satellites=(MODEL,),
)
