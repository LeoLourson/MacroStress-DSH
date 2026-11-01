"""The version of the model SDK and the rule a model plugin's range is checked by.

The model SDK follows semantic versioning like the position SDK (``mst.plugin_sdk.version``):
a minor release only adds, a major release may break. A model plugin declares the range it was
written for in ``Plugin.model_sdk``; a plugin outside the range blocks start-up with its name.
"""

from ..sdk_version import compatible as _compatible
from ..sdk_version import parse

MODEL_SDK_VERSION = "2.0.0"

__all__ = ["MODEL_SDK_VERSION", "model_compatible", "parse"]


def model_compatible(spec: str, version: str = MODEL_SDK_VERSION) -> bool:
    """Whether ``version`` satisfies every comma-separated clause of ``spec``."""
    return _compatible(spec, version)
