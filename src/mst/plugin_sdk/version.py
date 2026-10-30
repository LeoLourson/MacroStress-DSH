"""The SDK version and the rule a plugin's declared compatibility is checked by.

The SDK follows semantic versioning: a minor release only adds, a major release may break.
A plugin declares the range it was written for (``">=1.0,<2.0"``); a plugin outside the
range blocks start-up rather than failing in the middle of a run.
"""

from ..sdk_version import compatible as _compatible
from ..sdk_version import parse

SDK_VERSION = "1.0.0"

__all__ = ["SDK_VERSION", "compatible", "parse"]


def compatible(spec: str, version: str = SDK_VERSION) -> bool:
    """Whether ``version`` satisfies every comma-separated clause of ``spec``."""
    return _compatible(spec, version)
