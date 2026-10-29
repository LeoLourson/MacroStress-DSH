"""``bond.paths.v1``: a coupon bond on the run's paths of its spread driver."""

from .model import MODEL
from .plugin import plugin

__all__ = ["MODEL", "plugin"]
