"""``credit.paths.v1``: a loan or a deposit on the run's paths of its overdue-share driver."""

from .model import MODEL
from .plugin import plugin

__all__ = ["MODEL", "plugin"]
