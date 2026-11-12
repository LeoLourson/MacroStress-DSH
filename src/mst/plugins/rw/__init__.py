"""Joint Student random walk, selected explicitly as ``rw.joint``."""

from .model import JointWalk, RwModel
from .plugin import MODEL, plugin

__all__ = ["MODEL", "JointWalk", "RwModel", "plugin"]
