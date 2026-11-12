"""Declaration of the joint random walk macro plugin."""

from mst.plugin_sdk import Plugin

from .model import RwModel

MODEL = RwModel()

plugin = Plugin(
    plugin_id="rw",
    version="0.1.0",
    model_sdk=">=4.0,<5.0",
    label=MODEL.label,
    description=MODEL.description,
    macro=(MODEL,),
)
