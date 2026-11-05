"""The interview a model with declared inputs is asked in small parts.

``Session`` owns the parts, the tickets and the publication; the model owns what is asked and
what a valid answer is. ``address`` reads a ticket back to the model and date it names, for
the server that has to build the model's context before answering.
"""

from .session import Session, address

__all__ = ["Session", "address"]
