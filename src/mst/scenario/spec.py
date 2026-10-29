"""``scenarios/<id>.json``: the scenario (N, C, D, O) of one run and the versions it takes.

The file is ``Scenario`` from ``contracts.py`` plus the two model versions a run of it reads:
the macro version whose panel the conditional forecast starts from and the satellite version
estimated on that same panel. The id is the file name; ``mst.scenario.tool`` reads and writes
the files, ``mst.scenario.build`` turns one into paths.
"""

from ..contracts import Sha256
from .contracts import Scenario


class Spec(Scenario):
    """One scenario file: the scenario, the macro version and the satellites of that version."""

    macro: Sha256
    satellites: Sha256
