#!/usr/bin/env python
"""The gates G2–G5 of ``plans/macro-models.md`` from the saved runs of a workspace.

Run from the repository root::

    uv run python scripts/macro_gates.py --workspace DIR RUN_ID [RUN_ID ...]

The logic lives in ``mst.models.gates``; this file is the executable the plan names.
"""

import sys

from mst.models.gates import main

if __name__ == "__main__":
    sys.exit(main())
