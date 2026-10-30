"""``python -m mst.metrics``: the metrics commands, run as a task."""

import os
import sys

# The samplers solve many small dense systems; a BLAS thread pool costs more than it saves.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from .cli import main  # noqa: E402

sys.exit(main())
