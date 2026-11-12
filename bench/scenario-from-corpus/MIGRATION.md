# Migration of the bench scenarios to the intervention format

The scenarios of the 100-event bench of 2026-11-10 were converted on 2026-11-12 with
`python -m mst.scenario migrate`, one staging folder per event, so that every file went through
the shipped translator and writer. 84 of the 100 files are now in the intervention format; the
16 below stay in the legacy format because the translator refuses to turn them into the new
canon without changing a number, and a written-out refusal is better than a silent shift.

The refusals fall into three classes, all of them shapes the old format allowed and the new one
does not:

- `log100 observation on ru.usdrub has unsupported weights` — E007, E014, E019. A check on a
  variable measured in log units carried weights other than one, which the new contract does not
  express.
- `expectations on X: E(log X) is not log E(X)` — E030, E033, E044, E091, E092. A mean of the
  variable was written in the log scale; the new contract states an expectation of the level, and
  the translation of the mean is not the mean of the translation.
- `no shared unit preserves every value` / `no shared unit preserves bounds` — E017, E034, E045,
  E066 and E035, E039, E054, E085. The numbers were written in the model scale of the panel, and
  no single own unit (`level`, `change`, `percent`) reproduces them exactly.

These 16 files remain readable as the archive of that bench: they are not executed by the
platform, and re-running one means rewriting it under the current canon, with the numbers
re-derived from the event text and the corpus. A legacy calculation result keeps its scenario copy
in its own meta, so the past runs of the workspace stay readable as well.

Four of the migrated files named driving causes: E002, E004 and E083 named `ru.monetary`, E023
named `gl.oil`. The command keeps their intervention numbers as they were and moves the former
causes into the narrative as `прежние причины: …`, dropping the approver, so each of the four
needs a fresh classification under the current guide before it is computed again. The scenario
files of the workspace went through the same command on the same day, and its 13 cause-naming
files are listed in `workspace/scenarios-migration-causes.txt`.
