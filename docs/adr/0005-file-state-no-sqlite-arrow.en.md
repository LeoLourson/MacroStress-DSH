# ADR 0005: State stays in files, without SQLite or Arrow

Date: 2026-11-02
Status: accepted

## Context

An external review called file storage a defect and proposed three migrations: move the job
registry, version metadata and catalogues to SQLite in WAL mode (`workspace/state.db`), series to
Parquet or DuckDB, and run paths to Arrow IPC read through `mmap`. Its arguments were races when
writing `workspace.json`, `PermissionError` on Windows, slow `as_of` slices, and serialising paths
to JSON or CSV.

We checked each argument against the code and a working workspace:

- A model version is a folder named by the sha256 of `meta.json` and `arrays.npz`. The draft is
  prepared next to it and renamed once fully written, so an interrupted estimation leaves no
  version. A finished version never changes and needs no transaction.
- Flat JSON goes through a draft and an atomic replace, so a reader sees the whole old file or the
  whole new one.
- Two jobs do not write to one resource: `O_CREAT | O_EXCL` creates the lock, and the lock of a
  process that has ended counts as stale and is taken over.
- `workspace.json` is a snapshot of the latest versions and counts: 473 bytes on a workspace with
  33 versions and 24 runs. History does not enter it, and its size does not grow with history.
- `PermissionError` on Windows is real: a reader blocks the replace. Retrying the replace treats
  it (ADR 0004), with no database needed.
- Paths live in `paths.npz`; neither JSON nor CSV is used for them. The largest file, 48.6 MB with
  arrays $(2048, 13, 23)$ and $(2048, 13, 205)$, loads in 0.17 s.
- An `as_of` slice over all 610 series (1.8 MB of CSV) took 15.8 s with the processor at 100 % from
  other calculations, which is 26 ms a series. The panel is assembled once before an estimation
  that takes minutes.

## Decision

File storage stays: JSON for state, CSV for series, content-addressed folders with `npz` for
versions, lock files for jobs. For a single-user local tool, a migration would be rebuilding for
its own sake: it closes no observed failure, adds the `pyarrow` dependency and requires moving the
data.

The decision is reopened if at least one condition holds:

- several people use one workspace at once, or one resource needs parallel writers;
- the workspace sits on a network drive, where atomic rename and `O_EXCL` are not guaranteed;
- path arrays stop fitting in memory;
- assembling an `as_of` slice becomes a noticeable share of the calculation time.

## Consequences

State can be read and checked with ordinary tools, and the hashes of existing versions do not
change. The price is no multi-file transactions and no queries over the catalogue. A failure
between two writes leaves traces the code handles: the next job takes over a stale lock, and
`jobs(action="state")` closes unfinished job records. Searching versions and runs remains a walk
over folders.
