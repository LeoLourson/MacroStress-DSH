# ADR 0004: Atomic writes retry when a Windows reader holds the target

Date: 2026-11-02
Status: accepted

## Context

State is written atomically everywhere: a draft file (or folder) is prepared next to the
target and `os.replace` moves it into place. On POSIX, renaming past a file another
process is reading is allowed and the reader keeps the old version. On Windows,
`os.replace` raises `PermissionError` for as long as any process holds the target open —
and `workspace.json` is read exactly that way by the DSH plugin, from its own Node
process, whenever it rebuilds the prompt section. A write landing in that window took the
whole tool call down.

## Decision

All atomic writes — `store.write_json` for the flat JSON state, `versions.save` for a
version folder, `SeriesStore.write` for a series and the download of a corpus document — move the draft through `store.replace`, which retries `os.replace` a few
times (six tries, 20 ms apart) before letting the `PermissionError` through. A reader
working through the file gives up the handle inside the retry window; a holder that never
lets go still gets the error, and the draft is cleaned up as before.

## Consequences

A state write that coincides with the prompt section's read no longer fails the call that caused it;
the worst case is a delay of about a tenth of a second. The retry changes nothing on
POSIX, where the first `os.replace` succeeds anyway. Locking the reader instead — a
shared advisory lock protocol between the Python server and the Node plugin — was
rejected: it would couple two runtimes to agree on locking semantics for the sake of a
window measured in milliseconds.

The window is sized for a rare short read. A Node process reading the file non-stop lets only
5 % of replaces through (49 of 955 without retries), and six tries are not enough against it: 2
writes of 25 passed. A Python reader itself gets a `PermissionError` at the moment of a replace
in isolated cases (over 14,628 reads); `read_json` does not retry, so such a call has to be
repeated. A holder that keeps the file longer than 100 ms, an antivirus for one, still gets the
error.
