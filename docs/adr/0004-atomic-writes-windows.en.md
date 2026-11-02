# ADR 0004: Atomic writes retry when a Windows reader holds the target

Date: 2026-11-03
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

Both atomic writes — `store.write_json` for the flat JSON state and `versions.save` for a
version folder — move the draft through `store.replace`, which retries `os.replace` a few
times (six tries, 20 ms apart) before letting the `PermissionError` through. A reader
working through the file gives up the handle inside the retry window; a holder that never
lets go still gets the error, and the draft is cleaned up as before.

## Consequences

A state write racing the prompt section's read no longer fails the call that caused it;
the worst case is a delay of about a tenth of a second. The retry changes nothing on
POSIX, where the first `os.replace` succeeds anyway. Locking the reader instead — a
shared advisory lock protocol between the Python server and the Node plugin — was
rejected: it would couple two runtimes to agree on locking semantics for the sake of a
window measured in milliseconds.
