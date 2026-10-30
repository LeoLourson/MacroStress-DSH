"""Semantic version parsing and range checks, shared by the SDKs.

Both SDKs (``mst.plugin_sdk`` for position models, ``mst.model_sdk`` for macro and satellite
models) follow semantic versioning: a minor release only adds, a major release may break. A
plugin declares the range it was written for; a plugin outside the range blocks start-up with
its name rather than failing in the middle of a run.
"""

import re

_VERSION = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?$")
_CLAUSE = re.compile(r"^(>=|<=|==|<|>)\s*(\d+(?:\.\d+){0,2})$")


def parse(version: str) -> tuple[int, int, int]:
    match = _VERSION.match(version.strip())
    if not match:
        raise ValueError(f"not a version: {version!r}")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


def compatible(spec: str, version: str) -> bool:
    """Whether ``version`` satisfies every comma-separated clause of ``spec``."""
    current = parse(version)
    clauses = [part.strip() for part in spec.split(",") if part.strip()]
    if not clauses:
        raise ValueError("an empty compatibility range")
    for clause in clauses:
        match = _CLAUSE.match(clause)
        if not match:
            raise ValueError(f"not a version clause: {clause!r}")
        operator, bound = match.groups()
        target = parse(bound)
        ok = {
            ">=": current >= target,
            "<=": current <= target,
            "==": current == target,
            "<": current < target,
            ">": current > target,
        }[operator]
        if not ok:
            return False
    return True
