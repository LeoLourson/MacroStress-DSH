"""The versions of every date of a check, estimated in parallel before the windows open.

A check estimates its models once per calculation date, and the dates do not depend on one
another: the macro version of a date reads only the data known on it, the satellite bundle only
that version and the driver series. ``prepare`` estimates the missing ones in a pool of worker
processes and leaves them in the cache under ``<models>/backtest/``; the windows then open from
the cache in the order of the check, so the report is the same whatever the number of workers.

A refusal of a model is cached too (``windows.refuse``); the window that reads it is refused
with the same reason. Any other error is a defect and ends the check.

Each worker runs its linear algebra on one thread: the pool is the parallelism, and a pool of
multithreaded workers would only fight for the cores.
"""

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import date
from pathlib import Path

from mst.data.store import SeriesStore
from mst.macro.registry import Registry
from mst.model_sdk import ShortHistory
from mst.model_sdk.checkpoint import report

from .windows import Window, _data_key, cached, macro_name, satellite_name, simulate

THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
"""The variables that set the threads of the linear algebra a worker process starts with."""


def default_workers() -> int:
    """Half the logical processors, at least one: numerical work gains little from the second
    thread of a core, and the half leaves the memory of the machine for the versions."""
    return max(1, (os.cpu_count() or 1) // 2)


def prepare(
    plugin,
    data: SeriesStore,
    registry: Registry,
    dates: Iterable[date],
    root: Path,
    *,
    inputs: Path | None = None,
    drivers: Path | None = None,
    workers: int = 1,
) -> None:
    """The macro versions of ``dates`` and, with a driver catalog, their satellite bundles,
    estimated where the cache does not hold them, by ``workers`` processes."""
    chosen = sorted(set(dates))
    missing = [
        when
        for when in chosen
        if not _known(root / _data_key(plugin, data, registry, when, inputs), root / "macro")
    ]
    _run(
        "macro versions",
        _macro,
        [(plugin, data.root, registry, when, root, inputs) for when in missing],
        workers,
    )
    if drivers is None:
        return
    tasks = []
    for when in chosen:
        try:
            name = macro_name(plugin, data, registry, when, root, inputs)
        except ValueError:
            continue
        tasks.append((name, when, data.root, root, drivers))
    _run("satellite bundles", _satellites, tasks, workers)


def simulated(
    build: Callable[..., Window], items: list[tuple], registry, plan, *, workers: int = 1
) -> list[Window | str]:
    """The windows ``build(*item)`` opens, simulated by ``workers`` processes, in the order of
    ``items``; a window the build refuses gives the reason instead.

    ``build`` is a module-level function, so a worker can run it; the windows read their
    versions from the cache ``prepare`` filled. The simulation is seeded by the plan, so the
    paths do not depend on the process that drew them.
    """
    tasks = [(build, item, registry, plan) for item in items]
    count = min(workers, len(tasks))
    if count <= 1:
        return [_simulated(*task) for task in tasks]
    report(f"windows: {len(tasks)} to simulate, {count} process(es)")
    found: list[Window | str] = []
    with _single_threaded(), ProcessPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(_simulated, *task) for task in tasks]
        for done, future in enumerate(futures, start=1):
            found.append(future.result())
            report(f"windows: {done} of {len(tasks)}")
    return found


class Short(str):
    """The reason a window was not opened because its date is short of history for the model:
    every variable the model was given is too short (``ShortHistory``)."""


def _simulated(build: Callable[..., Window], item: tuple, registry, plan) -> Window | str:
    """One window opened and simulated; the reason when its build refuses it, a ``Short``
    reason when the date is short of history for the model."""
    try:
        window = build(*item)
    except ShortHistory as short:
        return Short(short)
    except ValueError as error:
        return str(error)
    simulate(window, registry, plan)
    return window


def _known(pointer: Path, folder: Path) -> bool:
    """Whether the cache answers ``pointer``: with a version or with a refusal of this code."""
    try:
        return cached(pointer, folder) is not None
    except ValueError:
        return True


def _macro(plugin, store: Path, registry: Registry, when: date, root: Path, inputs) -> None:
    macro_name(plugin, SeriesStore(store), registry, when, root, inputs)


def _satellites(macro: str, when: date, store: Path, root: Path, drivers: Path) -> None:
    satellite_name(macro, when, SeriesStore(store), root, drivers)


def _run(what: str, task: Callable, arguments: list[tuple], workers: int) -> None:
    """``task`` on every argument tuple; a refusal is already cached by the task itself."""
    if not arguments:
        return
    count = min(workers, len(arguments))
    report(f"{what}: {len(arguments)} to estimate, {count} process(es)")
    if count == 1:
        for done, item in enumerate(arguments, start=1):
            _refusal_kept(task, item)
            report(f"{what}: {done} of {len(arguments)}")
        return
    with _single_threaded(), ProcessPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(_refusal_kept, task, item) for item in arguments]
        for done, future in enumerate(as_completed(futures), start=1):
            future.result()
            report(f"{what}: {done} of {len(arguments)}")


def _refusal_kept(task: Callable, item: tuple) -> None:
    """The task, whose refusal the cache has kept; the window reads it from there."""
    try:
        task(*item)
    except ValueError:
        pass


@contextmanager
def _single_threaded():
    """One thread of linear algebra in the worker processes started inside the block."""
    saved = {name: os.environ.get(name) for name in THREADS}
    os.environ.update({name: "1" for name in THREADS})
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
