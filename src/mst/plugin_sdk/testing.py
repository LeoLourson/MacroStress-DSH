"""The conformance check every plugin passes: declarations and the demo calculation.

``check_plugin`` reads the plugin's declaration and calculates each of its demo cases through
the platform's own dispatcher, so a plugin is judged by the same path a run takes: the model is
resolved by instrument type, its driver requirements are served by the demo paths, and the
result is checked against the model's own declaration. The same calculation repeated must give
the same numbers.

``calculate_demo`` lets a plugin's own tests run a demo case (or a variant of it) without a
workspace, a satellite version or a network connection.
"""

import json
from dataclasses import dataclass

from ..engine.model_catalog import Catalog
from ..engine.run import run
from ..plugins import PluginError, build, declaration_problems
from .plugin import DemoCase, Plugin


@dataclass(frozen=True)
class Finding:
    """One problem the conformance check found."""

    plugin_id: str
    check: str
    message: str
    case: str = ""

    def __str__(self) -> str:
        where = f" [{self.case}]" if self.case else ""
        return f"{self.plugin_id}{where} {self.check}: {self.message}"


@dataclass(frozen=True)
class DemoDrivers:
    """The drivers of one demo case: the requirement quantity names its path driver."""

    drivers: dict[str, str]

    def resolve(self, quantity: str, *, object_=None, codes=None, among=None) -> str | None:
        """The path driver of ``quantity``; the demo case names one driver per quantity."""
        driver = self.drivers.get(quantity)
        if driver is None or (among is not None and driver not in among):
            return None
        return driver


def calculate_demo(case: DemoCase, registry):
    """The demo case calculated by the engine on its own paths."""
    catalog = Catalog(tuple(registry.models.values()))
    return run([case.position], case.paths, drivers=DemoDrivers(case.drivers), catalog=catalog)


def _fingerprint(batch) -> str:
    return json.dumps(
        [result.model_dump(mode="json") for result in batch.results], sort_keys=True
    )


def check_plugin(plugin: Plugin, registry=None) -> list[Finding]:
    """Every problem found with ``plugin``; an empty list means it conforms."""
    findings = [
        Finding(plugin.plugin_id, "declaration", problem)
        for problem in declaration_problems(plugin)
    ]
    if findings:
        return findings
    if registry is None:
        try:
            registry = build([plugin])
        except PluginError as error:
            return [
                Finding(plugin.plugin_id, "registry", problem) for problem in error.problems
            ]

    for case in plugin.demo:
        try:
            first = calculate_demo(case, registry)
            again = calculate_demo(case, registry)
        except Exception as error:  # noqa: BLE001 - a broken demo is a finding
            findings.append(
                Finding(
                    plugin.plugin_id,
                    "demo",
                    f"{type(error).__name__}: {error}",
                    case.name,
                )
            )
            continue
        if _fingerprint(first) != _fingerprint(again):
            findings.append(
                Finding(
                    plugin.plugin_id,
                    "determinism",
                    "the same demo gave different results",
                    case.name,
                )
            )
        if not case.completes:
            continue
        for result in first.results:
            if result.status != "invalid":
                continue
            reason = "; ".join(
                f"{item.code} {item.message}".strip()
                for item in result.diagnostics
                if item.severity == "error"
            )
            findings.append(
                Finding(
                    plugin.plugin_id,
                    "outputs",
                    f"{result.position_id} did not complete: "
                    f"{reason or 'the position was not calculated'}",
                    case.name,
                )
            )
    return findings
