"""One position through its model, with no branch on its type.

``prepare`` asks the model for its requirements, resolves them against the driver catalog of
the run's satellite version and the macro paths, and lets the model judge the result. A model
that breaks its declaration does not reach the portfolio: its position stays in the run as
``invalid`` with the reason. ``calculate`` runs the model on the run's paths and checks what it
returned against the model's declaration — identity, metrics and the number of realisations.
"""

from pydantic import ValidationError

from ..drivers.registry import Registry
from ..plugin_sdk import fx
from ..plugin_sdk.model import (
    CalculationContext,
    DataRequirements,
    Diagnostic,
    Position,
    PreparedInputs,
    RunOptions,
)
from ..plugin_sdk.paths import Paths
from ..plugin_sdk.results import CalculationResult
from .model_catalog import Catalog
from .model_catalog import catalog as default_catalog


def prepare(
    position: Position,
    options: RunOptions,
    paths: Paths,
    drivers: Registry,
    served,
    catalog: Catalog = default_catalog,
) -> tuple[PreparedInputs | None, list[Diagnostic]]:
    """The prepared inputs of one position and what the platform and its model found.

    ``drivers`` is the driver registry of the run's satellite version; ``served`` are the
    drivers that version serves. A ``DriverRequirement`` the run cannot serve stays unresolved
    and the position is not calculated.
    """
    try:
        model = catalog.model_for(position)
    except (KeyError, ValueError) as error:
        return None, [Diagnostic(code="no_model", message=str(error))]
    if model.terms_schema is not None:
        problems = terms_problems(model.terms_schema, position.terms)
        if problems:
            return None, problems
    try:
        needs = model.requirements(position, options)
    except (ValidationError, ValueError) as error:
        return None, [Diagnostic(code="requirements_failed", message=str(error))]
    problems = [
        Diagnostic(
            code=f"macro_missing:{variable}",
            message=f"the paths carry no macro variable {variable}",
        )
        for variable in needs.macro
        if variable not in paths.variables
    ]
    value_current = 0.0
    if not problems:
        try:
            value_current = fx.exposure(position, paths, options.base_currency)
        except ValueError as error:
            problems.append(Diagnostic(code=f"fx_missing:{position.currency}", message=str(error)))
    prepared = PreparedInputs(
        position=position,
        model_id=model.model_id,
        as_of=options.as_of,
        horizon=options.horizon,
        base_currency=options.base_currency,
        value_current=value_current,
        requirements=needs,
    )
    for requirement in needs.drivers:
        driver = drivers.resolve(
            requirement.quantity,
            object_=requirement.object,
            codes=requirement.codes or position.codes,
            among=served,
        )
        if driver is None:
            prepared.missing[requirement.key] = (
                f"no served {requirement.quantity} for {selector(requirement, position)}"
            )
        else:
            prepared.drivers[requirement.key] = driver
    try:
        found = model.validate(prepared)
    except Exception as error:  # noqa: BLE001 - one broken model must not end the run
        found = [
            Diagnostic(
                code=f"model_error:{type(error).__name__}", message=str(error)[:500]
            )
        ]
    problems += found
    problems += required_driver_problems(needs, prepared.missing, problems, position)
    return prepared, problems


def calculate(
    prepared: PreparedInputs,
    diagnostics: list[Diagnostic],
    context: CalculationContext,
    catalog: Catalog = default_catalog,
) -> CalculationResult:
    """The model's result, checked against its declaration; a broken one is ``invalid``."""
    model = catalog.model(prepared.model_id)
    if any(item.severity == "error" for item in diagnostics):
        return not_run(prepared, diagnostics, model)
    try:
        result = model.calculate(prepared, context)
        problems = output_problems(result, prepared, context, model)
    except Exception as error:  # noqa: BLE001 - one position must not end the run
        problems = [
            Diagnostic(code=f"model_error:{type(error).__name__}", message=str(error)[:500])
        ]
    if problems:
        return not_run(prepared, [*diagnostics, *problems], model)
    # Warnings of the input check travel with the result.
    known = {(item.code, item.message) for item in result.diagnostics}
    return result.model_copy(
        update={
            "diagnostics": [
                *(item for item in diagnostics if (item.code, item.message) not in known),
                *result.diagnostics,
            ]
        }
    )


def not_run(
    prepared: PreparedInputs, diagnostics: list[Diagnostic], model
) -> CalculationResult:
    """A position that was not calculated stays in the run with its reason, without numbers."""
    blocking = [item for item in diagnostics if item.severity == "error"] or [
        Diagnostic(code="not_calculated", message="the position was not run")
    ]
    return CalculationResult(
        position_id=prepared.position.id,
        name=prepared.position.name,
        instrument_type=prepared.position.type,
        model_id=model.model_id,
        model_version=model.version,
        status="invalid",
        base_currency=prepared.base_currency,
        value_current=prepared.value_current,
        flags=[item.code for item in blocking],
        diagnostics=diagnostics,
    )


def terms_problems(schema, terms: dict) -> list[Diagnostic]:
    """The position's terms against the schema its model declares."""
    try:
        schema.model_validate(terms)
    except ValidationError as error:
        return [
            Diagnostic(
                code="terms_invalid",
                message=f"/{'.'.join(str(part) for part in item['loc'])}: {item['msg']}",
            )
            for item in error.errors()
        ]
    return []


def selector(requirement, position: Position) -> str:
    """What a driver requirement selects, in the words of the position."""
    if requirement.object is not None:
        return requirement.object
    codes = requirement.codes or position.codes
    return ", ".join(f"{kind} {codes[kind]}" for kind in sorted(codes)) or "no codes"


def required_driver_problems(
    needs: DataRequirements,
    missing: dict[str, str],
    found: list[Diagnostic],
    position: Position,
) -> list[Diagnostic]:
    """A required driver the run does not serve stops the position unless the model said so."""
    stated = {item.code for item in found}
    problems = []
    for requirement in needs.drivers:
        code = f"driver_missing:{requirement.quantity}"
        if not requirement.required or requirement.key not in missing or code in stated:
            continue
        problems.append(
            Diagnostic(
                code=code,
                message=(
                    f"нет драйвера {requirement.quantity} "
                    f"для {selector(requirement, position)}"
                ),
            )
        )
    return problems


def output_problems(
    result: CalculationResult,
    prepared: PreparedInputs,
    context: CalculationContext,
    model,
) -> list[Diagnostic]:
    """How a result departs from its model's declaration."""
    expected = {
        "position_id": prepared.position.id,
        "instrument_type": prepared.position.type,
        "model_id": model.model_id,
        "model_version": model.version,
        "base_currency": prepared.base_currency,
        "value_current": prepared.value_current,
    }
    problems = [
        Diagnostic(
            code=f"output:{name}",
            message=f"{name} is {getattr(result, name)!r}, expected {value!r}",
        )
        for name, value in expected.items()
        if getattr(result, name) != value
    ]
    if result.status not in ("ok", "needs_review"):
        problems.append(
            Diagnostic(code="output:status", message=f"a calculated result is {result.status!r}")
        )
    if len(result.value_change) != context.paths.draws:
        problems.append(
            Diagnostic(
                code="output:value_change",
                message=(
                    f"{len(result.value_change)} realisations, "
                    f"the run has {context.paths.draws}"
                ),
            )
        )
    for metric_id in result.metrics:
        if metric_id not in model.metrics:
            problems.append(
                Diagnostic(
                    code=f"output:metric:{metric_id}",
                    message=f"{metric_id} is not a declared metric of the model",
                )
            )
    for metric_id in model.required_metrics:
        if metric_id not in result.metrics:
            problems.append(
                Diagnostic(
                    code=f"output:metric:{metric_id}",
                    message=f"required metric {metric_id} is absent",
                )
            )
    return problems
