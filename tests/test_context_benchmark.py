"""Context policies keep the same questions without bringing back whole history."""

import importlib.util
from pathlib import Path


def benchmark():
    spec = importlib.util.spec_from_file_location(
        'elicitation_check', Path('bench/elicitation-context/check.py')
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reset_policies_keep_four_separate_horizon_questions():
    module = benchmark()
    for _h in (1, 3, 6, 12):
        assert module.reset_context('bounded', 'call', 'gl.brent', 'gl.brent')
        assert not module.reset_context('history', 'call', 'gl.brent', 'gl.brent')
        assert not module.reset_context('per-variable', 'call', 'gl.brent', 'gl.brent')
    assert module.reset_context('per-variable', 'call', 'gl.fed_rate', 'gl.brent')
    assert module.reset_context('history', None, 'gl.brent', None)


def test_benchmark_uses_exactly_one_current_tool_and_separate_horizon_questions():
    module = benchmark()
    assert 'single current question' in module.SYSTEM
    assert 'rate percentage points' in module.SYSTEM
    assert module.compact({'q5': -1, 'q50': 0, 'q95': 1}) == '{"q5":-1,"q50":0,"q95":1}'
