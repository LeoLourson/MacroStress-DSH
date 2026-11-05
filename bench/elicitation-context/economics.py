"""Descriptive economic checks for saved sample answers, not acceptance rules."""

import math


def describe(parts):
    sizes = parts['law']['size']
    oil = {}
    rate = {}
    for horizon in (1, 3, 6, 12):
        for variable, target in [('gl.brent', oil), ('gl.fed_rate', rate)]:
            step = f'response:{variable}:{horizon}'
            if step not in parts:
                continue
            medians = [parts[step][f'at_{node}']['q50'] for node in ('q5', 'q50', 'q95')]
            target[str(horizon)] = medians
    reference = [100 * math.log1p(sizes[node] / 100) if sizes[node] > -100 else None
                 for node in ('q5', 'q50', 'q95')]
    return {
        'own_impact_reference_if_size_is_causal_contribution': reference,
        'brent_median_log100': oil,
        'brent_median_percent': {h: [100 * math.expm1(x / 100) for x in values]
                                 for h, values in oil.items()},
        'fed_rate_median_percentage_points': rate,
        'reference_is_not_hard_validation': 'Measured total price movement and isolated causal '
        'contribution are not the same conditioning object; definition remains unresolved.',
        'uncertainty_collapsed_cells': [
            f'{step}/{node}' for step, answer in parts.items() if step.startswith('response:')
            for node, quantiles in answer.items()
            if quantiles['q5'] == quantiles['q50'] == quantiles['q95']
        ],
    }
