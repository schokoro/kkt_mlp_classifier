"""Row-paired conditional NI statistics; never reads a dataset or model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .s07 import digest, require

BOOTSTRAP = {'B': 5000, 'rng_seed': 808, 'alpha': .025,
             'stratification': 'true_L3', 'unit': 'row', 'shared_indices_across_seeds': True,
             'percentile_method': 'linear', 'zero_division': 0, 'seed_order': [42, 43, 44]}


def _encode(values, labels):
    lookup = {v: i for i, v in enumerate(labels)}
    try:
        return np.asarray([lookup[v] for v in values], dtype=np.int64)
    except (KeyError, TypeError) as exc:
        raise ValueError('Missing or out-of-universe prediction/label') from exc


def _f1(y, pred, k):
    true = np.bincount(y, minlength=k)
    predicted = np.bincount(pred, minlength=k)
    tp = np.bincount(y[y == pred], minlength=k)
    denom = true + predicted
    return float(np.divide(2. * tp, denom, out=np.zeros(k), where=denom != 0).mean())


def paired_bootstrap(y_true, full_predictions, candidate_predictions, *, labels,
                     row_ids=None, groups=None, B=5000, rng_seed=808, alpha=.025,
                     fixture=False):
    """Same class-stratified sampled row indices in every seed pair and method.

    Inputs have shape (3, n), in seed order 42/43/44. Non-singleton groups
    require a separately validated design and are rejected. No resampling of
    seed-level F1 values; each replicate recomputes all six fixed-universe F1s.
    """
    labels = list(labels)
    require(len(labels) == len(set(labels)) and len(labels) >= 2
            and not pd.isna(labels).any(), 'Invalid class universe')
    require(fixture or len(labels) == 293, 'Production requires fixed 293 L3')
    require(isinstance(B, int) and not isinstance(B, bool) and B >= 5000, 'B must be >=5000')
    require(alpha == .025, 'S08 alpha is fixed at .025')
    require(isinstance(rng_seed, int) and not isinstance(rng_seed, bool), 'Integer RNG seed required')
    truth = _encode(y_true, labels)
    n, k = len(truth), len(labels)
    require(n > 0, 'No rows')
    if row_ids is not None:
        require(len(row_ids) == n and len(set(row_ids)) == n
                and not pd.isna(row_ids).any(), 'Invalid/duplicate row identity')
    if groups is not None:
        require(len(groups) == n and len(set(groups)) == n
                and not pd.isna(groups).any(),
                'Non-singleton/missing groups incompatible with class-stratified row design')
    full, candidate = np.asarray(full_predictions), np.asarray(candidate_predictions)
    require(full.shape == candidate.shape == (3, n), 'Missing/unaligned seed predictions; require (3,n)')
    full = np.stack([_encode(row, labels) for row in full])
    candidate = np.stack([_encode(row, labels) for row in candidate])
    strata = [np.flatnonzero(truth == c) for c in range(k)]
    require(min(map(len, strata)) >= 2, 'Each fixed class needs >=2 rows; otherwise design is insufficient')
    seed_deltas = [_f1(truth, candidate[s], k) - _f1(truth, full[s], k) for s in range(3)]
    rng = np.random.default_rng(rng_seed)
    replicates = np.empty(B)
    for b in range(B):
        idx = np.concatenate([rng.choice(rows, size=len(rows), replace=True) for rows in strata])
        replicates[b] = np.mean([_f1(truth[idx], candidate[s, idx], k)
                                 - _f1(truth[idx], full[s, idx], k) for s in range(3)])
    lower, upper = np.quantile(replicates, [alpha, 1-alpha], method='linear')
    return {'delta': float(np.mean(seed_deltas)), 'seed_deltas': seed_deltas,
            'lower_bound': float(lower), 'upper_descriptive_bound': float(upper),
            'bootstrap_std': float(replicates.std()), 'class_support': list(map(len, strata)),
            'recipe': {**BOOTSTRAP, 'B': B, 'rng_seed': rng_seed,
                       'generator': 'numpy.default_rng/PCG64', 'labels_sha256': digest(labels),
                       'inputs_sha256': digest({'truth': truth.tolist(), 'full': full.tolist(),
                                                'candidate': candidate.tolist()}),
                       'numpy_version': np.__version__},
            'interpretation': 'Conditional on rows/class composition and three fixed seeds; no dataset-shift claim'}


def ni_decision(result, margin):
    """Pure arithmetic only, not an authorization API or confirmatory report."""
    if margin is None:
        return {'status': 'not_run_awaiting_approval', 'delta_NI': None, 'decision': None}
    require(isinstance(margin, (int, float)) and not isinstance(margin, bool)
            and np.isfinite(margin) and 0 < margin < 1, 'Invalid NI margin')
    bound = result['lower_bound']
    require(np.isfinite(bound), 'Nonfinite bound')
    return {'status': 'non_inferior' if bound > -margin else 'not_established',
            'delta_NI': margin, 'lower_bound': bound, 'strict_rule': 'lower_bound > -delta_NI'}


def simulated_feasibility(pool, labels):
    """Explicit hypothetical scenarios on a dev-derived support composition.

    This does not estimate sealed performance. Sparse supports use a deterministic
    10%-of-development projection with minimum 2, never sealed label access.
    """
    labels = list(labels)
    supports = [max(2, int(round(v * .1))) for v in pool.L3.value_counts().reindex(labels)]
    y = np.repeat(np.arange(len(labels)), supports)
    rng = np.random.default_rng(809)
    full = np.tile(y, (3, 1))
    common_error = rng.random(len(y)) < .25
    wrong = (y + rng.integers(1, len(labels), len(y))) % len(labels)
    for s in range(3):
        errors = common_error | (rng.random(len(y)) < .03)
        full[s, errors] = wrong[errors]
    scenarios = []
    for disagreement in (.03, .10):
        cand = full.copy()
        # Both gains and losses; identical rows remain paired across all seeds.
        changes = rng.random(len(y)) < disagreement
        cand[:, changes] = np.where(full[:, changes] == y[changes], wrong[changes], y[changes])
        out = paired_bootstrap(y, full, cand, labels=list(range(len(labels))), fixture=True)
        scenarios.append({'hypothetical_disagreement_probability': disagreement, **out,
                          'percentile_width': out['upper_descriptive_bound'] - out['lower_bound'],
                          'downward_uncertainty': out['delta'] - out['lower_bound'],
                          'null_effect_centered_lower_bound': out['lower_bound'] - out['delta'],
                          'centering_note': 'Exploratory precision projection only: shift observed simulated '
                                            'point to zero; not a bootstrap under a fitted null or power estimate.',
                          'illustrative_margin_only': .01,
                          'illustrative_bound_clears_minus_01': out['lower_bound'] > -.01})
    return {'source': 'explicitly_simulated_categorical_predictions; dev-derived support projection',
            'exploratory': True, 'sealed_quality_used': False, 'projected_rows': len(y),
            'projected_support_histogram': {str(v): supports.count(v) for v in sorted(set(supports))},
            'scenarios': scenarios, 'delta_NI': None, 'margin_status': 'unapproved',
            'conclusion': 'Precision depends on unknown paired error correlation. Illustrative .01 may be unresolved; '
                          'these scenarios cannot approve a margin or establish NI. Sparse-class bootstrap is conditional '
                          'and cannot represent unobserved error modes; collect independent annotations if precision is inadequate.'}
