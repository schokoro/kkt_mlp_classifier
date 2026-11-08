"""Statistical contract tests derived from the original S08 pre-registration.

Only generated categorical predictions are used. No production holdout reads.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from kkt_clf import s08_stats as stats


def paired_rows(n_classes=3):
    labels = [f'class_{i:03d}' for i in range(n_classes)]
    truth = np.repeat(labels, 3)
    full = np.tile(truth, (3, 1))
    candidate = full.copy()
    for seed in range(3):
        # Distinct seed-local errors force pairing to matter.
        full[seed, seed] = labels[1]
        candidate[seed, 3 + seed] = labels[0]
        candidate[seed, seed] = labels[2 % n_classes]
    return truth, full, candidate, labels


def macro_f1_from_counts(truth_ids, pred_ids, n_labels):
    """Independent sufficient-statistic implementation, retaining absent labels."""
    support = np.bincount(truth_ids, minlength=n_labels)
    predicted = np.bincount(pred_ids, minlength=n_labels)
    tp = np.bincount(truth_ids[truth_ids == pred_ids], minlength=n_labels)
    denominator = support + predicted
    return np.divide(2 * tp, denominator, out=np.zeros(n_labels, dtype=float),
                     where=denominator != 0).mean()


def oracle(truth, full, candidate, labels, B=5000, rng_seed=808):
    mapping = {label: i for i, label in enumerate(labels)}
    y = np.array([mapping[label] for label in truth])
    f = np.array([[mapping[label] for label in row] for row in full])
    c = np.array([[mapping[label] for label in row] for row in candidate])
    strata = [np.flatnonzero(y == i) for i in range(len(labels))]
    rng = np.random.default_rng(rng_seed)
    deltas = []
    for _ in range(B):
        # Draw ONCE per replicate, then reuse across methods AND all seed pairs.
        idx = np.concatenate([rng.choice(rows, size=len(rows), replace=True) for rows in strata])
        assert np.array_equal(np.bincount(y[idx], minlength=len(labels)),
                              np.bincount(y, minlength=len(labels)))
        deltas.append(np.mean([
            macro_f1_from_counts(y[idx], c[seed, idx], len(labels)) -
            macro_f1_from_counts(y[idx], f[seed, idx], len(labels))
            for seed in range(3)]))
    seed_deltas = [f1_score(truth, c_row, labels=labels, average='macro', zero_division=0) -
                   f1_score(truth, f_row, labels=labels, average='macro', zero_division=0)
                   for c_row, f_row in zip(candidate, full)]
    return np.array(deltas), np.array(seed_deltas)


def test_ci_matches_independent_row_level_paired_oracle():
    truth, full, candidate, labels = paired_rows()
    expected, seed_deltas = oracle(truth, full, candidate, labels)
    actual = stats.paired_bootstrap(truth, full, candidate, labels=labels, fixture=True)
    assert actual['delta'] == pytest.approx(seed_deltas.mean(), abs=1e-15)
    np.testing.assert_allclose(actual['seed_deltas'], seed_deltas, atol=1e-15)
    assert actual['lower_bound'] == pytest.approx(np.quantile(expected, .025, method='linear'), abs=1e-15)
    assert actual['upper_descriptive_bound'] == pytest.approx(np.quantile(expected, .975, method='linear'), abs=1e-15)
    assert actual == stats.paired_bootstrap(truth, full, candidate, labels=labels, fixture=True)


def test_production_fixed_293_universe_and_candidate_minus_full_sign():
    truth, full, candidate, labels = paired_rows(293)
    actual = stats.paired_bootstrap(truth, full, candidate, labels=labels)
    seed_deltas = [f1_score(truth, c, labels=labels, average='macro', zero_division=0) -
                   f1_score(truth, f, labels=labels, average='macro', zero_division=0)
                   for c, f in zip(candidate, full)]
    assert actual['delta'] == pytest.approx(np.mean(seed_deltas), abs=1e-15)
    assert actual['delta'] < 0
    assert len(actual['seed_deltas']) == 3
    assert actual['class_support'] == [3] * 293


def test_strict_inequality_null_margin_and_inconclusive_do_not_prove_inferiority():
    result = {'delta': -.01, 'lower_bound': -.01, 'upper_bound': .03}
    equal = stats.ni_decision(result, .01)
    assert equal['status'] == 'not_established'
    favorable = stats.ni_decision({**result, 'lower_bound': np.nextafter(-.01, np.inf)}, .01)
    assert favorable['status'] == 'non_inferior'
    unfavorable = stats.ni_decision({**result, 'lower_bound': -.02}, .01)
    assert unfavorable['status'] == 'not_established'
    pending = stats.ni_decision(result, None)
    assert pending['status'] == 'not_run_awaiting_approval'
    assert pending['delta_NI'] is None


@pytest.mark.parametrize('margin', [0, -.01, float('nan'), float('inf'), True])
def test_invalid_margin_never_gives_ni(margin):
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        stats.ni_decision({'delta': 0, 'lower_bound': 0, 'upper_bound': 0}, margin)


@pytest.mark.parametrize('damage', [
    'missing_prediction', 'unknown_prediction', 'missing_seed', 'extra_seed',
    'missing_true', 'absent_class', 'duplicate_label', 'single_support',
    'short_rows', 'duplicate_rows', 'shared_group', 'cross_class_group',
    'B_too_small', 'alpha_changed', 'production_small_universe'])
def test_statistical_design_fails_closed(damage):
    truth, full, candidate, labels = paired_rows()
    kwargs = {'labels': labels, 'fixture': True, 'row_ids': np.arange(len(truth)),
              'groups': np.arange(len(truth)).astype(str)}
    if damage == 'missing_prediction':
        candidate = candidate.astype(object)
        candidate[0, 0] = None
    elif damage == 'unknown_prediction':
        candidate = candidate.astype(object)
        candidate[0, 0] = 'unknown'
    elif damage == 'missing_seed':
        candidate = candidate[:2]
    elif damage == 'extra_seed':
        candidate = np.concatenate([candidate, candidate[:1]])
    elif damage == 'missing_true':
        truth = truth.astype(object)
        truth[0] = None
    elif damage == 'absent_class':
        kwargs['labels'] = labels + ['missing']
    elif damage == 'duplicate_label':
        kwargs['labels'] = labels + [labels[0]]
    elif damage == 'single_support':
        idx = np.arange(len(truth))[2:]
        truth, full, candidate = truth[idx], full[:, idx], candidate[:, idx]
        kwargs['row_ids'] = np.arange(len(truth))
        kwargs['groups'] = np.arange(len(truth)).astype(str)
    elif damage == 'short_rows':
        kwargs['row_ids'] = kwargs['row_ids'][:-1]
    elif damage == 'duplicate_rows':
        kwargs['row_ids'][1] = kwargs['row_ids'][0]
    elif damage in ('shared_group', 'cross_class_group'):
        kwargs['groups'][3 if damage == 'cross_class_group' else 1] = kwargs['groups'][0]
    elif damage == 'B_too_small':
        kwargs['B'] = 4999
    elif damage == 'alpha_changed':
        kwargs['alpha'] = .05
    else:
        kwargs['fixture'] = False
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        stats.paired_bootstrap(truth, full, candidate, **kwargs)


def test_all_ties_with_unpredicted_class_are_finite_zero_delta():
    truth, full, _, labels = paired_rows()
    full[:] = labels[0]  # Other classes receive zero predicted positives.
    result = stats.paired_bootstrap(truth, full, full.copy(), labels=labels, fixture=True)
    assert result['delta'] == result['lower_bound'] == result['upper_descriptive_bound'] == 0
    assert result['seed_deltas'] == [0, 0, 0]


@pytest.mark.parametrize('field', ['row_ids', 'groups'])
@pytest.mark.parametrize('missing', [None, float('nan')])
def test_missing_structural_identity_is_not_a_valid_singleton(field, missing):
    truth, full, candidate, labels = paired_rows()
    identity = np.arange(len(truth)).astype(object)
    identity[0] = missing
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        stats.paired_bootstrap(truth, full, candidate, labels=labels, fixture=True, **{field: identity})


def test_regeneration_recipe_freezes_counts_rng_ci_and_zero_division():
    truth, full, candidate, labels = paired_rows()
    result = stats.paired_bootstrap(truth, full, candidate, labels=labels, fixture=True)
    assert result['class_support'] == [3, 3, 3]
    recipe = result['recipe']
    assert recipe['B'] >= 5000 and recipe['rng_seed'] == 808 and recipe['alpha'] == .025
    assert recipe['percentile_method'] == 'linear' and recipe['zero_division'] == 0
    assert recipe['shared_indices_across_seeds'] is True
    assert recipe['seed_order'] == [42, 43, 44]
    assert recipe['stratification'] == 'true_L3' and recipe['unit'] == 'row'
