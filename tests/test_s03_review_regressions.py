import json
import shutil

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.feature_selection import chi2

from kkt_clf import experiment as exp, selection as sel
from kkt_clf.vectorizer import refit_selected_vectorizer
from tests.test_s03_selection import fitted
from tests.test_s03_experiment_contract import saved_experiment


def test_multiclass_global_matches_sklearn_imbalanced():
    X = sp.csr_matrix([[10., 0., 1.], [0., 1., 2.], [0., 1., 1.], [0., 3., 1.], [1., 0., 4.]])
    y = np.array([0, 1, 1, 1, 2])
    expected, _ = chi2(X, y)
    result = sel._select_branch_global(X, y, X.shape[1])
    np.testing.assert_allclose(result.scores, expected[result.indices], rtol=1e-6)


def test_class_aware_positive_sign_uses_class_frequency():
    X = sp.csr_matrix([[10., 0.5], [0., 0.75], [0., 0.75]])
    result, _ = sel._select_branch_class_aware(X, np.array([0, 1, 1]), 2)
    # Для признака 1: observed=1.5 > expected=2 * 2 / 3.
    assert dict(zip(result.indices, result.origin)) == {0: 'class:0', 1: 'class:1'}


def test_class_aware_equal_observed_and_expected_uses_fallback():
    X = sp.csr_matrix([[10., 1.], [0., 1.], [0., 1.]])
    result, _ = sel._select_branch_class_aware(X, np.array([0, 1, 1]), 2)
    # Для признака 1: observed=expected=3 * 2 / 3=2.
    assert dict(zip(result.indices, result.origin)) == {0: 'class:0', 1: sel.ORIGIN_GLOBAL_FALLBACK}


@pytest.mark.parametrize('share', [0., 1.])
def test_empty_branch_refit_or_explicit_early_rejection(fitted, tmp_path, share):
    path, names, full, meta, _ = fitted
    X = full.transform(names)
    try:
        result = sel.select_features(X, np.arange(len(names)) % 2, sel.SelectionConfig(k=2, char_share=share), full.n_char_features)
    except (ValueError, sel.SelectionError):
        return
    selected = refit_selected_vectorizer(str(path), full, meta, result.to_payload(), out_dir=str(tmp_path / 'out'))
    assert selected.builder.transform(names).shape[1] == 2


def test_stages_include_failed_and_skipped():
    planned = [{'run_id': 'a'}, {'run_id': 'b'}]
    results = {'a': {'status': 'failed', 'stages': [{'run_id': 'a', 'stage': 'selection', 'status': 'failed', 'seconds': 1.0}]}, 'b': {'status': 'skipped', 'stages': []}}
    frame = exp._stages_frame(planned, results)
    for run in ('a', 'b'):
        assert set(frame[frame.run_id == run].stage) >= {'selection', 'selected_tfidf_refit', 'mlp_train', 'validation_inference'}
    assert frame.query("run_id == 'a' and stage == 'selection'").iloc[0]['status'] == 'failed'
    assert (frame[frame.run_id == 'b']['status'] == 'skipped').all()


def test_graph_does_not_connect_across_missing_budget():
    svg = exp._svg_line_chart('test', [50, 100, 200], [('global', 'red', {50: .4, 200: .8})])
    import re
    for points in re.findall(r'<polyline[^>]*points="([^"]*)"', svg):
        assert len(points.split()) <= 1
    assert svg.count('<circle') == 2


def test_report_regeneration_from_four_files(saved_experiment):
    paths = exp.build_report(str(saved_experiment))
    shutil.rmtree(saved_experiment / 'runs')
    (saved_experiment / 'experiment.json').unlink()
    (saved_experiment / 'report/report.html').unlink()
    exp.build_report(str(saved_experiment))
    assert 'global_50000' in (saved_experiment / 'report/report.html').read_text()


def test_manifest_contains_full_tfidf_config(saved_experiment):
    path = saved_experiment / 'experiment.json'
    metadata = json.loads(path.read_text())
    config = {'char': {'ngram_range': [2, 4], 'min_df': 2}, 'word': {'ngram_range': [1, 1], 'min_df': 3}}
    metadata['inputs']['full_tfidf']['tfidf_config'] = config
    path.write_text(json.dumps(metadata))
    paths = exp.build_report(str(saved_experiment))
    manifest = json.loads(open(paths['manifest_json']).read())
    assert config in (manifest.get('configs', {}).get('tfidf'), manifest.get('inputs', {}).get('full_tfidf', {}).get('tfidf_config'))
