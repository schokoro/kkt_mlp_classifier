"""Regression tests for the five findings in S03's second review."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from kkt_clf import experiment as exp, selection as sel
from kkt_clf.vectorizer import fit_and_save_vectorizer
from tests.test_s03_selection import fitted
from tests.test_s03_experiment_contract import saved_experiment
from tests.test_s01_vectorizer import make_cfg


def test_class_aware_under_nofile_limit_closes_descriptors(tmp_path):
    pytest.importorskip('resource')
    script = r'''
import os, resource
import numpy as np
import scipy.sparse as sp
from kkt_clf import selection as sel
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
def fds(): return len(os.listdir('/proc/self/fd'))
before = fds()
n = 100
X = sp.eye(n, format='csr', dtype=np.float64)
for _ in range(2):
    result, report = sel._select_branch_class_aware(X, np.arange(n), n)
    assert len(result.indices) == n
    assert report['n_classes'] == n
    assert fds() == before, (before, fds())
'''
    proc = subprocess.run([sys.executable, '-c', script], text=True, capture_output=True,
                          env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}, timeout=60)
    assert proc.returncode == 0, proc.stderr


def test_saved_global_ranking_preserves_close_float64_order(tmp_path):
    X = sp.csr_matrix([[1.0, 1.00000001], [0.0, 0.0]], dtype=np.float64)
    result = sel._select_branch_global(X, np.array([0, 1]), 2, str(tmp_path), 'char')
    assert result.indices[0] == 1
    raw = (tmp_path / 'char_global.npy').read_bytes()
    assert len(raw) == 2 * np.dtype('float64').itemsize
    scores = np.frombuffer(raw, dtype=np.float64)
    assert scores[1] > scores[0]
    np.testing.assert_array_equal(sel._rank_topk(scores, 2), result.indices)


def test_real_experiment_has_one_row_per_stage(fitted, tmp_path):
    base = make_cfg(tmp_path)
    corpus = base.data.path
    vectorizer = tmp_path / 'full'
    fit_and_save_vectorizer(str(vectorizer), base.features, corpus)
    cfg = exp.ExperimentConfig(output_dir=str(tmp_path / 'experiment'), full_tfidf_path=str(vectorizer),
        corpus_path=str(corpus), data=base.data, model=base.model, loss=base.loss, train=base.train,
        selection=exp.SelectionPlanConfig(methods=['global'], budgets=[4], char_share=.5))
    result = exp.run_experiment(cfg)
    assert result['statuses'] == {'none': 'success', 'global_4': 'success'}, result['results']
    paths = exp.build_report(cfg.output_dir)
    stages = pd.read_csv(paths['stages_csv'])
    assert not stages.duplicated(['run_id', 'stage']).any(), stages.to_string()
    assert set(stages.query("run_id == 'none'").stage) >= {'selection', 'selected_tfidf_refit', 'mlp_train', 'validation_inference'}


def test_report_survives_corrupt_metrics_and_incomplete_per_class(saved_experiment):
    path = saved_experiment / 'runs/global_50000/run_result.json'
    result = json.loads(path.read_text())
    result['metrics']['l3_macro_f1'] = 'broken'
    result['n_features'] = 'bad'
    result['per_class'] = [{'L3': 'a', 'support': 2, 'f1': 'broken'}, {'L3': 'b'}]
    path.write_text(json.dumps(result))
    paths = exp.build_report(str(saved_experiment))
    runs = pd.read_csv(paths['runs_csv']).set_index('run_id')
    assert len(runs) == 7 and runs.loc['none', 'status'] == 'success'
    assert runs.loc['global_50000', 'status'] != 'success' or pd.isna(runs.loc['global_50000', 'delta_macro_f1'])
    pc = pd.read_parquet(paths['per_class_parquet'])
    assert set(pc.query("run_id == 'none'").L3) == {'a', 'b', 'absent'}
    assert 'global_50000' in open(paths['report_html'], encoding='utf8').read()


def test_overwrite_failure_preserves_old_artifact_and_success_replaces(fitted, tmp_path, monkeypatch):
    _, names, full, _, _ = fitted
    X = full.transform(names)
    y = np.arange(len(names)) % 2
    cfg = sel.SelectionConfig(method='global', k=8, char_share=.5)
    path = tmp_path / 'selection'
    old = sel.select_features(X, y, cfg, full.n_char_features)
    sel.save_selection_artifact(str(path), old, {'version': 'old'})
    before = {p.relative_to(path): p.read_bytes() for p in path.rglob('*') if p.is_file()}
    new_cfg = sel.SelectionConfig(method='global', k=6, char_share=.5)
    new = sel.select_features(X, y, new_cfg, full.n_char_features)
    original = sel.joblib.dump
    def interrupted(value, filename, *args, **kwargs):
        original(value, filename, *args, **kwargs)
        raise OSError('simulated interrupted write')
    with monkeypatch.context() as m:
        m.setattr(sel.joblib, 'dump', interrupted)
        with pytest.raises(OSError, match='simulated'):
            sel.save_selection_artifact(str(path), new, {'version': 'new'}, overwrite=True)
    after = {p.relative_to(path): p.read_bytes() for p in path.rglob('*') if p.is_file()}
    assert after == before
    assert sel.load_selection_artifact(str(path))[1]['provenance'] == {'version': 'old'}
    replacement = sel.select_features(X, y, new_cfg, full.n_char_features)
    sel.save_selection_artifact(str(path), replacement, {'version': 'new'}, overwrite=True)
    assert sel.load_selection_artifact(str(path))[1]['provenance'] == {'version': 'new'}
    assert not (path / 'ratings' / 'ratings').exists()
