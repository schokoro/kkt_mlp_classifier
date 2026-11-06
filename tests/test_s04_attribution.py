"""Independent checks of S04 margin attribution and persisted rankings."""
import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch import nn

from kkt_clf.attribution import (MarginAttributions, build_gxi_ratings,
    competitor_indices, compute_margin_attributions, select_attributed_rows,
    select_features_gradient_x_input, AttributionConfig)
from kkt_clf.model import SparseLinear, csr_to_batch
from kkt_clf.selection import SelectionConfig, save_selection_artifact, load_selection_artifact


class TinyModel(nn.Module):
    n_classes = (3, 3, 3)

    def __init__(self, activation):
        super().__init__()
        self.input = SparseLinear(4, 2)
        self.activation = activation
        self.head = nn.Linear(2, 3)
        with torch.no_grad():
            self.input.bag.weight.copy_(torch.tensor([[1., .5], [.5, 1.], [1., -0.5], [.25, .5]]))
            self.input.bias.copy_(torch.tensor([1., 1.]))
            self.head.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., -1.]]))
            self.head.bias.copy_(torch.tensor([0., 0., .1]))

    def forward(self, batch):
        z = self.head(self.activation(self.input(batch)))
        return z, z, z


def test_competitor_excludes_truth_for_correct_wrong_and_ties():
    logits = np.array([[3, 2, 1], [3, 2, 1], [2, 2, 2], [4, 4, 1]])
    np.testing.assert_array_equal(competitor_indices(logits, np.array([0, 1, 1, 2])), [1, 0, 0, 0])


@pytest.mark.parametrize('activation', [nn.ReLU, nn.Tanh])
def test_sparse_attribution_matches_finite_differences_and_preserves_model(activation):
    model = TinyModel(activation())
    model.train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    X = sp.csr_matrix(np.array([[.8, 0, .3, 0], [0, .6, 0, .2], [0, 0, 0, 0]], dtype=np.float32))
    y = np.array([0, 0, 2])
    result = compute_margin_attributions(model, X, y, torch.device('cpu'), batch_size=2)
    assert model.training
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    np.testing.assert_array_equal(result.values.indices, X.indices)
    np.testing.assert_array_equal(result.values.indptr, X.indptr)
    np.testing.assert_array_equal(result.empty, [False, False, True])
    model.eval()
    for i in range(X.shape[0]):
        row = X[i].copy()
        with torch.no_grad():
            logits = model(csr_to_batch(row))[2][0].numpy()
        rival = result.competitor[i]
        assert rival != y[i]
        assert result.margins[i] == pytest.approx(float(logits[y[i]] - logits[rival]), abs=1e-6)
        for j in range(row.nnz):
            eps = 1e-3
            plus, minus = row.copy(), row.copy()
            plus.data[j] += eps
            minus.data[j] -= eps
            with torch.no_grad():
                zp = model(csr_to_batch(plus))[2][0]
                zm = model(csr_to_batch(minus))[2][0]
            derivative = ((zp[y[i]] - zp[rival]) - (zm[y[i]] - zm[rival])).item() / (2 * eps)
            assert result.values.data[X.indptr[i] + j] == pytest.approx(row.data[j] * derivative, abs=2e-4)
    if activation is nn.ReLU:
        # First row: margin z0-z1, gradient [0.5, 1.5] on active columns 0 and 2.
        np.testing.assert_allclose(result.values.data[:2], [.4, .45], atol=1e-6)


def _attributions():
    vals = sp.csr_matrix(np.array([[2., -1., 4., 0.], [0., 0., 0., 0.],
                                   [0., 2., 0., -1.], [0., 0., 0., 0.]]))
    return MarginAttributions(vals, np.array([1., -1., -2., 0.]), np.array([1, 1, 0, 0]),
                              np.array([True, False, False, False]), np.array([False, True, False, True]),
                              np.array([0, 0, 1, 2]), np.arange(4), 3)


def test_class_means_include_empty_and_wrong_examples_and_empty_classes(tmp_path):
    manifest, report = build_gxi_ratings(_attributions(), 2, str(tmp_path))
    branch = manifest['branches']['char']
    indices = np.fromfile(tmp_path / branch['rank_indices'], dtype=np.int64)
    scores = np.fromfile(tmp_path / branch['rank_scores'], dtype=np.float64)
    counts = np.fromfile(tmp_path / branch['rank_counts'], dtype=np.int64)
    classes = np.fromfile(tmp_path / branch['classes'], dtype=np.int64)
    np.testing.assert_array_equal(classes, [0, 1, 2])
    np.testing.assert_array_equal(counts, [1, 1, 0])
    np.testing.assert_array_equal(indices, [0, 1])
    np.testing.assert_allclose(scores, [1., 2.])
    assert report['n_empty_rows'] == 2
    assert report['incorrect']['count'] == 3
    assert report['contributions']['negative']['count'] == 2
    assert report['n_classes_without_positive'] == 1


def test_round_robin_fallback_persistence_and_branch_offsets(tmp_path):
    ratings = tmp_path / 'ratings'
    build_gxi_ratings(_attributions(), 2, str(ratings))
    cfg = SelectionConfig(method='gradient_x_input', k=4, char_share=.5, seed=17)
    a = select_features_gradient_x_input(str(ratings), cfg, 2, 4)
    b = select_features_gradient_x_input(str(ratings), cfg, 2, 4)
    np.testing.assert_array_equal(a.char.indices, b.char.indices)
    np.testing.assert_array_equal(a.word.indices, b.word.indices)
    assert set(a.char.indices) == {0, 1}
    assert set(a.word.indices) == {2, 3}
    assert a.report['word']['n_remainder_fill'] == 1
    out = tmp_path / 'selection'
    save_selection_artifact(str(out), a, {'vectorizer': 'fixture', 'checkpoint': 'fixture'}, ratings_copy=True)
    restored, meta = load_selection_artifact(str(out))
    np.testing.assert_array_equal(restored.word.indices, a.word.indices)
    assert meta['provenance']['checkpoint'] == 'fixture'
    assert (out / 'ratings' / 'meta.json').is_file()


def test_sampling_is_train_index_deterministic():
    cfg = AttributionConfig(max_rows=4, sampling='random')
    a = select_attributed_rows(20, cfg, 42)
    np.testing.assert_array_equal(a, select_attributed_rows(20, cfg, 42))
    assert len(set(a)) == 4 and np.all((a >= 0) & (a < 20))


from tests.test_s03_selection import fitted
def test_gxi_selected_refit_and_two_fresh_training_roundtrips(fitted, tmp_path, monkeypatch):
    from tests.test_s03_selection import make_cfg, assert_features_equal, verify_roundtrip
    from kkt_clf.vectorizer import refit_selected_vectorizer
    from kkt_clf.features import FeatureBuilder
    from kkt_clf.pipeline import run_training
    from kkt_clf.inference import Predictor

    path, names, full, meta, _ = fitted
    X = full.transform(names)
    torch.manual_seed(7)
    from kkt_clf.config import ModelConfig
    from kkt_clf.model import SparseMLP
    model = SparseMLP(X.shape[1], (2, 2, 2), ModelConfig(hidden=[4], dropout=[0.0]))
    att = compute_margin_attributions(model, X, np.arange(len(names)) % 2, torch.device('cpu'), batch_size=3)
    ratings = tmp_path / 'ratings'
    build_gxi_ratings(att, full.n_char_features, str(ratings))
    selected_indices = select_features_gradient_x_input(str(ratings),
        SelectionConfig(method='gradient_x_input', k=5, char_share=.6, seed=7), full.n_char_features, X.shape[1])
    out = tmp_path / 'selected'
    selected = refit_selected_vectorizer(str(path), full, meta, selected_indices.to_payload(), out_dir=str(out)).builder
    for branch, offset in [('char', 0), ('word', full.n_char_features)]:
        ids = getattr(selected_indices, branch).indices - offset
        np.testing.assert_allclose(getattr(selected, branch).idf_, getattr(full, branch).idf_[ids])
        matrix = getattr(selected, branch).transform(selected.preprocess(names))
        norms = np.sqrt(np.asarray(matrix.power(2).sum(axis=1)).ravel())
        np.testing.assert_allclose(norms[norms > 0], 1, atol=1e-6)
    def forbidden(*args, **kwargs):
        pytest.fail('inference and pretrained training must not refit features')
    monkeypatch.setattr(FeatureBuilder, 'fit', forbidden)
    cfg = make_cfg(tmp_path, 'pretrained', str(out))
    for i in range(2):
        cfg.output_dir = str(tmp_path / f'run_gxi_{i}')
        run = run_training(cfg)
        predictor = Predictor.load(run.artifact_dir, 'cpu')
        assert_features_equal(selected, predictor.features, names + ['', None])
        assert verify_roundtrip(run)


def test_report_with_gxi_broken_run_and_unknown_memory(saved_experiment, monkeypatch):
    import json
    import pandas as pd
    from kkt_clf import experiment as exp
    metadata_path = saved_experiment / 'experiment.json'
    metadata = json.loads(metadata_path.read_text())
    cfg = exp.ExperimentConfig()
    cfg.selection.methods = ['global', 'class_aware', 'gradient_x_input']
    metadata['planned_runs'] = [dict(run_id=p.run_id, method=p.method, requested_k=p.requested_k)
                                for p in exp.planned_runs(cfg)]
    metadata['attribution'] = {'target': 'true_l3_vs_best_other_l3_logit_margin', 'summary': {}}
    metadata_path.write_text(json.dumps(metadata))
    broken = saved_experiment / 'runs' / 'gradient_x_input_50000'
    broken.mkdir(parents=True)
    (broken / 'run_result.json').write_text('{invalid')
    shared = saved_experiment / 'shared'
    shared.mkdir()
    (shared / 'attribution_stages.json').write_text(json.dumps({'stages': [
        {'stage': 'attribution', 'status': 'success', 'seconds': 1,
         'peak_rss_bytes': None, 'peak_vram_bytes': None, 'measurement_note': 'unavailable'}]}))
    monkeypatch.setattr(exp, '_train_mlp', lambda *a, **k: pytest.fail('report retrained'))
    paths = exp.build_report(str(saved_experiment))
    runs = pd.read_csv(paths['runs_csv']).set_index('run_id')
    assert len(runs) == 10
    assert runs.loc['gradient_x_input_50000', 'status'] == 'failed'
    assert runs.loc['gradient_x_input_100000', 'status'] == 'skipped'
    assert pd.isna(runs.loc['gradient_x_input_50000', 'delta_macro_f1'])
    stages = pd.read_csv(paths['stages_csv'])
    shared_row = stages.query("run_id == 'shared_attribution' and stage == 'attribution'")
    assert len(shared_row) == 1 and shared_row.peak_rss_bytes.isna().all()
    assert json.loads(open(paths['manifest_json'], encoding='utf8').read())['attribution']['target'] == metadata['attribution']['target']


from tests.test_s03_experiment_contract import saved_experiment
