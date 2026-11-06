"""Regression checks for S04 review 1, using actual persisted artefacts."""
import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from kkt_clf import experiment as exp
from kkt_clf.attribution import compute_margin_attributions
from kkt_clf.vectorizer import fit_and_save_vectorizer
from tests.test_s01_vectorizer import make_cfg
from tests.test_s03_experiment_contract import saved_experiment, _ExternalAssets, _saved_result
from tests.test_s04_comparison_cost_contract import IDENTITY
from tests.test_s04_attribution import TinyModel


@pytest.fixture
def checkpoint_context(tmp_path):
    base = make_cfg(tmp_path)
    full = tmp_path / 'full'
    fit_and_save_vectorizer(str(full), base.features, base.data.path)
    cfg = exp.ExperimentConfig(output_dir=str(tmp_path / 'experiment'),
        full_tfidf_path=str(full), corpus_path=str(base.data.path), data=base.data,
        model=base.model, loss=base.loss, train=base.train,
        selection=exp.SelectionPlanConfig(methods=['gradient_x_input'], budgets=[4], char_share=.5))
    ctx = exp._prepare_context(cfg, torch.device('cpu'))
    trained = exp._train_attribution_model(ctx, cfg.output_dir)
    cfg.attribution.checkpoint_path = trained['checkpoint_dir']
    exp._load_attribution_model(ctx)  # The unchanged checkpoint must be accepted.
    return ctx


@pytest.mark.parametrize('change', ['vocabulary', 'idf', 'train_split'])
def test_loaded_checkpoint_rejects_changed_actual_inputs(checkpoint_context, change):
    ctx = checkpoint_context
    if change == 'vocabulary':
        vocab = ctx.full_builder.char.vocabulary_
        a, b = list(vocab)[:2]
        vocab[a], vocab[b] = vocab[b], vocab[a]
    elif change == 'idf':
        idf = ctx.full_builder.char.idf_.copy()
        idf[0] += 1
        ctx.full_builder.char.idf_ = idf
    else:
        train = ctx.data.splits['train']
        original = train.iloc[0]['name']
        data = pd.read_parquet(ctx.cfg.data.path)
        data.loc[data['name'] == original, 'name'] = original + ' altered'
        data.to_parquet(ctx.cfg.data.path, index=False)
        ctx = exp._prepare_context(ctx.cfg, torch.device('cpu'))
    with pytest.raises(Exception, match='(?i)(checkpoint|fingerprint|совмест|идентич|split|tf.idf)'):
        exp._load_attribution_model(ctx)


@pytest.mark.parametrize('field', ['train_sha256', 'full_tfidf_sha256', 'model_config_sha256',
                                   'seed', 'eval_protocol_sha256', 'val_sha256', 'label_map_sha256'])
def test_report_never_compares_different_identities(saved_experiment, field):
    metadata_path = saved_experiment / 'experiment.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['schema_version'] = 's04-experiment@1'
    metadata_path.write_text(json.dumps(metadata))
    common = {'train_sha256': 'train', 'full_tfidf_sha256': 'features',
              'model_config_sha256': 'model', 'eval_protocol_sha256': 'protocol'}
    for rid in ('none', 'global_50000'):
        path = saved_experiment / 'runs' / rid / 'run_result.json'
        obj = json.loads(path.read_text())
        obj.update(common)
        if rid != 'none':
            obj[field] = 43 if field == 'seed' else 'different'
        path.write_text(json.dumps(obj))
    paths = exp.build_report(str(saved_experiment))
    runs = pd.read_csv(paths['runs_csv']).set_index('run_id')
    assert pd.isna(runs.loc['global_50000', 'delta_macro_f1'])
    pc = pd.read_parquet(paths['per_class_parquet']).query("run_id == 'global_50000'")
    assert pc[['baseline_f1', 'baseline_recall', 'delta_f1', 'delta_recall']].isna().all().all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_sparse_attributions_match_cpu():
    cpu = TinyModel(torch.nn.Tanh()).eval()
    gpu = TinyModel(torch.nn.Tanh()).cuda().eval()
    gpu.load_state_dict(cpu.state_dict())
    X = sp.csr_matrix(np.array([[.8, 0, .3, 0], [0, .6, 0, .2], [0, 0, 0, 0]], dtype=np.float32))
    y = np.array([0, 1, 2])
    expected = compute_margin_attributions(cpu, X, y, torch.device('cpu'), batch_size=2)
    actual = compute_margin_attributions(gpu, X, y, torch.device('cuda'), batch_size=2)
    np.testing.assert_allclose(actual.values.toarray(), expected.values.toarray(), atol=1e-5)
    np.testing.assert_allclose(actual.margins, expected.margins, atol=1e-5)
    np.testing.assert_array_equal(actual.competitor, expected.competitor)


def test_manifest_contains_attributed_counts_for_each_l3(checkpoint_context):
    ctx = checkpoint_context
    exp._prepare_attribution(ctx)
    assert ctx.attribution_error is None, ctx.attribution_error
    meta = ctx.attribution_meta
    report = ctx.attribution.report
    # The self-contained manifest metadata must carry counts, not merely a path.
    expected = {str(row['label']): row['n_attributed'] for row in report['per_class']}
    def mappings(value):
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from mappings(child)
        elif isinstance(value, list):
            for child in value:
                yield from mappings(child)
    assert any(all(mapping.get(label) == count for label, count in expected.items())
               for mapping in mappings(meta)), expected


def test_gxi_report_shared_cost_and_offline_html(saved_experiment):
    metadata_path = saved_experiment / 'experiment.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['schema_version'] = 's04-experiment@1'
    metadata['attribution'] = {'target': 'true_l3_vs_best_other_l3_logit_margin', 'summary': {}}
    for run_id in ('none', 'global_50000'):
        path = saved_experiment / 'runs' / run_id / 'run_result.json'
        result = json.loads(path.read_text())
        result.update(IDENTITY)
        path.write_text(json.dumps(result))
    for budget in (50000, 100000, 200000):
        run_id = f'gradient_x_input_{budget}'
        metadata['planned_runs'].append({'run_id': run_id, 'method': 'gradient_x_input',
                                         'requested_k': budget})
        result = _saved_result(run_id, .6)
        result.update(IDENTITY)
        run_dir = saved_experiment / 'runs' / run_id
        run_dir.mkdir()
        (run_dir / 'run_result.json').write_text(json.dumps(result))
    metadata_path.write_text(json.dumps(metadata))
    shared = saved_experiment / 'shared'
    shared.mkdir()
    stages = [{'stage': name, 'status': 'success', 'seconds': seconds,
               'peak_rss_bytes': 1024, 'peak_vram_bytes': None, 'measurement_note': 'RSS sample'}
              for name, seconds in [('attribution_model_train', 11), ('attribution', 13)]]
    (shared / 'attribution_stages.json').write_text(json.dumps({'stages': stages}))
    paths = exp.build_report(str(saved_experiment))
    runs = pd.read_csv(paths['runs_csv']).set_index('run_id')
    for budget in (50000, 100000, 200000):
        run = runs.loc[f'gradient_x_input_{budget}']
        assert run['status'] == 'success' and bool(run['comparable_with_baseline'])
        assert run['delta_macro_f1'] == pytest.approx(.1)
    frame = pd.read_csv(paths['stages_csv'])
    for stage, seconds in [('attribution_model_train', 11), ('attribution', 13)]:
        rows = frame.query('stage == @stage and run_id == "shared_attribution"')
        assert len(rows) == 1 and rows.iloc[0].seconds == seconds
    html = open(paths['report_html'], encoding='utf8').read()
    parser = _ExternalAssets()
    parser.feed(html)
    assert not parser.external
    # A computed scenario total includes both shared costs and the run's own stages.
    section = html.split('<h2>Качество ↔ ресурсы</h2>', 1)[1].split('<h2>', 1)[0]
    assert '<svg' in section
    assert '28.0' in section or '28.00' in section
    assert '36.00' in section  # Three four-second runs plus shared 11 + 13, once.


def test_ranking_peak_memory_does_not_scale_as_all_class_feature_pairs(tmp_path):
    import os
    import subprocess
    import sys
    script = r'''
import json, os, threading
import numpy as np
import scipy.sparse as sp
from kkt_clf.attribution import MarginAttributions, build_gxi_ratings
C, F = 100, 12000
# Existing sparse inputs have one positive contribution per class/feature.
X = sp.csr_matrix(np.ones((C, F), dtype=np.float32))
y = np.arange(C)
a = MarginAttributions(X, np.ones(C), np.zeros(C, dtype=int),
    np.ones(C, dtype=bool), np.zeros(C, dtype=bool), y, y, C)
def rss():
    with open('/proc/self/statm') as f:
        return int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')
base = rss(); peak = [base]; stop = threading.Event()
def sample():
    while not stop.wait(.001): peak[0] = max(peak[0], rss())
t = threading.Thread(target=sample); t.start()
try:
    build_gxi_ratings(a, F // 2, OUT)
    peak[0] = max(peak[0], rss())
finally:
    stop.set(); t.join()
print(json.dumps({'extra': peak[0] - base, 'pairs': C * F}))
'''.replace('OUT', repr(str(tmp_path / 'ratings')))
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    proc = subprocess.run([sys.executable, '-c', script], capture_output=True,
                          text=True, env=env, timeout=60, check=True)
    measurement = json.loads(proc.stdout.strip().splitlines()[-1])
    # Allow for CSR working copies, but not two additional full C×F arrays
    # for both index and score plus their concatenated copies.
    assert measurement['extra'] < measurement['pairs'] * 40, measurement
