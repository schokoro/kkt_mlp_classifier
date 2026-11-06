"""Independent review-1 regressions; all data and outputs are disposable."""
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from kkt_clf import s05_baseline as baseline, s05_report as report
from tests.test_s05_baseline_independent import synthetic
from tests.test_s05_report_independent import saved
from tests.test_s05_split_independent import rows, split


def test_plateau_refit_is_rejected_or_made_nonadaptive(synthetic, monkeypatch):
    cfg, manifest, loaded, _, _ = synthetic
    cfg.train['scheduler'] = 'plateau'
    observed = []
    original = baseline.train_model
    def capture(*args, **kwargs):
        observed.append(args[6].scheduler)
        return original(*args, **kwargs)
    monkeypatch.setattr(baseline, 'train_model', capture)
    try:
        baseline.run_refit(cfg, manifest, loaded, torch.device('cpu'),
                       tfidf_verification={'decision': 'ok', 'full_tfidf_id': 'full-id'})
    except (baseline.S05BaselineError, ValueError):
        pass
    assert all(s != 'plateau' for s in observed), 'refit LR must not depend on monitor loss'


@pytest.mark.parametrize('changed', ['train_fit', 'train_monitor', 'tfidf'])
def test_selection_provenance_mismatch_stops_before_training(synthetic, monkeypatch, changed):
    cfg, manifest, loaded, _, _ = synthetic
    selection = baseline.run_selection(cfg, manifest, loaded, torch.device('cpu'))
    if changed == 'tfidf':
        selection['full_tfidf_id'] = 'different'
    else:
        selection['splits'][changed] = 'different'
    monkeypatch.setattr(baseline, 'train_model', lambda *a, **k: pytest.fail('trained with mismatched provenance'))
    with pytest.raises((baseline.S05BaselineError, ValueError)):
        baseline.run_refit(cfg, manifest, loaded, torch.device('cpu'), selection=selection,
                           tfidf_verification={'decision': 'ok', 'full_tfidf_id': 'full-id'})


def test_sweep_different_metrics_cannot_be_compared(synthetic, monkeypatch):
    cfg, manifest, loaded, _, _ = synthetic
    cfg.sweep = [{'name': 'other', 'train': {'early_stopping_metric': 'val_loss'}}]
    cfg.train['early_stopping_metric'] = 'l3_macro_f1'
    with pytest.raises((baseline.S05BaselineError, ValueError)):
        baseline.run_selection(cfg, manifest, loaded, torch.device('cpu'))


def test_cli_sweep_accepts_json_list(synthetic, monkeypatch):
    from scripts import run_s05_baseline as cli
    cfg, _, _, _, _ = synthetic
    seen = []
    monkeypatch.setattr(cli, '_cfg_from', lambda args: cfg)
    def contour(config):
        seen.extend(config.sweep)
        return {'statuses': {}, 'seed_stats_comparison_val': []}
    monkeypatch.setattr(cli, 'run_training_contour', contour)
    assert cli.main(['all', '--sweep', '[{"name":"x"}]']) == 0
    assert seen == [{'name': 'x'}]


@pytest.mark.parametrize('failure', ['reference', 'evaluation'])
def test_failures_preserve_machine_readable_diagnostics(synthetic, monkeypatch, failure):
    cfg, manifest, loaded, _, _ = synthetic
    if failure == 'reference':
        original = baseline.train_model
        def train(*args, **kwargs):
            if args[9] == 42:
                raise RuntimeError('reference failed')
            return original(*args, **kwargs)
        monkeypatch.setattr(baseline, 'train_model', train)
    else:
        original = baseline.predict
        count = [0]
        def predict(*args, **kwargs):
            count[0] += 1
            if count[0] == 1:
                raise RuntimeError('evaluation failed')
            return original(*args, **kwargs)
        monkeypatch.setattr(baseline, 'predict', predict)
    try:
        baseline.run_refit(cfg, manifest, loaded, torch.device('cpu'),
                       tfidf_verification={'decision': 'ok', 'full_tfidf_id': 'full-id'})
    except (baseline.S05BaselineError, RuntimeError):
        pass
    out = Path(cfg.output_dir)
    for name in ('runs.csv', 'history.csv', 'stages.csv', 'per_class.parquet', 'manifest.json'):
        assert (out / name).is_file(), name
    runs = pd.read_csv(out / 'runs.csv')
    assert set(runs.train_seed) == set(cfg.seeds)
    assert 'failed' in set(runs.status)
    assert 'success' in set(runs.status)
    report.build_s05_report(str(out), str(out / 'report'))


def test_roundtrip_uses_loaded_feature_transform(synthetic, monkeypatch):
    cfg, manifest, loaded, _, _ = synthetic
    original = baseline.load_artifact
    calls = []
    def load(*args, **kwargs):
        art = original(*args, **kwargs)
        transform = art.features.transform
        def spy(*a, **kw):
            calls.append(True)
            return transform(*a, **kw)
        art.features.transform = spy
        return art
    monkeypatch.setattr(baseline, 'load_artifact', load)
    baseline.run_refit(cfg, manifest, loaded, torch.device('cpu'),
                       tfidf_verification={'decision': 'ok', 'full_tfidf_id': 'full-id'})
    assert calls, 'roundtrip must transform comparison names using checkpoint features'


def test_refit_history_contains_actual_selection_curves(synthetic):
    cfg, manifest, loaded, _, _ = synthetic
    selection = baseline.run_selection(cfg, manifest, loaded, torch.device('cpu'))
    baseline.run_refit(cfg, manifest, loaded, torch.device('cpu'), selection=selection,
                           tfidf_verification={'decision': 'ok', 'full_tfidf_id': 'full-id'})
    history = pd.read_csv(Path(cfg.output_dir) / 'history.csv')
    internal = history[history.run_id.astype(str).str.startswith('selection:')]
    assert {'train', 'monitor'} <= set(internal['split'])
    assert internal.loc[internal['split'] == 'train', ['l3_macro_f1', 'l3_accuracy']].notna().all().all()


def test_rare_class_support_survives_save(rows, tmp_path):
    from kkt_clf.freeze_split import save_frozen_split
    source = tmp_path / 'input.parquet'
    rows.to_parquet(source, index=False)
    frozen = split(rows)
    save_frozen_split(frozen, str(source), str(tmp_path / 'splits'), str(tmp_path / 'audit'),
                      10, (0.8, 0.1, 0.1), 0.1, 42, 0.1)
    support = pd.read_csv(tmp_path / 'audit/class_support_split.csv').set_index('L3')
    assert support.loc['rare', 'support_all'] == 3


def test_incomplete_history_and_majority_gate(saved, tmp_path):
    runs = pd.read_csv(saved / 'runs.csv').iloc[:3].copy()
    runs['config_id'] = 'A'
    runs['l3_macro_f1'] = .1
    runs.to_csv(saved / 'runs.csv', index=False)
    (saved / 'majority_baseline.json').write_text(json.dumps({'l3_macro_f1': .2}))
    pd.DataFrame(columns=report.HISTORY_REQUIRED).to_csv(saved / 'history.csv', index=False)
    report.build_s05_report(str(saved), str(tmp_path / 'report'))
    gate = json.loads((tmp_path / 'report/manifest.json').read_text())['gate']
    assert gate['decision'] == 'not_ready'


def test_report_rerender_preserves_context(saved, tmp_path):
    (saved / 'majority_baseline.json').write_text(json.dumps({'l3_macro_f1': .123456}))
    (saved / 'selection').mkdir()
    (saved / 'selection/selection_report.json').write_text(json.dumps({'chosen': {'name': 'unique_choice', 'fixed_epochs': 7}}))
    report.build_s05_report(str(saved), str(tmp_path / 'first'))
    report.build_s05_report(str(tmp_path / 'first'), str(tmp_path / 'second'))
    first = (tmp_path / 'first/report.html').read_text()
    second = (tmp_path / 'second/report.html').read_text()
    assert 'unique_choice' in first and 'unique_choice' in second
    assert json.loads((tmp_path / 'first/manifest.json').read_text())['gate'] == json.loads((tmp_path / 'second/manifest.json').read_text())['gate']


def test_missing_resource_time_is_not_zero(saved, tmp_path):
    row = dict.fromkeys(report.STAGES_REQUIRED)
    row.update(run_id='seed_42', stage='train', status='success', seconds=None)
    pd.DataFrame([row]).to_csv(saved / 'stages.csv', index=False)
    report.build_s05_report(str(saved), str(tmp_path / 'report'))
    html = (tmp_path / 'report/report.html').read_text()
    section = html.split('Ресурсные затраты (stages.csv)', 1)[1].split('<h2>', 1)[0]
    assert '0.0000' not in section


@pytest.mark.parametrize('corruption', ['vocabulary', 'idf'])
def test_checkpoint_rejects_equal_dimension_feature_corruption(synthetic, monkeypatch, corruption):
    cfg, manifest, loaded, _, _ = synthetic
    original = baseline.load_artifact
    def load(*args, **kwargs):
        art = original(*args, **kwargs)
        branch = art.features.word
        if corruption == 'idf':
            idf = branch.idf_.copy()
            idf[0] += 1
            branch.idf_ = idf
        else:
            keys = list(branch.vocabulary_)[:2]
            a, b = keys
            branch.vocabulary_[a], branch.vocabulary_[b] = branch.vocabulary_[b], branch.vocabulary_[a]
        return art
    monkeypatch.setattr(baseline, 'load_artifact', load)
    try:
        baseline.run_refit(cfg, manifest, loaded, torch.device('cpu'),
                       tfidf_verification={'decision': 'ok', 'full_tfidf_id': 'full-id'})
    except (baseline.S05BaselineError, ValueError):
        return
    selected = Path(cfg.output_dir) / 'selected_checkpoint.json'
    assert not selected.exists() or json.loads(selected.read_text())['roundtrip']['status'] != 'success'


@pytest.mark.parametrize('command', ['select', 'refit'])
def test_cli_verifies_tfidf_before_training(synthetic, monkeypatch, command):
    import kkt_clf.vectorizer as vectorizer
    from scripts import run_s05_baseline as cli
    cfg, manifest, loaded, _, _ = synthetic
    monkeypatch.setattr(cli, '_cfg_from', lambda args: cfg)
    monkeypatch.setattr(cli, 'load_split_manifest', lambda path: manifest)
    monkeypatch.setattr(vectorizer, 'load_vectorizer_artifact', lambda path: loaded)
    monkeypatch.setattr(cli, 'verify_full_tfidf', lambda *args: (_ for _ in ()).throw(baseline.S05BaselineError('invalid corpus')))
    monkeypatch.setattr(cli, 'run_selection', lambda *a, **k: pytest.fail('selection before TF-IDF verification'))
    monkeypatch.setattr(cli, 'run_refit', lambda *a, **k: pytest.fail('refit before TF-IDF verification'))
    assert cli.main([command]) != 0
