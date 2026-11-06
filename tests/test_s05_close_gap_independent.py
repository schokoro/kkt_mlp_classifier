"""Regression checks for selection loss gaps from saved S05 results, without training."""
import html
import json
import re
import shutil

import pandas as pd
import pytest

from kkt_clf import s05_report as report


def _history(rows):
    return pd.DataFrame([(run, epoch, split, loss, .4, .5, .01)
                         for run, epoch, split, loss in rows],
                        columns=report.HISTORY_REQUIRED)


def _saved(root, rows):
    root.mkdir()
    run = dict.fromkeys(report.RUNS_REQUIRED)
    run.update(run_id='refit_42', config_id='A', train_seed=42, split_seed=7,
               status='success', metric_evaluated_on='comparison_val',
               l3_macro_f1=.4, l3_accuracy=.5)
    pd.DataFrame([run]).to_csv(root / 'runs.csv', index=False)
    _history(rows).to_csv(root / 'history.csv', index=False)
    pd.DataFrame(columns=report.STAGES_REQUIRED).to_csv(root / 'stages.csv', index=False)
    pd.DataFrame(columns=report.PER_CLASS_REQUIRED).to_parquet(root / 'per_class.parquet', index=False)
    (root / 'manifest.json').write_text(json.dumps({'seeds': {'planned': [42]}}))


def _render(source, destination):
    report.build_s05_report(str(source), str(destination))
    manifest = json.loads((destination / 'manifest.json').read_text())
    page = html.unescape((destination / 'report.html').read_text())
    return manifest, page


def _gap_reasons(manifest):
    return ' '.join(reason for reason in manifest['gate']['reasons']
                    if 'gap' in reason.lower())


def test_maximum_across_candidates_and_epochs_and_final_gap_survive_render(tmp_path):
    source = tmp_path / 'source'
    _saved(source, [
        ('selection:A', 1, 'train', .5), ('selection:A', 1, 'monitor', 3.0),
        ('selection:A', 2, 'train', .4), ('selection:A', 2, 'monitor', .7),
        ('selection:B', 1, 'train', .5), ('selection:B', 1, 'monitor', 2.0),
        ('selection:B', 2, 'train', .4), ('selection:B', 2, 'monitor', .6),
        ('refit_42', 1, 'train', .1), ('refit_42', 1, 'monitor', 20.0),
    ])
    first = tmp_path / 'first'
    manifest, page = _render(source, first)
    reasons = _gap_reasons(manifest)
    assert '2.5000' in reasons
    assert '19.9000' not in reasons
    # The maximum must remain attributable to its candidate and epoch; the
    # final-epoch difference is a separate value for the same candidate.
    rows = re.findall(r'<tr[^>]*>.*?</tr>', page, flags=re.S | re.I)
    candidate = [re.sub(r'<[^>]+>', ' ', row) for row in rows if 'selection:A' in row]
    assert any('2.5000' in row and re.search(r'\b1\b', row) and '0.3000' in row
               for row in candidate)
    assert 'selection:B' in page and '1.5000' in page and '0.2000' in page
    assert set(p.name for p in first.iterdir()) == set(report.REPORT_FILES)
    shutil.rmtree(source)
    rebuilt_manifest, rebuilt_page = _render(first, tmp_path / 'rebuilt')
    assert _gap_reasons(rebuilt_manifest) == reasons
    assert '2.5000' in rebuilt_page and 'selection:A' in rebuilt_page
    assert '0.3000' in rebuilt_page


def test_missing_and_nan_losses_do_not_form_pairs_or_create_zero_gap(tmp_path):
    source = tmp_path / 'source'
    _saved(source, [
        ('selection:A', 1, 'train', .2), ('selection:A', 1, 'monitor', .7),
        ('selection:A', 2, 'train', .1), ('selection:A', 2, 'monitor', float('nan')),
        ('selection:A', 3, 'monitor', 50.0),
        ('selection:B', 1, 'train', float('nan')), ('selection:B', 1, 'monitor', 80.0),
        ('selection:B', 2, 'train', .1),
        ('refit_42', 1, 'train', .1), ('refit_42', 1, 'monitor', 90.0),
    ])
    manifest, page = _render(source, tmp_path / 'out')
    assert not _gap_reasons(manifest)
    assert '0.5000' in page
    assert '89.9000' not in _gap_reasons(manifest)
    assert '49.9000' not in page and '79.9000' not in page
    # No final paired losses in A's last epoch: do not silently report zero.
    gaps = report._train_monitor_gap(_history([
        ('selection:A', 1, 'train', .2), ('selection:A', 1, 'monitor', .7),
        ('selection:A', 3, 'monitor', 50.0),
    ]), pd.DataFrame({'run_id': ['refit_42']}))
    a = gaps.set_index('run_id').loc['selection:A']
    assert pd.isna(a['gap_monitor_minus_train'])
