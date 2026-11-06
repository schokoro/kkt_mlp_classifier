"""Independent regression checks for the five findings in S04 review 2."""
import json

import pandas as pd
import pytest
import torch

from kkt_clf import experiment as exp
from kkt_clf.attribution import build_gxi_ratings
from tests.test_s04_attribution import _attributions
from tests.test_s04_review1_independent import checkpoint_context  # noqa: F401
from tests.test_s04_comparison_cost_contract import IDENTITY
from tests.test_s03_experiment_contract import saved_experiment  # noqa: F401


def alternate_preprocess(texts):
    return [str(x).upper() for x in texts]


@pytest.mark.parametrize('change', ['binary', 'sublinear_tf', 'norm', 'ngram_range',
                                    'tokenizer', 'preprocessing'])
def test_checkpoint_rejects_changed_transform_contract(checkpoint_context, change):
    ctx = checkpoint_context
    word = ctx.full_builder.word
    if change == 'preprocessing':
        ctx.full_builder.preprocess = alternate_preprocess
    elif change == 'tokenizer':
        word.tokenizer = str.split
    else:
        setattr(word, change, {'binary': True, 'sublinear_tf': not word.sublinear_tf,
                               'norm': None, 'ngram_range': (1, 2)}[change])
    with pytest.raises(Exception, match='(?i)(checkpoint|fingerprint|совмест|идентич|tf.idf|контракт)'):
        exp._load_attribution_model(ctx)


def test_checkpoint_rejects_replaced_compatible_weights(checkpoint_context):
    ctx = checkpoint_context
    path = ctx.cfg.attribution.checkpoint_path + '/model.pt'
    weights = torch.load(path, map_location='cpu', weights_only=True)
    key = next(k for k, value in weights.items() if torch.is_floating_point(value))
    weights[key] = weights[key].clone()
    weights[key].view(-1)[0] += 1
    torch.save(weights, path)
    with pytest.raises(Exception, match='(?i)(checkpoint|fingerprint|вес|weight|идентич|hash)'):
        exp._load_attribution_model(ctx)


def test_all_attributed_rows_include_empty_in_both_reports(tmp_path):
    manifest, report = build_gxi_ratings(_attributions(), 2, str(tmp_path))
    n = report['n_attributed_rows']
    assert n == report['n_rows_total'] == 4
    assert manifest['attribution']['n_attributed_rows'] == n
    assert report['n_empty_rows'] == 2
    assert report['correct']['count'] + report['incorrect']['count'] == n
    assert sum(row['n_attributed'] for row in report['per_class']) == n


def test_zero_seed_remains_comparable_in_saved_report(saved_experiment):
    meta_path = saved_experiment / 'experiment.json'
    meta = json.loads(meta_path.read_text())
    meta['schema_version'] = 's04-experiment@1'
    meta_path.write_text(json.dumps(meta))
    for run_id in ('none', 'global_50000'):
        path = saved_experiment / 'runs' / run_id / 'run_result.json'
        result = json.loads(path.read_text())
        result.update({**IDENTITY, 'seed': 0})
        path.write_text(json.dumps(result))
    paths = exp.build_report(str(saved_experiment))
    runs = pd.read_csv(paths['runs_csv']).set_index('run_id')
    assert bool(runs.loc['global_50000', 'comparable_with_baseline'])
    assert runs.loc['global_50000', 'delta_macro_f1'] == pytest.approx(.25)


@pytest.mark.parametrize('status', ['failed', 'skipped'])
def test_planned_unsuccessful_budget_makes_method_total_unknown(status):
    runs = pd.DataFrame([{'run_id': f'g{i}', 'method': 'gradient_x_input',
                          'requested_k': i, 'l3_macro_f1': .5 if i < 3 else None,
                          'status': 'success' if i < 3 else status,
                          'comparable_with_baseline': i < 3} for i in (1, 2, 3)])
    stages = pd.DataFrame([{'run_id': run, 'stage': 'train', 'status': 'success', 'seconds': sec}
                           for run, sec in [('g1', 4), ('g2', 5), ('shared_attribution', 11)]])
    _, totals = exp._scenario_cost_rows(stages, runs)
    total = next(row for row in totals if row['method'] == 'gradient_x_input')
    assert total['n_budgets'] == 3
    assert total['scenario_total_seconds'] is None


def test_shared_vectorization_cost_applies_to_each_method_once():
    runs = pd.DataFrame([{'run_id': name, 'method': method, 'requested_k': 50,
                          'l3_macro_f1': .5, 'status': 'success', 'comparable_with_baseline': True}
                         for name, method in [('none', 'none'), ('chi', 'global'), ('g1', 'gradient_x_input')]])
    # Interface assumption: a separately identified shared_vectorization run records
    # the common full train/validation transform; attribution remains separate.
    stages = pd.DataFrame([{'run_id': run, 'stage': stage, 'status': 'success', 'seconds': sec}
                           for run, stage, sec in [('none', 'train', 1), ('chi', 'train', 2),
                               ('g1', 'train', 3), ('shared_vectorization', 'vectorize', 7),
                               ('shared_attribution', 'attribution', 11)]])
    rows, totals = exp._scenario_cost_rows(stages, runs)
    rows = {r['run_id']: r for r in rows}
    assert rows['none']['scenario_total_seconds'] == 8
    assert rows['chi']['scenario_total_seconds'] == 9
    assert rows['g1']['scenario_total_seconds'] == 21
    assert {r['method']: r['scenario_total_seconds'] for r in totals}['global'] == 9


def test_external_checkpoint_unknown_training_cost_is_not_full_total():
    runs = pd.DataFrame([{'run_id': 'g1', 'method': 'gradient_x_input', 'requested_k': 50,
                          'l3_macro_f1': .5, 'status': 'success', 'comparable_with_baseline': True}])
    # Unknown external training cost is represented by a null training stage.
    stages = pd.DataFrame([{'run_id': run, 'stage': stage, 'status': 'success', 'seconds': sec}
                           for run, stage, sec in [('g1', 'train', 4),
                               ('shared_attribution', 'attribution_model_train', None),
                               ('shared_attribution', 'attribution', 13)]])
    rows, totals = exp._scenario_cost_rows(stages, runs)
    assert rows[0]['scenario_total_seconds'] is None
    assert totals[0]['scenario_total_seconds'] is None


def test_loaded_checkpoint_records_unknown_training_cost(checkpoint_context):
    ctx = checkpoint_context
    exp._prepare_attribution(ctx)
    assert ctx.attribution_error is None, ctx.attribution_error
    training = [row for row in ctx.shared_stage_rows if row['stage'] == 'attribution_model_train']
    assert len(training) == 1
    assert training[0]['seconds'] is None


def test_report_passes_all_planned_budgets_to_scenario_accounting(saved_experiment, monkeypatch):
    seen = []
    original = exp._scenario_cost_rows

    def capture(stages, runs, *args, **kwargs):
        seen.extend(runs.to_dict('records'))
        return original(stages, runs, *args, **kwargs)

    monkeypatch.setattr(exp, '_scenario_cost_rows', capture)
    exp.build_report(str(saved_experiment))
    global_runs = [r for r in seen if r['method'] == 'global']
    assert len(global_runs) == 3
    assert {r['status'] for r in global_runs} >= {'success', 'skipped'}
