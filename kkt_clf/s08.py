"""S08 frozen preparation and explicitly authorized, single-use evaluation.

Consent is a trusted caller input, never established by an integrity hash alone.
"""
from __future__ import annotations

from contextlib import contextmanager
import dataclasses
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import torch

from . import s07
from .artifact import environment_versions, load_artifact, save_artifact
from .attribution import feature_contract_fingerprint
from .config import LossConfig, ModelConfig, TrainConfig
from .experiment import _Measurement
from .labels import LabelMaps, LEVELS
from .model import SparseMLP
from .reject import RejectRule
from .selection import (SelectionConfig, load_selection_artifact, save_selection_artifact,
                        select_features)
from .s07 import (digest, fingerprint, fresh_root, group_keys, now, read_json, read_sealed,
                  require, row_identity, sealed, sha, tree, verify_tree, write_new)
from .s08_stats import BOOTSTRAP, simulated_feasibility
from .train import compute_class_weights, resolve_device, set_seed, train_model
from .vectorizer import load_vectorizer_artifact, refit_selected_vectorizer

METHODS = ('none', 'class_aware', 'global')
SEEDS = (42, 43, 44)
TEST_FILES = ('paired_test.csv', 'per_class.parquet', 'support_bins.csv',
              'predictions_test.parquet', 'confusion_pairs.csv')
LIMITATIONS = [
    'ODS PoC only; no industrial/external generalization or superiority claim.',
    'Development labels already explored; feasibility projections are exploratory.',
    'Unlabeled TF-IDF fit is historically partially transductive.',
    'Sparse-class bootstrap is conditional on observed errors, class composition and fixed seeds.',
    'Primary class_aware and illustrative margin .01 are unapproved; delta_NI is null.',
    'Evaluation requires a caller-trusted detached decision digest and a local atomic single-use claim.',
    'CPU/GPU costs are local observations; unavailable RSS/VRAM is null, peaks are not additive.',
]


@dataclasses.dataclass(frozen=True)
class S08Config(s07.S07Config):
    """Production pins inherited unchanged; explicit synthetic fixtures stay synthetic."""


def verify_sources(cfg: S08Config) -> dict:
    source = s07.verify_sources(cfg)
    m, refs, pool = source['manifest'], source['refs'], source['pool']
    frames = {}
    # Verify all original partitions, including dev-only epoch-selection provenance.
    for part, info in m['parts'].items():
        require(part in ('outer_train', 'comparison_val', 'sealed_test', 'train_fit', 'train_monitor'),
                'Unexpected source partition')
        require(fingerprint(info['file']) == {k: info[k] for k in ('sha256', 'bytes')},
                f'Original {part} fingerprint mismatch')
        data = pd.read_parquet(info['file'], columns=['row_id', 'name'])
        require(len(data) == info['n_rows'] and data.row_id.is_unique, 'Original partition count/IDs mismatch')
        keys = group_keys(data)
        require(len(set(keys)) == len(keys),
                'Non-singleton groups incompatible with preregistered row-bootstrap design')
        require(info['n_groups'] == len(set(keys)), 'Original group count mismatch')
        frames[part] = data
        refs[part] = {'path': str(Path(info['file']).resolve()), **fingerprint(info['file']), **row_identity(data)}
    require(set(frames) == {'outer_train', 'comparison_val', 'sealed_test', 'train_fit', 'train_monitor'},
            'Missing original partitions')
    for a, b in [('outer_train', 'comparison_val'), ('train_fit', 'train_monitor')]:
        require(not set(frames[a].row_id) & set(frames[b].row_id)
                and not set(group_keys(frames[a])) & set(group_keys(frames[b])), 'Original partition leakage')
    union = pd.concat([frames['train_fit'], frames['train_monitor']]).sort_values('row_id').reset_index(drop=True)
    require(union.equals(frames['outer_train'].sort_values('row_id').reset_index(drop=True)),
            'Original inner partition union differs from outer_train')
    require(len(set(group_keys(pool))) == len(pool), 'Development groups not singleton')
    return source


def implementation_files():
    repo = Path(__file__).resolve().parents[1]
    paths = sorted((repo / 'kkt_clf').glob('*.py')) + [repo / 'scripts' / 'run_s08.py']
    return {str(p.relative_to(repo)): fingerprint(p) for p in paths}


def protocol_contract(labels):
    return {'schema': 's08-protocol@1', 'primary_candidate': 'class_aware', 'primary_status': 'unapproved',
            'reference': 'none', 'secondary_descriptive': 'global', 'delta_NI': None,
            'margin_status': 'unapproved', 'practical_rationale': None,
            'planning_example_only': .01, 'status': 'awaiting_user_approval',
            'estimand': 'mean_seed(macro_F1(candidate)-macro_F1(full))',
            'bootstrap': dict(BOOTSTRAP), 'labels': list(labels), 'prediction': 'raw top-1 L3',
            'reject': False, 'tie_rule': 'torch.topk(k=1), pinned runtime; bootstrap percentile linear',
            'decision': 'lower_bound > -delta_NI', 'one_time_use': True,
            'approval': 'caller-trusted detached decision SHA256 + local atomic single-use claim required'}


def run_matrix(cfg):
    return [{'run_id': f'{m}__seed_{s}', 'method': m, 'seed': s,
             'n_char': (cfg.full_counts if m == 'none' else cfg.selected_counts)[0],
             'n_word': (cfg.full_counts if m == 'none' else cfg.selected_counts)[1],
             'initial_status': 'pending'} for m in METHODS for s in SEEDS]


def training_contract(cfg, source):
    p = source['protocol']
    return {'model': p['resolved_sections']['model'], 'loss': p['resolved_sections']['loss'],
            'train': {**p['resolved_sections']['train'], 'epochs': p['fixed_epochs'],
                      'patience': p['fixed_epochs'], 'device': cfg.device, 'num_threads': cfg.num_threads},
            'restore_best': False, 'weights': 'final_epoch', 'training_rows': 'full_development_union',
            'diagnostic': 'in_sample_only; no validation/checkpoint/epoch tuning'}


def preflight(cfg: S08Config, output_root):
    root = fresh_root(output_root)
    audit = {'schema': 's08-preflight@1', 'config': dataclasses.asdict(cfg), 'created_at': now(),
             'sealed_labels_read': False, 'sealed_inference': False, 'workload_runs': 0}
    try:
        source = verify_sources(cfg)
        feasibility = simulated_feasibility(source['pool'], source['manifest']['label_maps']['L3'])
        write_new(root / 'feasibility.json', sealed(feasibility))
        audit.update(passed=True, sources=source['refs'], development_pool=row_identity(source['pool']),
                     training=training_contract(cfg, source), feasibility_sha256=sha(root / 'feasibility.json'))
    except Exception as exc:
        audit.update(passed=False, error=f'{type(exc).__name__}: {exc}')
    write_new(root / 'preflight.json', sealed(audit))
    return audit


def write_plan(cfg: S08Config, output_root):
    source = verify_sources(cfg)
    feasibility = simulated_feasibility(source['pool'], source['manifest']['label_maps']['L3'])
    root = fresh_root(output_root)
    protocol = protocol_contract(source['manifest']['label_maps']['L3'])
    protocol.update(sources=source['refs'], training=training_contract(cfg, source),
                    study_kind='synthetic' if cfg.fixture else 'ODS_internal_holdout',
                    code=implementation_files(), versions=environment_versions())
    write_new(root / 'protocol.json', sealed(protocol))
    write_new(root / 'feasibility.json', sealed(feasibility))
    prereg = ('# S08 pre-registration — UNSIGNED\n\n'
              'Primary Class-aware Chi² is proposed, UNAPPROVED. delta_NI is null/unapproved. '
              '.01 is an illustrative planning scenario, not an acceptable-loss decision. '
              'Practical rationale/owner/approval remain absent. No holdout access is authorized.\n\n'
              'Three paired seeds 42/43/44; full development union; frozen final-epoch training. '
              'Fixed L3 universe; raw top-1 macro-F1, zero_division=0. Class-stratified paired row bootstrap, '
              'B=5000, PCG64 seed808, alpha=.025, linear percentile; identical sampled rows across all seeds. '
              'Non-singleton groups fail closed. Strict lower_bound > -delta_NI. '
              'Global is descriptive only. No superiority claim.\n\n'
              'Approval must be an explicit externally authenticated orchestrator-recorded user decision '
              'binding margin, rationale, candidate, one-time use, protocol/plan/training/model/feature hashes. '
              'The caller must supply a trusted digest only after actual user consent; a local claim consumes use permanently. '
              'After any future one-time evaluation, adaptive confirmation requires genuinely new labels.\n\n'
              f'Frozen at {now()}; protocol file SHA256 {sha(root / "protocol.json")}.\n')
    (root / 'pre_registration.md').write_text(prereg, encoding='utf-8')
    plan = {'schema': 's08-plan@1', 'created_at': now(), 'config': dataclasses.asdict(cfg),
            'sources': source['refs'], 'label_maps': source['manifest']['label_maps'],
            'development_pool': row_identity(source['pool']),
            'corpus_fit_selection': source['corpus_fit_selection'],
            'training': training_contract(cfg, source), 'runs': run_matrix(cfg),
            'dependencies': [{'method': m, 'ranking': 'full_development_union_only',
                              'budget': list(cfg.selected_counts), 'refit_corpus': source['refs']['corpus'],
                              'shared_seeds': list(SEEDS)} for m in METHODS[1:]],
            'protocol': fingerprint(root / 'protocol.json'), 'feasibility': fingerprint(root / 'feasibility.json'),
            'pre_registration': fingerprint(root / 'pre_registration.md'),
            'implementation': implementation_files(), 'versions': environment_versions(),
            'limitations': LIMITATIONS, 'resume': False,
            'output_schema': {'pending': ['manifest.json', 'runs.csv', 'stages.csv', 'ni_decision.json', 'report.html'],
                              'test_only': list(TEST_FILES)},
            'evaluation': 'requires_caller_trusted_anchor_and_explicit_user_decision'}
    # Normalize tuple-valued fixture config before returning the frozen plan.
    import json
    plan = json.loads(json.dumps(plan, allow_nan=False))
    write_new(root / 'plan.json', sealed(plan))
    return plan


def load_plan(output_root, verify_inputs=True):
    root = Path(output_root)
    plan = read_sealed(root / 'plan.json')
    require(plan['schema'] == 's08-plan@1', 'Wrong plan schema')
    cfg = S08Config(**plan['config'])
    cfg.validate()
    require(plan['runs'] == run_matrix(cfg) and len(plan['runs']) == 9, 'Run matrix mismatch')
    require(plan['implementation'] == implementation_files(), 'Implementation changed after freeze')
    require(plan['versions'] == environment_versions(), 'Runtime versions changed after freeze')
    require(plan['resume'] is False and plan['limitations'] == LIMITATIONS, 'Freeze contract changed')
    for name, file in [('protocol', 'protocol.json'), ('feasibility', 'feasibility.json'),
                       ('pre_registration', 'pre_registration.md')]:
        require(fingerprint(root / file) == plan[name], f'Frozen {name} changed')
    protocol = read_sealed(root / 'protocol.json')
    for k, value in protocol_contract(plan['label_maps']['L3']).items():
        require(protocol[k] == value, f'Protocol contract changed: {k}')
    require(protocol['sources'] == plan['sources'] and protocol['training'] == plan['training']
            and protocol['code'] == plan['implementation'] and protocol['versions'] == plan['versions'],
            'Protocol/plan binding mismatch')
    require(protocol['study_kind'] == ('synthetic' if cfg.fixture else 'ODS_internal_holdout'),
            'Synthetic/production provenance mismatch')
    require(plan['evaluation'] == 'requires_caller_trusted_anchor_and_explicit_user_decision',
            'Evaluation gate changed')
    require(plan['output_schema'] == {
        'pending': ['manifest.json', 'runs.csv', 'stages.csv', 'ni_decision.json', 'report.html'],
        'test_only': list(TEST_FILES)}, 'Output schema changed')
    if verify_inputs:
        source = verify_sources(cfg)
        require(source['refs'] == plan['sources'] and source['manifest']['label_maps'] == plan['label_maps']
                and row_identity(source['pool']) == plan['development_pool'], 'Source/development identity changed')
        require(source['corpus_fit_selection'] == plan['corpus_fit_selection'], 'Corpus fit selection changed')
        require(training_contract(cfg, source) == plan['training'], 'Frozen training contract changed')
        expected_deps = [{'method': m, 'ranking': 'full_development_union_only',
                          'budget': list(cfg.selected_counts), 'refit_corpus': source['refs']['corpus'],
                          'shared_seeds': list(SEEDS)} for m in METHODS[1:]]
        require(plan['dependencies'] == expected_deps, 'Dependency plan changed')
    return plan


def verify_frozen_runtime(root, plan):
    require(read_sealed(Path(root) / 'plan.json') == plan, 'Plan changed during operation')
    require(implementation_files() == plan['implementation'], 'Implementation changed during operation')
    require(fingerprint(Path(root) / 'protocol.json') == plan['protocol'], 'Protocol changed during operation')
    for info in plan['sources'].values():
        require(fingerprint(info['path']) == {k: info[k] for k in ('sha256', 'bytes')}, 'Source changed during operation')


@contextmanager
def measured_stage(root, method, run_id, stage, device, scope):
    measurement = _Measurement(device, 'S08 dev-only; shared costs charged once' if scope == 'shared'
                               else 'S08 fixed-epoch final training; no sealed inference')
    status = 'failed'
    try:
        with measurement:
            yield
        status = 'success'
    finally:
        if hasattr(measurement, '_rss'):
            row = measurement.stage_row(run_id or method, stage, status)
        else:
            row = {'stage': stage, 'run_id': run_id or method, 'status': status, 'seconds': None,
                   'peak_rss_bytes': None, 'peak_vram_bytes': None}
        row.update(scope=scope, method=method, charge_count=1)
        write_new(Path(root) / 'stages' / f'{uuid.uuid4().hex}.json', sealed(row))


def _provenance(plan, run, feature_id):
    return {'plan_sha256': digest(plan), 'protocol': plan['protocol'], 'run': run,
            'feature_id': feature_id, 'development_pool': plan['development_pool'],
            'label_maps_sha256': digest(plan['label_maps']), 'training_sha256': digest(plan['training'])}


def _verify_refit(full, selected, selection, cfg):
    require((selected.builder.n_char_features, selected.builder.n_word_features) == cfg.selected_counts,
            'Selected branch dimension mismatch')
    require(selected.meta['source'] == full.meta['source'] and selected.meta['fit'] == full.meta['fit'],
            'Refit used different corpus/row selection')
    for branch, offset in [('char', 0), ('word', full.builder.n_char_features)]:
        indices = np.sort(np.asarray(getattr(selection, branch).indices, dtype=np.int64) - offset)
        old, new = getattr(full.builder, branch), getattr(selected.builder, branch)
        require(len(np.unique(indices)) == len(indices) and (indices >= 0).all()
                and (indices < len(old.vocabulary_)).all(), 'Invalid branch indices')
        require(new.get_feature_names_out().tolist() == old.get_feature_names_out()[indices].tolist(),
                'Selected vocabulary identity/order mismatch')
        require(np.allclose(new.idf_, old.idf_[indices], rtol=1e-6, atol=1e-6), 'Refit IDF invariant failed')
        require(new.norm == 'l2' and selected.meta['refit_checks'][f'{branch}_idf_max_abs_diff'] <= 1e-5,
                'Branch normalization/refit checks failed')


def _prepare_selected(root, plan, method, source, device):
    cfg, full, pool = S08Config(**plan['config']), source['full'], source['pool']
    require(method in METHODS[1:], 'S08 permits Chi2 methods only')
    verified = verify_sources(cfg)
    require(pool.equals(verified['pool']) and full.meta == verified['full'].meta
            and feature_contract_fingerprint(full.builder) == feature_contract_fingerprint(verified['full'].builder),
            'Ranking input differs from verified development/full TF-IDF')
    base = root / 'shared' / method
    base.mkdir(parents=True, exist_ok=False)
    verify_frozen_runtime(root, plan)
    with measured_stage(root, method, None, 'ranking_transform', device, 'shared'):
        X = full.builder.transform(pool.name.tolist(), chunk_rows=cfg.chunk_rows)
    with measured_stage(root, method, None, 'ranking_selection', device, 'shared'):
        selection = select_features(X, LabelMaps.from_dict(plan['label_maps']).encode('L3', pool.L3),
                                    SelectionConfig(method=method, k=sum(cfg.selected_counts), char_share=.75, seed=42),
                                    full.builder.n_char_features, l3_labels=plan['label_maps']['L3'])
        require((len(selection.char.indices), len(selection.word.indices)) == cfg.selected_counts,
                'Strict selected budget unavailable')
        save_selection_artifact(str(base / 'selection'), selection,
                                {'s08_plan_sha256': digest(plan), 'development_pool': plan['development_pool']},
                                ratings_copy=True)
        selection, _ = load_selection_artifact(str(base / 'selection'))
    with measured_stage(root, method, None, 'refit', device, 'shared'):
        refit = refit_selected_vectorizer(cfg.corpus, full.builder, full.meta, selection,
                                          out_dir=str(base / 'tfidf'), row_norm_sample=1000)
        reloaded = load_vectorizer_artifact(str(base / 'tfidf'))
        require(feature_contract_fingerprint(refit.builder) == feature_contract_fingerprint(reloaded.builder),
                'Feature reload mismatch')
        _verify_refit(full, reloaded, selection, cfg)
    write_new(base / 'receipt.json', sealed({'plan_sha256': digest(plan), 'method': method,
                                            'files': {n: tree(base / n) for n in ('selection', 'tfidf')},
                                            'feature_contract': feature_contract_fingerprint(reloaded.builder),
                                            'refit_checks': refit.checks}))
    verify_frozen_runtime(root, plan)
    return reloaded


def verify_selected(root, plan, method, full):
    require(method in METHODS[1:], 'Unknown S08 selection method')
    base = Path(root) / 'shared' / method
    receipt = read_sealed(base / 'receipt.json')
    require(receipt['plan_sha256'] == digest(plan) and receipt['method'] == method, 'Selection provenance mismatch')
    for name in ('selection', 'tfidf'):
        verify_tree(base / name, receipt['files'][name])
    selection, meta = load_selection_artifact(str(base / 'selection'))
    cfg = S08Config(**plan['config'])
    require(meta['provenance'] == {'s08_plan_sha256': digest(plan), 'development_pool': plan['development_pool']}
            and selection.method == method and selection.k_requested == sum(cfg.selected_counts),
            'Selection method/budget/development provenance mismatch')
    from .s06_comparison import _selection_from_saved_ratings, _verify_ratings_meta
    require(_verify_ratings_meta(selection.ratings_dir, method) == selection.ratings_manifest,
            'Saved ranking manifest mismatch')
    replay = _selection_from_saved_ratings(method, selection.ratings_dir, sum(cfg.selected_counts), .75,
                                           full.builder.n_char_features, full.builder.n_features,
                                           l3_labels=plan['label_maps']['L3'])
    for branch in ('char', 'word'):
        saved, expected = getattr(selection, branch), getattr(replay, branch)
        require(np.array_equal(saved.indices, expected.indices)
                and np.allclose(saved.scores, expected.scores, equal_nan=True),
                'Selection does not reproduce from frozen rankings')
    selected = load_vectorizer_artifact(str(base / 'tfidf'))
    _verify_refit(full, selected, selection, S08Config(**plan['config']))
    require(feature_contract_fingerprint(selected.builder) == receipt['feature_contract'], 'Feature contract changed')
    return selected


def _train_one(root, plan, run, builder, feature_id, pool, device):
    verify_frozen_runtime(root, plan)
    cfg = S08Config(**plan['config'])
    verified = verify_sources(cfg)
    require(pool.equals(verified['pool']), 'Training rows/content differ from frozen full development union')
    require(run in plan['runs'], 'Unplanned training run')
    expected = verified['full'] if run['method'] == 'none' else verify_selected(root, plan, run['method'], verified['full'])
    expected_id = plan['sources']['features']['sha256'] if run['method'] == 'none' else sha(
        root / 'shared' / run['method'] / 'tfidf' / 'features.joblib')
    require(feature_id == expected_id
            and feature_contract_fingerprint(builder) == feature_contract_fingerprint(expected.builder),
            'Training feature contract/identity differs from frozen dependency')
    run_dir = root / 'runs' / run['run_id']
    run_dir.mkdir(parents=True, exist_ok=False)
    maps = LabelMaps.from_dict(plan['label_maps'])
    mc, lc, tc = ModelConfig(**plan['training']['model']), LossConfig(**plan['training']['loss']), TrainConfig(**plan['training']['train'])
    y = tuple(maps.encode(l, pool[l]) for l in LEVELS)
    sizes = tuple(maps.n_classes(l) for l in LEVELS)
    weights = {h: compute_class_weights(y[i], sizes[i], lc.class_weight_mode[h], lc.class_weight_alpha)
               for i, h in enumerate(('l1', 'l2', 'l3'))}
    with measured_stage(root, run['method'], run['run_id'], 'train_transform', device, 'run'):
        X = builder.transform(pool.name.tolist(), chunk_rows=cfg.chunk_rows)
    set_seed(run['seed'])
    model = SparseMLP(builder.n_features, sizes, mc)
    # Required training API monitor is explicitly in-sample, never a held-out decision source.
    with measured_stage(root, run['method'], run['run_id'], 'train', device, 'run'):
        history, best, trace = train_model(model, X, y, X, y, lc, tc, weights, device,
                                           run['seed'], restore_best=False)
    require(len(history) == tc.epochs and not best.get('restored_best'), 'Final epoch rule violated')
    provenance = _provenance(plan, run, feature_id)
    record = s07._training_record(provenance, history, best, trace, tc)
    s07._require_finite_model(model)
    save_artifact(str(run_dir / 'checkpoint'), model, mc, builder, maps, RejectRule(.5, False, False),
                  {'note': 'S08 raw top-1/no reject; no holdout evaluation'},
                  {'mode': lc.class_weight_mode, 'alpha': lc.class_weight_alpha,
                   'weights': {k: None if v is None else v.tolist() for k, v in weights.items()}},
                  plan['training'], {'s08': provenance, 'seed': run['seed']})
    write_new(run_dir / 'training.json', record)
    write_new(run_dir / 'checkpoint_receipt.json', sealed({'provenance': provenance, 'files': tree(run_dir / 'checkpoint')}))
    verify_checkpoint(root, plan, run, builder, feature_id, device)
    verify_frozen_runtime(root, plan)
    write_new(run_dir / 'result.json', sealed({'status': 'success', 'provenance': provenance,
                                              'checkpoint_receipt': fingerprint(run_dir / 'checkpoint_receipt.json'),
                                              'training': fingerprint(run_dir / 'training.json'),
                                              'validation_macro_f1': None, 'test_macro_f1': None,
                                              'model_bytes': (run_dir / 'checkpoint' / 'model.pt').stat().st_size}))


def verify_checkpoint(root, plan, run, builder, feature_id, device=None):
    base = Path(root) / 'runs' / run['run_id']
    receipt = read_sealed(base / 'checkpoint_receipt.json')
    expected = _provenance(plan, run, feature_id)
    require(receipt['provenance'] == expected, 'Checkpoint provenance mismatch')
    verify_tree(base / 'checkpoint', receipt['files'])
    require(read_json(base / 'checkpoint' / 'model_spec.json') == {
        'n_features': builder.n_features, 'n_classes': [len(plan['label_maps'][l]) for l in LEVELS],
        'model': plan['training']['model']}, 'Model architecture/input spec changed')
    art = load_artifact(str(base / 'checkpoint'), device or torch.device('cpu'))
    require(art.maps.to_dict() == plan['label_maps'] and art.meta['s08'] == expected
            and art.config == plan['training'], 'Loaded checkpoint contracts differ')
    require(feature_contract_fingerprint(art.features) == feature_contract_fingerprint(builder),
            'Loaded checkpoint vocabulary/IDF/index contract differs')
    s07._require_finite_model(art.model)
    return art


def execute(output_root):
    """Nine fresh fixed-epoch development models; never performs sealed inference."""
    root = Path(output_root)
    plan = load_plan(root)
    for name in ('execution_started.json', 'execution_finished.json', 'training_freeze.json', 'runs', 'shared', 'stages'):
        require(not (root / name).exists(), 'Unsafe reuse/resume rejected; use a fresh experiment')
    cfg = S08Config(**plan['config'])
    device = resolve_device(cfg.device)
    if cfg.num_threads:
        torch.set_num_threads(cfg.num_threads)
    write_new(root / 'execution_started.json', sealed({'plan_sha256': digest(plan), 'started_at': now()}))
    source = verify_sources(cfg)
    summary = {'success': [], 'failed': [], 'skipped': [], 'pending': []}
    active = None
    try:
        for method in METHODS:
            active = None
            features = source['full'] if method == 'none' else _prepare_selected(root, plan, method, source, device)
            fid = plan['sources']['features']['sha256'] if method == 'none' else sha(root / 'shared' / method / 'tfidf' / 'features.joblib')
            for run in [r for r in plan['runs'] if r['method'] == method]:
                active = run
                _train_one(root, plan, run, features.builder, fid, source['pool'], device)
                summary['success'].append(run['run_id'])
                active = None
        verify_frozen_runtime(root, plan)
    except Exception as exc:
        summary['stopped_reason'] = f'{type(exc).__name__}: {exc}'
        if active:
            summary['failed'].append(active['run_id'])
            path = root / 'runs' / active['run_id'] / 'result.json'
            if not path.exists():
                write_new(path, sealed({'status': 'failed', 'reason': summary['stopped_reason'],
                                        'plan_sha256': digest(plan)}))
        else:
            for run in plan['runs']:
                if run['method'] == method and run['run_id'] not in summary['success']:
                    summary['skipped'].append(run['run_id'])
                    write_new(root / 'runs' / run['run_id'] / 'result.json', sealed({
                        'status': 'skipped', 'reason': summary['stopped_reason'], 'plan_sha256': digest(plan)}))
    completed = set(summary['success'] + summary['failed'] + summary['skipped'])
    summary['pending'] = [r['run_id'] for r in plan['runs'] if r['run_id'] not in completed]
    write_new(root / 'execution_finished.json', sealed({'plan_sha256': digest(plan), 'finished_at': now(),
                                                       'summary': summary, 'runs': tree(root / 'runs') if (root / 'runs').exists() else {},
                                                       'shared': tree(root / 'shared') if (root / 'shared').exists() else {},
                                                       'stages': tree(root / 'stages') if (root / 'stages').exists() else {}}))
    if len(summary['success']) == 9 and not summary.get('stopped_reason'):
        # All model/feature/label map files are covered recursively, not just weights.
        write_new(root / 'training_freeze.json', sealed({
            'schema': 's08-training-freeze@1', 'plan': fingerprint(root / 'plan.json'),
            'protocol': fingerprint(root / 'protocol.json'), 'runs': tree(root / 'runs'),
            'shared': tree(root / 'shared'), 'execution': fingerprint(root / 'execution_finished.json'),
            'prediction_interface': {'top_k': 1, 'head': 'L3', 'reject': False,
                                     'empty_rows': 'bias_path', 'batch_size': plan['training']['train']['eval_batch_size']},
            'gate': 'awaiting_user_approval'}))
        verify_training_freeze(root)
    return summary


def verify_training_freeze(output_root):
    root = Path(output_root)
    plan = load_plan(root)
    freeze = read_sealed(root / 'training_freeze.json')
    require(freeze['schema'] == 's08-training-freeze@1' and freeze['plan'] == fingerprint(root / 'plan.json')
            and freeze['protocol'] == fingerprint(root / 'protocol.json'), 'Training freeze binding mismatch')
    require(freeze['gate'] == 'awaiting_user_approval' and freeze['prediction_interface'] == {
        'top_k': 1, 'head': 'L3', 'reject': False, 'empty_rows': 'bias_path',
        'batch_size': plan['training']['train']['eval_batch_size']}, 'Frozen prediction interface changed')
    verify_tree(root / 'runs', freeze['runs'])
    verify_tree(root / 'shared', freeze['shared'])
    require(freeze['execution'] == fingerprint(root / 'execution_finished.json'), 'Execution receipt changed')
    finished = read_sealed(root / 'execution_finished.json')
    require(finished['summary'] == {'success': [r['run_id'] for r in plan['runs']],
                                   'failed': [], 'skipped': [], 'pending': []}, 'Missing/failed final models')
    require(finished['plan_sha256'] == digest(plan), 'Execution plan mismatch')
    for name in ('runs', 'shared', 'stages'):
        verify_tree(root / name, finished[name])
    full = load_vectorizer_artifact(S08Config(**plan['config']).full_tfidf)
    for method in METHODS:
        features = full if method == 'none' else verify_selected(root, plan, method, full)
        fid = plan['sources']['features']['sha256'] if method == 'none' else sha(root / 'shared' / method / 'tfidf' / 'features.joblib')
        for run in [r for r in plan['runs'] if r['method'] == method]:
            base = root / 'runs' / run['run_id']
            result = read_sealed(base / 'result.json')
            require(result['status'] == 'success' and result['provenance'] == _provenance(plan, run, fid),
                    'Stale or unsuccessful result')
            require(result['checkpoint_receipt'] == fingerprint(base / 'checkpoint_receipt.json')
                    and result['training'] == fingerprint(base / 'training.json'), 'Run receipt/training changed')
            verify_checkpoint(root, plan, run, features.builder, fid)
    return freeze


def evaluate(output_root, approval=None, trusted_anchor=None, *, report_dir=None):
    """Evaluate once. Only the orchestrator may supply the trusted consent digest.

    The digest must come from actual user consent, never from untrusted payload.
    No approval record or anchor is created by this API.
    """
    from .s08_evaluation import evaluate_authorized
    return evaluate_authorized(output_root, approval, trusted_anchor, report_dir=report_dir)
