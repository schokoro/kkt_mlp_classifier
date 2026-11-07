"""S07 repeated-development-split study. Production gates are deliberately strict.

No S06 supervised artifact is consumed. Planning never fits or scores a model.
Execution is single-use: after an attempt (even a crash), use a fresh plan/root.
Fixture mode is explicitly marked synthetic and cannot yield a production study.
"""
from __future__ import annotations

import dataclasses
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .artifact import environment_versions, load_artifact, save_artifact
from .attribution import (build_gxi_ratings, compute_margin_attributions,
                          feature_contract_fingerprint, select_features_gradient_x_input)
from .config import LossConfig, ModelConfig, TrainConfig
from .experiment import _Measurement, _l3_val_metrics
from .features import preprocess, strip_whitespace
from .labels import LEVELS, LabelMaps
from .model import SparseMLP
from .reject import RejectRule
from .s06_split import assign_groups, search_feasible_assignment
from .selection import (SelectionConfig, load_selection_artifact, save_selection_artifact,
                        select_features, validate_ratings_dir)
from .train import compute_class_weights, predict, resolve_device, set_seed, train_model
from .vectorizer import load_vectorizer_artifact, refit_selected_vectorizer

METHODS = ('none', 'global', 'class_aware', 'gradient_x_input')
SPLIT_SEEDS = (101, 202, 303)
MODEL_SEEDS = (42, 43, 44)
MINIMA = {'train_fit': 6, 'train_monitor': 2, 'comparison_val': 2}
SOURCE_PINS = {
    'manifest': 'ff44250867e864af23c72da94c38ed4d5a2796af9a658b83787193432ee2cd5d',
    'protocol': 'caffae8c2609d9f9cff28c47cf368cedddf7c38e7c4a4c8801bb553802c5cb0d',
    'corpus': 'f9a057a00db10ed115dd5f47e035f3a3c56c710d9463f47d38812d4cceef3dbc',
    'features': '7ff42885729d0bcfb2b6e9e0893b1be4ddf59562705e3ed048614183cd952052',
}
WARNINGS = [
    'ODS mechanics and robustness study only; no industrial validation.',
    'Development data was previously inspected in S06; these are not independent external holdouts.',
    'Related split resamples and seed pairs are correlated; no independence-based confidence interval.',
    'Full TF-IDF fit is partially transductive; no validation/test labels used in feature fit.',
    'Rare L3 estimates (especially 2 validation examples) are unstable.',
    'Unavailable VRAM measurements are null; stage RSS peaks are not additive.',
]


class S07Error(RuntimeError):
    """A failed integrity gate; never silently substitute or relax inputs."""


def require(ok: Any, message: str) -> None:
    if not ok:
        raise S07Error(message)


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding='utf-8') as f:
        value = json.load(f)
    require(isinstance(value, dict), f'JSON object required: {path}')
    return value


def write_new(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def sealed(value: dict) -> dict:
    require('integrity_sha256' not in value, 'Already sealed')
    return {**value, 'integrity_sha256': digest(value)}


def read_sealed(path: str | Path) -> dict:
    value = read_json(path)
    expected = value.pop('integrity_sha256', None)
    require(expected == digest(value), f'Invalid/missing integrity seal: {path}')
    return value


def fingerprint(path: str | Path) -> dict:
    p = Path(path)
    require(p.is_file() and not p.is_symlink(), f'Missing/unsafe file: {p}')
    return {'sha256': sha(p), 'bytes': p.stat().st_size}


def tree(path: str | Path) -> dict:
    path = Path(path)
    require(path.is_dir() and not path.is_symlink(), f'Missing/unsafe directory: {path}')
    return {str(p.relative_to(path)): fingerprint(p) for p in sorted(path.rglob('*')) if p.is_file()}


def verify_tree(path: str | Path, expected: dict) -> None:
    require(bool(expected) and tree(path) == expected, f'Artifact tree fingerprint mismatch: {path}')


def row_identity(df: pd.DataFrame) -> dict:
    return {'rows': len(df), 'row_ids_sha256': digest(sorted(df.row_id.astype(int).tolist())),
            'groups_sha256': digest(sorted(group_keys(df)))}


def group_keys(df: pd.DataFrame) -> list[str]:
    return [strip_whitespace(t) for t in preprocess(df['name'].tolist())]


@dataclasses.dataclass(frozen=True)
class S07Config:
    manifest: str = 'outputs/S06/splits/manifest.json'
    protocol: str = 'outputs/S06_final_cpu_20261107/shared/training_protocol.json'
    full_tfidf: str = 'outputs/S02/tfidf_1m'
    corpus: str = 'data/corpus.pqt'
    device: str = 'cpu'
    num_threads: int | None = None
    chunk_rows: int = 50000
    attribution_batch_size: int = 256
    # Explicit fixture profile only: all source hashes must still be pinned.
    fixture: bool = False
    fixture_pins: dict | None = None
    fixture_full_counts: tuple[int, int] = (300, 100)
    fixture_selected_counts: tuple[int, int] = (75, 25)

    def validate(self) -> None:
        require(self.chunk_rows > 0 and self.attribution_batch_size > 0, 'Invalid batch/chunk size')
        require(self.device in ('cpu', 'cuda', 'auto'), 'Invalid device')
        require(self.num_threads is None or self.num_threads > 0, 'Invalid thread count')
        if self.fixture:
            require(isinstance(self.fixture_pins, dict) and set(self.fixture_pins) == set(SOURCE_PINS),
                    'Synthetic fixture requires all four explicit source pins')
            c, w = self.fixture_full_counts
            sc, sw = self.fixture_selected_counts
            require(0 < sc <= c and 0 < sw <= w and sc * 3 == sw * 9,
                    'Fixture budgets must keep the 75/25 allocation')
        else:
            require(self.fixture_pins is None, 'Production source pins cannot be overridden')

    @property
    def full_counts(self) -> tuple[int, int]:
        return tuple(self.fixture_full_counts) if self.fixture else (300000, 100000)

    @property
    def selected_counts(self) -> tuple[int, int]:
        return tuple(self.fixture_selected_counts) if self.fixture else (75000, 25000)


def split_development_pool(pool: pd.DataFrame, label_maps: dict,
                           seeds: tuple[int, ...] = SPLIT_SEEDS) -> tuple[dict, list]:
    """Constructive witness first; exact bounded fallback distinguishes proof/search failure.

    Three leaf parts are assigned jointly, so monitor/fit minima cannot invalidate
    a previously greedy outer split. Fractions are targets; minima take priority.
    """
    require(tuple(seeds) == SPLIT_SEEDS, 'S07 fixes exactly seeds 101/202/303')
    require(pool.row_id.is_unique, 'Duplicate development row IDs')
    require(LabelMaps.from_df(pool).to_dict() == label_maps, 'Development label-map mismatch')
    work = pool.copy()
    work['_group'] = group_keys(work)
    require(work.groupby('_group').L3.nunique().max() == 1, 'Group spans conflicting L3 labels')
    results, diagnostics = {}, []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        leaves = {p: [] for p in MINIMA}
        failures = []
        for label, data in work.groupby('L3', sort=True):
            groups = [g for _, g in data.groupby('_group', sort=True)]
            sizes = np.asarray([len(g) for g in groups], dtype=np.int64)
            n = len(data)
            val = max(2, int(round(n * .1)))
            mon = max(2, int(round((n - val) * .1)))
            quota = np.array([n - val - mon, mon, val])
            minima = np.array(list(MINIMA.values()))
            if quota[0] < 6:
                quota = minima.copy()
                quota[0] += n - sum(minima)
            assignment, rows = assign_groups(sizes, quota, minima, rng)
            if np.any(rows < minima):
                assignment, proven, nodes = search_feasible_assignment(sizes, minima)
                if assignment is None:
                    failures.append({'L3': label, 'group_sizes': sizes.tolist(),
                                     'proven_infeasible': bool(proven), 'search_nodes': nodes})
                    continue
            for group, part in zip(groups, assignment):
                leaves[list(MINIMA)[int(part)]].extend(group.row_id.astype(int).tolist())
        diagnostics.append({'seed': seed, 'feasible': not failures, 'failures': failures})
        if not failures:
            parts = {p: pool[pool.row_id.isin(ids)].sort_values('row_id').reset_index(drop=True)
                     for p, ids in leaves.items()}
            parts['outer_train'] = pd.concat([parts['train_fit'], parts['train_monitor']]).sort_values(
                'row_id').reset_index(drop=True)
            validate_partition(pool, parts, label_maps)
            results[f'split_{seed}'] = parts
    return results, diagnostics


def validate_partition(pool: pd.DataFrame, parts: dict, label_maps: dict) -> None:
    require(set(parts) == {*MINIMA, 'outer_train'}, 'Incorrect partition set')
    row_sets, group_sets = [], []
    for part, minimum in MINIMA.items():
        data = parts[part]
        require(data.row_id.is_unique, f'{part}: duplicate IDs')
        counts = data.L3.value_counts().reindex(label_maps['L3'], fill_value=0)
        require((counts >= minimum).all(), f'{part}: class minimum {minimum} failed')
        require(set(data.L3) == set(label_maps['L3']), f'{part}: class universe changed')
        row_sets.append(set(data.row_id))
        group_sets.append(set(group_keys(data)))
    for i in range(3):
        for j in range(i):
            require(not row_sets[i] & row_sets[j] and not group_sets[i] & group_sets[j],
                    'Row/normalized-name group leakage')
    require(set.union(*row_sets) == set(pool.row_id), 'Development pool coverage mismatch')
    require(set(parts['outer_train'].row_id) == row_sets[0] | row_sets[1], 'Outer union mismatch')
    indexed = pool.set_index('row_id')
    for part, data in parts.items():
        expected = indexed.loc[data.row_id, ['name', *LEVELS]].reset_index()
        require(expected.equals(data[['row_id', 'name', *LEVELS]].reset_index(drop=True)),
                f'{part}: source row content mismatch')


def verify_sources(cfg: S07Config) -> dict:
    """Only sealed-test row_id/name projected; never query sealed label columns."""
    cfg.validate()
    pins = cfg.fixture_pins if cfg.fixture else SOURCE_PINS
    sources = {'manifest': Path(cfg.manifest), 'protocol': Path(cfg.protocol),
               'corpus': Path(cfg.corpus), 'features': Path(cfg.full_tfidf) / 'features.joblib'}
    refs = {}
    for key, path in sources.items():
        fp = fingerprint(path)
        require(fp['sha256'] == pins[key], f'{key}: authoritative SHA256 mismatch')
        refs[key] = {'path': str(path.resolve()), **fp}
    meta_path = Path(cfg.full_tfidf) / 'vectorizer_meta.json'
    refs['vectorizer_meta'] = {'path': str(meta_path.resolve()), **fingerprint(meta_path)}
    m, p = read_json(cfg.manifest), read_json(cfg.protocol)
    require(m['policy']['min_count'] == 12, 'Original S06 support filter must stay >=12')
    frames = {}
    for part in ('outer_train', 'comparison_val', 'sealed_test'):
        spec = m['parts'][part]
        path = Path(spec['file'])
        require(fingerprint(path) == {k: spec[k] for k in ('sha256', 'bytes')},
                f'S06 {part}: source fingerprint mismatch')
        data = pd.read_parquet(path, columns=['row_id', 'name'] if part == 'sealed_test' else None)
        require(len(data) == spec['n_rows'] and data.row_id.is_unique, f'{part}: identity/count mismatch')
        require(not data[['row_id', 'name']].isna().any().any(), f'{part}: null structural identity')
        frames[part] = data
        refs[part] = {'path': str(path.resolve()), **fingerprint(path), **row_identity(data)}
    pool = pd.concat([frames['outer_train'], frames['comparison_val']], ignore_index=True).sort_values(
        'row_id').reset_index(drop=True)
    require(pool.row_id.is_unique and not pool[['name', *LEVELS]].isna().any().any(),
            'Invalid development rows')
    require(not set(pool.row_id) & set(frames['sealed_test'].row_id), 'Sealed row identity leakage')
    require(not set(group_keys(pool)) & set(group_keys(frames['sealed_test'])), 'Sealed group leakage')
    require(LabelMaps.from_df(pool).to_dict() == m['label_maps'], 'S06 label maps not preserved')
    require(p['frozen'] is True and p['fixed_epochs'] == p['chosen']['fixed_epochs'] > 0,
            'Protocol not frozen/invalid epoch rule')
    require(p['full_tfidf_id'] == refs['features']['sha256'], 'Protocol/full TF-IDF mismatch')
    require(p['corpus_fingerprint']['sha256'] == refs['corpus']['sha256'], 'Protocol/corpus mismatch')
    for part in frames:
        require(all(p['splits'][part][k] == m['parts'][part][k] for k in ('sha256', 'bytes', 'n_rows')),
                f'Protocol/source {part} mismatch')
    sections = p['resolved_sections']
    require(sections['train']['scheduler'] == 'none', 'Only frozen constant LR supported')
    if not cfg.fixture:
        require(len(pool) == 14652 and len(m['label_maps']['L3']) == 293, 'Production universe mismatch')
        require(p['fixed_epochs'] == 14, 'Production final epochs must be 14')
        require(sections['model'] == {'hidden': [256], 'dropout': [.3], 'activation': 'relu', 'norm': None},
                'Frozen production model mismatch')
        for k, v in {'optimizer': 'adamw', 'lr': .001, 'weight_decay': 0., 'batch_size': 256}.items():
            require(sections['train'][k] == v, f'Frozen optimizer mismatch: {k}')
    full = load_vectorizer_artifact(cfg.full_tfidf)
    require((full.builder.n_char_features, full.builder.n_word_features) == cfg.full_counts,
            'Full feature sizes mismatch')
    meta = full.meta
    require(meta['source']['sha256'] == refs['corpus']['sha256']
            and meta['source']['bytes'] == refs['corpus']['bytes'], 'Full source/corpus mismatch')
    require(meta['source']['strategy'] == 'head', 'Unsupported corpus selection strategy')
    if not cfg.fixture:
        require(meta['source']['rows_used'] == 981850 and meta['source']['max_rows'] == 1000000
                and meta['fit'] == {'rows_in': 981850, 'rows_fit': 981850, 'subsampled': False},
                'Full corpus fit selection mismatch')
    return {'refs': refs, 'manifest': m, 'protocol': p, 'pool': pool, 'full': full,
            'corpus_fit_selection': {'source': meta['source'], 'fit': meta['fit'],
                                     'tfidf_config': meta['tfidf_config'], 'seed': meta['seed']}}


def fresh_root(root: str | Path) -> Path:
    path = Path(root)
    path.mkdir(parents=True, exist_ok=False)
    return path


def preflight(cfg: S07Config, output_root: str | Path) -> dict:
    """Persist compact pass/fail evidence even if a mandatory gate fails."""
    root = fresh_root(output_root)
    audit = {'schema': 's07-preflight@1', 'created_at': now(), 'config': dataclasses.asdict(cfg),
             'study_kind': 'synthetic' if cfg.fixture else 'ODS_development', 'workload_runs': 0,
             'sealed_labels_read': False, 'sealed_inference': False}
    try:
        source = verify_sources(cfg)
        splits, diagnostics = split_development_pool(source['pool'], source['manifest']['label_maps'])
        audit.update(refs=source['refs'], corpus_fit_selection=source['corpus_fit_selection'],
                     label_maps_sha256=digest(source['manifest']['label_maps']),
                     development_pool=row_identity(source['pool']), feasibility=diagnostics,
                     partitions={s: {p: {**row_identity(d), 'minimum_l3_support': int(d.L3.value_counts().min())}
                                     for p, d in parts.items()} for s, parts in splits.items()},
                     resolved_protocol=source['protocol']['resolved_sections'],
                     fixed_epochs=source['protocol']['fixed_epochs'])
        require(len(splits) == 3, 'Mandatory grouped split minima not satisfied; see feasibility diagnostics')
        audit.update(passed=True)
    except Exception as exc:
        audit.update(passed=False, error=f'{type(exc).__name__}: {exc}')
    write_new(root / 'preflight.json', sealed(audit))
    return audit


def implementation_files() -> dict:
    repo = Path(__file__).resolve().parents[1]
    paths = sorted((repo / 'kkt_clf').glob('*.py')) + [repo / 'scripts' / 'run_s07.py']
    return {str(p.relative_to(repo)): fingerprint(p) for p in paths}


def _run_matrix(cfg: S07Config) -> list[dict]:
    return [{'run_id': f'split_{s}__{method}__seed_{seed}', 'split_id': f'split_{s}',
             'method': method, 'seed': seed,
             'n_char': cfg.full_counts[0] if method == 'none' else cfg.selected_counts[0],
             'n_word': cfg.full_counts[1] if method == 'none' else cfg.selected_counts[1],
             'feature_dependency': 'full' if method == 'none' else f'split_{s}/{method}',
             'initial_status': 'pending'}
            for s in SPLIT_SEEDS for method in METHODS for seed in MODEL_SEEDS]


def dependency_matrix(cfg: S07Config, split_specs: dict, corpus_sha256: str) -> list[dict]:
    return [
        {'dependency_id': f'{sid}/{method}', 'split_id': sid, 'method': method,
         'ranking_rows': spec['parts']['outer_train']['sha256'],
         'reference_checkpoint_run': f'{sid}__none__seed_42' if method == 'gradient_x_input' else None,
         'reference_seed': 42 if method == 'gradient_x_input' else None,
         'selection': {'k': sum(cfg.selected_counts), 'char_share': .75, 'seed': 42},
         'attribution': {'target': 'L3 true-minus-strongest-competitor margin',
                         'positive_only': True, 'aggregation': 'class_aware', 'rows': 'all_outer_train'}
         if method == 'gradient_x_input' else None,
         'refit_source_sha256': corpus_sha256, 'shared_across_seeds': list(MODEL_SEEDS)}
        for sid, spec in split_specs.items() for method in METHODS[1:]]


def write_plan(cfg: S07Config, output_root: str | Path) -> dict:
    """Freeze the full matrix, source/split identities and dependency DAG before scoring."""
    # Fail before touching output if source/split gates fail.
    source = verify_sources(cfg)
    splits, diagnostic = split_development_pool(source['pool'], source['manifest']['label_maps'])
    require(len(splits) == 3, f'Feasibility failed: {diagnostic}')
    root = fresh_root(output_root)
    split_specs = {}
    for sid, parts in splits.items():
        spec = {}
        supports = []
        for part, data in parts.items():
            path = root / 'splits' / sid / f'{part}.parquet'
            path.parent.mkdir(parents=True, exist_ok=True)
            data.to_parquet(path, index=False)
            spec[part] = {'file': str(path.relative_to(root)), **fingerprint(path), **row_identity(data),
                          'row_l3_sha256': digest(data[['row_id', 'L3']].sort_values('row_id').values.tolist())}
            counts = data.L3.value_counts()
            supports.extend({'part': part, 'L3': label, 'support': int(counts.get(label, 0))}
                            for label in source['manifest']['label_maps']['L3'])
        support_path = root / 'splits' / sid / 'supports.csv'
        pd.DataFrame(supports).to_csv(support_path, index=False)
        split_specs[sid] = {'seed': int(sid.split('_')[1]), 'parts': spec,
                            'supports': {'file': str(support_path.relative_to(root)), **fingerprint(support_path)}}
    dependencies = dependency_matrix(cfg, split_specs, source['refs']['corpus']['sha256'])
    protocol = source['protocol']
    final_train = {**protocol['resolved_sections']['train'], 'epochs': protocol['fixed_epochs'],
                   'patience': protocol['fixed_epochs'], 'device': cfg.device,
                   'num_threads': cfg.num_threads}
    plan = {'schema': 's07-plan@1', 'created_at': now(), 'config': dataclasses.asdict(cfg),
            'study_kind': 'synthetic' if cfg.fixture else 'ODS_development',
            'sources': source['refs'], 'corpus_fit_selection': source['corpus_fit_selection'],
            'label_maps': source['manifest']['label_maps'], 'minima': MINIMA,
            'development_pool': row_identity(source['pool']), 'splits': split_specs,
            'runs': _run_matrix(cfg), 'dependencies': dependencies,
            'training': {'model': protocol['resolved_sections']['model'],
                         'loss': protocol['resolved_sections']['loss'], 'train': final_train,
                         'restore_best': False, 'weights': 'final_epoch',
                         'no_validation_tuning': True},
            'metrics': {'prediction': 'raw top-1 L3', 'reject': False, 'universe': 'fixed label_maps.L3',
                        'zero_division': 0},
            'versions': environment_versions(), 'implementation': implementation_files(), 'warnings': WARNINGS,
            'resume': 'unsupported; execution_started.json makes a plan single-use'}
    plan = json.loads(json.dumps(plan, ensure_ascii=False, allow_nan=False))
    write_new(root / 'plan.json', sealed(plan))
    return plan


def load_plan(output_root: str | Path, verify_inputs: bool = True) -> dict:
    root = Path(output_root)
    plan = read_sealed(root / 'plan.json')
    require(plan.get('schema') == 's07-plan@1', 'Unknown plan schema')
    cfg = S07Config(**plan['config'])
    cfg.validate()
    require(plan['runs'] == _run_matrix(cfg) and len(plan['runs']) == 36, 'Run matrix/config mismatch')
    require(plan['study_kind'] == ('synthetic' if cfg.fixture else 'ODS_development'), 'Study kind/config mismatch')
    require(plan['metrics'] == {'prediction': 'raw top-1 L3', 'reject': False,
                                'universe': 'fixed label_maps.L3', 'zero_division': 0}, 'Metric protocol changed')
    require(plan['resume'] == 'unsupported; execution_started.json makes a plan single-use',
            'Resume semantics changed')
    require(plan['warnings'] == WARNINGS, 'Mandatory interpretation warnings changed')
    require(plan['minima'] == MINIMA and set(plan['splits']) == {f'split_{s}' for s in SPLIT_SEEDS},
            'Split/minima plan mismatch')
    require(plan['dependencies'] == dependency_matrix(cfg, plan['splits'], plan['sources']['corpus']['sha256']),
            'Ranking/refit dependency DAG differs from fixed split-local design')
    require(plan['implementation'] == implementation_files(), 'Implementation changed since plan freeze')
    if verify_inputs:
        source = verify_sources(cfg)
        require(source['refs'] == plan['sources'], 'Source provenance changed since plan freeze')
        require(source['corpus_fit_selection'] == plan['corpus_fit_selection'], 'Corpus fit selection changed')
        require(source['manifest']['label_maps'] == plan['label_maps'], 'Label maps changed')
        require(row_identity(source['pool']) == plan['development_pool'], 'Development pool identity changed')
        require(environment_versions() == plan['versions'], 'Runtime dependencies differ from frozen plan')
        p = source['protocol']
        expected_train = {**p['resolved_sections']['train'], 'epochs': p['fixed_epochs'],
                          'patience': p['fixed_epochs'], 'device': cfg.device, 'num_threads': cfg.num_threads}
        require(plan['training'] == {'model': p['resolved_sections']['model'], 'loss': p['resolved_sections']['loss'],
                                    'train': expected_train, 'restore_best': False, 'weights': 'final_epoch',
                                    'no_validation_tuning': True}, 'Training protocol/config changed')
        rebuilt, diagnostics = split_development_pool(source['pool'], plan['label_maps'])
        require(len(rebuilt) == 3, f'Frozen split no longer feasible: {diagnostics}')
        for sid, spec in plan['splits'].items():
            require(spec['seed'] == int(sid.split('_')[1]), 'Recorded split seed mismatch')
            parts = load_split(root, plan, sid)
            validate_partition(source['pool'], parts, plan['label_maps'])
            for part in parts:
                require(parts[part].equals(rebuilt[sid][part]), 'Saved split differs from deterministic seed')
            info = spec['supports']
            require(fingerprint(root / info['file']) == {k: info[k] for k in ('sha256', 'bytes')},
                    'Saved support table changed')
    return plan


def load_split(root: Path, plan: dict, split_id: str) -> dict:
    result = {}
    for part, spec in plan['splits'][split_id]['parts'].items():
        path = root / spec['file']
        require(path.resolve().is_relative_to(root.resolve()), 'Unsafe split path')
        require(fingerprint(path) == {k: spec[k] for k in ('sha256', 'bytes')}, 'Split fingerprint changed')
        data = pd.read_parquet(path)
        require(row_identity(data) == {k: spec[k] for k in ('rows', 'row_ids_sha256', 'groups_sha256')},
                'Split row/group identities changed')
        result[part] = data
    return result


def verify_frozen_runtime(root: Path, plan: dict) -> None:
    require(read_sealed(root / 'plan.json') == plan, 'Plan changed during execution')
    require(plan['implementation'] == implementation_files(), 'Implementation changed during execution')
    for info in plan['sources'].values():
        require(fingerprint(info['path']) == {k: info[k] for k in ('sha256', 'bytes')},
                'Frozen source changed during execution')


def run_provenance(plan: dict, run: dict, feature_id: str) -> dict:
    sid = run['split_id']
    return {'plan_sha256': digest(plan), 'run': run, 'feature_id': feature_id,
            'outer_train_sha256': plan['splits'][sid]['parts']['outer_train']['sha256'],
            'comparison_val_sha256': plan['splits'][sid]['parts']['comparison_val']['sha256'],
            'label_maps_sha256': digest(plan['label_maps']), 'training_sha256': digest(plan['training'])}


def dependency_provenance(plan: dict, sid: str, method: str) -> dict:
    deps = [d for d in plan['dependencies'] if d['split_id'] == sid and d['method'] == method]
    require(len(deps) == 1, 'Missing/ambiguous ranking dependency')
    return {'plan_sha256': digest(plan), 'dependency': deps[0],
            'label_maps_sha256': digest(plan['label_maps']),
            'full_features_sha256': plan['sources']['features']['sha256']}


def stage_cost_role(method: str, run_id: str | None, stage: str) -> str:
    """One allocation rule for the comparator reused as GxI checkpoint."""
    return ('attribution_checkpoint_and_full_comparator'
            if method == 'none' and stage in ('train', 'train_transform')
            and run_id and run_id.endswith('seed_42') else 'ordinary')


def _record_stage(root: Path, scope: str, method: str, sid: str, run_id: str | None,
                  name: str, measurement: _Measurement, status: str = 'success') -> None:
    if hasattr(measurement, '_rss'):
        record = measurement.stage_row(run_id or f'{sid}/{method}', name, status)
    else:
        record = {'run_id': run_id or f'{sid}/{method}', 'stage': name, 'status': status,
                  'seconds': None, 'peak_rss_bytes': None, 'peak_vram_bytes': None,
                  'measurement_note': measurement.note + '; measurement setup failed, resources unavailable'}
    cost_role = stage_cost_role(method, run_id, name)
    record.update(scope=scope, method=method, split_id=sid, charge_count=1,
                  plan_sha256=digest(read_sealed(root / 'plan.json')),
                  cost_role=cost_role,
                  checkpoint_incremental_training_seconds=0.0
                  if cost_role == 'attribution_checkpoint_and_full_comparator' else None)
    # One immutable event file per measurement, no repeated append/resume charging.
    import uuid
    write_new(root / 'stages' / f'{uuid.uuid4().hex}.json', sealed(record))


@contextmanager
def measured_stage(root: Path, scope: str, method: str, sid: str, run_id: str | None,
                   stage: str, device: torch.device, note: str):
    measurement = _Measurement(device, note)
    try:
        with measurement:
            yield measurement
    except BaseException:
        _record_stage(root, scope, method, sid, run_id, stage, measurement, 'failed')
        raise
    else:
        _record_stage(root, scope, method, sid, run_id, stage, measurement)


def _verify_checkpoint(root: Path, plan: dict, run: dict, feature_builder: Any,
                       feature_id: str, device: torch.device) -> Any:
    run_dir = root / 'runs' / run['run_id']
    receipt = read_sealed(run_dir / 'checkpoint_receipt.json')
    expected = run_provenance(plan, run, feature_id)
    require(receipt['provenance'] == expected, 'Checkpoint dependency provenance mismatch')
    verify_tree(run_dir / 'checkpoint', receipt['files'])
    require(read_json(run_dir / 'checkpoint' / 'model_spec.json') == {
        'n_features': feature_builder.n_features,
        'n_classes': [len(plan['label_maps'][l]) for l in LEVELS],
        'model': plan['training']['model']}, 'Checkpoint model specification differs from frozen architecture')
    art = load_artifact(str(run_dir / 'checkpoint'), device)
    require(art.maps.to_dict() == plan['label_maps'] and art.meta['s07'] == expected,
            'Checkpoint label map/run provenance mismatch')
    require(art.config == plan['training'], 'Checkpoint resolved training config mismatch')
    require(feature_contract_fingerprint(art.features) == feature_contract_fingerprint(feature_builder),
            'Checkpoint feature contract mismatch')
    require(art.model.n_features == feature_builder.n_features, 'Checkpoint input dimension mismatch')
    _require_finite_model(art.model)
    return art


def verify_selected(root: Path, plan: dict, sid: str, method: str, full: Any) -> Any:
    """Verify entire ranking/refit chain before any downstream transform/checkpoint use."""
    base = root / 'shared' / sid / method
    receipt = read_sealed(base / 'dependency_receipt.json')
    expected = dependency_provenance(plan, sid, method)
    require(receipt['provenance'] == expected, 'Split-local ranking provenance mismatch')
    expected_files = {'selection', 'tfidf', 'attribution_ratings'} if method == 'gradient_x_input' else {'selection', 'tfidf'}
    require(set(receipt['files']) == expected_files, 'Incomplete dependency artifact inventory')
    for name in expected_files:
        verify_tree(base / name, receipt['files'][name])
    result, meta = load_selection_artifact(str(base / 'selection'))
    require(meta['provenance'] == {**expected, 'checkpoint': receipt['checkpoint']},
            'Selection/checkpoint provenance mismatch')
    require(result.method == method and result.k_requested == sum(S07Config(**plan['config']).selected_counts),
            'Selection method/budget mismatch')
    from .s06_comparison import _verify_ratings_meta
    ratings_meta = _verify_ratings_meta(result.ratings_dir, method)
    require(ratings_meta == result.ratings_manifest, 'Saved ranking manifest mismatch')
    if method != 'global':
        validate_ratings_dir(result.ratings_dir, result.ratings_manifest, method=method,
                             widths={'char': full.builder.n_char_features, 'word': full.builder.n_word_features})
    if method == 'gradient_x_input':
        require(fingerprint(base / 'attribution.json') == receipt['attribution_report'], 'Attribution report changed')
        att_report = read_sealed(base / 'attribution.json')
        require(att_report['provenance'] == expected and att_report['checkpoint'] == receipt['checkpoint'],
                'Attribution report/checkpoint provenance mismatch')
        reference = next(r for r in plan['runs'] if r['run_id'] == expected['dependency']['reference_checkpoint_run'])
        ref_dir = root / 'runs' / reference['run_id']
        ref_receipt = read_sealed(ref_dir / 'checkpoint_receipt.json')
        require(receipt['checkpoint'] == {'run_id': reference['run_id'],
                                         'receipt_sha256': sha(ref_dir / 'checkpoint_receipt.json')},
                'GxI reference checkpoint changed')
        _verify_checkpoint(root, plan, reference, full.builder, plan['sources']['features']['sha256'],
                           torch.device('cpu'))
        require(ref_receipt['provenance']['run']['seed'] == 42, 'GxI reference seed is not 42')
    else:
        require(receipt['checkpoint'] is None, 'Chi2 unexpectedly depends on checkpoint')
    from .s06_comparison import _selection_from_saved_ratings
    replay_cfg = SelectionConfig(method=method, k=result.k_requested, char_share=.75, seed=42)
    if method == 'gradient_x_input':
        replay = select_features_gradient_x_input(result.ratings_dir, replay_cfg, full.builder.n_char_features,
                                                  full.builder.n_features, l3_labels=plan['label_maps']['L3'])
    else:
        replay = _selection_from_saved_ratings(method, result.ratings_dir, result.k_requested, .75,
                                               full.builder.n_char_features, full.builder.n_features,
                                               l3_labels=plan['label_maps']['L3'])
    for branch in ('char', 'word'):
        saved, expected_branch = getattr(result, branch), getattr(replay, branch)
        require(np.array_equal(saved.indices, expected_branch.indices)
                and np.allclose(saved.scores, expected_branch.scores, equal_nan=True),
                'Selected vocabulary does not reproduce from saved ranking')
    loaded = load_vectorizer_artifact(str(base / 'tfidf'))
    cfg = S07Config(**plan['config'])
    require((loaded.builder.n_char_features, loaded.builder.n_word_features) == cfg.selected_counts,
            'Selected branch sizes mismatch')
    require(loaded.meta['source'] == full.meta['source'] and loaded.meta['fit'] == full.meta['fit'],
            'Refit corpus snapshot/fit subset mismatch')
    for branch, indices, offset in [('char', result.char.indices, 0),
                                     ('word', result.word.indices, full.builder.n_char_features)]:
        selected = np.sort(np.asarray(indices, dtype=np.int64) - offset)
        old, new = getattr(full.builder, branch), getattr(loaded.builder, branch)
        require(len(selected) == cfg.selected_counts[0 if branch == 'char' else 1], 'Branch count mismatch')
        require(len(np.unique(selected)) == len(selected) and np.all(selected >= 0)
                and np.all(selected < len(old.vocabulary_)), 'Invalid full-space branch indices')
        names = old.get_feature_names_out()[selected].tolist()
        require(new.get_feature_names_out().tolist() == names, 'Stable feature identity/index mapping mismatch')
        require(np.allclose(new.idf_, old.idf_[selected], rtol=1e-6, atol=1e-6), 'IDF invariant failed')
        require(new.norm == 'l2', 'Per-branch L2 normalization required')
        require(loaded.meta['refit_checks'][f'{branch}_idf_max_abs_diff'] <= 1e-5,
                'Refit recorded IDF check failed')
    return loaded


def _require_finite_model(model: torch.nn.Module) -> None:
    require(all(torch.isfinite(value).all().item() for value in model.state_dict().values()),
            'Nonfinite model parameter/buffer')


def _training_record(provenance: dict, history: list[dict], best: dict,
                     trace: list[dict], train_cfg: TrainConfig) -> dict:
    """Serialize the LR controller's uncomputed subset diagnostic as JSON null.

    Only plateau computes val_loss_subset. Other schedulers emit a NaN sentinel;
    neither infinity nor a nonfinite computed loss is an unavailable diagnostic.
    Strict sealing still rejects every other nonfinite training value.
    """
    serialized_trace = []
    for entry in trace:
        entry = dict(entry)
        if (train_cfg.scheduler in ('none', 'cosine', 'onecycle')
                and np.isnan(entry['val_loss_subset'])):
            entry['val_loss_subset'] = None
        serialized_trace.append(entry)
    return sealed({'provenance': provenance, 'history': history,
                   'best_monitor_diagnostic_only': best, 'lr_trace': serialized_trace})


def _train_one(root: Path, plan: dict, run: dict, builder: Any, feature_id: str,
               parts: dict, device: torch.device) -> None:
    verify_frozen_runtime(root, plan)
    cfg = S07Config(**plan['config'])
    maps = LabelMaps.from_dict(plan['label_maps'])
    run_dir = root / 'runs' / run['run_id']
    run_dir.mkdir(parents=True, exist_ok=False)
    frozen_parts = load_split(root, plan, run['split_id'])
    require(all(parts[p].equals(frozen_parts[p]) for p in frozen_parts),
            'Training rows differ from frozen split')
    train, monitor = parts['outer_train'], parts['train_monitor']
    model_cfg = ModelConfig(**plan['training']['model'])
    loss_cfg = LossConfig(**plan['training']['loss'])
    train_cfg = TrainConfig(**plan['training']['train'])
    y = tuple(maps.encode(l, train[l]) for l in LEVELS)
    ym = tuple(maps.encode(l, monitor[l]) for l in LEVELS)
    nclasses = tuple(maps.n_classes(l) for l in LEVELS)
    weights = {h: compute_class_weights(y[i], nclasses[i], loss_cfg.class_weight_mode[h],
                                        loss_cfg.class_weight_alpha) for i, h in enumerate(('l1', 'l2', 'l3'))}
    with measured_stage(root, 'run', run['method'], run['split_id'], run['run_id'], 'train_transform',
                        device, 'Train/monitor transform; monitor belongs to outer_train, no tuning') as m:
        X = builder.transform(train.name.tolist(), chunk_rows=cfg.chunk_rows)
        Xm = builder.transform(monitor.name.tolist(), chunk_rows=cfg.chunk_rows)
    set_seed(run['seed'])
    model = SparseMLP(builder.n_features, nclasses, model_cfg)
    with measured_stage(root, 'run', run['method'], run['split_id'], run['run_id'], 'train',
                        device, 'Fresh fixed-epoch model; final weights, no validation tuning') as m:
        history, best, trace = train_model(model, X, y, Xm, ym, loss_cfg, train_cfg, weights,
                                           device, run['seed'], restore_best=False)
    require(len(history) == train_cfg.epochs and not best.get('restored_best'), 'Final epoch rule violated')
    provenance = run_provenance(plan, run, feature_id)
    training_record = _training_record(provenance, history, best, trace, train_cfg)
    _require_finite_model(model)
    save_artifact(str(run_dir / 'checkpoint'), model, model_cfg, builder, maps,
                  RejectRule(.5, False, False), {'note': 'S07 raw top-1, no reject'},
                  {'mode': loss_cfg.class_weight_mode, 'alpha': loss_cfg.class_weight_alpha,
                   'weights': {k: None if v is None else v.tolist() for k, v in weights.items()}},
                  plan['training'], {'s07': provenance, 'seed': run['seed']})
    write_new(run_dir / 'training.json', training_record)
    write_new(run_dir / 'checkpoint_receipt.json', sealed({'provenance': provenance,
                                                          'files': tree(run_dir / 'checkpoint')}))
    _verify_checkpoint(root, plan, run, builder, feature_id, device)


def _prepare_selected(root: Path, plan: dict, sid: str, method: str, full: Any,
                      parts: dict, device: torch.device) -> None:
    verify_frozen_runtime(root, plan)
    cfg = S07Config(**plan['config'])
    base = root / 'shared' / sid / method
    base.mkdir(parents=True, exist_ok=False)
    frozen_parts = load_split(root, plan, sid)
    # Integrity gate checks supplied partitions; ranking below consumes outer_train only.
    require(all(parts[p].equals(frozen_parts[p]) for p in frozen_parts),
            'Ranking rows differ from frozen split')
    maps = LabelMaps.from_dict(plan['label_maps'])
    with measured_stage(root, 'shared', method, sid, None, 'ranking_transform',
                        device, 'Split outer_train transform for supervised ranking'):
        X = full.builder.transform(parts['outer_train'].name.tolist(), chunk_rows=cfg.chunk_rows)
    verified_full = load_vectorizer_artifact(cfg.full_tfidf)
    require(feature_contract_fingerprint(full.builder) == feature_contract_fingerprint(verified_full.builder)
            and full.meta == verified_full.meta, 'Ranking full TF-IDF input contract mismatch')
    y = maps.encode('L3', parts['outer_train'].L3)
    k = sum(cfg.selected_counts)
    sel_cfg = SelectionConfig(method=method, k=k, char_share=.75, seed=42)
    checkpoint = None
    ranking_stage = 'attribution_ranking_selection' if method == 'gradient_x_input' else 'ranking_selection'
    with measured_stage(root, 'shared', method, sid, None, ranking_stage,
                        device, 'Split-local outer_train-only ranking/selection'):
        if method == 'gradient_x_input':
            ref = next(r for r in plan['runs'] if r['split_id'] == sid and r['method'] == 'none' and r['seed'] == 42)
            art = _verify_checkpoint(root, plan, ref, full.builder, plan['sources']['features']['sha256'], device)
            checkpoint = {'run_id': ref['run_id'], 'receipt_sha256': sha(root / 'runs' / ref['run_id'] /
                                                                       'checkpoint_receipt.json')}
            att = compute_margin_attributions(art.model, X, y, device, batch_size=cfg.attribution_batch_size)
            ratings_dir = base / 'attribution_ratings'
            _, report = build_gxi_ratings(att, full.builder.n_char_features, str(ratings_dir), l3_labels=maps.l3)
            write_new(base / 'attribution.json', sealed({'provenance': dependency_provenance(plan, sid, method),
                                                        'checkpoint': checkpoint, 'report': report}))
            result = select_features_gradient_x_input(str(ratings_dir), sel_cfg, full.builder.n_char_features,
                                                      full.builder.n_features, l3_labels=maps.l3)
        else:
            result = select_features(X, y, sel_cfg, full.builder.n_char_features, l3_labels=maps.l3)
        require((len(result.char.indices), len(result.word.indices)) == cfg.selected_counts,
                'Strict S07 branch budget unavailable')
        save_selection_artifact(str(base / 'selection'), result,
                                {**dependency_provenance(plan, sid, method), 'checkpoint': checkpoint},
                                ratings_copy=True)
        selection, _ = load_selection_artifact(str(base / 'selection'))
    with measured_stage(root, 'shared', method, sid, None, 'refit',
                        device, 'Fixed vocabulary refit on exact original corpus selection; branch normalization'):
        refit = refit_selected_vectorizer(cfg.corpus, full.builder, full.meta, selection,
                                          out_dir=str(base / 'tfidf'), row_norm_sample=1000)
        reloaded = load_vectorizer_artifact(str(base / 'tfidf'))
        require(feature_contract_fingerprint(refit.builder) == feature_contract_fingerprint(reloaded.builder),
                'Refit serialization changed feature contract')
    files = {name: tree(base / name) for name in (('selection', 'tfidf', 'attribution_ratings')
                                                 if method == 'gradient_x_input' else ('selection', 'tfidf'))}
    write_new(base / 'dependency_receipt.json', sealed({'provenance': dependency_provenance(plan, sid, method),
                                                       'checkpoint': checkpoint, 'files': files,
                                                       'attribution_report': fingerprint(base / 'attribution.json')
                                                       if method == 'gradient_x_input' else None,
                                                       'checks': refit.checks}))
    verify_selected(root, plan, sid, method, full)


def _evaluate_one(root: Path, plan: dict, run: dict, builder: Any, feature_id: str,
                  parts: dict, device: torch.device) -> None:
    # Revalidate the immutable plan/source/split fingerprints immediately before inference.
    verify_frozen_runtime(root, plan)
    parts = load_split(root, plan, run['split_id'])
    art = _verify_checkpoint(root, plan, run, builder, feature_id, device)
    val = parts['comparison_val']
    cfg = S07Config(**plan['config'])
    with measured_stage(root, 'run', run['method'], run['split_id'], run['run_id'], 'inference',
                        device, 'Active development comparison_val raw top-1; no reject'):
        X = art.features.transform(val.name.tolist(), chunk_rows=cfg.chunk_rows)
        predictions, _ = predict(art.model, X, device, plan['training']['train']['eval_batch_size'], top_k=1)
    # Preserve S06 raw top-1 semantics, including the bias path for empty rows.
    pred = np.asarray(predictions.l3_top1).copy()
    truth = art.maps.encode('L3', val.L3)
    f1, acc, pc = _l3_val_metrics(truth, pred, art.maps.l3)
    labels = np.asarray(art.maps.l3, dtype=object)
    frame = pd.DataFrame({'row_id': val.row_id.to_numpy(), 'run_id': run['run_id'],
                          'split_id': run['split_id'], 'true_L3': val.L3.to_numpy(),
                          'predicted_L3': [labels[x] if x >= 0 else None for x in pred],
                          'empty_features': np.diff(X.indptr) == 0})
    path = root / 'runs' / run['run_id'] / 'predictions_val.parquet'
    require(not path.exists(), 'Predictions already exist; unsafe resume rejected')
    frame.to_parquet(path, index=False)
    receipt = {'schema': 's07-run@1', 'status': 'success', 'reason': None,
               'provenance': run_provenance(plan, run, feature_id), 'predictions': fingerprint(path),
               'checkpoint_receipt_sha256': sha(path.parent / 'checkpoint_receipt.json'),
               'training_sha256': sha(path.parent / 'training.json'),
               'metrics': {'l3_macro_f1': f1, 'l3_accuracy': acc}, 'per_class': pc}
    write_new(path.parent / 'result.json', sealed(receipt))


def execute(output_root: str | Path) -> dict:
    """Single-use execution; no reused checkpoints/results, including after a crash.

    Full seed42 is trained split-locally before GxI. It is the full comparator
    too: its training cost is charged once, with GxI allocation disclosed.
    """
    root = Path(output_root)
    plan = load_plan(root)
    require(not (root / 'execution_started.json').exists(), 'Resume unsupported: use a fresh output root/plan')
    for name in ('runs', 'shared', 'stages'):
        require(not (root / name).exists(), f'Untrusted pre-existing {name}; fresh root required')
    cfg = S07Config(**plan['config'])
    device = resolve_device(cfg.device)
    if cfg.num_threads is not None:
        torch.set_num_threads(cfg.num_threads)
    write_new(root / 'execution_started.json', sealed({'plan_sha256': digest(plan), 'started_at': now(),
                                                      'device': str(device)}))
    full = load_vectorizer_artifact(cfg.full_tfidf)
    summary = {'success': [], 'failed': [], 'skipped': [], 'pending': []}
    try:
        for sid in plan['splits']:
            parts = load_split(root, plan, sid)
            for method in METHODS:
                if method == 'none':
                    builder, fid = full.builder, plan['sources']['features']['sha256']
                else:
                    try:
                        _prepare_selected(root, plan, sid, method, full, parts, device)
                    except Exception as exc:
                        for dependent in [r for r in plan['runs'] if r['split_id'] == sid and r['method'] == method]:
                            skipped_dir = root / 'runs' / dependent['run_id']
                            write_new(skipped_dir / 'result.json', sealed({
                                'schema': 's07-run@1', 'status': 'skipped',
                                'reason': f'Shared dependency failed: {type(exc).__name__}: {exc}',
                                'provenance': run_provenance(plan, dependent, 'unavailable')}))
                            summary['skipped'].append(dependent['run_id'])
                        raise
                    selected = verify_selected(root, plan, sid, method, full)
                    builder, fid = selected.builder, sha(root / 'shared' / sid / method / 'tfidf' / 'features.joblib')
                for run in [r for r in plan['runs'] if r['split_id'] == sid and r['method'] == method]:
                    try:
                        _train_one(root, plan, run, builder, fid, parts, device)
                        if method != 'none':
                            verify_selected(root, plan, sid, method, full)
                        _evaluate_one(root, plan, run, builder, fid, parts, device)
                        summary['success'].append(run['run_id'])
                    except Exception as exc:
                        result_path = root / 'runs' / run['run_id'] / 'result.json'
                        if not result_path.exists():
                            write_new(result_path, sealed({'schema': 's07-run@1', 'status': 'failed',
                                                           'reason': f'{type(exc).__name__}: {exc}',
                                                           'provenance': run_provenance(plan, run, fid)}))
                        summary['failed'].append(run['run_id'])
                        raise  # Integrity failures stop the workload; no silent continuation.
    except Exception as exc:
        summary['stopped_reason'] = f'{type(exc).__name__}: {exc}'
    completed = set(summary['success'] + summary['failed'] + summary['skipped'])
    summary['pending'] = [r['run_id'] for r in plan['runs'] if r['run_id'] not in completed]
    # Missing dependencies/stages leave planned runs pending, never success.
    write_new(root / 'execution_finished.json', sealed({'plan_sha256': digest(plan), 'finished_at': now(),
                                                       'summary': summary,
                                                       'results': {r['run_id']: fingerprint(root / 'runs' / r['run_id'] / 'result.json')
                                                                   for r in plan['runs'] if (root / 'runs' / r['run_id'] / 'result.json').exists()},
                                                       'stages': tree(root / 'stages')
                                                       if (root / 'stages').exists() else {}}))
    return summary
