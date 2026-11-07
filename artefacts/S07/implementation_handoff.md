# S07 implementation handoff (delegated author)

No production model training, comparison-validation inference, attribution or large refit was launched. This is an implementation handoff, not completion of the S07 study. The parent owns final DoD accounting, STATUS/DECISION_LOG/result updates, and the two-round read-only review budget.

## Files authored

- `kkt_clf/s07.py`: source verification, compact preflight, grouped split witnesses/exact bounded fallback, frozen 36-run plan and dependency DAG, single-use execution, split-local ranking/refit, checkpoint/provenance validation and immutable measurement events.
- `kkt_clf/s07_report.py`: verified prediction/metric aggregation, paired effects and separate variability, per-class/support-bin/confusion diagnostics, stable-name branch Jaccard, cost allocation, all ten report files, and table-only HTML rebuilding with semantic cross-checks.
- `scripts/run_s07.py`: production CLI for preflight, plan, execute, report and rebuild.
- This document. No tests, prior sprint artifacts, source S07 documents, AGENTS.md, notebooks or user archives were edited by this author. Independently authored `tests/test_s07*.py` appeared concurrently in the shared tree and were run without modification.

## Preflight evidence

The FIRST gate, before implementation, passed in `outputs/S07/preflight_20261107T135054_cba50c95/preflight.json`. Its diagnostic driver remains `/tmp/s07_preflight_gate.py`. The packaged implementation also passed `outputs/S07/preflight_implementation_20261107_final/preflight.json`.

Verified development pool: **14,652 rows, 293 L3, minimum class support 10, 14,652 distinct normalized-name groups**. Seeds 101/202/303 each have fit 11,766 / monitor 1,378 / comparison validation 1,508, and outer_train 13,144. Minimum supports are 6/2/2. Actual assignments/row-ID hashes differ across seeds. These valid assignments are constructive feasibility witnesses, not heuristic guesses. In general the bounded exact fallback records whether a failed search proves infeasibility or only exhausts its budget.

Only `row_id` and `name` were projected from sealed-test parquet to verify structural identity/group exclusion; no sealed labels, inference or quality metrics. Sealed SHA256: `1842de401867e21a42c1dd342716548e1c40ef1ea57cb6109845ca8e308015f4` (1,637 rows).

Verified authoritative SHA256:

| Input | SHA256 |
|---|---|
| S06 split manifest | `ff44250867e864af23c72da94c38ed4d5a2796af9a658b83787193432ee2cd5d` |
| Full S02 features | `7ff42885729d0bcfb2b6e9e0893b1be4ddf59562705e3ed048614183cd952052` |
| Exact corpus | `f9a057a00db10ed115dd5f47e035f3a3c56c710d9463f47d38812d4cceef3dbc` |
| Final S06 protocol | `caffae8c2609d9f9cff28c47cf368cedddf7c38e7c4a4c8801bb553802c5cb0d` |

Corpus fit selection: head up to 1,000,000 nonempty names, all 981,850 rows used, no vectorizer subsampling, duplicates preserved. Full TF-IDF reload verified 300,000 char + 100,000 word. Frozen final training uses **14 epochs**, hidden [256], dropout [.3], ReLU/no norm, AdamW .001, weight decay 0, batch 256, constant LR, final epoch weights (`restore_best=False`). Historical `resolved_sections.train.epochs=15` describes epoch selection and is overridden by the verified `fixed_epochs=14` final rule. The monitor is included in outer_train during final training and is a diagnostic only.

## Operational contract

`plan.json` includes exact split files/supports and row/label identities, all 36 keys, nine ranking/refit dependencies, predetermined split-local full seed42 GxI checkpoint, label maps, complete resolved final configuration, source fingerprints, corpus fit selection, runtime/library versions and implementation-file fingerprints. Changing implementation after planning requires a fresh plan/root. Legacy S03–S06 modules/APIs were not modified; their existing low-level feature, ranking, refit, sparse training and artifact contracts are reused.

Execution is single-use. `execution_started.json`, existing `runs/`, `shared/` or `stages/` reject reuse before training; even an interrupted attempt requires a fresh root. Integrity failures stop further work. Failed model results remain failed; dependent runs whose shared preparation fails become skipped with a reason; untouched planned runs remain pending. Existing outputs are never overwritten.

Rankings use only the active outer_train matrix and labels. Integrity gates verify supplied partitions against their frozen files; comparison-validation labels do not enter ranking algorithms or select hyperparameters/checkpoints. Selected vocabularies replay from saved rankings before use. Refit artifacts bind exact original corpus selection, full-space branch indices, feature identities, IDF invariants and per-branch L2 normalization. Checkpoints, predictions, results and stage ledgers are fingerprinted and cross-validated. Raw top-1 includes the existing S06 bias path for empty feature rows; empty-row flags are saved.

Report seed variability is population SD within each split; between-split variability is population SD across complete three-seed split means. Partial pairs never imply a complete split. Shared ranking/refit costs are charged once per split/method. Full seed42 checkpoint training and input transform are already counted once as comparator costs; the separate GxI standalone scenario allocates that checkpoint cost explicitly, while incremental comparison cost adds zero checkpoint training. Stage windows are not end-to-end elapsed time; unavailable VRAM stays null.

`S07Config(fixture=True, fixture_pins=..., fixture_full_counts=..., fixture_selected_counts=...)` enables tiny synthetic mechanics checks with all source hashes still mandatory. Production CLI exposes no fixture override, and production budgets/class universe/protocol cannot be reduced through fixture fields unless the output is explicitly marked synthetic. Minima 6/2/2 and the 36-key design also remain mandatory for fixtures.

## Commands and next steps

These documented workload commands are **not executed** by this author. Run only after the parent completes independent checks/review. Use unused root names; do not reuse a root from an interrupted execution, and regenerate the plan after code changes.

```bash
.venv/bin/python scripts/run_s07.py preflight --output-root outputs/S07/preflight_NEW_ID
.venv/bin/python scripts/run_s07.py plan --output-root outputs/S07/study_NEW_ID --device cpu --num-threads 2
.venv/bin/python scripts/run_s07.py execute --output-root outputs/S07/study_NEW_ID
.venv/bin/python scripts/run_s07.py report --output-root outputs/S07/study_NEW_ID
.venv/bin/python scripts/run_s07.py rebuild --report-dir outputs/S07/study_NEW_ID/report --output-html /tmp/s07_NEW_ID_rebuilt.html
```

For an isolated table-only rebuild, copy `manifest.json` and the eight saved CSV/parquet tables into a new directory, excluding `report.html`; pass that directory to `rebuild`. No split files, source data, `runs/`, `shared/` or models are required.

Actual planning-only diagnostic root: `outputs/S07/plan_handoff_20261107/`; report must show **0 success, 0 failed, 0 skipped, 36 pending**, `complete=false`. Earlier diagnostic roots remain preserved and may bind earlier code fingerprints; they are not formal study results.

Actual commands/checks executed include:

```text
PYTHONPATH=. .venv/bin/python /tmp/s07_preflight_gate.py
.venv/bin/python scripts/run_s07.py preflight --output-root outputs/S07/preflight_implementation_20261107_01
.venv/bin/python scripts/run_s07.py plan --output-root outputs/S07/plan_diagnostic_20261107_01
.venv/bin/python scripts/run_s07.py report --output-root outputs/S07/plan_diagnostic_20261107_01
.venv/bin/python scripts/run_s07.py preflight --output-root outputs/S07/preflight_implementation_20261107_final
.venv/bin/python scripts/run_s07.py plan --output-root outputs/S07/plan_diagnostic_20261107_final
.venv/bin/python scripts/run_s07.py report --output-root outputs/S07/plan_diagnostic_20261107_final
.venv/bin/python scripts/run_s07.py rebuild --report-dir /tmp/s07-report-tables-only-105co5hr --output-html /tmp/s07-report-tables-only-105co5hr/rebuilt.html
.venv/bin/python scripts/run_s07.py plan --output-root outputs/S07/plan_handoff_20261107
.venv/bin/python scripts/run_s07.py report --output-root outputs/S07/plan_handoff_20261107
.venv/bin/python -m compileall -q kkt_clf/s07.py kkt_clf/s07_report.py scripts/run_s07.py
.venv/bin/python -m pytest -q tests/test_s07_integrity_independent.py tests/test_s07_report_independent.py
.venv/bin/python -m pytest -q -m 'not slow'
.venv/bin/python -m pytest -q
git diff --check
```

Targeted tests grew concurrently during independent authorship; the last dedicated run before the final metadata guards passed **68 tests** (`/tmp/s07_independent_all_20261107_05.log`), including a synthetic 36-run/nine-refit smoke and isolated table-only rebuild. The fast suite passed **503, skipped 1, deselected 1** (`/tmp/s07_fast_final_20261107.log`). An initial full suite passed 435, skipped 1 and failed the notebook execution test because sandbox socket creation is denied (`PermissionError: [Errno 1] Operation not permitted`); see `/tmp/s07_existing_pytest_20261107.log`. The final full suite passed **503**, skipped **1**, and failed **1** (`tests/test_notebooks.py::test_notebooks_run_all`) for the same sandbox socket restriction; all S07 checks passed. Final log: `/tmp/s07_full_final_20261107.log`. The production-data pending-report rebuild succeeded in `/tmp/s07-report-tables-only-105co5hr/` using tables only; evidence `/tmp/s07_table_only_rebuild_20261107.json`. No tests were authored by this implementation session.

## Acceptance limits

Production runs: **0/36**. Nine full ODS models, nine selected full-corpus refits, production attribution, paired research outcomes, production resource measurements, GPU behavior and independent Kimi review remain pending. This implementation/synthetic verification does not complete D05–D12 of the study or justify a feature-budget recommendation. D13 remains partial until parent verification/review, with the sandbox notebook limitation reported explicitly. No commits, stash, reset, merge, push or destructive cleanup.
