# S08 independent testing handoff

Independent test author only. Authored five `tests/test_s08*.py` files and this handoff; **no implementation code was edited**. Read AGENTS, original S08 Value/DoD and orchestrator prompt, S07 preflight/result, S06 result, actual S06 manifest/final training protocol, and the implementation handoff. Verified `dev` at `dd7559a` and current `sprint/S08-noninferiority`. Original requirements, SPEC, prior outputs and unrelated untracked notebook were not modified by this worker. No Kimi invocation, production S08 training/refit, holdout quality evaluation, commit, merge, push or deletion.

## Results

- **81 independent S08 tests collected.** The combined 78-test run passed; the later tie/empty-feature test passed; six late targeted checks passed, including two newly added original-inner-partition cases and inference-with-fit-forbidden.
- Full repository run, before the last two added partition cases: **657 passed,1 skipped,1 failed in96.96s**. The only failure is `tests/test_notebooks.py::test_notebooks_run_all`: Jupyter cannot create its socket in this sandbox (`PermissionError: [Errno 1] Operation not permitted`). No notebook test or environment restriction was bypassed. Log `/tmp/s08-independent-full-pytest.log`.
- Fast repository run including all81 S08 tests: **659 passed,1 skipped,1 deselected in115.16s**. Log `/tmp/s08-independent-fast-pytest.log`. All81 S08 tests passed in this final run.
- Initial independent negatives exposed two genuine defects: bootstrap accepted NaN row IDs and NaN group IDs as valid singleton identities. The implementation worker fixed them; tester did not implement fixes. Both regressions passed in the combined rerun.
- An earlier integrity run happened while the concurrent implementation lacked `scripts/run_s08.py`: 19 failures/6 passes, all19 failures from the missing CLI fingerprint dependency. After the real CLI appeared, integrity+smoke passed34. No stub, fake implementation, approval record or holdout-dependent output was created to make tests pass.

## Independent coverage

`tests/test_s08_fixtures.py` builds all sources independently from generated names. The20-row development pool has two classes with support10 each, so reapplying support>=12 would break the smoke. All five original partitions include count/group/hash evidence. The synthetic sealed file deliberately contains only `row_id/name`, with no labels. Source final epochs14 deliberately differ from historical selection epochs15. Full32/selected8 dimensions exercise the same paths with6 char+2 word selected; production defaults are separately asserted as300k+100k and75k+25k.

`tests/test_s08_integrity_independent.py` covers source corpus/features/protocol/manifest corruption; every original partition; row/group overlap, conflicting hierarchy and changed label maps even after repinning synthetic sources; protocol/plan mutation; unauthenticated and mismatched approval attempts before any forbidden label read/inference; unsafe execution reuse and byte preservation; fixed nine-key methods/seeds matrix; immutable production pins and budgets.

`tests/test_s08_statistics_independent.py` independently recomputes a class-stratified row bootstrap and seed-wise sklearn macro-F1 from generated row-level predictions. Drawn indices are reused across all three seed pairs and methods, with fixed class support. It verifies candidate-minus-full sign, fixed293 scoring, B>=5000, seed808, alpha.025, linear percentile, zero_division0, strict lower_bound>-margin, equality/inconclusive/null states, ties, missing/out-of-universe predictions, seed/row mismatches, insufficient class support, duplicate/null identities, and group-design incompatibility. The regeneration recipe is checked. These are pure synthetic arithmetic tests, not holdout evaluation or margin approval.

`tests/test_s08_smoke_independent.py` executes the real tiny nine-run flow with seeds42/43/44 and frozen14 epochs on the **full** generated development union. Spies independently check the supervised ranking matrix/labels, exact corpus/refit source, all20 fitting rows, final-epoch rule, and only two shared refits across six selected models. Checks include immutable full artifact bytes, selected branch counts/order/IDF, same corpus sampling metadata, per-branch L2 normalization and a measurable difference from slicing already normalized Full. All nine checkpoints load twice and predict on development names only; TF-IDF `fit`/`fit_transform` are prohibited during inference. Weight/map/vocab/training/stale-success/freeze tampering fails. Raw top1 ties and zero-feature bias predictions match the pinned runtime. Production holdout is not used.

`tests/test_s08_report_independent.py` checks nine pending statuses and honest nulls, unapproved margin/primary, no five holdout-dependent files, exploratory numerical feasibility, failed/incomplete/stale-success reporting, model evidence verification, and single charging of two shared refits. HTML is byte-identical after rebuilding twice from only `manifest.json`, `runs.csv`, `stages.csv`, `ni_decision.json` copied into an isolated directory. Both `Path.open` and built-in `open` are guarded against outside reads; model/source/experiment APIs are forbidden. CLI rebuild also matches byte-for-byte. Table hash and resealed semantic tampering are rejected.

## Exact verification commands

Executed from the repository root:

```text
.venv/bin/python -m pytest -q tests/test_s08_statistics_independent.py
.venv/bin/python -m pytest -q tests/test_s08_statistics_independent.py --tb=short
.venv/bin/python -m pytest -q tests/test_s08_integrity_independent.py --tb=short
.venv/bin/python -m pytest -q tests/test_s08_integrity_independent.py tests/test_s08_smoke_independent.py --tb=short > /tmp/s08-independent-study-tests.log 2>&1
.venv/bin/python -m pytest -q tests/test_s08_report_independent.py --tb=short > /tmp/s08-independent-report-tests.log 2>&1
.venv/bin/python -m pytest -q tests/test_s08*.py --tb=short > /tmp/s08-independent-all-tests.log 2>&1
.venv/bin/python -m pytest -q tests/test_s08_smoke_independent.py::test_raw_top1_ties_and_empty_feature_bias_path_follow_frozen_runtime --tb=short > /tmp/s08-independent-tie-test.log 2>&1
.venv/bin/python -m pytest -q --tb=short > /tmp/s08-independent-full-pytest.log 2>&1
.venv/bin/python -m pytest -q tests/test_s08_smoke_independent.py::test_nine_run_smoke_exact_dev_only_rank_refit_train_save_load_inference tests/test_s08_integrity_independent.py::test_changed_partition_fails_before_training --tb=short > /tmp/s08-independent-late-tests.log 2>&1
.venv/bin/python -m pytest -q -m 'not slow' --tb=short > /tmp/s08-independent-fast-pytest.log 2>&1
.venv/bin/python -m pytest --collect-only -q tests/test_s08_fixtures.py tests/test_s08_integrity_independent.py tests/test_s08_report_independent.py tests/test_s08_smoke_independent.py tests/test_s08_statistics_independent.py
git diff --check
```

Outcome history: initial statistics24 passed8.12s; expanded statistics27 passed/2 failed8.82s (NaN findings); study34 passed20.68s; report14 passed11.09s; combined78 passed40.62s; tie1 passed2.16s; late targeted6 passed5.37s. New test files were also checked individually using `git diff --no-index --check /dev/null <test-file>` with no whitespace diagnostics. `git diff --check` passed.

## Limits / remaining work

This is mechanics and security regression evidence, not an S08 NI result or production resource measurement. No full-size selected refit, nine production models, production GPU check, holdout score, confirmatory CI or user approval was performed by this worker. Existing repository regression tests retain their usual temporary ODS-derived fixtures; the new S08 smoke is entirely synthetic and never opens the original sealed parquet.

Current implementation deliberately **cannot execute even a legitimately approved evaluation**: no trusted external approval anchor/atomic consumption ledger is provisioned, and `evaluate` always rejects. Tests verify fail-closed behavior, not an authorized positive holdout path. Conditional D09 and post-unseal reports remain unimplemented/unverified; they cannot be claimed complete. Margin and proposed primary remain unapproved, delta_NI remains null. Production execution/review/DoD accounting remain orchestrator responsibilities. No independent Kimi review budget was consumed by this test worker.

## Final verification

Final `.venv/bin/python -m pytest -q -m 'not slow' --tb=short > /tmp/s08-independent-fast-pytest.log 2>&1` completed successfully: **659 passed,1 skipped,1 deselected in115.16s**. This covers all81 current S08 tests, including both NaN regressions, both late partition cases, and inference with TF-IDF fitting forbidden. The only outstanding repository-test issue observed is the full-suite notebook socket restriction described above; no S08 test remains failing. Final `git diff --check` passed. Test authorship and handoff are complete; no implementation fix or independent review was performed by this worker.
