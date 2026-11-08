# S08 H1 independent regression handoff

Independent test author only. Read AGENTS, original value_dod/orchestrator prompt, implementation/testing handoffs, review1 and review_fixes_handoff. No implementation edits, production approval, original holdout access, production training/evaluation or review invocation. Actual study delta_NI=null; primary remains unapproved.

New tests: `tests/test_s08_authorization_independent.py`. A SEPARATE generated labeled holdout is written before planning, with separate generated manifest/protocol and fixture pins. Existing label-free sealed fixture is preserved and checked. Synthetic detached records are constructed independently from frozen on-disk identities; trusted test anchors are only synthetic test inputs.

Initial targeted run is ongoing: `/tmp/s08-review-regression-initial.log`. No final passing claim yet. Tests cover raw independent anchor, semantically mismatched authenticated payload, freeze mutation, one-shot claim concurrency/replay/failure, post-claim integrity, persisted row alignment, independent sklearn/per-class/accuracy/confusion/oracle bootstrap checks, isolated table-only report rebuild, and resealed fake evaluated statuses/metrics. Further defects/results will be appended here.

## First findings for implementation worker

Initial run: 24 passed/17 failed in56.91s. Fifteen failures were an overly broad **test guard** blocking authorized DEVELOPMENT label reads and are corrected in the test (guard now permits development and rejects generated holdout label/BytesIO reads). Two genuine report defects remain:

- `test_evaluated_reporting_rejects_missing_or_resealed_fake_success[fake_run_status]`: evaluated renderer accepts `runs.csv` row status changed from success to pending with recomputed table integrity; falsely preserves confirmatory complete.
- Same test `[fake_metric]`: evaluated renderer accepts arbitrary test_macro_f1=.987654321 in runs.csv after resealing; no consistency with persisted predictions/paired tables.

Reproduce `.venv/bin/python -m pytest -q tests/test_s08_authorization_independent.py -k 'evaluated_reporting' --tb=short`. Implementation worker should fix semantic evaluated table validation, not weaken these assertions. Independent tests will rerun once stable. Source files are still changing concurrently; initial run not final evidence.

## Corrected guard / first clean expanded run

`.venv/bin/python -m pytest -q tests/test_s08_authorization_independent.py --tb=short > /tmp/s08-review-regression-second.log 2>&1`: **44 passed in66.24s**. Includes both reported evaluated-table semantic regressions, now fixed by implementation worker. Independent test author made no implementation edits. Three later additions (read-time mutation, two simultaneous complete evaluations, fourth-inference interruption) are pending the next combined run.

## Combined S08 run

`.venv/bin/python -m pytest -q tests/test_s08*.py --tb=short > /tmp/s08-review-regression-all.log 2>&1`: **128 passed in120.96s**. Covers all prior81 S08 tests plus47 authorization tests, including read-time mutation, genuine simultaneous evaluation callers (exactly one9-model inference pass), and a fourth-inference interruption. Five final pending unauthorized-output cases, stricter partial-stage assertions and post-consumption pending-report rejection were added during this run and are not claimed covered by it. Final file collects52 new tests,133 S08 total. Full repository run with this frozen final test file is ongoing at `/tmp/s08-review-regression-full.log`.

Guard hardening: the forbidden holdout-label read now calls `pytest.fail` rather than raising an `AssertionError`; the evaluation API catches ordinary `Exception` to publish consumed failures, so the test prohibition must not itself be swallowed as an expected failed evaluation. This is a test-only strengthening, requiring the post-claim/read-time tests to rerun; no implementation change.

## Final independent verification

- `.venv/bin/python -m pytest -q tests/test_s08_authorization_independent.py -k 'pending_report_rejects or interrupted_fourth or generated_authorized' --tb=short > /tmp/s08-review-regression-final-targeted.log 2>&1`: **7 passed,45 deselected in9.83s**. Synthetic end-to-end includes nine aligned predictions, independent fixed-label seed F1/Delta/CI recomputation, nondefault .023 synthetic margin, strict inequality, descriptive Global, per-class/bin/confusion/accuracy, shared-refit/inference costs, immutable training/protocol outputs and rejection of a new pending report after consumption. Also checks final interrupted-stage statuses and five unauthorized pending report files.
- `.venv/bin/python -m pytest -q tests/test_s08_authorization_independent.py --tb=short > /tmp/s08-review-regression-final-authorization.log 2>&1`: **52 passed in83.26s**, with the hardened non-swallowable label-read guard. All final52 new tests passed; original81 S08 tests unchanged. Combined original81+new47 run passed128 as recorded above; full repository run includes all133 S08 tests.
- `.venv/bin/python -m pytest -q --tb=short > /tmp/s08-review-regression-full.log 2>&1`: **711 passed,1 skipped,1 failed in185.83s**. Sole failure `tests/test_notebooks.py::test_notebooks_run_all` is Jupyter socket creation blocked by sandbox (`PermissionError: [Errno 1] Operation not permitted`), with IPython temp-directory/local-interface warnings. Every S08 test passed. No socket escalation or notebook modification was attempted; orchestrator's earlier broader-environment success is separate evidence, not this worker's result. Non-notebook suite passed in this full run; a redundant standalone quick rerun was not launched by this worker.
- `.venv/bin/python -m pytest --collect-only -q tests/test_s08_authorization_independent.py`: **52 collected**.
- `git diff --check`: passed; `git diff --no-index --check /dev/null tests/test_s08_authorization_independent.py` and the same for this handoff emitted no whitespace diagnostics (normal exit1 for a new-file diff). Branch remains `sprint/S08-noninferiority`.

## Coverage and closure

All negative fixture reads remain label-free. The new generated labeled holdout lives in pytest temporary directories, separately named and pinned before planning. No production decision file, actual margin approval, production refit/train/inference/quality, original S06 sealed-label access, Kimi invocation, commit/merge/push or user-file deletion occurred. Independent author edited only the new test file and this handoff; implementation, original requirements, prior outputs and user notebook were not edited.

New52 tests cover missing/wrong/malformed independent anchors; authenticated wrong margin/schema/scope/primary/owner/evidence/one-time setting and exact frozen source/plan/protocol/training/model/map/vocabulary bindings; changed weights/maps/features before claim; no quality before freeze/consent; atomic O_EXCL helper race plus two full concurrent calls (only one nine-model inference pass); success replay and permanently consumed failures; post-claim model/source mutation and byte-read race; failed fourth inference with explicit3-success/1-failed/5-pending statuses; reordered/missing/duplicate persisted prediction keys; independent sklearn metric and shared-row bootstrap oracle; failed/pending honesty; unauthorized output preservation; evaluated table-only byte-identical rebuild under outside-read guards; and missing/resealed fake reports. Existing independent bootstrap tests additionally prove fixed293 scoring, shared draw indices across all seeds, sign, strict equality rejection, zero division, unsupported identities and incompatible groups.

Two genuine evaluated-renderer defects (resealed pending status and false F1) were reported, fixed by the code worker and verified by passing unchanged scientific regression assertions. No observed implementation regression remains open from this test pass. The one full-suite environment failure remains explicit. Future Kimi round2 and final production preparation/DoD accounting belong to orchestrator; this worker consumed no review rounds and does not claim independent re-review.

Actual study: original S06 holdout CLOSED, delta_NI=null, primary Class-aware unapproved; D09 remains conditional/blocked pending actual user consent. Synthetic passing NI mechanics are not a study result, feasibility proof, industrial generalization or production resource measurement.
