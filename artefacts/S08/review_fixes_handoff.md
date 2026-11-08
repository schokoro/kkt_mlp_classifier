# S08 H1 implementation/fixes handoff (in progress)

Implementation worker only; tests are owned by the independent test worker. No production training/evaluation or approval record has been generated. Original S06 holdout remains CLOSED; actual delta_NI=null and primary unapproved. Original requirements, notebook and previous outputs preserved.

API: `s08.evaluate(output_root, approval=None, trusted_anchor=None, *, report_dir=None)`. Approval must be a separate JSON file path; trusted_anchor is its **raw file SHA256**, supplied independently by the orchestrator ONLY after actual user consent. Never auto-derive the trusted argument from payload. Integrity is not consent. Local ledger assumes trusted operator/storage (cannot resist deliberate deletion/rollback by account owner); no external issuer/service is required.

`kkt_clf.s08_evaluation.approval_bindings(root)` is a read-only identity template (not approval). It verifies the complete freeze, returns exact `protocol`, `plan`, `training_freeze` fingerprints, `sources`, recursive `runs`/`shared` identities, and `label_maps_sha256`. Separate decision has EXACT keys:

```
schema: s08-detached-decision@1
scope: S08_original_S06_sealed_test_once
primary_candidate: class_aware
delta_NI: explicit finite number 0<margin<1 (NO DEFAULT)
practical_rationale: nonempty string
owner: nonempty string
consent_evidence: nonempty string recording actual user decision
one_time_use: true
bindings: exact approval_bindings(root) result
```

No arbitrary `approved` flag; global and GxI cannot be primary. Protocol and unsigned pre-registration remain unapproved and unchanged; detached decision binds frozen identities. Synthetic fixture approvals may exercise the mechanics only, with synthetic provenance, never imply real consent or unlock production. Independent tester may generate synthetic sealed labels and a synthetic record **before** planning/freeze, repinning fixture sources as required.

`evaluation_claim.json` is created by `os.open(O_CREAT|O_EXCL|O_WRONLY,0400)` before any sealed label read/inference; never retry or resume, even if stage fails. Concurrent loser fails without label access. Fully verify all 9 checkpoints/features before claim. After claim, checkpoint bytes are individually matched to frozen hashes and saved in new protected snapshots; inference loads those copies, with before/after hash checks. Live freeze is revalidated after claim and before each sealed read/inference and after inference. Source parquet is parsed from a byte snapshot verified against the frozen source hash.

Fresh `report-evaluation-<uuid>` default (or fresh versioned explicit directory), never overwrite pre-unseal report. Raw anchored approval bytes saved as `approval_decision.json`; nine keyed predictions persisted and re-read with exact ordered rowID/truth/model checks before statistics. Bootstrap remains B5000/seed808/alpha.025, class-stratified shared row indices, primary only. Global descriptive. Evaluated and failed-consumed report schemas rebuild entirely from manifest+listed tables+decision; no source/model reads. A failed/partial stage remains permanently consumed. Abrupt interruption can leave claim without final report; this is consumed, never resumable.

CLI:
`.venv/bin/python scripts/run_s08.py evaluate --output-root <frozen-root> --approval-record <separate-decision.json> --trusted-approval-sha256 <independently-trusted-raw-file-sha256> [--report-dir <fresh-versioned-report>]`

No production command of this form has been run. Further verification and exact results appended after checks.

## Implemented reporting and integrity details

- H1: complete explicit authorized path exists before formal production freeze;
  caller trust and local-storage assumptions documented, no invented issuer.
- L2: immutable per-run prediction parts are saved before progressing to the next
  model, then combined without computing interim scores. Final stash is re-read;
  exact ordered rowIDs, truth, method/seed and model hashes must align before CI.
- L3: pending renderer status validation uses the literal allowed status set.
- Live guards also bind pre-registration, feasibility, execution receipt, stages,
  protocol/plan bytes, all sources and recursively all run/shared files. Inference
  loads separately verified checkpoint byte snapshots rather than mutable live
  checkpoints. Snapshot copies and generated partial output are preserved.
- Evaluated runs include `inference_status`/`inference_reason`, with each inference
  cost measured (CPU transform+predict, seconds/RSS/VRAM when available). Training
  and the two shared refits retain their original accounting. Gate verification,
  byte snapshotting and statistical/report overhead are not inference timings.
- All post-claim exceptions consume use. Failed reports carry null final quality,
  `evaluation_blocked`, `failed_consumed`, error, and no retry/resume. An abrupt
  process interruption or reporting I/O failure can leave only the durable claim
  and partial files; the claim still forbids another attempt. Pending reports are
  rejected after any claim to prevent falsely presenting consumed use as unopened.
- Renderer validates anchored raw decision bytes, table hashes, fixed nine-row
  matrix and statuses, primary pairing/delta/rule, bootstrap recipe and stage
  accounting. Evaluated/failed rebuild uses only listed tables+decision+manifest;
  protected model snapshots are not needed for rebuild.

## Verification performed so far

- `.venv/bin/python -m pytest -q tests/test_s08_integrity_independent.py tests/test_s08_report_independent.py tests/test_s08_smoke_independent.py tests/test_s08_statistics_independent.py > /tmp/s08-fixes-existing.log 2>&1`: **81 passed in41.31s** (before final additive evaluated-report validation).
- `PYTHONPATH=. .venv/bin/python /tmp/s08_manual_smoke.py > /tmp/s08-h1-manual-smoke-v3.log 2>&1`: manual synthetic mechanics only; nine tiny fresh models, generated labeled rows, anchored synthetic decision, immutable keyed predictions, evaluated tables, byte-identical rebuild, replay rejection. Output `/tmp/s08-h1-manual-synthetic-llpabzlb/`. Synthetic status `non_inferior` is fixture arithmetic, **not an actual study claim or margin approval**. Earlier smoke outputs preserved; initial invocation without PYTHONPATH failed import before any workload and was rerun correctly.
- `.venv/bin/python scripts/run_s08.py evaluate --help`: explicit record and independent trusted digest arguments present.
- `.venv/bin/python -m compileall -q kkt_clf/s08.py kkt_clf/s08_evaluation.py kkt_clf/s08_report.py scripts/run_s08.py`: passed.
- `git diff --check`: passed. New files are untracked, so individual no-index whitespace checks will also be recorded.

Independent expanded authorization tests and broad regressions are in progress;
exact final results will be appended. No review round was invoked by this worker.

Additional completed checks:

- `.venv/bin/python -m pytest -q tests/test_s08_authorization_independent.py > /tmp/s08-fixes-authorization.log 2>&1`: **44 passed in67.10s**. Independently authored generated-data authorization tests, including authenticated identity mismatches, positive evaluation, O_EXCL race, replay, post-claim mutation/failure, row stash alignment and isolated offline rebuild. This worker did not author/edit that file.
- `.venv/bin/python -m pytest -q -x --tb=short tests/test_s08_integrity_independent.py tests/test_s08_report_independent.py tests/test_s08_smoke_independent.py tests/test_s08_statistics_independent.py > /tmp/s08-fixes-existing-final.log 2>&1`: **81 passed in49.46s**, after final implementation edits.
- First broad run `/tmp/s08-fixes-fast.log`: **17 failed,686 passed,1 skipped,1 deselected in167.31s**. The authorization test file was being updated independently during that process: the collected guard asserted `columns is not None` for legitimate development-label reads (`s07.verify_sources`), whereas the current guard restricts only the generated holdout/byte streams. Its traceback shows the earlier assertion and the on-disk source now differs. No production/sealed access occurred; no implementation change was made to suppress these guard failures. The standalone current44 tests passed. Broad run is being repeated against the current test file; retain the first log as historical evidence.

Final broad fast regression against the current independent tests:

```text
.venv/bin/python -m pytest -q -m 'not slow' --tb=short > /tmp/s08-fixes-fast-final.log 2>&1
711 passed, 1 skipped, 1 deselected in187.46s
```

This includes the independently expanded authorization tests available at collection
time, beyond the initial44-test standalone check. No implementation change was
needed after the earlier broad guard mismatch. Full pytest is still running;
its actual outcome is appended below when complete.

## Final worker outcome

```text
.venv/bin/python -m pytest -q --tb=short > /tmp/s08-fixes-full.log 2>&1
1 failed, 711 passed, 1 skipped, 2 warnings in183.54s
```

The sole full-suite failure is `tests/test_notebooks.py::test_notebooks_run_all`:
Jupyter's `socket.socket()` raises `PermissionError: [Errno 1] Operation not
permitted` in this sandbox. This is the known notebook socket environment limit;
no approval escalation, notebook edit, skip or restriction bypass was performed.
The final fast run covers all collected non-slow tests successfully. This worker
does not claim a fresh fully passing notebook/full-suite execution; the earlier
orchestrator-provided660-pass full run predates these fixes.

`git diff --check` is clean. Individual
`git diff --no-index --check /dev/null <path>` checks produced no whitespace
diagnostics for `kkt_clf/s08.py`, `kkt_clf/s08_evaluation.py`,
`kkt_clf/s08_report.py`, `kkt_clf/s08_stats.py` and `scripts/run_s08.py`
(no-index returns1 for added-file differences). CLI top-level/evaluate help and
compilation passed. The generated synthetic v3 smoke is the last manual smoke;
no implementation edits followed it. Documentation alone was appended later.

H1/L2/L3 implementation is complete and ready for the orchestrator's remaining
Kimi review. No Kimi invocation or claim of independent re-review by this worker.
No commit/merge/push/delete, no production9, no full-size refit, no GPU benchmark,
no real authorization record, and no original S06 sealed label/inference/quality
access. Actual study remains `awaiting_user_approval`, `delta_NI=null`, primary
unapproved; conditional confirmatory D09 remains blocked by missing real consent.
Trusted caller input and trusted local ledger storage are explicit assumptions.
