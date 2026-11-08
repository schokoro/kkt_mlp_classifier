# S08 final read-only Kimi review, round 2 of 2

Raw log `/tmp/s08-kimi-review2.log`; snapshot `/tmp/s08-review2.diff`, SHA256 `ed60002fdd9808e3c7d12fa817c591dacbab259270090ec72c62f89f9e35352c`. Final verdict PASS, no blockers. Optional hardening is not implemented after review. Reviewer line references sometimes address concatenated diff positions; use named functions in current source.

• # S08 Round-2 Independent Review — findings

  Review complete. I read the exact diff at `/tmp/s08-review2.diff` (SHA256 verified, 2637 lines) against integrated dev `dd7559a`, plus the round-1 review, both fix/regression handoffs, and the current on-disk sources for line-accurate citation. No edits, no tests run, no sealed data opened.

  ## Round-1 items — verified resolved

  - **H1 (required):** `kkt_clf/s08.py:591-598` now delegates to a complete authorized path in `kkt_clf/s08_evaluation.py`. Authentication (`_approval`, s08_evaluation.py:41-65) requires a separate caller-supplied SHA256 anchor, exact schema, explicit finite margin with no default, primary `class_aware`, `one_time_use`, nonempty rationale/owner/evidence, and exact `approval_bindings(root)` identity equality. The O_EXCL claim (s08_evaluation.py:200) is written only after `_approval` → `load_plan` → `verify_training_freeze`, i.e., after full freeze verification and **before** any sealed label read or inference. No external issuer is required; trusted caller/operator/storage is documented as an explicit boundary. Independent tests (44, `tests/test_s08_authorization_independent.py`) cover valid/mismatched/replayed/failed/simultaneous authority, post-claim mutation, and row alignment.
  - **L2:** `_aligned_predictions` (s08_evaluation.py:117-133) re-reads the persisted stash and proves per-run exact ordered `row_id`, truth, model-hash and in-universe predictions before any positional statistics; mutation tests (reordered/missing/duplicate) force `failed_consumed`.
  - **L3:** pending renderer now validates against the literal status set (`s08_report.py` `render_report`, on-disk ~line 1118).

  ## Changed authorization/evaluation/reporting — examined for new defects

  - **No label read before valid approval:** sealed access occurs only at s08_evaluation.py:212-221, after authentication, full freeze verification, and the atomic claim. Replay fails at the claim (outside the try) without reading anything; pre-claim failures leave the tree byte-identical (tested).
  - **Mutation windows:** live checkpoints are hash-matched at snapshot time, copied O_EXCL, re-verified before *and after* `load_artifact`; the sealed parquet is read as a whole byte snapshot with hash check after read; `_guard` revalidates plan/protocol/freeze/execution/stages/runs/shared and all source fingerprints before each stage. Tested including mutation between final guard and byte read.
  - **Statistics:** primary call uses frozen defaults B=5000, PCG64 seed 808, alpha=.025, linear percentile, shared class-stratified indices across all three seed pairs, F1 recomputed from row-level predictions; strict `lower_bound > -delta_NI`; secondary Global kept descriptive-only (role enforced in table and renderer). Independent oracle confirms arithmetic to 1e-15.
  - **Failure truth:** failed/partial runs keep null final quality, `evaluation_blocked`/`failed_consumed`, explicit `sealed_access_may_have_occurred: true`; interrupted-4th-inference test verifies honest partial statuses and permanent consumption.
  - **Report integrity:** evaluated/failed rebuild reads only manifest-listed tables + decision; success requires all five test files plus exact paired/delta/bootstrap-recipe consistency; failed case requires null delta/bound; fake complete/status/metric and resealed manifest forgeries rejected; `build_report` refuses pending reports after any claim (s08_report.py:69).
  - **No regressions** in the refactored `s08.py` (`evaluate` delegation, `load_plan` gate-string check, protocol `approval` contract key are internally consistent; fresh experiments only).

  ## New findings

  **LOW 1 — Redundant sealed-identity check (cannot independently fire).** `s08_evaluation.py:216-217` compares `row_identity(frame[['row_id','name']])` against `{k: info[k] for k in row_identity(...)}` on bytes whose sha256/bytes were already verified twice (`_approval` → `load_plan(verify_inputs=True)` → `verify_sources`, then `_guard`). Harmless belt-and-suspenders, but like round-1 L3 it can never add a failure mode. Optional: drop it or compare against the frozen part entry directly.

  **LOW 2 — Orphanable half-start in `execute`.** `s08.py:410-411` writes `execution_started.json` before a second `verify_sources(cfg)` that sits *outside* the try. A failure there propagates without `execution_finished.json`/statuses, and `execute` permanently refuses reuse of that root (fail-closed, no corruption; `build_report` still renders an honest all-pending report). Optional: move the marker write after source verification.

  **Observation (not a defect):** an evaluation that fails *after* `predictions_test.parquet` is written (e.g., the min-support gate) keeps real sealed predictions in the failed-consumed report (s08_evaluation.py:292, rendered under an explicit `evaluation_blocked` header). Honest and audit-friendly, but note the one-time use is then consumed **without any NI answer** — most plausibly via `paired_bootstrap`'s `min support >= 2` requirement (s08_stats.py:61) if any of the 293 classes has a single sealed row. Pre-approval label-derived support counts are forbidden, so this cannot be excluded in advance; recommend stating it explicitly in `result.md` residual risks.

  ## Verdict

  **PASS — no blocking findings.** H1 is resolved in line with the round-1 repair specification; L2/L3 confirmed fixed; the authorization/evaluation/reporting changes are fail-closed, honest about consumed/failed states, and faithful to the pre-registered statistical contract and the trusted-caller boundary. The two LOW items are optional hardening, not required for freeze.

  ## Residual risks (carried, not re-proven)

  - No actual stakeholder approval exists; `delta_NI=null`, primary candidate unapproved, no production `execute`/evaluation has run. Synthetic-authority tests exercise mechanics on generated fixtures only.
  - Conditional class-stratified bootstrap on a finite, class-imbalanced ODS holdout with ~2–4 support for most L3 classes does not guarantee nominal population coverage; a wide/inconclusive CI is a real possibility, and the consumption-without-answer scenario above is live.
  - Claim/ledger resistance assumes trusted local operator/storage; deliberate deletion/rollback by the account owner is out of boundary (documented).
  - ODS is unrepresentative of the industrial corpus; results can never support industrial non-inferiority or superiority claims. CPU-only; no RSS/VRAM on GPU.
  - Test status (128 passed, final 133/full runs pending) is as reported by the independent tester; I did not re-run suites per read-only scope.

To resume this session: kimi -r session_102fa541-f9ea-4f71-9673-e65b350c3598
