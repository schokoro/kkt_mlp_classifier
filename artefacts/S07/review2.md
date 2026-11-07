# S07 Independent Review — Round 2 (Final)

  **Verdict: PASS — no critical or material blockers. All round-1 findings M1/M2/L1–L5 are correctly repaired and independently regression-tested; the M3 risk flag is resolved by the confirmed real-data gate. The implementation is fit to proceed to the production 36-run workload.**

  ## Verification of round-1 repairs (all confirmed in the final diff)

  **M1 — FIXED (kkt_clf/s07.py, `_evaluate_one`).** The function now does `parts = load_split(root, plan, run['split_id'])` and scores on that freshly verified `comparison_val` for both transform and inference; the caller-supplied `parts` is fully shadowed. `tests/test_s07_review_regressions.py::test_evaluation_uses_verified_frozen_texts_not_caller_substitutions` proves substituted caller names never reach inference and that saved predictions/metrics match frozen validation rows.

  **M2 — FIXED (kkt_clf/s07_report.py:994-1003).** Expected feature IDs are now derived only from verified dependencies (features pin for `none`, live SHA of the verified refit for selected methods with receipts) or the fixed `'unavailable'` sentinel; a result's own provenance is never trusted as the expectation. Crucially, `status in ('skipped','pending')` is required before the sentinel path, so `failed`/`success` selected runs cannot bypass dependency verification — covered by three parametrized negative tests including forged and self-supplied feature IDs.

  **L1 — FIXED.** Renamed to `gxi_total_measured_seconds` with an explicit `gxi_standalone_scenario_seconds` column; the manifest formula now correctly states full seed-42 training *and input transform* are charged once to the full comparator, standalone adds that cost, incremental adds 0. The cost-allocation test (101/9/110) verifies the arithmetic and that baseline/other-seed costs are not charged to GxI.

  **L2 — FIXED (by documentation).** `effect_counts` formula plus HTML prose now explicitly state within-split gains/losses/ties count seed pairs, across-split counts complete split means, and `split_count` is 0/1 within a split. Regression test asserts the documentation and the counts (6/3/0 vs 2/1/0).

  **L3 — FIXED.** Extremes view filters `pc.delta_f1.notna()` before sorting; the presentation oracle confirms no NaN rows and correct empty-set behavior when everything is unpaired.

  **L4 — FIXED.** Single `stage_cost_role()` in `kkt_clf/s07.py:507` is the only rule; `_record_stage` uses it at write time and `validate_report_tables` re-derives and requires equality per row (plus `checkpoint_incremental_training_seconds` consistency), so stage renames can no longer silently drift.

  **L5 — ADDRESSED by documentation.** Module docstring and argparse help/ description for `report` now state explicitly that exit 0 means construction succeeded, not study completion, and direct automation to `complete=true` / `statuses.success=36` in JSON/manifest. Acceptable per the round-1 "document or fold" options.

  **M3 — RESOLVED.** I independently re-ran the structural check: 14,652 pool rows = 14,652 unique normalized groups, zero multi-L3 groups, zero sealed identity overlap, 293 L3, `min_count` 12. The strict single-L3-group `require` cannot trigger on the real pool.

  ## New guards added since round 1 — all sound

  `load_plan` now additionally pins `study_kind`, the metric protocol (`raw top-1, no reject, fixed universe, zero_division=0`), resume semantics, the full warnings list, development-pool identity, and per-split recorded seeds. These close residual tamper/repin gaps without weakening any legitimate path.

  ## Focus-area reassessment (no new findings)

  - **Fail-closed chain**: source pins → feasibility → sealed immutable plan (36 rows + 9-entry dependency DAG re-derived and compared) → per-train and per-inference `verify_frozen_runtime` → checkpoint/selection/receipt tree verification → prediction checksum + row/label identity re-derived from frozen split digests (`row_l3_sha256`) → execution-ledger consistency → table/manifest fingerprint cross-checks. Statuses are truthful: non-success rows carry reasons and no metrics (enforced in `validate_report_tables`), partial reports carry an explicit PARTIAL warning, and `complete` is cross-checked against the status table.
  - **Frozen 14-epoch protocol, no validation tuning**: `patience == epochs == fixed_epochs`, `restore_best=False`, history-length and no-restore assertions at train time and again at report time; S06 protocol pins enforced structurally (scheduler `none`, hidden [256], adamw/lr 1e-3/batch 256 in production profile).
  - **Split-local ranking / 9 refits**: GxI reference is predeclared `none/seed_42` with receipt-hash chaining; refit verified on the identical corpus snapshot (source/fit metadata equality), stable feature-name identity, IDF invariant vs the full vectorizer (rtol 1e-6), per-branch L2 renormalization, recorded refit checks, replay-reproduction of selected vocabularies.
  - **Statistics**: unrounded `(split, seed)`-paired deltas; across-split rows require all three seeds (`split_count` explicit); within-split seed SD and between-split mean SD reported separately; no independence-based CI anywhere.
  - **Costs**: one immutable event per measurement, `charge_count==1`, duplicate-event and cost-role validation, shared ranking/refit charged once per split, VRAM honestly null on CPU.

  ## Residual notes (non-blocking, optional hardening)

  1. `_evaluate_one` still accepts a `parts` parameter it no longer uses (kkt_clf/s07.py:745). Cosmetic; shadowing is deliberate and tested.
  2. `stage_cost_role` matches `run_id.endswith('seed_42')`; run IDs are frozen by `_run_matrix`, so this is not exploitable, but an exact-suffix/`split__none__seed_42` match would be marginally stricter.
  3. The GxI-vs-Chi² comparison table in the HTML averages over however many paired seeds exist; `paired_seeds` makes partial counts explicit, so this is honest, but effects are unweighted across splits if read carelessly.

  None of these affect integrity, DoD gates, or the acceptance boundary. Review budget (2 rounds) is exhausted; no further independent review is available, and no fixes after this review have occurred.

To resume this session: kimi -r session_81aa6325-b23c-4410-8b14-2721764ba2ec
