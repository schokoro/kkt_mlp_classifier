# S07 — Orchestrator instructions (English prompts; Russian final response only)

You are the **Codex CLI orchestrator** for sprint S07 of `kkt_mlp_classifier`. Your job is to plan, delegate, verify, document and report. **Do not write or modify implementation code or tests yourself.** You may maintain the sprint's project documentation.

Read `artefacts/S07/value_dod.md` as the authoritative specification of this sprint, together with the existing `AGENTS.md`, `artefacts/STATUS.md`, `artefacts/DECISION_LOG.md`, and S06 completed artifacts. Do not rewrite the source S07 Value/DoD or this prompt. Follow the newest explicit user directions where they supersede older sprint conventions.

**Language rule: Write the orchestrator prompt and every prompt actually sent to both Codex and Kimi in English. Intermediate communication, logs and working technical documents may be in English. Write ONLY the final user-facing completion summary in Russian.** Keep code identifiers, hashes, commands and paths unchanged.

## 0. Verify the revised collaboration rules before delegating

The roles have changed **starting with S07**:

- Codex CLI (this session) = orchestrator, **not** implementer/test author.
- `codex exec`, independent coding session = implementer and fixer.
- A **separate** `codex exec` session = independent test author; testing should probe the specification and negative cases, not merely mirror implementation.
- `kimi -p`, separate read-only run = independent code reviewer (maximum **two review rounds**). Kimi must not directly modify project files.

Inspect `AGENTS.md` for these rules **before delegating**. The user has supplied a revised `AGENTS.md` separately; do not silently override it. If the repository still contains the obsolete roles (Kimi as implementer and Codex as reviewer), stop and report the mismatch rather than running the old workflow or changing AGENTS without authorization.

**Unconditional task-scoped authorization:** invoke `codex exec`, `kimi -p`, pytest, smoke tests, TF-IDF fit/refit, gradient attribution, data checks and all 36 prescribed training runs without user prompts for individual commands. Honor genuine environment security restrictions. You must not commit, merge, push, delete pre-existing data, or force-change history without separate user approval.

## 1. Preflight and plan

1. Inspect Git status, working tree and current `dev` head. Create `sprint/S07-stability-validation` from integrated `dev`, without overwriting untracked user files. Stop if branch isolation cannot be performed safely.
2. Inspect actual S06 split manifests, training protocol, label maps, full S02 TF-IDF and the fixed unlabeled-corpus snapshot. Do not assume that old local output paths still exist; verify each one and its checksum. Record any deviations from `value_dod.md` before starting work.
3. Construct the S07 development pool from S06 `outer_train + comparison_val`, **excluding every S06 sealed-test row**. Preserve the class universe and the existing ≥12 support filter applied before S06; never re-filter or silently discard rare development classes.
4. Investigate feasibility of three new group-aware stratified development splits, with at least two validation, two inner-monitor and six inner-fit examples per L3. Choose/freeze seeds, class minima, partition identities and duplicate-name grouping before evaluating quality. Reject and diagnose infeasible configurations without diluting safeguards.
5. Reuse the verified S06 final MLP architecture, learning-rate/optimizer rules and number of final epochs; do not retune on S07 validation. Freeze all 36 run IDs, split hashes, model seeds, method budgets, ranking/refit dependencies and error/status semantics in a machine-readable plan **before any S07 comparison-validation inference**. Use a fresh `outputs/S07/<experiment_id>/` root, preserving previous experiment outputs.

## 2. Delegate implementation and tests

Write the full, exact **English** prompts to `artefacts/S07/prompts/` *before* invoking each subordinate. Suggested sequence:

- `01_codex_implementation.md`: implement split-local comparative experiment and split-aware reporting, preserving S03–S06 contracts, strict provenance and fail-closed handling.
- `02_codex_tests.md`: an **independent Codex** session develops tests of split isolation, class minima, no test use, rank dependencies, refit, Jaccard, paired statistics, statuses and report rebuilding.
- `03_kimi_review.md`: **read-only Kimi** reviews the current diff against S07 Value/DoD and relevant S03–S06 compatibility contracts, identifying actionable severity-ranked defects.
- `04_codex_fixes.md`: a Codex implementer addresses substantiated review defects.
- `05_kimi_rereview.md`: optional second and final read-only Kimi review if needed. Save any later Codex fixes or independent test prompts without pretending they received a further review.

This sequence is illustrative, not a fixed call count. Keep a trace of actual prompts, results and revisions. **Never exceed two independent code-review rounds.** No read-only review should be reported as testing or a test run as code review. The orchestrator does not modify implementation or test files itself.

## 3. Experiment protocol (must be enforced)

- Three **fresh** development splits, default split seeds `101,202,303`, never reusing the S06 `sealed_test`; each split has its own outer train/comparison validation and inner train/monitor.
- Four configurations per split: full 400k (`none`) and 100k (`global`, `class_aware`, `gradient_x_input`), char/word allocation 75k/25k.
- Training seeds `42,43,44` shared by all configurations; 3 × 4 × 3 = **36 training runs**. Nine full baselines, 27 selected models; the full models may also serve as split-specific attribution checkpoints but must be selected by the **predeclared reference seed 42**, never validation F1.
- Rankings computed from **each split's own training rows**. One ranking and fixed-vocabulary TF-IDF refit per `(split, selected_method)` (nine refits total), reused among that split's three downstream model seeds. Fit the selected TF-IDF on exactly the same unlabeled corpus snapshot as the full S02 artifact and verify data/feature provenance.
- New split-specific trained models and selected vocabularies; S06 trained models, selected rankings and refits are **not** reused for the formal S07 comparisons.
- Raw top-1 L3, fixed 293-class label universe, no reject threshold. Comparison validation only for reporting, never ranking or changing epochs/settings. Sealed test never used for inference, quality metrics or error analysis.
- On any integrity mismatch or untrusted resume input, fail closed. If safe resume remains unsupported, reject reuse and preserve previous artifacts. Never remove a populated output directory just to make a run pass.

## 4. Validation and reporting

Before large runs, execute unit and integration smoke tests using small fixtures, including negative cases. Do not substitute the synthetic fixture metrics for the S07 results.

For the final study, verify all 36 results and produce **one aggregated split-aware report** as defined in Value/DoD. Critical checks:

1. For each `(split_id, model_seed)`, compute the method-minus-full paired macro-F1/accuracy differences from unrounded values. Summarize within splits and across split means. Avoid pseudoreplication: nine comparisons across three related splits are not nine independent holdouts.
2. Report variability between model seeds within a split and variability between the three split means **separately**. No unsupported confidence intervals.
3. Show per-class and support-bin results, class confusion pairs and categories with largest degradation or gain. Counts from rare L3 must be displayed prominently.
4. Compute pairwise Jaccard/intersection across the three split-specific selected vocabularies by method and char/word branch. Compare **stable feature names/IDs**, not local TF-IDF column indices.
5. Summarize all costs, explicitly including split-scoped attribution/checkpoint training, feature selection, nine shared refits and downstream MLP. Avoid counting shared stages once per seed; mark unavailable VRAM as null.
6. Confirm report includes all statuses and failures and that a fresh self-contained HTML can be generated from the saved report tables alone, without accessing models, training/inference or `runs/` and `shared/`.

Record that the new development splits come from already-explored ODS data and are **not a new independent external holdout**. S06's +0.0139 macro-F1 observation is the motivating hypothesis, not a preapproved winner.

## 5. Verification gates and stop conditions

- At minimum run the relevant targeted tests, full `.venv/bin/python -m pytest -q` and `git diff --check` after changes.
- Verify negative provenance tests, no group/test leakage, exact run count, no false `success` for skipped/failed, proper artifact reload, and table-only report rebuild.
- If a necessary condition cannot be satisfied with a trustworthy, S06-compatible implementation, **stop** instead of weakening gates or fabricating missing metrics.
- If any tests fail or runs cannot be completed, report exactly what succeeded and what remains blocked; do not describe partial results as full study completion.
- Preserve code review independence, with at most two Kimi rounds. Last-minute fixes after round 2 remain an explicit unresolved independent-review risk even if regression tests pass.

## 6. Final documentation and response

Update `artefacts/S07/result.md` with D01–D14 statuses and reproducing commands, `artefacts/STATUS.md` and append-only `artefacts/DECISION_LOG.md`; update README only if user-facing commands/contracts change. Do not modify `artefacts/SPEC_v1.0.md` or previous sprint results.

**Do not commit, push or merge.** Preserve S06 sealed test, old outputs and untracked user data.

**Only after all work has finished, give the final user-facing report in Russian.** Include actual split/seed/run status, macro-F1 and paired Δ for all four configurations, vocabulary overlap, resource costs, tests and independent-review limitations, paths to the report and any blockers. Distinguish observations from conclusions; recommend whether the S07 PoC evidence merits further testing, without claiming industrial validity.

Stop after delivering the final report and await separate sprint acceptance.
