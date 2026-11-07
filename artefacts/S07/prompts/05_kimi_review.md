You are the independent READ-ONLY reviewer for S07. This is review round 1 of at most 2. Do not modify any project files, create files, run training, run tests, or delegate. Review code only; do not claim execution. All response text English. The orchestrator uses Codex implementation and a separate Codex test author under revised AGENTS.md. Ignore obsolete S06 role assignments.

Review complete new S07 implementation and independent tests against the original authoritative requirements below and S03-S06 compatibility. The diff boundary is five NEW files: kkt_clf/s07.py, kkt_clf/s07_report.py, scripts/run_s07.py, tests/test_s07_integrity_independent.py, tests/test_s07_report_independent.py, all relative to integrated dev c7df0d4. Exact diff snapshot is /tmp/s07-review1.diff ; read it in full. You may read existing source/contracts for context. User AGENTS.md edits, source S07 docs, user notebook and ZIP are outside implementation scope. No S03-S06 implementation files changed. Docs may still be updated.

Assess actionable correctness and methodology defects with severity, precise file/line, concrete triggering scenario, requirement violated and recommended repair. Focus on fail-closed dependencies and semantic validation even for re-sealed untrusted inputs, no validation/test labels in supervised selection, fixed 36-run plan before inference, complete statuses, exact frozen S06 14-epoch protocol, split-local GxI seed42, nine shared refits and correct source selection/IDF/normalization, truthful null/failed/partial reporting, stable-name Jaccard, correct paired statistics without pseudoreplication, and report rebuild integrity from tables only. Examine all report costs (checkpoint/attribution/transform/shared stages charged once), rare-class support bins/confusion/error deltas, empty-feature handling, label universe and train-save-load. Distinguish critical integrity blockers from optional hardening. Current synthetic checks: 68 independent tests pass incl 36 tiny runs/nine refits. Full pytest at earlier checkpoint 500 passed/1 skipped; final run pending. Formal ODS workload has not begun. Historical S06 +.0139 is a hypothesis only. Return explicit verdict and unresolved limitations. Do not silently weaken acceptance gates.

ORIGINAL AUTHORITATIVE SPECIFICATION:

# S07 — Robustness of supervised feature selection across development splits

## Value

Determine whether the apparent benefit of reducing TF-IDF from 400k to 100k features, especially the +0.0139 L3 macro-F1 observed for Class-aware Chi² in S06, is stable when the composition of training and validation data changes. Assess the incremental value of Gradient × Input relative to its extra attribution-model cost. Produce an auditable decision on whether 100k is a promising **PoC research configuration**, not an industrial recommendation.

The deliverable is a split-aware, aggregated comparison with separate estimates of variation across *data splits* and *MLP training seeds*, plus feature-vocabulary stability and resource accounting. No new selection algorithm or MLP architecture is introduced.

## Confirmed background and interpretation

- S06 used 16,289 labeled ODS names across 293 L3 after applying `L3 support >= 12` to the original labeled dataset, with a frozen `outer_train`, `comparison_val`, and `sealed_test`.
- S06 used a full 300k-char + 100k-word TF-IDF fitted on the fixed 981,850-row unlabeled corpus, and reported mean macro-F1 0.7229 (full), 0.7222 (Global Chi² 100k), 0.7368 (Class-aware Chi² 100k), and 0.7296 (Gradient × Input 100k), based on three model seeds on one comparison-validation split.
- The S06 validation set has been inspected. S07 uses the **same development data** with new split assignments; therefore, even the new validation splits do not constitute a new independent external test set. The S06 sealed test remains closed.
- S06's old checkpoints/rankings/selected TF-IDF and its metrics are *historical references only*; S07 may reuse the verified full TF-IDF and frozen training hyperparameters but must not reuse supervised artifacts whose inputs depend on a different development split.

## Scope and frozen experimental design

### A. Data and split contract

1. Define the S07 development pool **only** as the union of the original S06 `outer_train` and `comparison_val` rows. Exclude every original `sealed_test` row and preserve its fingerprint, identity and sealed status. Do not query its labels for selection, training, inference, validation, diagnostics or report quality metrics; reading existing structural manifest hashes/counts to verify exclusion is permitted.
2. Keep the S06 class universe (293 L3), original `support >= 12` filtering and label maps. The threshold applies to the original S06 filtered dataset; **do not re-filter the development pool** and silently drop L3 classes after excluding the sealed test.
3. Create exactly three fresh, deterministic, group-aware development partitions (proposed split seeds: `101, 202, 303`; record the actual values before execution). For each partition, create disjoint `outer_train`/`comparison_val`, and a disjoint inner `train_fit`/`train_monitor` partition of its `outer_train`. Enforce, for every L3, at least 2 names in `comparison_val`, 2 in `train_monitor`, and 6 in `train_fit`; do not split duplicate-normalized-name groups across partitions. Existing S06 sealed-test rows cannot be reassigned.
4. Verify feasibility (including group constraints) **before** launching models. If the splitter cannot find a valid partition, report whether infeasibility is proven or merely not found by the current search; do not misrepresent a heuristic failure as mathematical impossibility. No silent relaxation of minima, changed class universe or new external data.
5. Freeze split artifacts, row IDs, class support tables, label maps, seeds, hashes, and the **complete plan of 36 model runs before validation scoring**. Each model run must have a unique `split_id × method × seed` key. The complete plan also fixes ranking/refit dependencies, common training protocol and provenance.

### B. Four configurations and training protocol

For **each of the three development splits**, compare exactly:

| Method | Feature count | Character / word count |
|---|---:|---:|
| Full TF-IDF (`none`) | 400,000 | 300,000 / 100,000 |
| Global Chi² (`global`) | 100,000 | 75,000 / 25,000 |
| Class-aware Chi² (`class_aware`) | 100,000 | 75,000 / 25,000 |
| Gradient × Input (`gradient_x_input`) | 100,000 | 75,000 / 25,000 |

- Use training seeds `42, 43, 44` for all four configurations in every split: **3 splits × 4 configurations × 3 seeds = 36 MLP training runs**. Refit/selection work is shared across model seeds within a split, not across different splits.
- Reuse S06's verified MLP architecture, optimizer and frozen final-training rules, including the already chosen number of epochs, **without retuning against S07 validation**. Read and verify the exact frozen protocol from S06 artifacts before recording it for S07; do not assume copied default values are correct. Use identical classification metric definitions and fixed L3 universe.
- Reuse the **same verified full S02 TF-IDF** and the same exact unlabeled corpus snapshot for every split. Recompute supervised rankings **from the current split's `outer_train` only**. For Gradient × Input, use the newly trained full-TF-IDF checkpoint of a predetermined training seed (default `42`) from that split, independently of its validation score. Keep the agreed L3 margin, positive-only aggregation and class-aware selection.
- Refit each of the three selected TF-IDF variants **once per split**, using fixed selected vocabularies on the same original unlabeled-corpus snapshot as full TF-IDF. Verify char/word vocabulary sizes, IDF, normalization, serializing/reloading, rankings and provenance. Selected downstream MLPs start from scratch; no reuse of S06 selected models.
- Compare raw top-1 L3 predictions with no rejection threshold. Both model training and feature selection must remain label-disjoint from the active split's comparison validation.

### C. Pre-flight gates, recovery and safety

- Do not start the 36-run workload until the data, complete machine-readable plan, frozen training configuration and source artifacts pass structural and fingerprint checks. Do not silently substitute artifacts, skip groups, or reassign test rows.
- Use a fresh configurable `outputs/S07/...` root; preserve all S01–S06 outputs and user files. Apply fail-closed validation to stored selected vocabularies, checkpoint dependencies, predictions and reporting inputs. If safe resume is unavailable, reject an attempted resume rather than blindly reusing results.
- Each planned run must be represented as `success`, `failed`, `skipped`, or `pending`, with a reason where applicable. Any incomplete run means its DoD is partial; never report a partial matrix as a complete experiment.
- Start with compact functional checks/smoke on synthetic fixtures and run the full workload only after those checks pass. No destructive cleanup of existing result directories.

## Questions answered by the aggregated report

1. Across three development splits, does Class-aware Chi² 100k retain a positive **paired** Δ macro-F1 against full TF-IDF? How many splits and seed pairs show gains/losses?
2. Does Gradient × Input 100k outperform simpler Chi² choices **enough to justify training the attribution checkpoint**, after accounting for shared attribution and ranking costs?
3. Are observed improvements concentrated in rare L3 categories, or distributed across support groups? Which categories and confusion pairs improve/degrade most?
4. How stable are the chosen 100k vocabularies across the three splits? Show pairwise Jaccard and intersection sizes separately for char and word per method. Do not compare raw indices without resolving features to the same full-vocabulary identity.
5. How much do model size, refit time, ranking/attribution time, model training and inference cost change? Shared costs are counted **once per split** in cost comparisons.
6. How sensitive are conclusions to the particular train/validation partition versus model initialization? Keep variation across splits **separate** from variation across seeds inside a split.

**Primary analysis:** for each selected method and each `(split_id, training_seed)`, compute `Δ macro-F1 = macro-F1(method, split, seed) − macro-F1(full, split, seed)` using unrounded metrics. Summarize per-split paired means and distributions across split means. Never treat the nine correlated observations as nine independent data samples, or report a 95% CI justified solely by them.

**Secondary analysis:** accuracy, per-L3 precision/recall/F1, per-class paired Δ, validation support bins `2–4`, `5–9`, `10–19`, `>=20` (or empty bins explicitly), most frequent `true_L3 → predicted_L3` mistakes, selection-vocabulary Jaccard across splits, timing/RSS/VRAM/model size. Count each L3 once per bin; quantify class count and example support. Clearly mark unsupported GPU measurements as unavailable.

## Required artifacts

Create an S07-specific output root with preserved split IDs and provenance. Exact internal layout is flexible, but the report must include:

```text
outputs/S07/<experiment_id>/
├── plan.json                # frozen split × method × seed matrix (36 rows)
├── splits/                  # 3 development splits, support checks, identities and hashes
├── shared/                  # references, split-scoped rankings, refit and attribution provenance
├── runs/                    # checkpoints, split-scoped results and verified val predictions
└── report/
    ├── report.html           # single self-contained aggregated report
    ├── manifest.json         # plan, input hashes, statuses, formulas, warnings and provenance
    ├── runs.csv              # all 36 runs, split, seed, method, metrics, paired Δ, status
    ├── per_class.parquet     # per-run/per-L3 metrics and baseline-paired changes
    ├── predictions_val.parquet
    ├── support_bins.csv      # support-bin deltas per split/seed/method
    ├── confusion_pairs.csv
    ├── feature_stability.csv # Jaccard/intersection by method, split pair and branch
    ├── stages.csv            # elapsed/RSS/VRAM and shared cost attribution
    └── paired_summary.csv    # per-split paired effects and across-split summaries
```

`report.html` must be rebuildable from saved report tables alone, without model loading, retraining, inference or access to `runs/` and `shared/`. Provide a reproducible command, a table-only rebuild test, and a manifest that distinguishes real experimental measurements from reused static references. Record failed/skipped/pending runs accurately.

## Out of scope

- New supervised feature-selection methods, additional budgets (50k/200k), model architecture changes or feature-engineering changes.
- Retuning hyperparameters, choosing the best split, adapting budgets/epochs using S07 validation, or changing the S06 label universe.
- Inference, metrics, error analysis or model selection involving the S06 `sealed_test`.
- Claims of industrial accuracy, inference throughput on 300m receipt lines or definitive superiority from correlated development resamples.
- Any destructive cleanup or migration of S01–S06 outputs; changing `artefacts/SPEC_v1.0.md`.
- Commit, merge, push, or deletion of a sprint branch without explicit user authorization.

## Definition of Done

- [ ] **D01 — Sprint isolation.** Branch `sprint/S07-stability-validation` created from current integrated `dev`, existing user changes preserved, current AGENTS roles confirmed.
- [ ] **D02 — Sealed-test exclusion.** S06 test identities are permanently excluded from the S07 development pool; structural provenance and duplicate-group separation validated; no test predictions or quality metrics.
- [ ] **D03 — Split integrity.** Three new reproducible group-aware development splits pass class minimums and disjointness checks, with fixed seed, row IDs, counts, fingerprints and identical 293-L3 label universe; any infeasible case is reported honestly.
- [ ] **D04 — Immutable plan.** A 36-run matrix and every shared artifact dependency, configuration and training protocol are fingerprinted and frozen before S07 validation inference; tampering/mismatch fails closed.
- [ ] **D05 — Full baseline.** Nine full TF-IDF models (3 splits × 3 model seeds) trained afresh; all checkpoints and per-split validation metrics recorded; frozen S06 hyperparameters and training rule reused without validation tuning.
- [ ] **D06 — Split-local ranking.** Global Chi², Class-aware Chi² and Gradient × Input rankings computed independently for each split on its own train; reference checkpoint is split-local; provenance, reload and train-only property tested.
- [ ] **D07 — Selected refit.** Nine fixed-vocabulary TF-IDF refits created on the same unlabeled corpus snapshot; each is 75k char + 25k word, with verified IDF/normalization/serialization and provenance.
- [ ] **D08 — Complete comparison.** All 36 planned model runs complete, or failures/skips are explicitly surfaced as partial; shared seeds, same split and training rules, clean downstream train/inference.
- [ ] **D09 — Correct paired statistics.** All 27 method-versus-full paired deltas checked; split effects and model-seed effects reported distinctly without invalid independence claims.
- [ ] **D10 — Error and rare-class analysis.** Per-class metrics, support-bin paired deltas and confusion pairs computed from verified predictions on active development validation only.
- [ ] **D11 — Feature stability.** For every selected method, pairwise overlaps of 100k vocabulary across three splits computed separately by char and word from stable feature identities; Jaccard and counts verified with tests.
- [ ] **D12 — Resource and report contract.** All ten report files created; HTML is self-contained and reconstructible solely from report tables; costs are recorded by stage, with no repeated charging of shared work.
- [ ] **D13 — Tests and independent review.** Implementation/fixes by delegated Codex; independent tests by separate Codex; at most two read-only Kimi reviews following English prompts; full pytest, targeted negative tests and `git diff --check` run. Any fixes after last review flagged as not independently reviewed.
- [ ] **D14 — Completion report.** `artefacts/S07/result.md`, `STATUS.md` and append-only `DECISION_LOG.md` updated with exact commands, actual run counts, detailed DoD statuses, findings, limitations and a conditional research recommendation. No commits or merges.

## Acceptance boundary

S07 is an ODS robustness study, not an independent industrial validation. Passing DoD proves experiment integrity and documents repeated-split observations; it does not establish transferable gains for a different corpus or taxonomy. If any critical provenance, split isolation or report-integrity gate fails, stop and document the blocked step rather than weakening it to obtain 36/36 runs.


ORIGINAL ORCHESTRATOR REQUIREMENTS:

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


Read the exact complete diff from /tmp/s07-review1.diff. It is a required input to this review. The earlier prompt draft exceeded the OS single-argument length and never started a review; this remains round 1.
