You are the independent READ-ONLY reviewer for S07. This is review round 1 of at most 2. Do not modify any project files, create files, run training, run tests, or delegate. Review code only; do not claim execution. All response text English. The orchestrator uses Codex implementation and a separate Codex test author under revised AGENTS.md. Ignore obsolete S06 role assignments.

Review complete new S07 implementation and independent tests against the original authoritative requirements below and S03-S06 compatibility. The diff boundary is five NEW files: kkt_clf/s07.py, kkt_clf/s07_report.py, scripts/run_s07.py, tests/test_s07_integrity_independent.py, tests/test_s07_report_independent.py, all relative to integrated dev c7df0d4. Exact diff snapshot is /tmp/s07-review1.diff and is included below. You may read existing source/contracts for context. User AGENTS.md edits, source S07 docs, user notebook and ZIP are outside implementation scope. No S03-S06 implementation files changed. Docs may still be updated.

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


EXACT REVIEW DIFF:

diff --git a/kkt_clf/s07.py b/kkt_clf/s07.py
new file mode 100644
index 0000000..d94e31c
--- /dev/null
+++ b/kkt_clf/s07.py
@@ -0,0 +1,821 @@
+"""S07 repeated-development-split study. Production gates are deliberately strict.
+
+No S06 supervised artifact is consumed. Planning never fits or scores a model.
+Execution is single-use: after an attempt (even a crash), use a fresh plan/root.
+Fixture mode is explicitly marked synthetic and cannot yield a production study.
+"""
+from __future__ import annotations
+
+import dataclasses
+from contextlib import contextmanager
+import datetime as dt
+import hashlib
+import json
+from pathlib import Path
+from typing import Any
+
+import numpy as np
+import pandas as pd
+import torch
+
+from .artifact import environment_versions, load_artifact, save_artifact
+from .attribution import (build_gxi_ratings, compute_margin_attributions,
+                          feature_contract_fingerprint, select_features_gradient_x_input)
+from .config import LossConfig, ModelConfig, TrainConfig
+from .experiment import _Measurement, _l3_val_metrics
+from .features import preprocess, strip_whitespace
+from .labels import LEVELS, LabelMaps
+from .model import SparseMLP
+from .reject import RejectRule
+from .s06_split import assign_groups, search_feasible_assignment
+from .selection import (SelectionConfig, load_selection_artifact, save_selection_artifact,
+                        select_features, validate_ratings_dir)
+from .train import compute_class_weights, predict, resolve_device, set_seed, train_model
+from .vectorizer import load_vectorizer_artifact, refit_selected_vectorizer
+
+METHODS = ('none', 'global', 'class_aware', 'gradient_x_input')
+SPLIT_SEEDS = (101, 202, 303)
+MODEL_SEEDS = (42, 43, 44)
+MINIMA = {'train_fit': 6, 'train_monitor': 2, 'comparison_val': 2}
+SOURCE_PINS = {
+    'manifest': 'ff44250867e864af23c72da94c38ed4d5a2796af9a658b83787193432ee2cd5d',
+    'protocol': 'caffae8c2609d9f9cff28c47cf368cedddf7c38e7c4a4c8801bb553802c5cb0d',
+    'corpus': 'f9a057a00db10ed115dd5f47e035f3a3c56c710d9463f47d38812d4cceef3dbc',
+    'features': '7ff42885729d0bcfb2b6e9e0893b1be4ddf59562705e3ed048614183cd952052',
+}
+WARNINGS = [
+    'ODS mechanics and robustness study only; no industrial validation.',
+    'Development data was previously inspected in S06; these are not independent external holdouts.',
+    'Related split resamples and seed pairs are correlated; no independence-based confidence interval.',
+    'Full TF-IDF fit is partially transductive; no validation/test labels used in feature fit.',
+    'Rare L3 estimates (especially 2 validation examples) are unstable.',
+    'Unavailable VRAM measurements are null; stage RSS peaks are not additive.',
+]
+
+
+class S07Error(RuntimeError):
+    """A failed integrity gate; never silently substitute or relax inputs."""
+
+
+def require(ok: Any, message: str) -> None:
+    if not ok:
+        raise S07Error(message)
+
+
+def now() -> str:
+    return dt.datetime.now(dt.UTC).isoformat()
+
+
+def sha(path: str | Path) -> str:
+    h = hashlib.sha256()
+    with Path(path).open('rb') as f:
+        for block in iter(lambda: f.read(1 << 20), b''):
+            h.update(block)
+    return h.hexdigest()
+
+
+def digest(value: Any) -> str:
+    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
+                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()
+
+
+def read_json(path: str | Path) -> dict:
+    with Path(path).open(encoding='utf-8') as f:
+        value = json.load(f)
+    require(isinstance(value, dict), f'JSON object required: {path}')
+    return value
+
+
+def write_new(path: str | Path, value: dict) -> None:
+    path = Path(path)
+    path.parent.mkdir(parents=True, exist_ok=True)
+    with path.open('x', encoding='utf-8') as f:
+        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
+        f.write('\n')
+
+
+def sealed(value: dict) -> dict:
+    require('integrity_sha256' not in value, 'Already sealed')
+    return {**value, 'integrity_sha256': digest(value)}
+
+
+def read_sealed(path: str | Path) -> dict:
+    value = read_json(path)
+    expected = value.pop('integrity_sha256', None)
+    require(expected == digest(value), f'Invalid/missing integrity seal: {path}')
+    return value
+
+
+def fingerprint(path: str | Path) -> dict:
+    p = Path(path)
+    require(p.is_file() and not p.is_symlink(), f'Missing/unsafe file: {p}')
+    return {'sha256': sha(p), 'bytes': p.stat().st_size}
+
+
+def tree(path: str | Path) -> dict:
+    path = Path(path)
+    require(path.is_dir() and not path.is_symlink(), f'Missing/unsafe directory: {path}')
+    return {str(p.relative_to(path)): fingerprint(p) for p in sorted(path.rglob('*')) if p.is_file()}
+
+
+def verify_tree(path: str | Path, expected: dict) -> None:
+    require(bool(expected) and tree(path) == expected, f'Artifact tree fingerprint mismatch: {path}')
+
+
+def row_identity(df: pd.DataFrame) -> dict:
+    return {'rows': len(df), 'row_ids_sha256': digest(sorted(df.row_id.astype(int).tolist())),
+            'groups_sha256': digest(sorted(group_keys(df)))}
+
+
+def group_keys(df: pd.DataFrame) -> list[str]:
+    return [strip_whitespace(t) for t in preprocess(df['name'].tolist())]
+
+
+@dataclasses.dataclass(frozen=True)
+class S07Config:
+    manifest: str = 'outputs/S06/splits/manifest.json'
+    protocol: str = 'outputs/S06_final_cpu_20261107/shared/training_protocol.json'
+    full_tfidf: str = 'outputs/S02/tfidf_1m'
+    corpus: str = 'data/corpus.pqt'
+    device: str = 'cpu'
+    num_threads: int | None = None
+    chunk_rows: int = 50000
+    attribution_batch_size: int = 256
+    # Explicit fixture profile only: all source hashes must still be pinned.
+    fixture: bool = False
+    fixture_pins: dict | None = None
+    fixture_full_counts: tuple[int, int] = (300, 100)
+    fixture_selected_counts: tuple[int, int] = (75, 25)
+
+    def validate(self) -> None:
+        require(self.chunk_rows > 0 and self.attribution_batch_size > 0, 'Invalid batch/chunk size')
+        require(self.device in ('cpu', 'cuda', 'auto'), 'Invalid device')
+        require(self.num_threads is None or self.num_threads > 0, 'Invalid thread count')
+        if self.fixture:
+            require(isinstance(self.fixture_pins, dict) and set(self.fixture_pins) == set(SOURCE_PINS),
+                    'Synthetic fixture requires all four explicit source pins')
+            c, w = self.fixture_full_counts
+            sc, sw = self.fixture_selected_counts
+            require(0 < sc <= c and 0 < sw <= w and sc * 3 == sw * 9,
+                    'Fixture budgets must keep the 75/25 allocation')
+        else:
+            require(self.fixture_pins is None, 'Production source pins cannot be overridden')
+
+    @property
+    def full_counts(self) -> tuple[int, int]:
+        return tuple(self.fixture_full_counts) if self.fixture else (300000, 100000)
+
+    @property
+    def selected_counts(self) -> tuple[int, int]:
+        return tuple(self.fixture_selected_counts) if self.fixture else (75000, 25000)
+
+
+def split_development_pool(pool: pd.DataFrame, label_maps: dict,
+                           seeds: tuple[int, ...] = SPLIT_SEEDS) -> tuple[dict, list]:
+    """Constructive witness first; exact bounded fallback distinguishes proof/search failure.
+
+    Three leaf parts are assigned jointly, so monitor/fit minima cannot invalidate
+    a previously greedy outer split. Fractions are targets; minima take priority.
+    """
+    require(tuple(seeds) == SPLIT_SEEDS, 'S07 fixes exactly seeds 101/202/303')
+    require(pool.row_id.is_unique, 'Duplicate development row IDs')
+    require(LabelMaps.from_df(pool).to_dict() == label_maps, 'Development label-map mismatch')
+    work = pool.copy()
+    work['_group'] = group_keys(work)
+    require(work.groupby('_group').L3.nunique().max() == 1, 'Group spans conflicting L3 labels')
+    results, diagnostics = {}, []
+    for seed in seeds:
+        rng = np.random.default_rng(seed)
+        leaves = {p: [] for p in MINIMA}
+        failures = []
+        for label, data in work.groupby('L3', sort=True):
+            groups = [g for _, g in data.groupby('_group', sort=True)]
+            sizes = np.asarray([len(g) for g in groups], dtype=np.int64)
+            n = len(data)
+            val = max(2, int(round(n * .1)))
+            mon = max(2, int(round((n - val) * .1)))
+            quota = np.array([n - val - mon, mon, val])
+            minima = np.array(list(MINIMA.values()))
+            if quota[0] < 6:
+                quota = minima.copy()
+                quota[0] += n - sum(minima)
+            assignment, rows = assign_groups(sizes, quota, minima, rng)
+            if np.any(rows < minima):
+                assignment, proven, nodes = search_feasible_assignment(sizes, minima)
+                if assignment is None:
+                    failures.append({'L3': label, 'group_sizes': sizes.tolist(),
+                                     'proven_infeasible': bool(proven), 'search_nodes': nodes})
+                    continue
+            for group, part in zip(groups, assignment):
+                leaves[list(MINIMA)[int(part)]].extend(group.row_id.astype(int).tolist())
+        diagnostics.append({'seed': seed, 'feasible': not failures, 'failures': failures})
+        if not failures:
+            parts = {p: pool[pool.row_id.isin(ids)].sort_values('row_id').reset_index(drop=True)
+                     for p, ids in leaves.items()}
+            parts['outer_train'] = pd.concat([parts['train_fit'], parts['train_monitor']]).sort_values(
+                'row_id').reset_index(drop=True)
+            validate_partition(pool, parts, label_maps)
+            results[f'split_{seed}'] = parts
+    return results, diagnostics
+
+
+def validate_partition(pool: pd.DataFrame, parts: dict, label_maps: dict) -> None:
+    require(set(parts) == {*MINIMA, 'outer_train'}, 'Incorrect partition set')
+    row_sets, group_sets = [], []
+    for part, minimum in MINIMA.items():
+        data = parts[part]
+        require(data.row_id.is_unique, f'{part}: duplicate IDs')
+        counts = data.L3.value_counts().reindex(label_maps['L3'], fill_value=0)
+        require((counts >= minimum).all(), f'{part}: class minimum {minimum} failed')
+        require(set(data.L3) == set(label_maps['L3']), f'{part}: class universe changed')
+        row_sets.append(set(data.row_id))
+        group_sets.append(set(group_keys(data)))
+    for i in range(3):
+        for j in range(i):
+            require(not row_sets[i] & row_sets[j] and not group_sets[i] & group_sets[j],
+                    'Row/normalized-name group leakage')
+    require(set.union(*row_sets) == set(pool.row_id), 'Development pool coverage mismatch')
+    require(set(parts['outer_train'].row_id) == row_sets[0] | row_sets[1], 'Outer union mismatch')
+    indexed = pool.set_index('row_id')
+    for part, data in parts.items():
+        expected = indexed.loc[data.row_id, ['name', *LEVELS]].reset_index()
+        require(expected.equals(data[['row_id', 'name', *LEVELS]].reset_index(drop=True)),
+                f'{part}: source row content mismatch')
+
+
+def verify_sources(cfg: S07Config) -> dict:
+    """Only sealed-test row_id/name projected; never query sealed label columns."""
+    cfg.validate()
+    pins = cfg.fixture_pins if cfg.fixture else SOURCE_PINS
+    sources = {'manifest': Path(cfg.manifest), 'protocol': Path(cfg.protocol),
+               'corpus': Path(cfg.corpus), 'features': Path(cfg.full_tfidf) / 'features.joblib'}
+    refs = {}
+    for key, path in sources.items():
+        fp = fingerprint(path)
+        require(fp['sha256'] == pins[key], f'{key}: authoritative SHA256 mismatch')
+        refs[key] = {'path': str(path.resolve()), **fp}
+    meta_path = Path(cfg.full_tfidf) / 'vectorizer_meta.json'
+    refs['vectorizer_meta'] = {'path': str(meta_path.resolve()), **fingerprint(meta_path)}
+    m, p = read_json(cfg.manifest), read_json(cfg.protocol)
+    require(m['policy']['min_count'] == 12, 'Original S06 support filter must stay >=12')
+    frames = {}
+    for part in ('outer_train', 'comparison_val', 'sealed_test'):
+        spec = m['parts'][part]
+        path = Path(spec['file'])
+        require(fingerprint(path) == {k: spec[k] for k in ('sha256', 'bytes')},
+                f'S06 {part}: source fingerprint mismatch')
+        data = pd.read_parquet(path, columns=['row_id', 'name'] if part == 'sealed_test' else None)
+        require(len(data) == spec['n_rows'] and data.row_id.is_unique, f'{part}: identity/count mismatch')
+        require(not data[['row_id', 'name']].isna().any().any(), f'{part}: null structural identity')
+        frames[part] = data
+        refs[part] = {'path': str(path.resolve()), **fingerprint(path), **row_identity(data)}
+    pool = pd.concat([frames['outer_train'], frames['comparison_val']], ignore_index=True).sort_values(
+        'row_id').reset_index(drop=True)
+    require(pool.row_id.is_unique and not pool[['name', *LEVELS]].isna().any().any(),
+            'Invalid development rows')
+    require(not set(pool.row_id) & set(frames['sealed_test'].row_id), 'Sealed row identity leakage')
+    require(not set(group_keys(pool)) & set(group_keys(frames['sealed_test'])), 'Sealed group leakage')
+    require(LabelMaps.from_df(pool).to_dict() == m['label_maps'], 'S06 label maps not preserved')
+    require(p['frozen'] is True and p['fixed_epochs'] == p['chosen']['fixed_epochs'] > 0,
+            'Protocol not frozen/invalid epoch rule')
+    require(p['full_tfidf_id'] == refs['features']['sha256'], 'Protocol/full TF-IDF mismatch')
+    require(p['corpus_fingerprint']['sha256'] == refs['corpus']['sha256'], 'Protocol/corpus mismatch')
+    for part in frames:
+        require(all(p['splits'][part][k] == m['parts'][part][k] for k in ('sha256', 'bytes', 'n_rows')),
+                f'Protocol/source {part} mismatch')
+    sections = p['resolved_sections']
+    require(sections['train']['scheduler'] == 'none', 'Only frozen constant LR supported')
+    if not cfg.fixture:
+        require(len(pool) == 14652 and len(m['label_maps']['L3']) == 293, 'Production universe mismatch')
+        require(p['fixed_epochs'] == 14, 'Production final epochs must be 14')
+        require(sections['model'] == {'hidden': [256], 'dropout': [.3], 'activation': 'relu', 'norm': None},
+                'Frozen production model mismatch')
+        for k, v in {'optimizer': 'adamw', 'lr': .001, 'weight_decay': 0., 'batch_size': 256}.items():
+            require(sections['train'][k] == v, f'Frozen optimizer mismatch: {k}')
+    full = load_vectorizer_artifact(cfg.full_tfidf)
+    require((full.builder.n_char_features, full.builder.n_word_features) == cfg.full_counts,
+            'Full feature sizes mismatch')
+    meta = full.meta
+    require(meta['source']['sha256'] == refs['corpus']['sha256']
+            and meta['source']['bytes'] == refs['corpus']['bytes'], 'Full source/corpus mismatch')
+    require(meta['source']['strategy'] == 'head', 'Unsupported corpus selection strategy')
+    if not cfg.fixture:
+        require(meta['source']['rows_used'] == 981850 and meta['source']['max_rows'] == 1000000
+                and meta['fit'] == {'rows_in': 981850, 'rows_fit': 981850, 'subsampled': False},
+                'Full corpus fit selection mismatch')
+    return {'refs': refs, 'manifest': m, 'protocol': p, 'pool': pool, 'full': full,
+            'corpus_fit_selection': {'source': meta['source'], 'fit': meta['fit'],
+                                     'tfidf_config': meta['tfidf_config'], 'seed': meta['seed']}}
+
+
+def fresh_root(root: str | Path) -> Path:
+    path = Path(root)
+    path.mkdir(parents=True, exist_ok=False)
+    return path
+
+
+def preflight(cfg: S07Config, output_root: str | Path) -> dict:
+    """Persist compact pass/fail evidence even if a mandatory gate fails."""
+    root = fresh_root(output_root)
+    audit = {'schema': 's07-preflight@1', 'created_at': now(), 'config': dataclasses.asdict(cfg),
+             'study_kind': 'synthetic' if cfg.fixture else 'ODS_development', 'workload_runs': 0,
+             'sealed_labels_read': False, 'sealed_inference': False}
+    try:
+        source = verify_sources(cfg)
+        splits, diagnostics = split_development_pool(source['pool'], source['manifest']['label_maps'])
+        audit.update(refs=source['refs'], corpus_fit_selection=source['corpus_fit_selection'],
+                     label_maps_sha256=digest(source['manifest']['label_maps']),
+                     development_pool=row_identity(source['pool']), feasibility=diagnostics,
+                     partitions={s: {p: {**row_identity(d), 'minimum_l3_support': int(d.L3.value_counts().min())}
+                                     for p, d in parts.items()} for s, parts in splits.items()},
+                     resolved_protocol=source['protocol']['resolved_sections'],
+                     fixed_epochs=source['protocol']['fixed_epochs'])
+        require(len(splits) == 3, 'Mandatory grouped split minima not satisfied; see feasibility diagnostics')
+        audit.update(passed=True)
+    except Exception as exc:
+        audit.update(passed=False, error=f'{type(exc).__name__}: {exc}')
+    write_new(root / 'preflight.json', sealed(audit))
+    return audit
+
+
+def implementation_files() -> dict:
+    repo = Path(__file__).resolve().parents[1]
+    paths = sorted((repo / 'kkt_clf').glob('*.py')) + [repo / 'scripts' / 'run_s07.py']
+    return {str(p.relative_to(repo)): fingerprint(p) for p in paths}
+
+
+def _run_matrix(cfg: S07Config) -> list[dict]:
+    return [{'run_id': f'split_{s}__{method}__seed_{seed}', 'split_id': f'split_{s}',
+             'method': method, 'seed': seed,
+             'n_char': cfg.full_counts[0] if method == 'none' else cfg.selected_counts[0],
+             'n_word': cfg.full_counts[1] if method == 'none' else cfg.selected_counts[1],
+             'feature_dependency': 'full' if method == 'none' else f'split_{s}/{method}',
+             'initial_status': 'pending'}
+            for s in SPLIT_SEEDS for method in METHODS for seed in MODEL_SEEDS]
+
+
+def dependency_matrix(cfg: S07Config, split_specs: dict, corpus_sha256: str) -> list[dict]:
+    return [
+        {'dependency_id': f'{sid}/{method}', 'split_id': sid, 'method': method,
+         'ranking_rows': spec['parts']['outer_train']['sha256'],
+         'reference_checkpoint_run': f'{sid}__none__seed_42' if method == 'gradient_x_input' else None,
+         'reference_seed': 42 if method == 'gradient_x_input' else None,
+         'selection': {'k': sum(cfg.selected_counts), 'char_share': .75, 'seed': 42},
+         'attribution': {'target': 'L3 true-minus-strongest-competitor margin',
+                         'positive_only': True, 'aggregation': 'class_aware', 'rows': 'all_outer_train'}
+         if method == 'gradient_x_input' else None,
+         'refit_source_sha256': corpus_sha256, 'shared_across_seeds': list(MODEL_SEEDS)}
+        for sid, spec in split_specs.items() for method in METHODS[1:]]
+
+
+def write_plan(cfg: S07Config, output_root: str | Path) -> dict:
+    """Freeze the full matrix, source/split identities and dependency DAG before scoring."""
+    # Fail before touching output if source/split gates fail.
+    source = verify_sources(cfg)
+    splits, diagnostic = split_development_pool(source['pool'], source['manifest']['label_maps'])
+    require(len(splits) == 3, f'Feasibility failed: {diagnostic}')
+    root = fresh_root(output_root)
+    split_specs = {}
+    for sid, parts in splits.items():
+        spec = {}
+        supports = []
+        for part, data in parts.items():
+            path = root / 'splits' / sid / f'{part}.parquet'
+            path.parent.mkdir(parents=True, exist_ok=True)
+            data.to_parquet(path, index=False)
+            spec[part] = {'file': str(path.relative_to(root)), **fingerprint(path), **row_identity(data),
+                          'row_l3_sha256': digest(data[['row_id', 'L3']].sort_values('row_id').values.tolist())}
+            counts = data.L3.value_counts()
+            supports.extend({'part': part, 'L3': label, 'support': int(counts.get(label, 0))}
+                            for label in source['manifest']['label_maps']['L3'])
+        support_path = root / 'splits' / sid / 'supports.csv'
+        pd.DataFrame(supports).to_csv(support_path, index=False)
+        split_specs[sid] = {'seed': int(sid.split('_')[1]), 'parts': spec,
+                            'supports': {'file': str(support_path.relative_to(root)), **fingerprint(support_path)}}
+    dependencies = dependency_matrix(cfg, split_specs, source['refs']['corpus']['sha256'])
+    protocol = source['protocol']
+    final_train = {**protocol['resolved_sections']['train'], 'epochs': protocol['fixed_epochs'],
+                   'patience': protocol['fixed_epochs'], 'device': cfg.device,
+                   'num_threads': cfg.num_threads}
+    plan = {'schema': 's07-plan@1', 'created_at': now(), 'config': dataclasses.asdict(cfg),
+            'study_kind': 'synthetic' if cfg.fixture else 'ODS_development',
+            'sources': source['refs'], 'corpus_fit_selection': source['corpus_fit_selection'],
+            'label_maps': source['manifest']['label_maps'], 'minima': MINIMA,
+            'development_pool': row_identity(source['pool']), 'splits': split_specs,
+            'runs': _run_matrix(cfg), 'dependencies': dependencies,
+            'training': {'model': protocol['resolved_sections']['model'],
+                         'loss': protocol['resolved_sections']['loss'], 'train': final_train,
+                         'restore_best': False, 'weights': 'final_epoch',
+                         'no_validation_tuning': True},
+            'metrics': {'prediction': 'raw top-1 L3', 'reject': False, 'universe': 'fixed label_maps.L3',
+                        'zero_division': 0},
+            'versions': environment_versions(), 'implementation': implementation_files(), 'warnings': WARNINGS,
+            'resume': 'unsupported; execution_started.json makes a plan single-use'}
+    plan = json.loads(json.dumps(plan, ensure_ascii=False, allow_nan=False))
+    write_new(root / 'plan.json', sealed(plan))
+    return plan
+
+
+def load_plan(output_root: str | Path, verify_inputs: bool = True) -> dict:
+    root = Path(output_root)
+    plan = read_sealed(root / 'plan.json')
+    require(plan.get('schema') == 's07-plan@1', 'Unknown plan schema')
+    cfg = S07Config(**plan['config'])
+    cfg.validate()
+    require(plan['runs'] == _run_matrix(cfg) and len(plan['runs']) == 36, 'Run matrix/config mismatch')
+    require(plan['minima'] == MINIMA and set(plan['splits']) == {f'split_{s}' for s in SPLIT_SEEDS},
+            'Split/minima plan mismatch')
+    require(plan['dependencies'] == dependency_matrix(cfg, plan['splits'], plan['sources']['corpus']['sha256']),
+            'Ranking/refit dependency DAG differs from fixed split-local design')
+    require(plan['implementation'] == implementation_files(), 'Implementation changed since plan freeze')
+    if verify_inputs:
+        source = verify_sources(cfg)
+        require(source['refs'] == plan['sources'], 'Source provenance changed since plan freeze')
+        require(source['corpus_fit_selection'] == plan['corpus_fit_selection'], 'Corpus fit selection changed')
+        require(source['manifest']['label_maps'] == plan['label_maps'], 'Label maps changed')
+        require(environment_versions() == plan['versions'], 'Runtime dependencies differ from frozen plan')
+        p = source['protocol']
+        expected_train = {**p['resolved_sections']['train'], 'epochs': p['fixed_epochs'],
+                          'patience': p['fixed_epochs'], 'device': cfg.device, 'num_threads': cfg.num_threads}
+        require(plan['training'] == {'model': p['resolved_sections']['model'], 'loss': p['resolved_sections']['loss'],
+                                    'train': expected_train, 'restore_best': False, 'weights': 'final_epoch',
+                                    'no_validation_tuning': True}, 'Training protocol/config changed')
+        rebuilt, diagnostics = split_development_pool(source['pool'], plan['label_maps'])
+        require(len(rebuilt) == 3, f'Frozen split no longer feasible: {diagnostics}')
+        for sid, spec in plan['splits'].items():
+            parts = load_split(root, plan, sid)
+            validate_partition(source['pool'], parts, plan['label_maps'])
+            for part in parts:
+                require(parts[part].equals(rebuilt[sid][part]), 'Saved split differs from deterministic seed')
+            info = spec['supports']
+            require(fingerprint(root / info['file']) == {k: info[k] for k in ('sha256', 'bytes')},
+                    'Saved support table changed')
+    return plan
+
+
+def load_split(root: Path, plan: dict, split_id: str) -> dict:
+    result = {}
+    for part, spec in plan['splits'][split_id]['parts'].items():
+        path = root / spec['file']
+        require(path.resolve().is_relative_to(root.resolve()), 'Unsafe split path')
+        require(fingerprint(path) == {k: spec[k] for k in ('sha256', 'bytes')}, 'Split fingerprint changed')
+        data = pd.read_parquet(path)
+        require(row_identity(data) == {k: spec[k] for k in ('rows', 'row_ids_sha256', 'groups_sha256')},
+                'Split row/group identities changed')
+        result[part] = data
+    return result
+
+
+def verify_frozen_runtime(root: Path, plan: dict) -> None:
+    require(read_sealed(root / 'plan.json') == plan, 'Plan changed during execution')
+    require(plan['implementation'] == implementation_files(), 'Implementation changed during execution')
+    for info in plan['sources'].values():
+        require(fingerprint(info['path']) == {k: info[k] for k in ('sha256', 'bytes')},
+                'Frozen source changed during execution')
+
+
+def run_provenance(plan: dict, run: dict, feature_id: str) -> dict:
+    sid = run['split_id']
+    return {'plan_sha256': digest(plan), 'run': run, 'feature_id': feature_id,
+            'outer_train_sha256': plan['splits'][sid]['parts']['outer_train']['sha256'],
+            'comparison_val_sha256': plan['splits'][sid]['parts']['comparison_val']['sha256'],
+            'label_maps_sha256': digest(plan['label_maps']), 'training_sha256': digest(plan['training'])}
+
+
+def dependency_provenance(plan: dict, sid: str, method: str) -> dict:
+    deps = [d for d in plan['dependencies'] if d['split_id'] == sid and d['method'] == method]
+    require(len(deps) == 1, 'Missing/ambiguous ranking dependency')
+    return {'plan_sha256': digest(plan), 'dependency': deps[0],
+            'label_maps_sha256': digest(plan['label_maps']),
+            'full_features_sha256': plan['sources']['features']['sha256']}
+
+
+def _record_stage(root: Path, scope: str, method: str, sid: str, run_id: str | None,
+                  name: str, measurement: _Measurement, status: str = 'success') -> None:
+    if hasattr(measurement, '_rss'):
+        record = measurement.stage_row(run_id or f'{sid}/{method}', name, status)
+    else:
+        record = {'run_id': run_id or f'{sid}/{method}', 'stage': name, 'status': status,
+                  'seconds': None, 'peak_rss_bytes': None, 'peak_vram_bytes': None,
+                  'measurement_note': measurement.note + '; measurement setup failed, resources unavailable'}
+    record.update(scope=scope, method=method, split_id=sid, charge_count=1,
+                  plan_sha256=digest(read_sealed(root / 'plan.json')),
+                  cost_role='attribution_checkpoint_and_full_comparator' if method == 'none' and name == 'train'
+                  and run_id and run_id.endswith('seed_42') else 'ordinary',
+                  checkpoint_incremental_training_seconds=0.0 if method == 'none' and name == 'train'
+                  and run_id and run_id.endswith('seed_42') else None)
+    # One immutable event file per measurement, no repeated append/resume charging.
+    import uuid
+    write_new(root / 'stages' / f'{uuid.uuid4().hex}.json', sealed(record))
+
+
+@contextmanager
+def measured_stage(root: Path, scope: str, method: str, sid: str, run_id: str | None,
+                   stage: str, device: torch.device, note: str):
+    measurement = _Measurement(device, note)
+    try:
+        with measurement:
+            yield measurement
+    except BaseException:
+        _record_stage(root, scope, method, sid, run_id, stage, measurement, 'failed')
+        raise
+    else:
+        _record_stage(root, scope, method, sid, run_id, stage, measurement)
+
+
+def _verify_checkpoint(root: Path, plan: dict, run: dict, feature_builder: Any,
+                       feature_id: str, device: torch.device) -> Any:
+    run_dir = root / 'runs' / run['run_id']
+    receipt = read_sealed(run_dir / 'checkpoint_receipt.json')
+    expected = run_provenance(plan, run, feature_id)
+    require(receipt['provenance'] == expected, 'Checkpoint dependency provenance mismatch')
+    verify_tree(run_dir / 'checkpoint', receipt['files'])
+    require(read_json(run_dir / 'checkpoint' / 'model_spec.json') == {
+        'n_features': feature_builder.n_features,
+        'n_classes': [len(plan['label_maps'][l]) for l in LEVELS],
+        'model': plan['training']['model']}, 'Checkpoint model specification differs from frozen architecture')
+    art = load_artifact(str(run_dir / 'checkpoint'), device)
+    require(art.maps.to_dict() == plan['label_maps'] and art.meta['s07'] == expected,
+            'Checkpoint label map/run provenance mismatch')
+    require(art.config == plan['training'], 'Checkpoint resolved training config mismatch')
+    require(feature_contract_fingerprint(art.features) == feature_contract_fingerprint(feature_builder),
+            'Checkpoint feature contract mismatch')
+    require(art.model.n_features == feature_builder.n_features, 'Checkpoint input dimension mismatch')
+    return art
+
+
+def verify_selected(root: Path, plan: dict, sid: str, method: str, full: Any) -> Any:
+    """Verify entire ranking/refit chain before any downstream transform/checkpoint use."""
+    base = root / 'shared' / sid / method
+    receipt = read_sealed(base / 'dependency_receipt.json')
+    expected = dependency_provenance(plan, sid, method)
+    require(receipt['provenance'] == expected, 'Split-local ranking provenance mismatch')
+    expected_files = {'selection', 'tfidf', 'attribution_ratings'} if method == 'gradient_x_input' else {'selection', 'tfidf'}
+    require(set(receipt['files']) == expected_files, 'Incomplete dependency artifact inventory')
+    for name in expected_files:
+        verify_tree(base / name, receipt['files'][name])
+    result, meta = load_selection_artifact(str(base / 'selection'))
+    require(meta['provenance'] == {**expected, 'checkpoint': receipt['checkpoint']},
+            'Selection/checkpoint provenance mismatch')
+    require(result.method == method and result.k_requested == sum(S07Config(**plan['config']).selected_counts),
+            'Selection method/budget mismatch')
+    from .s06_comparison import _verify_ratings_meta
+    ratings_meta = _verify_ratings_meta(result.ratings_dir, method)
+    require(ratings_meta == result.ratings_manifest, 'Saved ranking manifest mismatch')
+    if method != 'global':
+        validate_ratings_dir(result.ratings_dir, result.ratings_manifest, method=method,
+                             widths={'char': full.builder.n_char_features, 'word': full.builder.n_word_features})
+    if method == 'gradient_x_input':
+        require(fingerprint(base / 'attribution.json') == receipt['attribution_report'], 'Attribution report changed')
+        att_report = read_sealed(base / 'attribution.json')
+        require(att_report['provenance'] == expected and att_report['checkpoint'] == receipt['checkpoint'],
+                'Attribution report/checkpoint provenance mismatch')
+        reference = next(r for r in plan['runs'] if r['run_id'] == expected['dependency']['reference_checkpoint_run'])
+        ref_dir = root / 'runs' / reference['run_id']
+        ref_receipt = read_sealed(ref_dir / 'checkpoint_receipt.json')
+        require(receipt['checkpoint'] == {'run_id': reference['run_id'],
+                                         'receipt_sha256': sha(ref_dir / 'checkpoint_receipt.json')},
+                'GxI reference checkpoint changed')
+        _verify_checkpoint(root, plan, reference, full.builder, plan['sources']['features']['sha256'],
+                           torch.device('cpu'))
+        require(ref_receipt['provenance']['run']['seed'] == 42, 'GxI reference seed is not 42')
+    else:
+        require(receipt['checkpoint'] is None, 'Chi2 unexpectedly depends on checkpoint')
+    from .s06_comparison import _selection_from_saved_ratings
+    replay_cfg = SelectionConfig(method=method, k=result.k_requested, char_share=.75, seed=42)
+    if method == 'gradient_x_input':
+        replay = select_features_gradient_x_input(result.ratings_dir, replay_cfg, full.builder.n_char_features,
+                                                  full.builder.n_features, l3_labels=plan['label_maps']['L3'])
+    else:
+        replay = _selection_from_saved_ratings(method, result.ratings_dir, result.k_requested, .75,
+                                               full.builder.n_char_features, full.builder.n_features,
+                                               l3_labels=plan['label_maps']['L3'])
+    for branch in ('char', 'word'):
+        saved, expected_branch = getattr(result, branch), getattr(replay, branch)
+        require(np.array_equal(saved.indices, expected_branch.indices)
+                and np.allclose(saved.scores, expected_branch.scores, equal_nan=True),
+                'Selected vocabulary does not reproduce from saved ranking')
+    loaded = load_vectorizer_artifact(str(base / 'tfidf'))
+    cfg = S07Config(**plan['config'])
+    require((loaded.builder.n_char_features, loaded.builder.n_word_features) == cfg.selected_counts,
+            'Selected branch sizes mismatch')
+    require(loaded.meta['source'] == full.meta['source'] and loaded.meta['fit'] == full.meta['fit'],
+            'Refit corpus snapshot/fit subset mismatch')
+    for branch, indices, offset in [('char', result.char.indices, 0),
+                                     ('word', result.word.indices, full.builder.n_char_features)]:
+        selected = np.sort(np.asarray(indices, dtype=np.int64) - offset)
+        old, new = getattr(full.builder, branch), getattr(loaded.builder, branch)
+        require(len(selected) == cfg.selected_counts[0 if branch == 'char' else 1], 'Branch count mismatch')
+        require(len(np.unique(selected)) == len(selected) and np.all(selected >= 0)
+                and np.all(selected < len(old.vocabulary_)), 'Invalid full-space branch indices')
+        names = old.get_feature_names_out()[selected].tolist()
+        require(new.get_feature_names_out().tolist() == names, 'Stable feature identity/index mapping mismatch')
+        require(np.allclose(new.idf_, old.idf_[selected], rtol=1e-6, atol=1e-6), 'IDF invariant failed')
+        require(new.norm == 'l2', 'Per-branch L2 normalization required')
+        require(loaded.meta['refit_checks'][f'{branch}_idf_max_abs_diff'] <= 1e-5,
+                'Refit recorded IDF check failed')
+    return loaded
+
+
+def _train_one(root: Path, plan: dict, run: dict, builder: Any, feature_id: str,
+               parts: dict, device: torch.device) -> None:
+    verify_frozen_runtime(root, plan)
+    cfg = S07Config(**plan['config'])
+    maps = LabelMaps.from_dict(plan['label_maps'])
+    run_dir = root / 'runs' / run['run_id']
+    run_dir.mkdir(parents=True, exist_ok=False)
+    frozen_parts = load_split(root, plan, run['split_id'])
+    require(all(parts[p].equals(frozen_parts[p]) for p in frozen_parts),
+            'Training rows differ from frozen split')
+    train, monitor = parts['outer_train'], parts['train_monitor']
+    model_cfg = ModelConfig(**plan['training']['model'])
+    loss_cfg = LossConfig(**plan['training']['loss'])
+    train_cfg = TrainConfig(**plan['training']['train'])
+    y = tuple(maps.encode(l, train[l]) for l in LEVELS)
+    ym = tuple(maps.encode(l, monitor[l]) for l in LEVELS)
+    nclasses = tuple(maps.n_classes(l) for l in LEVELS)
+    weights = {h: compute_class_weights(y[i], nclasses[i], loss_cfg.class_weight_mode[h],
+                                        loss_cfg.class_weight_alpha) for i, h in enumerate(('l1', 'l2', 'l3'))}
+    with measured_stage(root, 'run', run['method'], run['split_id'], run['run_id'], 'train_transform',
+                        device, 'Train/monitor transform; monitor belongs to outer_train, no tuning') as m:
+        X = builder.transform(train.name.tolist(), chunk_rows=cfg.chunk_rows)
+        Xm = builder.transform(monitor.name.tolist(), chunk_rows=cfg.chunk_rows)
+    set_seed(run['seed'])
+    model = SparseMLP(builder.n_features, nclasses, model_cfg)
+    with measured_stage(root, 'run', run['method'], run['split_id'], run['run_id'], 'train',
+                        device, 'Fresh fixed-epoch model; final weights, no validation tuning') as m:
+        history, best, trace = train_model(model, X, y, Xm, ym, loss_cfg, train_cfg, weights,
+                                           device, run['seed'], restore_best=False)
+    require(len(history) == train_cfg.epochs and not best.get('restored_best'), 'Final epoch rule violated')
+    provenance = run_provenance(plan, run, feature_id)
+    save_artifact(str(run_dir / 'checkpoint'), model, model_cfg, builder, maps,
+                  RejectRule(.5, False, False), {'note': 'S07 raw top-1, no reject'},
+                  {'mode': loss_cfg.class_weight_mode, 'alpha': loss_cfg.class_weight_alpha,
+                   'weights': {k: None if v is None else v.tolist() for k, v in weights.items()}},
+                  plan['training'], {'s07': provenance, 'seed': run['seed']})
+    write_new(run_dir / 'training.json', sealed({'provenance': provenance, 'history': history,
+                                                'best_monitor_diagnostic_only': best, 'lr_trace': trace}))
+    write_new(run_dir / 'checkpoint_receipt.json', sealed({'provenance': provenance,
+                                                          'files': tree(run_dir / 'checkpoint')}))
+    _verify_checkpoint(root, plan, run, builder, feature_id, device)
+
+
+def _prepare_selected(root: Path, plan: dict, sid: str, method: str, full: Any,
+                      parts: dict, device: torch.device) -> None:
+    verify_frozen_runtime(root, plan)
+    cfg = S07Config(**plan['config'])
+    base = root / 'shared' / sid / method
+    base.mkdir(parents=True, exist_ok=False)
+    frozen_parts = load_split(root, plan, sid)
+    # Integrity gate checks supplied partitions; ranking below consumes outer_train only.
+    require(all(parts[p].equals(frozen_parts[p]) for p in frozen_parts),
+            'Ranking rows differ from frozen split')
+    maps = LabelMaps.from_dict(plan['label_maps'])
+    with measured_stage(root, 'shared', method, sid, None, 'ranking_transform',
+                        device, 'Split outer_train transform for supervised ranking'):
+        X = full.builder.transform(parts['outer_train'].name.tolist(), chunk_rows=cfg.chunk_rows)
+    verified_full = load_vectorizer_artifact(cfg.full_tfidf)
+    require(feature_contract_fingerprint(full.builder) == feature_contract_fingerprint(verified_full.builder)
+            and full.meta == verified_full.meta, 'Ranking full TF-IDF input contract mismatch')
+    y = maps.encode('L3', parts['outer_train'].L3)
+    k = sum(cfg.selected_counts)
+    sel_cfg = SelectionConfig(method=method, k=k, char_share=.75, seed=42)
+    checkpoint = None
+    ranking_stage = 'attribution_ranking_selection' if method == 'gradient_x_input' else 'ranking_selection'
+    with measured_stage(root, 'shared', method, sid, None, ranking_stage,
+                        device, 'Split-local outer_train-only ranking/selection'):
+        if method == 'gradient_x_input':
+            ref = next(r for r in plan['runs'] if r['split_id'] == sid and r['method'] == 'none' and r['seed'] == 42)
+            art = _verify_checkpoint(root, plan, ref, full.builder, plan['sources']['features']['sha256'], device)
+            checkpoint = {'run_id': ref['run_id'], 'receipt_sha256': sha(root / 'runs' / ref['run_id'] /
+                                                                       'checkpoint_receipt.json')}
+            att = compute_margin_attributions(art.model, X, y, device, batch_size=cfg.attribution_batch_size)
+            ratings_dir = base / 'attribution_ratings'
+            _, report = build_gxi_ratings(att, full.builder.n_char_features, str(ratings_dir), l3_labels=maps.l3)
+            write_new(base / 'attribution.json', sealed({'provenance': dependency_provenance(plan, sid, method),
+                                                        'checkpoint': checkpoint, 'report': report}))
+            result = select_features_gradient_x_input(str(ratings_dir), sel_cfg, full.builder.n_char_features,
+                                                      full.builder.n_features, l3_labels=maps.l3)
+        else:
+            result = select_features(X, y, sel_cfg, full.builder.n_char_features, l3_labels=maps.l3)
+        require((len(result.char.indices), len(result.word.indices)) == cfg.selected_counts,
+                'Strict S07 branch budget unavailable')
+        save_selection_artifact(str(base / 'selection'), result,
+                                {**dependency_provenance(plan, sid, method), 'checkpoint': checkpoint},
+                                ratings_copy=True)
+        selection, _ = load_selection_artifact(str(base / 'selection'))
+    with measured_stage(root, 'shared', method, sid, None, 'refit',
+                        device, 'Fixed vocabulary refit on exact original corpus selection; branch normalization'):
+        refit = refit_selected_vectorizer(cfg.corpus, full.builder, full.meta, selection,
+                                          out_dir=str(base / 'tfidf'), row_norm_sample=1000)
+        reloaded = load_vectorizer_artifact(str(base / 'tfidf'))
+        require(feature_contract_fingerprint(refit.builder) == feature_contract_fingerprint(reloaded.builder),
+                'Refit serialization changed feature contract')
+    files = {name: tree(base / name) for name in (('selection', 'tfidf', 'attribution_ratings')
+                                                 if method == 'gradient_x_input' else ('selection', 'tfidf'))}
+    write_new(base / 'dependency_receipt.json', sealed({'provenance': dependency_provenance(plan, sid, method),
+                                                       'checkpoint': checkpoint, 'files': files,
+                                                       'attribution_report': fingerprint(base / 'attribution.json')
+                                                       if method == 'gradient_x_input' else None,
+                                                       'checks': refit.checks}))
+    verify_selected(root, plan, sid, method, full)
+
+
+def _evaluate_one(root: Path, plan: dict, run: dict, builder: Any, feature_id: str,
+                  parts: dict, device: torch.device) -> None:
+    # Revalidate the immutable plan/source/split fingerprints immediately before inference.
+    verify_frozen_runtime(root, plan)
+    load_split(root, plan, run['split_id'])
+    art = _verify_checkpoint(root, plan, run, builder, feature_id, device)
+    val = parts['comparison_val']
+    cfg = S07Config(**plan['config'])
+    with measured_stage(root, 'run', run['method'], run['split_id'], run['run_id'], 'inference',
+                        device, 'Active development comparison_val raw top-1; no reject'):
+        X = art.features.transform(val.name.tolist(), chunk_rows=cfg.chunk_rows)
+        predictions, _ = predict(art.model, X, device, plan['training']['train']['eval_batch_size'], top_k=1)
+    # Preserve S06 raw top-1 semantics, including the bias path for empty rows.
+    pred = np.asarray(predictions.l3_top1).copy()
+    truth = art.maps.encode('L3', val.L3)
+    f1, acc, pc = _l3_val_metrics(truth, pred, art.maps.l3)
+    labels = np.asarray(art.maps.l3, dtype=object)
+    frame = pd.DataFrame({'row_id': val.row_id.to_numpy(), 'run_id': run['run_id'],
+                          'split_id': run['split_id'], 'true_L3': val.L3.to_numpy(),
+                          'predicted_L3': [labels[x] if x >= 0 else None for x in pred],
+                          'empty_features': np.diff(X.indptr) == 0})
+    path = root / 'runs' / run['run_id'] / 'predictions_val.parquet'
+    require(not path.exists(), 'Predictions already exist; unsafe resume rejected')
+    frame.to_parquet(path, index=False)
+    receipt = {'schema': 's07-run@1', 'status': 'success', 'reason': None,
+               'provenance': run_provenance(plan, run, feature_id), 'predictions': fingerprint(path),
+               'checkpoint_receipt_sha256': sha(path.parent / 'checkpoint_receipt.json'),
+               'training_sha256': sha(path.parent / 'training.json'),
+               'metrics': {'l3_macro_f1': f1, 'l3_accuracy': acc}, 'per_class': pc}
+    write_new(path.parent / 'result.json', sealed(receipt))
+
+
+def execute(output_root: str | Path) -> dict:
+    """Single-use execution; no reused checkpoints/results, including after a crash.
+
+    Full seed42 is trained split-locally before GxI. It is the full comparator
+    too: its training cost is charged once, with GxI allocation disclosed.
+    """
+    root = Path(output_root)
+    plan = load_plan(root)
+    require(not (root / 'execution_started.json').exists(), 'Resume unsupported: use a fresh output root/plan')
+    for name in ('runs', 'shared', 'stages'):
+        require(not (root / name).exists(), f'Untrusted pre-existing {name}; fresh root required')
+    cfg = S07Config(**plan['config'])
+    device = resolve_device(cfg.device)
+    if cfg.num_threads is not None:
+        torch.set_num_threads(cfg.num_threads)
+    write_new(root / 'execution_started.json', sealed({'plan_sha256': digest(plan), 'started_at': now(),
+                                                      'device': str(device)}))
+    full = load_vectorizer_artifact(cfg.full_tfidf)
+    summary = {'success': [], 'failed': [], 'skipped': [], 'pending': []}
+    try:
+        for sid in plan['splits']:
+            parts = load_split(root, plan, sid)
+            for method in METHODS:
+                if method == 'none':
+                    builder, fid = full.builder, plan['sources']['features']['sha256']
+                else:
+                    try:
+                        _prepare_selected(root, plan, sid, method, full, parts, device)
+                    except Exception as exc:
+                        for dependent in [r for r in plan['runs'] if r['split_id'] == sid and r['method'] == method]:
+                            skipped_dir = root / 'runs' / dependent['run_id']
+                            write_new(skipped_dir / 'result.json', sealed({
+                                'schema': 's07-run@1', 'status': 'skipped',
+                                'reason': f'Shared dependency failed: {type(exc).__name__}: {exc}',
+                                'provenance': run_provenance(plan, dependent, 'unavailable')}))
+                            summary['skipped'].append(dependent['run_id'])
+                        raise
+                    selected = verify_selected(root, plan, sid, method, full)
+                    builder, fid = selected.builder, sha(root / 'shared' / sid / method / 'tfidf' / 'features.joblib')
+                for run in [r for r in plan['runs'] if r['split_id'] == sid and r['method'] == method]:
+                    try:
+                        _train_one(root, plan, run, builder, fid, parts, device)
+                        if method != 'none':
+                            verify_selected(root, plan, sid, method, full)
+                        _evaluate_one(root, plan, run, builder, fid, parts, device)
+                        summary['success'].append(run['run_id'])
+                    except Exception as exc:
+                        result_path = root / 'runs' / run['run_id'] / 'result.json'
+                        if not result_path.exists():
+                            write_new(result_path, sealed({'schema': 's07-run@1', 'status': 'failed',
+                                                           'reason': f'{type(exc).__name__}: {exc}',
+                                                           'provenance': run_provenance(plan, run, fid)}))
+                        summary['failed'].append(run['run_id'])
+                        raise  # Integrity failures stop the workload; no silent continuation.
+    except Exception as exc:
+        summary['stopped_reason'] = f'{type(exc).__name__}: {exc}'
+    completed = set(summary['success'] + summary['failed'] + summary['skipped'])
+    summary['pending'] = [r['run_id'] for r in plan['runs'] if r['run_id'] not in completed]
+    # Missing dependencies/stages leave planned runs pending, never success.
+    write_new(root / 'execution_finished.json', sealed({'plan_sha256': digest(plan), 'finished_at': now(),
+                                                       'summary': summary,
+                                                       'results': {r['run_id']: fingerprint(root / 'runs' / r['run_id'] / 'result.json')
+                                                                   for r in plan['runs'] if (root / 'runs' / r['run_id'] / 'result.json').exists()},
+                                                       'stages': tree(root / 'stages')
+                                                       if (root / 'stages').exists() else {}}))
+    return summary
diff --git a/kkt_clf/s07_report.py b/kkt_clf/s07_report.py
new file mode 100644
index 0000000..c437e9d
--- /dev/null
+++ b/kkt_clf/s07_report.py
@@ -0,0 +1,500 @@
+"""S07 verified split-aware tables and table-only, self-contained HTML rendering.
+
+Rebuilding HTML loads no models and reads only manifest plus the eight data
+files in the supplied report directory. Original report directories are immutable.
+"""
+from __future__ import annotations
+
+import html
+import itertools
+import json
+from pathlib import Path
+
+import numpy as np
+import pandas as pd
+
+from .s07 import (METHODS, MODEL_SEEDS, S07Config, WARNINGS, _l3_val_metrics,
+                  _verify_checkpoint, dependency_provenance, digest, fingerprint,
+                  fresh_root, load_plan, load_split, now, read_sealed, require,
+                  row_identity, run_provenance, sealed, sha, tree, verify_selected,
+                  verify_tree, write_new)
+
+TABLES = {'runs.csv': ['run_id', 'split_id', 'method', 'seed', 'status', 'reason',
+                       'l3_macro_f1', 'l3_accuracy', 'delta_macro_f1', 'delta_accuracy',
+                       'n_char', 'n_word', 'model_bytes'],
+          'per_class.parquet': ['run_id', 'split_id', 'method', 'seed', 'L3', 'support',
+                                'precision', 'recall', 'f1', 'delta_precision', 'delta_recall', 'delta_f1'],
+          'predictions_val.parquet': ['run_id', 'split_id', 'row_id', 'true_L3', 'predicted_L3', 'empty_features'],
+          'support_bins.csv': ['run_id', 'split_id', 'method', 'seed', 'bin', 'class_count',
+                               'example_support', 'macro_f1', 'delta_macro_f1'],
+          'confusion_pairs.csv': ['run_id', 'split_id', 'method', 'seed', 'true_L3', 'predicted_L3',
+                                  'count', 'baseline_count', 'delta_count'],
+          'feature_stability.csv': ['method', 'split_a', 'split_b', 'branch', 'status', 'reason',
+                                    'size_a', 'size_b', 'intersection', 'union', 'jaccard'],
+          'stages.csv': ['event_id', 'split_id', 'method', 'run_id', 'scope', 'stage', 'status',
+                         'seconds', 'peak_rss_bytes', 'peak_vram_bytes', 'charge_count', 'cost_role',
+                         'checkpoint_incremental_training_seconds', 'measurement_note'],
+          'paired_summary.csv': ['level', 'split_id', 'method', 'paired_count', 'split_count',
+                                 'mean_delta_macro_f1', 'within_split_seed_std', 'between_split_mean_std',
+                                 'min_delta_macro_f1', 'max_delta_macro_f1', 'gains', 'losses', 'ties',
+                                 'mean_delta_accuracy', 'mean_macro_f1', 'within_split_metric_std',
+                                 'between_split_metric_mean_std']}
+BINS = ('2-4', '5-9', '10-19', '>=20')
+
+
+def support_bin(support: int) -> str:
+    require(support >= 2, 'Comparison validation support below 2')
+    if support < 5:
+        return '2-4'
+    if support < 10:
+        return '5-9'
+    if support < 20:
+        return '10-19'
+    return '>=20'
+
+
+def feature_jaccard(a, b) -> dict:
+    """Stable feature identities only, independently for each TF-IDF branch."""
+    a, b = set(a), set(b)
+    intersection, union = len(a & b), len(a | b)
+    return {'size_a': len(a), 'size_b': len(b), 'intersection': intersection,
+            'union': union, 'jaccard': intersection / union if union else None}
+
+
+def paired_summary(runs: pd.DataFrame) -> pd.DataFrame:
+    """Within-split population seed SD and between-split population mean SD.
+
+    No pooling of nine correlated pairs for an independence-based uncertainty
+    estimate. Partial pair/split counts remain explicit and can never imply 3/3.
+    """
+    rows = []
+    for method in METHODS:
+        group = runs[(runs.method == method) & (runs.status == 'success')]
+        for sid in runs.split_id.unique():
+            g = group[group.split_id == sid]
+            paired = g.dropna(subset=['delta_macro_f1'])
+            d = paired.delta_macro_f1.to_numpy(float)
+            rows.append({'level': 'within_split', 'split_id': sid, 'method': method,
+                         'paired_count': len(d), 'split_count': int(bool(len(d))),
+                         'mean_delta_macro_f1': float(d.mean()) if len(d) else None,
+                         'within_split_seed_std': float(d.std(ddof=0)) if len(d) else None,
+                         'between_split_mean_std': None,
+                         'min_delta_macro_f1': float(d.min()) if len(d) else None,
+                         'max_delta_macro_f1': float(d.max()) if len(d) else None,
+                         'gains': int((d > 0).sum()), 'losses': int((d < 0).sum()), 'ties': int((d == 0).sum()),
+                         'mean_delta_accuracy': float(paired.delta_accuracy.mean()) if len(d) else None,
+                         'mean_macro_f1': float(g.l3_macro_f1.mean()) if len(g) else None,
+                         'within_split_metric_std': float(g.l3_macro_f1.std(ddof=0)) if len(g) else None,
+                         'between_split_metric_mean_std': None})
+        # All three matched seeds are required before a split enters across-split means.
+        within = pd.DataFrame([r for r in rows if r['method'] == method and r['level'] == 'within_split'])
+        complete = within[within.paired_count == len(MODEL_SEEDS)]
+        d = complete.mean_delta_macro_f1.to_numpy(float)
+        means = complete.mean_macro_f1.to_numpy(float)
+        rows.append({'level': 'across_split_means', 'split_id': None, 'method': method,
+                     'paired_count': int(complete.paired_count.sum()), 'split_count': len(complete),
+                     'mean_delta_macro_f1': float(d.mean()) if len(d) else None,
+                     'within_split_seed_std': None,
+                     'between_split_mean_std': float(d.std(ddof=0)) if len(d) else None,
+                     'min_delta_macro_f1': float(d.min()) if len(d) else None,
+                     'max_delta_macro_f1': float(d.max()) if len(d) else None,
+                     'gains': int((d > 0).sum()), 'losses': int((d < 0).sum()), 'ties': int((d == 0).sum()),
+                     'mean_delta_accuracy': float(complete.mean_delta_accuracy.mean()) if len(d) else None,
+                     'mean_macro_f1': float(means.mean()) if len(d) else None,
+                     'within_split_metric_std': None,
+                     'between_split_metric_mean_std': float(means.std(ddof=0)) if len(d) else None})
+    return pd.DataFrame(rows, columns=TABLES['paired_summary.csv'])
+
+
+def _verified_results(root: Path, plan: dict) -> tuple[list, list, list, dict]:
+    """Fail closed before reporting any successful row or deserializing its model."""
+    import torch
+    from .vectorizer import load_vectorizer_artifact
+
+    cfg = S07Config(**plan['config'])
+    full = load_vectorizer_artifact(cfg.full_tfidf)
+    all_runs, all_pc, all_preds, selections = [], [], [], {}
+    finished_path = root / 'execution_finished.json'
+    finished = read_sealed(finished_path) if finished_path.exists() else None
+    if finished is not None:
+        require(finished['plan_sha256'] == digest(plan), 'Execution/plan mismatch')
+        actual = {r['run_id']: fingerprint(root / 'runs' / r['run_id'] / 'result.json')
+                  for r in plan['runs'] if (root / 'runs' / r['run_id'] / 'result.json').exists()}
+        require(actual == finished['results'], 'Missing/modified result in execution ledger')
+        listed = finished['summary']
+        require(set(listed['success'] + listed['failed'] + listed['skipped'] + listed['pending'])
+                == {r['run_id'] for r in plan['runs']}, 'Execution statuses do not cover frozen plan')
+    for sid in plan['splits']:
+        for method in METHODS[1:]:
+            base = root / 'shared' / sid / method
+            if (base / 'dependency_receipt.json').exists():
+                selections[(sid, method)] = verify_selected(root, plan, sid, method, full)
+    for run in plan['runs']:
+        sid, method, rid = run['split_id'], run['method'], run['run_id']
+        path = root / 'runs' / rid / 'result.json'
+        row = {**run, 'status': 'pending', 'reason': 'No completed result',
+               'l3_macro_f1': None, 'l3_accuracy': None, 'delta_macro_f1': None, 'delta_accuracy': None,
+               'model_bytes': None}
+        if not path.exists():
+            all_runs.append(row)
+            continue
+        result = read_sealed(path)
+        status = result['status']
+        require(status in ('success', 'failed', 'skipped', 'pending'), 'Unknown run status')
+        row.update(status=status, reason=result.get('reason'))
+        require(status == 'success' or bool(result.get('reason')), 'Non-success status requires reason')
+        feature_id = plan['sources']['features']['sha256'] if method == 'none' else (
+            sha(root / 'shared' / sid / method / 'tfidf' / 'features.joblib')
+            if (sid, method) in selections else result['provenance']['feature_id'])
+        require(result['provenance'] == run_provenance(plan, run, feature_id), 'Result/plan provenance mismatch')
+        if finished:
+            require(rid in finished['summary'][status], 'Result/execution status mismatch')
+        if status != 'success':
+            all_runs.append(row)
+            continue
+        require(method == 'none' or (sid, method) in selections, 'Successful selected run lacks verified dependencies')
+        builder = full.builder if method == 'none' else selections[(sid, method)].builder
+        require(result['checkpoint_receipt_sha256'] == sha(path.parent / 'checkpoint_receipt.json'),
+                'Checkpoint receipt changed')
+        require(result['training_sha256'] == sha(path.parent / 'training.json'), 'Training record changed')
+        training = read_sealed(path.parent / 'training.json')
+        require(training['provenance'] == result['provenance']
+                and len(training['history']) == plan['training']['train']['epochs']
+                and not training['best_monitor_diagnostic_only'].get('restored_best'), 'Final training rule failed')
+        # Verify metadata/files and deserialized feature/model contracts, no inference.
+        art = _verify_checkpoint(root, plan, run, builder, feature_id, torch.device('cpu'))
+        del art
+        pred_path = path.parent / 'predictions_val.parquet'
+        require(fingerprint(pred_path) == result['predictions'], 'Prediction checksum/size mismatch')
+        df = pd.read_parquet(pred_path)
+        val = load_split(root, plan, sid)['comparison_val']
+        require(set(df.columns) == set(TABLES['predictions_val.parquet']), 'Prediction schema mismatch')
+        require(len(df) == len(val) and df.row_id.is_unique and set(df.row_id) == set(val.row_id),
+                'Prediction validation row identity mismatch')
+        require(set(df.run_id) == {rid} and set(df.split_id) == {sid}, 'Prediction run/split mismatch')
+        ordered = df.set_index('row_id').loc[val.row_id].reset_index()
+        require(ordered.true_L3.tolist() == val.L3.tolist(), 'Prediction labels not from active comparison validation')
+        labels = plan['label_maps']['L3']
+        require(ordered.predicted_L3.isin(labels).all(), 'Unknown/null predicted label')
+        require(ordered.empty_features.isin([True, False]).all(), 'Invalid empty feature flag')
+        ids = {label: i for i, label in enumerate(labels)}
+        truth = ordered.true_L3.map(ids).to_numpy(np.int64)
+        pred = ordered.predicted_L3.map(ids).to_numpy(np.int64)
+        f1, acc, pc = _l3_val_metrics(truth, pred, labels)
+        require(f1 == result['metrics']['l3_macro_f1'] and acc == result['metrics']['l3_accuracy']
+                and pc == result['per_class'], 'Saved predictions do not reproduce unrounded metrics')
+        row.update(l3_macro_f1=f1, l3_accuracy=acc, model_bytes=(path.parent / 'checkpoint' / 'model.pt').stat().st_size)
+        all_runs.append(row)
+        all_pc.extend({**c, **{k: run[k] for k in ('run_id', 'split_id', 'method', 'seed')}} for c in pc)
+        all_preds.extend(ordered.to_dict('records'))
+    return all_runs, all_pc, all_preds, selections
+
+
+def confusion_counts(predictions: pd.DataFrame) -> dict:
+    wrong = predictions[predictions.true_L3 != predictions.predicted_L3].copy()
+    # Missing labels represent an explicit error in diagnostic inputs, never a dropped group.
+    sentinel = '__S07_MISSING_PREDICTION__'
+    wrong['predicted_L3'] = wrong.predicted_L3.fillna(sentinel)
+    counts = wrong.groupby(['true_L3', 'predicted_L3'], dropna=False).size().to_dict()
+    return {(true, None if predicted == sentinel else predicted): int(count)
+            for (true, predicted), count in counts.items()}
+
+
+def _comparison_tables(runs: pd.DataFrame, pcs: pd.DataFrame, predictions: pd.DataFrame) -> tuple:
+    successful_ids = set(runs.loc[runs.status == 'success', 'run_id'])
+    pcs = pcs[pcs.run_id.isin(successful_ids)].copy()
+    predictions = predictions[predictions.run_id.isin(successful_ids)].copy()
+    baseline = runs[(runs.method == 'none') & (runs.status == 'success')].set_index(['split_id', 'seed'])
+    for i, row in runs[runs.status == 'success'].iterrows():
+        key = (row.split_id, row.seed)
+        if key in baseline.index:
+            runs.at[i, 'delta_macro_f1'] = row.l3_macro_f1 - baseline.loc[key, 'l3_macro_f1']
+            runs.at[i, 'delta_accuracy'] = row.l3_accuracy - baseline.loc[key, 'l3_accuracy']
+    for name in ('precision', 'recall', 'f1'):
+        pcs['delta_' + name] = np.nan
+    if not pcs.empty:
+        base_pc = pcs[pcs.method == 'none'].set_index(['split_id', 'seed', 'L3'])
+        for i, row in pcs.iterrows():
+            key = (row.split_id, row.seed, row.L3)
+            if key in base_pc.index:
+                require(row.support == base_pc.loc[key, 'support'], 'Paired per-class support mismatch')
+                for name in ('precision', 'recall', 'f1'):
+                    pcs.at[i, 'delta_' + name] = row[name] - base_pc.loc[key, name]
+    bins, confusion = [], []
+    for _, run in runs[runs.status == 'success'].iterrows():
+        g = pcs[pcs.run_id == run.run_id].copy()
+        g['bin'] = g.support.map(support_bin)
+        for b in BINS:
+            subset = g[g.bin == b]
+            bins.append({**{k: run[k] for k in ('run_id', 'split_id', 'method', 'seed')},
+                         'bin': b, 'class_count': len(subset), 'example_support': int(subset.support.sum()),
+                         'macro_f1': float(subset.f1.mean()) if len(subset) else None,
+                         'delta_macro_f1': float(subset.delta_f1.mean())
+                         if len(subset) and subset.delta_f1.notna().all() else None})
+        pred = predictions[predictions.run_id == run.run_id]
+        wrong = confusion_counts(pred)
+        base_run = baseline.loc[(run.split_id, run.seed), 'run_id'] if (run.split_id, run.seed) in baseline.index else None
+        bp = predictions[predictions.run_id == base_run]
+        base_wrong = confusion_counts(bp)
+        for true, predicted in sorted(set(wrong) | set(base_wrong), key=lambda pair: tuple(str(x) for x in pair)):
+            count = int(wrong.get((true, predicted), 0))
+            bc = int(base_wrong.get((true, predicted), 0)) if base_run is not None else None
+            confusion.append({**{k: run[k] for k in ('run_id', 'split_id', 'method', 'seed')},
+                              'true_L3': true, 'predicted_L3': predicted, 'count': count,
+                              'baseline_count': bc, 'delta_count': count - bc if bc is not None else None})
+    return (runs, pcs, pd.DataFrame(bins, columns=TABLES['support_bins.csv']),
+            pd.DataFrame(confusion, columns=TABLES['confusion_pairs.csv']))
+
+
+def build_report(output_root: str | Path, report_dir: str | Path | None = None) -> dict:
+    """Read verified saved results only; never train, refit, or run inference."""
+    root = Path(output_root)
+    plan = load_plan(root)
+    rows, pc, preds, selections = _verified_results(root, plan)
+    runs = pd.DataFrame(rows, columns=TABLES['runs.csv'])
+    per_class = pd.DataFrame(pc, columns=TABLES['per_class.parquet'])
+    predictions = pd.DataFrame(preds, columns=TABLES['predictions_val.parquet'])
+    runs, per_class, bins, confusion = _comparison_tables(runs, per_class, predictions)
+    stability = []
+    for method in METHODS[1:]:
+        for a, b in itertools.combinations(plan['splits'], 2):
+            for branch in ('char', 'word'):
+                row = {'method': method, 'split_a': a, 'split_b': b, 'branch': branch,
+                       'status': 'pending', 'reason': 'Both split-local verified refits required'}
+                if (a, method) in selections and (b, method) in selections:
+                    fa = getattr(selections[(a, method)].builder, branch).get_feature_names_out()
+                    fb = getattr(selections[(b, method)].builder, branch).get_feature_names_out()
+                    row.update(status='success', reason=None, **feature_jaccard(fa, fb))
+                stability.append(row)
+    stages = []
+    events = root / 'stages'
+    finished = read_sealed(root / 'execution_finished.json') if (root / 'execution_finished.json').exists() else None
+    if events.exists():
+        if finished:
+            require(tree(events) == finished['stages'], 'Stage event ledger changed')
+        for path in sorted(events.glob('*.json')):
+            event = read_sealed(path)
+            require(event['plan_sha256'] == digest(plan), 'Stage/plan provenance mismatch')
+            require(event['split_id'] in plan['splits'] and event['method'] in METHODS, 'Invalid stage identity')
+            require(event['charge_count'] == 1, 'Shared/run stage must be charged exactly once')
+            require(event['scope'] in ('shared', 'run') and event['status'] in ('success', 'failed', 'skipped', 'pending'),
+                    'Invalid stage scope/status')
+            event['event_id'] = path.stem
+            stages.append(event)
+    elif finished:
+        require(not finished['stages'], 'Missing stage ledger')
+    stages_df = pd.DataFrame(stages, columns=TABLES['stages.csv'])
+    if not stages_df.empty:
+        require(not stages_df.duplicated(['split_id', 'method', 'run_id', 'scope', 'stage']).any(),
+                'Duplicate stage cost event')
+    summary = paired_summary(runs)
+    tables = {'runs.csv': runs, 'per_class.parquet': per_class, 'predictions_val.parquet': predictions,
+              'support_bins.csv': bins, 'confusion_pairs.csv': confusion,
+              'feature_stability.csv': pd.DataFrame(stability, columns=TABLES['feature_stability.csv']),
+              'stages.csv': stages_df, 'paired_summary.csv': summary}
+    report = fresh_root(report_dir or root / 'report')
+    for name, data in tables.items():
+        if name.endswith('.parquet'):
+            data.to_parquet(report / name, index=False)
+        else:
+            data.to_csv(report / name, index=False, float_format='%.17g')
+    statuses = {status: int((runs.status == status).sum()) for status in ('success', 'failed', 'skipped', 'pending')}
+    manifest = {'schema': 's07-report@1', 'created_at': now(), 'plan': plan, 'plan_sha256': digest(plan),
+                'source_fingerprints': plan['sources'], 'statuses': statuses,
+                'complete': statuses['success'] == 36, 'study_kind': plan['study_kind'],
+                'warnings': WARNINGS + ([] if statuses['success'] == 36 else ['PARTIAL experiment; no completed-study conclusion.']),
+                'measurement_sources': {'S07': 'Verified saved predictions and stage event measurements',
+                                        'S06': 'Frozen protocol and full unsupervised TF-IDF references only; no reused quality metrics'},
+                'formulas': {'paired_delta': 'unrounded method(split,seed) minus none(split,seed)',
+                             'seed_variability': 'population SD of three paired deltas within a split (ddof=0)',
+                             'split_variability': 'population SD of complete three-seed split mean deltas (ddof=0)',
+                             'support_bins': 'each L3 counted once per bin; example_support=sum(validation support)',
+                             'jaccard': 'branch feature-name intersection / union, same full vocabulary provenance',
+                             'cost': 'sum distinct event seconds once; shared ranking/refit not multiplied by seeds',
+                             'attribution_checkpoint': 'full seed42 training charged once to full comparator; GxI standalone scenario adds that same cost, incremental comparison adds 0'},
+                'tables': {name: fingerprint(report / name) for name in TABLES}}
+    write_new(report / 'manifest.json', sealed(manifest))
+    render_report(report, report / 'report.html')
+    return {'report_dir': str(report), 'statuses': statuses, 'complete': manifest['complete']}
+
+
+def load_report_tables(report_dir: str | Path) -> tuple[dict, dict]:
+    """Only this directory is read; no source/output-root resolution in rebuild."""
+    report = Path(report_dir)
+    manifest = read_sealed(report / 'manifest.json')
+    require(manifest.get('schema') == 's07-report@1', 'Unknown report schema')
+    require(set(manifest['tables']) == set(TABLES), 'Incomplete table-only report manifest')
+    require(manifest['plan_sha256'] == digest(manifest['plan']), 'Saved report plan mismatch')
+    tables = {}
+    for name, columns in TABLES.items():
+        require(fingerprint(report / name) == manifest['tables'][name], f'Report table changed: {name}')
+        data = pd.read_parquet(report / name) if name.endswith('.parquet') else pd.read_csv(
+            report / name, float_precision='round_trip', keep_default_na=True)
+        require(data.columns.tolist() == columns, f'Report column schema changed: {name}')
+        tables[name] = data
+    runs = tables['runs.csv']
+    require(len(runs) == 36 and runs.run_id.is_unique
+            and set(runs.run_id) == {r['run_id'] for r in manifest['plan']['runs']}, 'Report run matrix mismatch')
+    require(runs.status.isin(['success', 'failed', 'skipped', 'pending']).all(), 'Invalid table statuses')
+    require(manifest['statuses'] == {s: int((runs.status == s).sum()) for s in manifest['statuses']},
+            'Table/manifest statuses disagree')
+    require(manifest['complete'] == bool((runs.status == 'success').all()), 'Incorrect completeness claim')
+    validate_report_tables(manifest, tables)
+    return manifest, tables
+
+
+def validate_report_tables(manifest: dict, tables: dict) -> None:
+    """Semantic cross-checks using table-only identities and the saved immutable plan."""
+    plan, runs = manifest['plan'], tables['runs.csv']
+    index = runs.set_index('run_id')
+    labels = plan['label_maps']['L3']
+    require(set(manifest['statuses']) == {'success', 'failed', 'skipped', 'pending'}, 'Incomplete status universe')
+    for planned in plan['runs']:
+        row = index.loc[planned['run_id']]
+        require(all(row[k] == planned[k] for k in ('split_id', 'method', 'seed', 'n_char', 'n_word')),
+                'Report run/config identity mismatch')
+        if row.status != 'success':
+            require(pd.isna(row.l3_macro_f1) and pd.isna(row.l3_accuracy)
+                    and pd.isna(row.delta_macro_f1) and pd.isna(row.delta_accuracy),
+                    'Non-success table row contains quality metrics')
+            require(pd.notna(row.reason), 'Non-success report row needs reason')
+    pcs, predictions = tables['per_class.parquet'], tables['predictions_val.parquet']
+    successful = set(runs.loc[runs.status == 'success', 'run_id'])
+    require(set(pcs.run_id) <= successful and set(predictions.run_id) <= successful,
+            'Predictions/per-class rows belong to non-success/unplanned run')
+    for rid in successful:
+        row = index.loc[rid]
+        pred = predictions[predictions.run_id == rid]
+        per_class = pcs[pcs.run_id == rid]
+        split_spec = plan['splits'][row.split_id]['parts']['comparison_val']
+        require(len(pred) == split_spec['rows'] and pred.row_id.is_unique
+                and digest(sorted(pred.row_id.astype(int).tolist())) == split_spec['row_ids_sha256'],
+                'Table prediction row identities differ from frozen validation')
+        require(digest(pred[['row_id', 'true_L3']].sort_values('row_id').values.tolist())
+                == split_spec['row_l3_sha256'], 'Table true labels differ from frozen validation')
+        require(set(pred.split_id) == {row.split_id} and pred.predicted_L3.isin(labels).all(),
+                'Table prediction split/label map mismatch')
+        require(len(per_class) == len(labels) and per_class.L3.is_unique and set(per_class.L3) == set(labels),
+                'Table per-class universe mismatch')
+        ids = {label: i for i, label in enumerate(labels)}
+        f1, accuracy, expected_pc = _l3_val_metrics(pred.true_L3.map(ids).to_numpy(np.int64),
+                                                  pred.predicted_L3.map(ids).to_numpy(np.int64), labels)
+        require(f1 == row.l3_macro_f1 and accuracy == row.l3_accuracy, 'Table prediction metrics mismatch')
+        expected_pc = pd.DataFrame(expected_pc).set_index('L3')
+        actual_pc = per_class.set_index('L3').loc[labels]
+        require(np.array_equal(actual_pc.support, expected_pc.support), 'Table per-class support mismatch')
+        for metric in ('precision', 'recall', 'f1'):
+            require(np.allclose(actual_pc[metric], expected_pc[metric], rtol=0, atol=1e-15),
+                    'Table per-class metric mismatch')
+    expected_runs, expected_pc, expected_bins, expected_confusion = _comparison_tables(
+        runs.copy(), pcs.copy(), predictions.copy())
+    def equal_frame(actual, expected, columns):
+        require(len(actual) == len(expected), 'Inconsistent table row count')
+        left, right = actual[columns].reset_index(drop=True).copy(), expected[columns].reset_index(drop=True).copy()
+        for column in columns:
+            if right[column].dtype == object:
+                left[column] = left[column].where(left[column].notna(), np.nan)
+                right[column] = right[column].where(right[column].notna(), np.nan)
+        try:
+            pd.testing.assert_frame_equal(left, right,
+                                          check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-15)
+        except AssertionError as exc:
+            require(False, 'Inconsistent derived report tables: ' + str(exc)[:200])
+    equal_frame(runs, expected_runs, ['delta_macro_f1', 'delta_accuracy'])
+    equal_frame(pcs, expected_pc, ['delta_precision', 'delta_recall', 'delta_f1'])
+    equal_frame(tables['support_bins.csv'], expected_bins, TABLES['support_bins.csv'])
+    equal_frame(tables['confusion_pairs.csv'], expected_confusion, TABLES['confusion_pairs.csv'])
+    equal_frame(tables['paired_summary.csv'], paired_summary(runs), TABLES['paired_summary.csv'])
+    stability = tables['feature_stability.csv']
+    expected_keys = {(m, a, b, branch) for m in METHODS[1:] for a, b in itertools.combinations(plan['splits'], 2)
+                     for branch in ('char', 'word')}
+    require(len(stability) == 18 and set(stability[['method', 'split_a', 'split_b', 'branch']].itertuples(
+        index=False, name=None)) == expected_keys, 'Incomplete feature stability table')
+    for _, row in stability[stability.status == 'success'].iterrows():
+        require(row.union == row.size_a + row.size_b - row.intersection
+                and 0 <= row.intersection <= min(row.size_a, row.size_b)
+                and abs(row.jaccard - row.intersection / row.union) <= 1e-15, 'Invalid Jaccard table values')
+    stages = tables['stages.csv']
+    require(stages.empty or (stages.charge_count == 1).all(), 'Repeated stage cost charging')
+
+
+def render_report(report_dir: str | Path, output_html: str | Path) -> str:
+    """Rebuild from manifest + eight tables only; refuses to overwrite HTML."""
+    manifest, tables = load_report_tables(report_dir)
+    esc = lambda value: html.escape(str(value))
+    pieces = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>S07 robustness study</title>',
+              '<style>body{font:15px system-ui;margin:2em;color:#182334}table{border-collapse:collapse;font-size:12px}'
+              'td,th{border:1px solid #ccd;padding:5px;text-align:left}section{overflow:auto;margin:2em 0}'
+              '.warning{background:#fff2cc;padding:1em}code{word-break:break-all}</style>',
+              '<body><h1>S07 development-split robustness</h1>',
+              f'<p class="warning">{"COMPLETE" if manifest["complete"] else "PARTIAL"}; '
+              f'{esc(manifest["study_kind"])}; {esc(manifest["statuses"])}</p>',
+              '<ul>' + ''.join(f'<li>{esc(w)}</li>' for w in manifest['warnings']) + '</ul>',
+              '<p>All effects are paired by split and model seed. Seed variation and split-mean variation '
+              'are shown separately; no confidence interval assumes nine independent samples. '
+              'Support-bin class counts and example counts are displayed, including empty bins.</p>']
+    for name in ('paired_summary.csv', 'runs.csv', 'support_bins.csv', 'feature_stability.csv'):
+        pieces.append(f'<section><h2>{esc(name)}</h2>{tables[name].to_html(index=False, escape=True)}</section>')
+    comparisons = []
+    run_table = tables['runs.csv']
+    successes = run_table[run_table.status == 'success'].set_index(['split_id', 'seed', 'method'])
+    for sid in run_table.split_id.unique():
+        for other in ('global', 'class_aware'):
+            differences = []
+            for seed in MODEL_SEEDS:
+                gx, alternative = (sid, seed, 'gradient_x_input'), (sid, seed, other)
+                if gx in successes.index and alternative in successes.index:
+                    differences.append(successes.loc[gx, 'l3_macro_f1'] - successes.loc[alternative, 'l3_macro_f1'])
+            comparisons.append({'split_id': sid, 'contrast': 'gradient_x_input minus ' + other,
+                                'paired_seeds': len(differences),
+                                'mean_delta_macro_f1': np.mean(differences) if differences else None,
+                                'within_split_seed_std': np.std(differences) if differences else None})
+    pieces.append('<section><h2>Incremental GxI effects versus Chi²</h2>' + pd.DataFrame(comparisons).to_html(
+        index=False, escape=True) + '<p>Compare these split-local effects with the checkpoint allocation and measured '
+        'shared ranking/refit costs below. There is no predeclared universal cost-benefit threshold.</p></section>')
+    pc = tables['per_class.parquet']
+    if not pc.empty:
+        extremes = pc[pc.method != 'none'].sort_values('delta_f1')
+        # Full per-class data remains embedded; this view highlights both tails.
+        pieces.append('<section><h2>Largest per-class losses and gains (validation support shown)</h2>' +
+                      pd.concat([extremes.head(30), extremes.tail(30)]).to_html(index=False, escape=True) + '</section>')
+    confusion = tables['confusion_pairs.csv']
+    pieces.append('<section><h2>Confusion changes</h2>' + confusion.sort_values(
+        'delta_count', ascending=False).head(60).to_html(index=False, escape=True) + '</section>')
+    stages = tables['stages.csv']
+    if not stages.empty:
+        costs = stages.groupby(['split_id', 'method', 'scope'], dropna=False).seconds.sum(min_count=1).reset_index()
+        pieces.append('<section><h2>Measured costs (distinct events charged once)</h2>' +
+                      costs.to_html(index=False, escape=True) + '</section>')
+        attribution_costs = []
+        for sid, g in stages.groupby('split_id'):
+            reference = g[(g.method == 'none') & g.stage.isin(['train', 'train_transform']) & g.run_id.astype('string').str.endswith('seed_42', na=False)]
+            checkpoint_seconds = reference.seconds.sum(min_count=1)
+            own = g[g.method == 'gradient_x_input'].seconds.sum(min_count=1)
+            attribution_costs.append({'split_id': sid, 'full_seed42_checkpoint_training_and_transform_seconds': checkpoint_seconds,
+                                      'gxi_incremental_measured_seconds': own,
+                                      'gxi_standalone_scenario_seconds': own + checkpoint_seconds})
+        pieces.append('<section><h2>GxI checkpoint cost allocation</h2><p>Full seed42 is both comparator and '
+                      'attribution checkpoint. Its training and input transform are charged once in measured totals; the standalone '
+                      'GxI scenario includes it explicitly and is not added to measured totals.</p>' +
+                      pd.DataFrame(attribution_costs).to_html(index=False, escape=True) + '</section>')
+    pieces.append('<section><h2>Stage resources (unavailable VRAM = null)</h2>' + stages.to_html(
+        index=False, escape=True, na_rep='null') + '</section>')
+    pieces.append('<section><h2>Interpretation</h2><p>A research recommendation requires a complete matrix, '
+                  'paired split effects, rare-class support and resource costs. Positive development deltas '
+                  'alone do not establish transferable or industrial gains. No S07 outcome is inferred from '
+                  'the historical S06 +0.0139 observation.</p></section>')
+    pieces.append('<section><h2>Provenance and formulas</h2><pre>' +
+                  esc(json.dumps({k: manifest[k] for k in ('plan_sha256', 'measurement_sources', 'formulas')},
+                                 ensure_ascii=False, indent=2)) + '</pre></section>')
+    # Self-contained full tables, safely encoded: no external assets/scripts/data requests.
+    embedded = {name: json.loads(df.to_json(orient='records', force_ascii=False, double_precision=15))
+                for name, df in tables.items()}
+    pieces.append('<script type="application/json" id="report-data">' + json.dumps(
+        embedded, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026') + '</script>')
+    pieces.append('</body></html>')
+    output = Path(output_html)
+    output.parent.mkdir(parents=True, exist_ok=True)
+    with output.open('x', encoding='utf-8') as f:
+        f.write('\n'.join(pieces))
+    return str(output)
diff --git a/scripts/run_s07.py b/scripts/run_s07.py
new file mode 100644
index 0000000..600698c
--- /dev/null
+++ b/scripts/run_s07.py
@@ -0,0 +1,57 @@
+"""S07: preflight, immutable planning, single-use execution, report and table-only rebuild."""
+from __future__ import annotations
+
+import argparse
+import json
+from pathlib import Path
+import sys
+
+sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
+
+
+def main() -> int:
+    parser = argparse.ArgumentParser(description=__doc__)
+    subs = parser.add_subparsers(dest='command', required=True)
+    for name in ('preflight', 'plan'):
+        p = subs.add_parser(name)
+        p.add_argument('--output-root', required=True, help='Fresh directory; never overwritten')
+        p.add_argument('--device', default='cpu', choices=('cpu', 'cuda', 'auto'))
+        p.add_argument('--num-threads', type=int, default=None)
+    p = subs.add_parser('execute')
+    p.add_argument('--output-root', required=True)
+    p = subs.add_parser('report')
+    p.add_argument('--output-root', required=True)
+    p.add_argument('--report-dir', default=None, help='Fresh report directory')
+    p = subs.add_parser('rebuild')
+    p.add_argument('--report-dir', required=True, help='Only saved report tables and manifest needed')
+    p.add_argument('--output-html', required=True, help='New HTML file')
+    args = parser.parse_args()
+    try:
+        if args.command == 'rebuild':
+            from kkt_clf.s07_report import render_report
+            result = {'html': render_report(args.report_dir, args.output_html)}
+        elif args.command == 'report':
+            from kkt_clf.s07_report import build_report
+            result = build_report(args.output_root, args.report_dir)
+        else:
+            from kkt_clf.s07 import S07Config, execute, preflight, write_plan
+            if args.command == 'execute':
+                result = execute(args.output_root)
+            else:
+                cfg = S07Config(device=args.device, num_threads=args.num_threads)
+                result = (preflight if args.command == 'preflight' else write_plan)(cfg, args.output_root)
+        # Compact CLI output; complete evidence lives on disk.
+        if args.command == 'plan':
+            result = {'output_root': args.output_root, 'runs': len(result['runs']),
+                      'dependencies': len(result['dependencies'])}
+        elif args.command == 'preflight':
+            result = {'output_root': args.output_root, 'passed': result['passed'], 'error': result.get('error')}
+        print(json.dumps(result, ensure_ascii=False, indent=2))
+        return 2 if result.get('passed') is False or result.get('failed') or result.get('stopped_reason') else 0
+    except Exception as exc:
+        print(f'S07 gate rejected: {type(exc).__name__}: {exc}', file=sys.stderr)
+        return 2
+
+
+if __name__ == '__main__':
+    raise SystemExit(main())
diff --git a/tests/test_s07_integrity_independent.py b/tests/test_s07_integrity_independent.py
new file mode 100644
index 0000000..995a9b6
--- /dev/null
+++ b/tests/test_s07_integrity_independent.py
@@ -0,0 +1,318 @@
+"""Independent S07 fixtures: no production inputs, models, or sealed labels."""
+from __future__ import annotations
+
+import copy
+import dataclasses
+import hashlib
+import itertools
+import json
+from pathlib import Path
+import re
+
+import numpy as np
+import pandas as pd
+import pytest
+
+from kkt_clf import s07
+from kkt_clf.config import BranchConfig, FeatureConfig, LossConfig, ModelConfig, TrainConfig
+from kkt_clf.vectorizer import fit_and_save_vectorizer
+
+
+def _hash(path):
+    return hashlib.sha256(path.read_bytes()).hexdigest()
+
+
+def _json(path, obj):
+    path.write_text(json.dumps(obj, sort_keys=True), encoding='utf-8')
+
+
+def _seal(obj):
+    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
+    return {**obj, 'integrity_sha256': hashlib.sha256(raw.encode()).hexdigest()}
+
+
+def _maps():
+    return {'L1': ['root'], 'L2': ['branch'], 'L3': ['alpha', 'beta'],
+            'parent_l2_of_l3': [0, 0], 'parent_l1_of_l2': [0]}
+
+
+def _pool(n=10):
+    return pd.DataFrame([{'row_id': c * 100 + i, 'name': f'{label} item{chr(97+i)} fresh',
+                          'L1': 'root', 'L2': 'branch', 'L3': label}
+                         for c, label in enumerate(('alpha', 'beta')) for i in range(n)])
+
+
+@pytest.fixture
+def synthetic(tmp_path):
+    """Build original sources independently, with no S06 output reads."""
+    pool = _pool()
+    corpus = tmp_path / 'corpus.parquet'
+    pd.DataFrame({'name': pool.name.tolist() + ['extra assorted grocery token']}).to_parquet(corpus)
+    full = tmp_path / 'full'
+    feat = FeatureConfig(char=BranchConfig((2, 3), 1, 24, True),
+                         word=BranchConfig((1, 1), 1, 8, True), vectorizer_fit_rows=None)
+    fit_and_save_vectorizer(str(full), feat, str(corpus), max_rows=100, seed=42)
+    parts = {}
+    frames = {'outer_train': pool.groupby('L3', sort=False).head(8),
+              'comparison_val': pool.groupby('L3', sort=False).tail(2),
+              # Deliberately no label columns: structural verification must suffice.
+              'sealed_test': pd.DataFrame({'row_id': [900, 901], 'name': ['sealed one', 'sealed two']})}
+    for part, frame in frames.items():
+        path = tmp_path / f'{part}.parquet'
+        frame.to_parquet(path, index=False)
+        parts[part] = {'file': str(path), 'sha256': _hash(path), 'bytes': path.stat().st_size,
+                       'n_rows': len(frame)}
+    manifest = tmp_path / 'manifest.json'
+    _json(manifest, {'policy': {'min_count': 12}, 'parts': parts, 'label_maps': _maps()})
+    protocol = tmp_path / 'protocol.json'
+    _json(protocol, {'frozen': True, 'fixed_epochs': 1, 'chosen': {'fixed_epochs': 1},
+                     'full_tfidf_id': _hash(full / 'features.joblib'),
+                     'corpus_fingerprint': {'sha256': _hash(corpus)}, 'splits': parts,
+                     'resolved_sections': {'model': dataclasses.asdict(ModelConfig([4], [0.])),
+                                           'loss': dataclasses.asdict(LossConfig()),
+                                           'train': dataclasses.asdict(TrainConfig(epochs=1, scheduler='none',
+                                                                                 batch_size=8, num_threads=1))}})
+    pins = {key: _hash(path) for key, path in {'manifest': manifest, 'protocol': protocol,
+                                             'corpus': corpus, 'features': full / 'features.joblib'}.items()}
+    cfg = s07.S07Config(manifest=str(manifest), protocol=str(protocol), full_tfidf=str(full),
+                        corpus=str(corpus), fixture=True, fixture_pins=pins,
+                        fixture_full_counts=(24, 8), fixture_selected_counts=(6, 2), num_threads=1)
+    return cfg, tmp_path / 'study'
+
+
+def test_three_deterministic_splits_keep_under12_development_classes():
+    pool = _pool()
+    first, diagnostics = s07.split_development_pool(pool, _maps())
+    second, _ = s07.split_development_pool(pool.sample(frac=1, random_state=99), _maps())
+    assert set(first) == {'split_101', 'split_202', 'split_303'}
+    assert all(d['feasible'] for d in diagnostics)
+    assignments = []
+    for sid, parts in first.items():
+        assignments.append(tuple(parts['comparison_val'].row_id))
+        leaf_ids = []
+        groups = []
+        for part, minimum in [('train_fit', 6), ('train_monitor', 2), ('comparison_val', 2)]:
+            frame = parts[part]
+            assert frame.L3.value_counts().to_dict() == {'alpha': minimum, 'beta': minimum}
+            assert frame.equals(second[sid][part])
+            leaf_ids.extend(frame.row_id)
+            groups.append(set(re.sub(r'\s', '', x.lower()) for x in frame.name))
+        assert sorted(leaf_ids) == sorted(pool.row_id)
+        assert all(not a & b for a, b in itertools.combinations(groups, 2))
+        assert set(parts['outer_train'].row_id) == set(parts['train_fit'].row_id) | set(parts['train_monitor'].row_id)
+    assert len(set(assignments)) == 3
+
+
+def test_duplicate_normalized_names_stay_together():
+    pool = _pool(12)
+    for label in ('alpha', 'beta'):
+        ids = pool.index[pool.L3 == label][:2]
+        pool.loc[ids, 'name'] = [f'{label} DUP', f'  {label.upper()}\tD U P ']
+    splits, _ = s07.split_development_pool(pool, _maps())
+    assert len(splits) == 3
+    for parts in splits.values():
+        for label in ('alpha', 'beta'):
+            ids = set(pool[pool.L3 == label].head(2).row_id)
+            assert sum(ids <= set(parts[p].row_id) for p in ('train_fit', 'train_monitor', 'comparison_val')) == 1
+
+
+@pytest.mark.parametrize('sizes', [(9,), (8, 1, 1), (6, 4)])
+def test_proven_group_infeasibility_preserves_class_and_diagnostics(sizes):
+    pool = _pool(sum(sizes))
+    for label in ('alpha', 'beta'):
+        names = [f'{label} group{g}' for g, size in enumerate(sizes) for _ in range(size)]
+        pool.loc[pool.L3 == label, 'name'] = names
+    splits, diag = s07.split_development_pool(pool, _maps())
+    assert not splits
+    assert len(diag) == 3
+    for d in diag:
+        assert not d['feasible']
+        assert {f['L3'] for f in d['failures']} == {'alpha', 'beta'}
+        assert all(f['proven_infeasible'] for f in d['failures'])
+
+
+def test_exhausted_search_is_not_a_proof(monkeypatch):
+    monkeypatch.setattr(s07, 'assign_groups', lambda sizes, quota, minima, rng: (np.zeros(len(sizes), int), np.array([10, 0, 0])))
+    monkeypatch.setattr(s07, 'search_feasible_assignment', lambda sizes, minima: (None, False, 7))
+    splits, diag = s07.split_development_pool(_pool(), _maps())
+    assert not splits
+    assert all(not f['proven_infeasible'] and f['search_nodes'] == 7 for d in diag for f in d['failures'])
+
+
+@pytest.mark.parametrize('damage', ['name', 'label', 'overlap', 'missing', 'outer'])
+def test_partition_validation_rejects_corruption(damage):
+    pool = _pool()
+    splits, _ = s07.split_development_pool(pool, _maps())
+    parts = copy.deepcopy(splits['split_101'])
+    if damage == 'name':
+        parts['train_fit'].loc[0, 'name'] = 'substitution'
+    elif damage == 'label':
+        parts['train_fit'].loc[0, 'L1'] = 'unknown'
+    elif damage == 'overlap':
+        parts['comparison_val'] = pd.concat([parts['comparison_val'], parts['train_fit'].head(1)])
+    elif damage == 'missing':
+        parts['train_fit'] = parts['train_fit'].iloc[1:]
+    else:
+        parts['outer_train'] = parts['outer_train'].iloc[1:]
+    with pytest.raises(s07.S07Error):
+        s07.validate_partition(pool, parts, _maps())
+
+
+def test_sealed_structural_projection_and_no_training_during_plan(synthetic, monkeypatch):
+    cfg, root = synthetic
+    original = pd.read_parquet
+    calls = []
+    def guarded(path, *args, **kwargs):
+        if str(path).endswith('sealed_test.parquet'):
+            assert kwargs.get('columns') == ['row_id', 'name']
+            calls.append(path)
+        return original(path, *args, **kwargs)
+    def forbidden(*args, **kwargs):
+        pytest.fail('Planning must not train, rank, refit or predict')
+    monkeypatch.setattr(pd, 'read_parquet', guarded)
+    for name in ('predict', 'train_model', 'select_features', 'compute_margin_attributions', 'refit_selected_vectorizer'):
+        monkeypatch.setattr(s07, name, forbidden)
+    plan = s07.write_plan(cfg, root)
+    assert calls
+    assert plan['study_kind'] == 'synthetic'
+    assert not (root / 'runs').exists()
+    keys = {(r['split_id'], r['method'], r['seed']) for r in plan['runs']}
+    assert keys == set(itertools.product(('split_101', 'split_202', 'split_303'),
+                                        ('none', 'global', 'class_aware', 'gradient_x_input'), (42, 43, 44)))
+    assert len(plan['runs']) == len({r['run_id'] for r in plan['runs']}) == 36
+    assert {r['initial_status'] for r in plan['runs']} == {'pending'}
+    assert len(plan['dependencies']) == 9
+    for dep in plan['dependencies']:
+        sid, method = dep['split_id'], dep['method']
+        consumers = [r for r in plan['runs'] if r['feature_dependency'] == dep['dependency_id']]
+        assert {r['seed'] for r in consumers} == {42, 43, 44}
+        assert {r['split_id'] for r in consumers} == {sid}
+        assert dep['ranking_rows'] == plan['splits'][sid]['parts']['outer_train']['sha256']
+        if method == 'gradient_x_input':
+            assert dep['reference_checkpoint_run'] == f'{sid}__none__seed_42'
+            assert dep['reference_seed'] == 42
+    assert s07.load_plan(root) == json.loads(json.dumps(plan))
+
+
+@pytest.mark.parametrize('source', ['manifest', 'protocol', 'corpus', 'features'])
+def test_authoritative_source_pins_fail_closed(synthetic, source):
+    cfg, root = synthetic
+    s07.write_plan(cfg, root)
+    paths = {'manifest': cfg.manifest, 'protocol': cfg.protocol, 'corpus': cfg.corpus,
+             'features': str(Path(cfg.full_tfidf) / 'features.joblib')}
+    with open(paths[source], 'ab') as f:
+        f.write(b'changed')
+    with pytest.raises(s07.S07Error):
+        s07.load_plan(root)
+
+
+@pytest.mark.parametrize('field', ['runs', 'training', 'label_maps', 'dependencies', 'config'])
+def test_unsigned_plan_mutation_fails_closed(synthetic, field):
+    cfg, root = synthetic
+    s07.write_plan(cfg, root)
+    path = root / 'plan.json'
+    plan = json.loads(path.read_text())
+    plan[field] = []
+    _json(path, plan)
+    with pytest.raises(s07.S07Error):
+        s07.load_plan(root)
+
+
+@pytest.mark.parametrize('damage', ['reference_seed', 'ranking_rows', 'cross_split', 'duplicate_dependency'])
+def test_semantically_invalid_dependency_fails_even_with_consistent_seal(synthetic, damage):
+    cfg, root = synthetic
+    plan = s07.write_plan(cfg, root)
+    dep = next(d for d in plan['dependencies'] if d['method'] == 'gradient_x_input')
+    if damage == 'reference_seed':
+        dep['reference_seed'] = 44
+        dep['reference_checkpoint_run'] = f"{dep['split_id']}__none__seed_44"
+    elif damage == 'ranking_rows':
+        dep['ranking_rows'] = plan['splits'][dep['split_id']]['parts']['comparison_val']['sha256']
+    elif damage == 'cross_split':
+        dep['reference_checkpoint_run'] = 'split_202__none__seed_42'
+    else:
+        plan['dependencies'][0] = copy.deepcopy(plan['dependencies'][1])
+    _json(root / 'plan.json', _seal(plan))
+    with pytest.raises(s07.S07Error):
+        s07.load_plan(root)
+
+
+@pytest.mark.parametrize('target', ['split', 'supports'])
+def test_saved_partition_files_are_immutable(synthetic, target):
+    cfg, root = synthetic
+    plan = s07.write_plan(cfg, root)
+    split = plan['splits']['split_101']
+    spec = split['parts']['train_fit'] if target == 'split' else split['supports']
+    with (root / spec['file']).open('ab') as f:
+        f.write(b'tampered')
+    with pytest.raises(s07.S07Error):
+        s07.load_plan(root)
+
+
+def test_populated_output_and_resume_are_rejected_without_changes(synthetic, monkeypatch):
+    cfg, root = synthetic
+    s07.write_plan(cfg, root)
+    sentinel = root / 'user.txt'
+    sentinel.write_text('preserve me')
+    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
+    with pytest.raises((FileExistsError, s07.S07Error)):
+        s07.write_plan(cfg, root)
+    (root / 'execution_started.json').write_text('{}')
+    monkeypatch.setattr(s07, 'train_model', lambda *a, **k: pytest.fail('unsafe resume trained'))
+    with pytest.raises(s07.S07Error, match='[Rr]esume'):
+        s07.execute(root)
+    assert all((root / path).read_bytes() == data for path, data in before.items())
+
+
+def test_production_profile_cannot_override_pins_or_budgets():
+    cfg = s07.S07Config(fixture_full_counts=(24, 8), fixture_selected_counts=(6, 2))
+    cfg.validate()
+    assert cfg.full_counts == (300000, 100000)
+    assert cfg.selected_counts == (75000, 25000)
+    with pytest.raises(s07.S07Error):
+        dataclasses.replace(cfg, fixture_pins={}).validate()
+
+
+@pytest.mark.parametrize('damage', ['sealed_id', 'sealed_group', 'label_map', 'policy', 'row_hash', 'epochs'])
+def test_invalid_references_rejected_even_when_source_files_are_repinned(synthetic, damage):
+    cfg, root = synthetic
+    manifest_path = Path(cfg.manifest)
+    protocol_path = Path(cfg.protocol)
+    manifest = json.loads(manifest_path.read_text())
+    protocol = json.loads(protocol_path.read_text())
+    if damage in ('sealed_id', 'sealed_group'):
+        path = Path(manifest['parts']['sealed_test']['file'])
+        frame = pd.read_parquet(path)
+        pool = _pool()
+        frame.loc[0, 'row_id' if damage == 'sealed_id' else 'name'] = (
+            int(pool.row_id.iloc[0]) if damage == 'sealed_id' else pool.name.iloc[0].upper().replace(' ', '\t'))
+        frame.to_parquet(path, index=False)
+        manifest['parts']['sealed_test'].update(sha256=_hash(path), bytes=path.stat().st_size)
+        protocol['splits'] = manifest['parts']
+    elif damage == 'label_map':
+        manifest['label_maps']['L3'].append('silently_missing_class')
+    elif damage == 'policy':
+        manifest['policy']['min_count'] = 10
+    elif damage == 'row_hash':
+        manifest['parts']['outer_train']['sha256'] = '0' * 64
+        protocol['splits'] = manifest['parts']
+    else:
+        protocol['chosen']['fixed_epochs'] = 2
+    _json(manifest_path, manifest)
+    _json(protocol_path, protocol)
+    pins = {**cfg.fixture_pins, 'manifest': _hash(manifest_path), 'protocol': _hash(protocol_path)}
+    with pytest.raises(s07.S07Error):
+        s07.write_plan(dataclasses.replace(cfg, fixture_pins=pins), root)
+    assert not root.exists()
+
+
+def test_failed_preflight_persists_honest_zero_workload(synthetic):
+    cfg, root = synthetic
+    cfg = dataclasses.replace(cfg, fixture_pins={**cfg.fixture_pins, 'corpus': '0' * 64})
+    audit = s07.preflight(cfg, root)
+    assert not audit['passed']
+    assert audit['workload_runs'] == 0 and audit['sealed_labels_read'] is False
+    assert audit['sealed_inference'] is False
+    assert 'SHA256' in audit['error']
+    assert (root / 'preflight.json').is_file()
+    assert not (root / 'runs').exists()
diff --git a/tests/test_s07_report_independent.py b/tests/test_s07_report_independent.py
new file mode 100644
index 0000000..96829dc
--- /dev/null
+++ b/tests/test_s07_report_independent.py
@@ -0,0 +1,439 @@
+"""S07 reporting oracles use hand-calculated data and synthetic execution only."""
+import itertools
+import json
+from pathlib import Path
+import shutil
+from types import SimpleNamespace
+
+import numpy as np
+import pandas as pd
+import pytest
+
+from kkt_clf import s07, s07_report as report
+from kkt_clf.vectorizer import load_vectorizer_artifact
+from tests.test_s07_integrity_independent import synthetic, _json, _seal
+
+
+def _tables():
+    runs, pcs, preds = [], [], []
+    for split, seed, method in itertools.product(range(3), (42, 43, 44), s07.METHODS):
+        baseline = .612345678912345 + split * .01 + (seed - 42) * .001
+        effect = 0 if method == 'none' else (.01, -.02, .04)[split] + (seed - 43) * .003
+        rid = f'split_{split}__{method}__{seed}'
+        runs.append({'run_id': rid, 'split_id': f'split_{split}', 'method': method,
+                     'seed': seed, 'status': 'success', 'reason': None,
+                     'l3_macro_f1': baseline + effect, 'l3_accuracy': baseline + effect / 2,
+                     'delta_macro_f1': None, 'delta_accuracy': None})
+        for label, support, f1 in [('rare_a', 2, .1), ('rare_b', 4, .9), ('common', 20, .7)]:
+            pcs.append({'run_id': rid, 'split_id': f'split_{split}', 'method': method, 'seed': seed,
+                        'L3': label, 'support': support, 'precision': f1, 'recall': f1,
+                        'f1': f1 + effect})
+        preds.extend([{'run_id': rid, 'split_id': f'split_{split}', 'row_id': i,
+                       'true_L3': 'rare_a', 'predicted_L3': v, 'empty_features': v is None}
+                      for i, v in enumerate(['rare_b', None])])
+    return pd.DataFrame(runs), pd.DataFrame(pcs), pd.DataFrame(preds)
+
+
+def test_unrounded_same_split_seed_pairing_and_separate_variability():
+    runs, pcs, predictions = _tables()
+    runs, _, _, _ = report._comparison_tables(runs, pcs, predictions)
+    full = runs[runs.method == 'none'].set_index(['split_id', 'seed'])
+    for _, row in runs.iterrows():
+        expected = row.l3_macro_f1 - full.loc[(row.split_id, row.seed), 'l3_macro_f1']
+        assert row.delta_macro_f1 == expected
+    summary = report.paired_summary(runs)
+    method = summary[summary.method == 'class_aware']
+    within = method[method.level == 'within_split'].sort_values('split_id')
+    assert within.paired_count.tolist() == [3, 3, 3]
+    np.testing.assert_allclose(within.mean_delta_macro_f1, [.01, -.02, .04], atol=1e-15)
+    np.testing.assert_allclose(within.within_split_seed_std, np.std([-.003, 0, .003]), atol=1e-15)
+    assert within.between_split_mean_std.isna().all()
+    across = method[method.level == 'across_split_means'].iloc[0]
+    assert across.split_count == 3
+    assert across.mean_delta_macro_f1 == pytest.approx(.01)
+    assert across.between_split_mean_std == pytest.approx(np.std([.01, -.02, .04]))
+    assert pd.isna(across.within_split_seed_std)
+    assert (across.gains, across.losses, across.ties) == (2, 1, 0)
+
+
+@pytest.mark.parametrize('status', ['failed', 'pending', 'skipped'])
+def test_incomplete_pairs_never_enter_complete_split_summary(status):
+    runs, pcs, preds = _tables()
+    runs.loc[(runs.split_id == 'split_1') & (runs.method == 'none') & (runs.seed == 43), 'status'] = status
+    runs, _, _, _ = report._comparison_tables(runs, pcs, preds)
+    selected = runs[(runs.split_id == 'split_1') & (runs.method == 'global') & (runs.seed == 43)].iloc[0]
+    assert pd.isna(selected.delta_macro_f1)
+    summary = report.paired_summary(runs)
+    across = summary[(summary.method == 'global') & (summary.level == 'across_split_means')].iloc[0]
+    assert across.split_count == 2
+    assert across.paired_count == 6
+
+
+def test_support_bins_count_classes_once_and_keep_empty_bins():
+    runs, pcs, preds = _tables()
+    _, _, bins, _ = report._comparison_tables(runs, pcs, preds)
+    rare = bins[(bins.run_id == runs.iloc[0].run_id) & (bins.bin == '2-4')].iloc[0]
+    assert rare.class_count == 2
+    assert rare.example_support == 6
+    assert rare.macro_f1 == pytest.approx(.5)  # Class mean, not example-weighted mean.
+    assert len(bins) == 36 * 4
+    empty = bins[bins.bin.isin(['5-9', '10-19'])]
+    assert (empty.class_count == 0).all() and (empty.example_support == 0).all()
+    assert empty.macro_f1.isna().all() and empty.delta_macro_f1.isna().all()
+
+
+def test_confusion_counts_empty_feature_prediction_as_an_error():
+    runs, pcs, preds = _tables()
+    _, _, _, confusion = report._comparison_tables(runs, pcs, preds)
+    first = confusion[confusion.run_id == runs.iloc[0].run_id]
+    assert first['count'].sum() == 2
+    assert first[first.predicted_L3.isna()]['count'].sum() == 1
+
+
+def test_feature_jaccard_uses_names_despite_reordered_local_columns():
+    a = np.array(['flour', 'milk', 'tea'])
+    b = np.array(['tea', 'milk', 'rice'])
+    overlap = report.feature_jaccard(a, b)
+    assert overlap == {'size_a': 3, 'size_b': 3, 'intersection': 2, 'union': 4, 'jaccard': .5}
+    assert report.feature_jaccard(a, a[::-1])['jaccard'] == 1
+    assert report.feature_jaccard(['x'], ['y'])['jaccard'] == 0
+
+
+def test_pending_matrix_reports_all_ten_files_and_rebuilds_with_only_tables(synthetic, tmp_path, monkeypatch):
+    cfg, root = synthetic
+    s07.write_plan(cfg, root)
+    result = report.build_report(root)
+    assert result['statuses'] == {'success': 0, 'failed': 0, 'skipped': 0, 'pending': 36}
+    assert result['complete'] is False
+    expected = {'report.html', 'manifest.json', 'runs.csv', 'per_class.parquet',
+                'predictions_val.parquet', 'support_bins.csv', 'confusion_pairs.csv',
+                'feature_stability.csv', 'stages.csv', 'paired_summary.csv'}
+    source = root / 'report'
+    assert {p.name for p in source.iterdir()} == expected
+    isolated = tmp_path / 'table_only'
+    isolated.mkdir()
+    for name in expected - {'report.html'}:
+        shutil.copyfile(source / name, isolated / name)
+    def forbidden(*a, **k):
+        pytest.fail('table-only rebuild accessed experiment API')
+    for name in ('load_plan', '_verify_checkpoint', 'verify_selected', 'load_split', '_verified_results'):
+        monkeypatch.setattr(report, name, forbidden)
+    original_open = Path.open
+    def guarded_open(path, *args, **kwargs):
+        assert path.resolve().is_relative_to(isolated.resolve()), f'external rebuild read: {path}'
+        return original_open(path, *args, **kwargs)
+    monkeypatch.setattr(Path, 'open', guarded_open)
+    report.render_report(isolated, isolated / 'rebuilt.html')
+    html = (isolated / 'rebuilt.html').read_text()
+    assert 'PARTIAL' in html and 'synthetic' in html
+    assert 'src="http' not in html and 'href="http' not in html
+    manifest, tables = report.load_report_tables(isolated)
+    assert len(tables['runs.csv']) == 36
+    assert manifest['complete'] is False
+
+
+@pytest.mark.parametrize('status', ['failed', 'skipped', 'pending'])
+def test_saved_non_success_status_keeps_reason_without_metrics(synthetic, status):
+    cfg, root = synthetic
+    s07.write_plan(cfg, root)
+    plan = s07.load_plan(root)
+    run = plan['runs'][0]
+    path = root / 'runs' / run['run_id'] / 'result.json'
+    path.parent.mkdir(parents=True)
+    _json(path, _seal({'status': status, 'reason': 'fixture interruption',
+                       'provenance': s07.run_provenance(plan, run, plan['sources']['features']['sha256'])}))
+    result = report.build_report(root)
+    rows = pd.read_csv(root / 'report' / 'runs.csv')
+    row = rows[rows.run_id == run['run_id']].iloc[0]
+    assert row.status == status and row.reason == 'fixture interruption'
+    assert pd.isna(row.l3_macro_f1)
+    assert not result['complete'] and result['statuses']['success'] == 0
+
+
+def test_shared_resource_event_charged_once_and_vram_unavailable(synthetic):
+    cfg, root = synthetic
+    s07.write_plan(cfg, root)
+    plan = s07.load_plan(root)
+    event = {'plan_sha256': s07.digest(plan), 'split_id': 'split_101', 'method': 'global', 'run_id': None, 'scope': 'shared',
+             'stage': 'refit', 'status': 'success', 'seconds': 7.25, 'peak_rss_bytes': 1024,
+             'peak_vram_bytes': None, 'charge_count': 1, 'cost_role': 'ordinary',
+             'checkpoint_incremental_training_seconds': None, 'measurement_note': 'synthetic'}
+    (root / 'stages').mkdir()
+    _json(root / 'stages' / 'one.json', _seal(event))
+    report.build_report(root)
+    stages = pd.read_csv(root / 'report' / 'stages.csv')
+    assert len(stages) == 1 and stages.seconds.sum() == 7.25
+    assert stages.peak_vram_bytes.isna().all()
+    _json(root / 'stages' / 'duplicate.json', _seal(event))
+    with pytest.raises(s07.S07Error, match='[Dd]uplicate'):
+        report.build_report(root, root / 'report_duplicate')
+
+
+def test_tiny_real_execution_refit_reload_and_selected_provenance(synthetic, tmp_path):
+    cfg, root = synthetic
+    s07.write_plan(cfg, root)
+    result = s07.execute(root)
+    assert len(result['success']) == 36, result
+    assert not result['failed'] and not result['pending']
+    plan = s07.load_plan(root)
+    full = load_vectorizer_artifact(cfg.full_tfidf)
+    assert len(list((root / 'shared').glob('*/*/dependency_receipt.json'))) == 9
+    corpus_names = pd.read_parquet(cfg.corpus).name.tolist()
+    for sid, method in itertools.product(plan['splits'], s07.METHODS[1:]):
+        loaded = s07.verify_selected(root, plan, sid, method, full)
+        X = loaded.builder.transform(corpus_names)
+        for branch, offset, count in [('char', 0, 6), ('word', 6, 2)]:
+            reduced = getattr(loaded.builder, branch)
+            original = getattr(full.builder, branch)
+            names = reduced.get_feature_names_out()
+            assert len(names) == count
+            indices = np.array([original.vocabulary_[name] for name in names])
+            np.testing.assert_allclose(reduced.idf_, original.idf_[indices], atol=1e-6)
+            norm = np.asarray(X[:, offset:offset+count].power(2).sum(axis=1)).ravel()
+            np.testing.assert_allclose(norm[norm > 0], 1, atol=1e-6)
+        path = root / 'shared' / sid / method / 'tfidf' / 'features.joblib'
+        before = path.read_bytes()
+        with path.open('ab') as f:
+            f.write(b'changed')
+        with pytest.raises(s07.S07Error):
+            s07.verify_selected(root, plan, sid, method, full)
+        # Restore only our generated fixture, to permit remaining independent checks.
+        path.write_bytes(before)
+    summary = report.build_report(root)
+    assert summary['complete']
+    runs = pd.read_csv(root / 'report' / 'runs.csv', float_precision='round_trip')
+    assert len(runs) == 36 and (runs.status == 'success').all()
+
+
+@pytest.mark.parametrize('method', ['global', 'class_aware', 'gradient_x_input'])
+def test_split_local_selection_refit_roundtrip_and_negative_inputs(synthetic, method, monkeypatch):
+    import torch
+    cfg, root = synthetic
+    s07.write_plan(cfg, root)
+    plan = s07.load_plan(root)
+    full = load_vectorizer_artifact(cfg.full_tfidf)
+    sid = 'split_101'
+    parts = s07.load_split(root, plan, sid)
+    if method == 'gradient_x_input':
+        run = next(r for r in plan['runs'] if r['run_id'] == f'{sid}__none__seed_42')
+        s07._train_one(root, plan, run, full.builder, plan['sources']['features']['sha256'], parts, torch.device('cpu'))
+    expected_y = np.array([plan['label_maps']['L3'].index(label) for label in parts['outer_train'].L3])
+    expected_X = full.builder.transform(parts['outer_train'].name.tolist())
+    called = []
+    selection_function = 'compute_margin_attributions' if method == 'gradient_x_input' else 'select_features'
+    original_function = getattr(s07, selection_function)
+    def checked_ranking(*args, **kwargs):
+        X, y = (args[1], args[2]) if method == 'gradient_x_input' else (args[0], args[1])
+        assert X.shape == expected_X.shape
+        assert (X != expected_X).nnz == 0
+        np.testing.assert_array_equal(y, expected_y)
+        called.append(True)
+        return original_function(*args, **kwargs)
+    monkeypatch.setattr(s07, selection_function, checked_ranking)
+    s07._prepare_selected(root, plan, sid, method, full, parts, torch.device('cpu'))
+    assert called == [True]
+    loaded = s07.verify_selected(root, plan, sid, method, full)
+    names = pd.read_parquet(cfg.corpus).name.tolist()
+    transformed = loaded.builder.transform(names)
+    for branch, start, count in [('char', 0, 6), ('word', 6, 2)]:
+        reduced, original = getattr(loaded.builder, branch), getattr(full.builder, branch)
+        indices = [original.vocabulary_[name] for name in reduced.get_feature_names_out()]
+        assert len(indices) == count
+        np.testing.assert_allclose(reduced.idf_, original.idf_[indices], atol=1e-6)
+        norm = np.asarray(transformed[:, start:start+count].power(2).sum(axis=1)).ravel()
+        np.testing.assert_allclose(norm[norm > 0], 1, atol=1e-6)
+        sliced = original.transform(names)[:, indices]
+        assert np.any(np.abs(np.asarray(sliced.power(2).sum(axis=1)).ravel() - norm) > 1e-5)
+    base = root / 'shared' / sid / method
+    for filename in ['selection/selection.json', 'selection/ratings.npy', 'tfidf/features.joblib']:
+        # Pick actual selection files via the immutable receipt rather than assuming a format.
+        target = (base / filename if filename.startswith('tfidf') else
+                  next((base / 'selection').rglob('*.npy' if filename.endswith('.npy') else '*.json')))
+        old = target.read_bytes()
+        target.write_bytes(old + b'alteration')
+        with pytest.raises(s07.S07Error):
+            s07.verify_selected(root, plan, sid, method, full)
+        target.write_bytes(old)
+    receipt_path = base / 'dependency_receipt.json'
+    receipt = json.loads(receipt_path.read_text())
+    receipt.pop('integrity_sha256')
+    receipt['provenance']['dependency']['split_id'] = 'split_202'
+    _json(receipt_path, _seal(receipt))
+    with pytest.raises(s07.S07Error):
+        s07.verify_selected(root, plan, sid, method, full)
+
+
+@pytest.fixture
+def baseline_result(synthetic):
+    import torch
+    cfg, root = synthetic
+    s07.write_plan(cfg, root)
+    plan = s07.load_plan(root)
+    full = load_vectorizer_artifact(cfg.full_tfidf)
+    run = plan['runs'][0]
+    parts = s07.load_split(root, plan, run['split_id'])
+    feature_id = plan['sources']['features']['sha256']
+    s07._train_one(root, plan, run, full.builder, feature_id, parts, torch.device('cpu'))
+    s07._evaluate_one(root, plan, run, full.builder, feature_id, parts, torch.device('cpu'))
+    return root, plan, run
+
+
+@pytest.mark.parametrize('damage', ['checkpoint', 'training', 'result_labels', 'prediction_hash',
+                                     'prediction_rows', 'prediction_labels', 'prediction_split', 'prediction_run'])
+def test_report_rejects_corrupt_checkpoint_training_or_predictions(baseline_result, damage):
+    root, plan, run = baseline_result
+    base = root / 'runs' / run['run_id']
+    result = json.loads((base / 'result.json').read_text())
+    result.pop('integrity_sha256')
+    if damage == 'checkpoint':
+        with (base / 'checkpoint' / 'model.pt').open('ab') as f:
+            f.write(b'changed')
+    elif damage == 'training':
+        with (base / 'training.json').open('ab') as f:
+            f.write(b'changed')
+    elif damage == 'result_labels':
+        result['provenance']['label_maps_sha256'] = '0' * 64
+        _json(base / 'result.json', _seal(result))
+    else:
+        path = base / 'predictions_val.parquet'
+        frame = pd.read_parquet(path)
+        if damage == 'prediction_rows':
+            frame.loc[0, 'row_id'] = 900  # Sealed structural identity must never be accepted as val.
+        elif damage == 'prediction_labels':
+            frame.loc[0, 'true_L3'] = 'beta' if frame.loc[0, 'true_L3'] == 'alpha' else 'alpha'
+        elif damage == 'prediction_split':
+            frame.loc[0, 'split_id'] = 'split_202'
+        else:
+            frame.loc[0, 'run_id'] = 'unknown_run'
+        frame.to_parquet(path, index=False)
+        if damage != 'prediction_hash':
+            result['predictions'] = s07.fingerprint(path)
+            _json(base / 'result.json', _seal(result))
+    with pytest.raises(s07.S07Error):
+        report.build_report(root)
+
+
+def test_verified_none_baseline_metrics_match_saved_predictions(baseline_result):
+    from sklearn.metrics import accuracy_score, f1_score
+    root, plan, run = baseline_result
+    outcome = report.build_report(root)
+    assert outcome['statuses']['success'] == 1 and outcome['statuses']['pending'] == 35
+    pred = pd.read_parquet(root / 'runs' / run['run_id'] / 'predictions_val.parquet')
+    table = pd.read_csv(root / 'report' / 'runs.csv', float_precision='round_trip')
+    row = table[table.run_id == run['run_id']].iloc[0]
+    assert row.l3_accuracy == accuracy_score(pred.true_L3, pred.predicted_L3)
+    assert row.l3_macro_f1 == f1_score(pred.true_L3, pred.predicted_L3, labels=['alpha', 'beta'], average='macro', zero_division=0)
+    assert row.delta_macro_f1 == 0
+
+
+def test_emitted_stability_table_resolves_feature_names(synthetic, monkeypatch):
+    cfg, root = synthetic
+    s07.write_plan(cfg, root)
+    original = report._verified_results
+    def synthetic_selections(root, plan):
+        rows, pcs, preds, _ = original(root, plan)
+        selections = {}
+        for sid in plan['splits']:
+            for method in s07.METHODS[1:]:
+                chars = {'split_101': ['a', 'b', 'c'], 'split_202': ['c', 'b', 'd'],
+                         'split_303': ['d', 'e', 'f']}[sid]
+                words = {'split_101': ['one', 'two'], 'split_202': ['two', 'one'],
+                         'split_303': ['three', 'four']}[sid]
+                # All objects use dense local 0..N columns; names determine actual overlap.
+                selections[(sid, method)] = SimpleNamespace(builder=SimpleNamespace(
+                    char=SimpleNamespace(get_feature_names_out=lambda names=chars: np.array(names)),
+                    word=SimpleNamespace(get_feature_names_out=lambda names=words: np.array(names))))
+        return rows, pcs, preds, selections
+    monkeypatch.setattr(report, '_verified_results', synthetic_selections)
+    report.build_report(root)
+    stability = pd.read_csv(root / 'report' / 'feature_stability.csv')
+    assert len(stability) == 18
+    pair = stability[(stability.split_a == 'split_101') & (stability.split_b == 'split_202')]
+    assert (pair[pair.branch == 'char'].intersection == 2).all()
+    assert (pair[pair.branch == 'char'].jaccard == .5).all()
+    assert (pair[pair.branch == 'word'].jaccard == 1).all()
+    different = stability[(stability.split_a == 'split_101') & (stability.split_b == 'split_303')]
+    assert (different.jaccard == 0).all()
+
+
+def test_report_table_hash_tampering_rejected(synthetic):
+    cfg, root = synthetic
+    s07.write_plan(cfg, root)
+    report.build_report(root)
+    path = root / 'report' / 'runs.csv'
+    with path.open('ab') as f:
+        f.write(b'\nchanged')
+    with pytest.raises(s07.S07Error):
+        report.render_report(root / 'report', root / 'tampered.html')
+
+
+def test_cli_table_only_rebuild(synthetic, tmp_path):
+    import subprocess
+    import sys
+    cfg, root = synthetic
+    s07.write_plan(cfg, root)
+    report.build_report(root)
+    isolated = tmp_path / 'cli_tables'
+    isolated.mkdir()
+    for path in (root / 'report').iterdir():
+        if path.name != 'report.html':
+            shutil.copyfile(path, isolated / path.name)
+    output = isolated / 'cli.html'
+    command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'scripts' / 'run_s07.py'),
+               'rebuild', '--report-dir', str(isolated), '--output-html', str(output)]
+    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
+    assert completed.returncode == 0, completed.stderr
+    assert output.is_file() and 'PARTIAL' in output.read_text()
+
+
+def test_empty_feature_predictions_preserve_s06_raw_bias_path(baseline_result, monkeypatch):
+    import torch
+    from kkt_clf.features import FeatureBuilder
+    root, plan, first = baseline_result
+    run = next(r for r in plan['runs'] if r['run_id'] == 'split_101__none__seed_43')
+    full = load_vectorizer_artifact(plan['config']['full_tfidf'])
+    parts = s07.load_split(root, plan, 'split_101')
+    feature_id = plan['sources']['features']['sha256']
+    s07._train_one(root, plan, run, full.builder, feature_id, parts, torch.device('cpu'))
+    original = FeatureBuilder.transform
+    validation_names = parts['comparison_val'].name.tolist()
+    def one_empty_row(builder, names, *args, **kwargs):
+        matrix = original(builder, names, *args, **kwargs)
+        if list(names) == validation_names:
+            matrix.data[matrix.indptr[0]:matrix.indptr[1]] = 0
+            matrix.eliminate_zeros()
+        return matrix
+    monkeypatch.setattr(FeatureBuilder, 'transform', one_empty_row)
+    raw_predictions = []
+    original_predict = s07.predict
+    def capture_raw(*args, **kwargs):
+        result = original_predict(*args, **kwargs)
+        raw_predictions.extend(result[0].l3_top1.tolist())
+        return result
+    monkeypatch.setattr(s07, 'predict', capture_raw)
+    s07._evaluate_one(root, plan, run, full.builder, feature_id, parts, torch.device('cpu'))
+    predictions = pd.read_parquet(root / 'runs' / run['run_id'] / 'predictions_val.parquet')
+    assert predictions.empty_features.iloc[0]
+    assert predictions.predicted_L3.tolist() == [plan['label_maps']['L3'][i] for i in raw_predictions]
+    summary = report.build_report(root)
+    assert summary['statuses']['success'] == 2
+    saved = pd.read_parquet(root / 'report' / 'predictions_val.parquet')
+    assert len(saved[saved.run_id == run['run_id']]) == len(validation_names)
+    assert saved[saved.run_id == run['run_id']].empty_features.sum() == 1
+
+
+@pytest.mark.parametrize('method', ['global', 'class_aware', 'gradient_x_input'])
+def test_changed_partition_fails_before_ranking(synthetic, method, monkeypatch):
+    import torch
+    cfg, root = synthetic
+    s07.write_plan(cfg, root)
+    plan = s07.load_plan(root)
+    full = load_vectorizer_artifact(cfg.full_tfidf)
+    parts = s07.load_split(root, plan, 'split_101')
+    parts['outer_train'] = parts['outer_train'].assign(L3='changed_label')
+    def forbidden(*a, **k):
+        pytest.fail('changed partition reached supervised ranking')
+    monkeypatch.setattr(s07, 'select_features', forbidden)
+    monkeypatch.setattr(s07, 'compute_margin_attributions', forbidden)
+    with pytest.raises(s07.S07Error, match='frozen split'):
+        s07._prepare_selected(root, plan, 'split_101', method, full, parts, torch.device('cpu'))
