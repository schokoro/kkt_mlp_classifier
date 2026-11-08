You are the independent READ-ONLY S08 code reviewer. This is round1 of maximum2. Do not edit/create files, run tests/training/inference, read sealed label/text data, or delegate. Use read-only inspection tools only. Return severity-ranked actionable findings with precise file/line, trigger, violated requirement and repair, plus explicit verdict. English only.

Review diff against integrated dev dd7559a, snapshot /tmp/s08-review1.diff (SHA256 418104e1c39f8a2a7c5b2fd1444e802b34da9eb2f8d99838d603806a4d8d2fef). Read that exact diff and current context files as needed. Boundaries: README, new kkt_clf/s08.py, s08_stats.py, s08_report.py, scripts/run_s08.py and tests/test_s08*.py; source S08 docs and unrelated notebook are excluded, no existing algorithms changed. Separate Codex implementation and independent test sessions. Current independent S08 tests79 passing incl synthetic9 models, full suite pending; no formal9 workload or sealed access. Read implementation_handoff/testing_handoff for current limitations but evaluate against originals below.

Focus: authoritative S08 contract; fixed293, original S06 identity/provenance/group checks; exact981850-row refit/100k branch identities; full dev union for nine final models, fixed14 epochs/seeds42/43/44; immutable protocol/plan and weights/maps/vocab freeze; negative approval and stale/failed/resume handling; paired bootstrap same rows across seeds, B>=5000 alpha.025 linear lower bound strict sign/inequality, conditional interpretation; useful numerical feasibility with no test labels; blocked null report and isolated deterministic rebuild; resources charged once and measured correctly.

Unresolved design issue: implementation permanently disables evaluate and post-approval reporting, claiming no trusted external issuer/ledger exists. Distinguish an honest conditional D09 limitation from unnecessary invented infrastructure that prevents an executable prepared study. User only requires explicit orchestrator-recorded stakeholder approval with authenticated/mismatched-record rejection, no external service requirement. Assess whether a trusted caller-supplied approval digest/authentication anchor outside the mutable experiment payload and single-use local claim can implement the authorized future path safely. No real approval exists; no production evaluation may run in this review or sprint preparation. Consider whether needed future implementation should be fixed before formal freeze.

ORIGINAL S08 REQUIREMENTS AND D01-D14:

# S08 — Pre-registered non-inferiority assessment of 100k TF-IDF

## Value

Determine whether a 100,000-feature TF-IDF/MLP pipeline can be treated as **practically non-inferior** to the full 400,000-feature pipeline **on the ODS proof-of-concept label universe**, while preserving the measurable model-size and training-cost advantage. This is a decision-focused verification sprint, not another contest for the highest validation macro-F1.

Produce: (1) an auditable, pre-registered non-inferiority protocol, including a defensible margin and a feasibility assessment; (2) reproducible final model artifacts trained without using the sealed test; and (3) **only after separate user authorization to unseal**, one final paired holdout evaluation with a non-inferiority decision or an explicit **inconclusive** result.

A successful sprint does **not** require a positive non-inferiority finding. Correctly reporting insufficient precision, insufficient sample size or a blocked holdout gate counts as useful evidence. Never claim non-inferiority of a 300-million-row industrial pipeline from this PoC.

## Confirmed background

- S06 used 16,289 labeled rows and 293 L3 after `L3 support >= 12`; the full TF-IDF is 300k character + 100k word features, fitted on the verified snapshot of 981,850 unlabeled names.
- S07 compared 36 trained models across three *development* splits (`101/202/303`) and three model seeds (`42/43/44`) per configuration. Its paired development macro-F1 differences versus Full 400k, averaged over split and training seed, were: Global Chi² 100k `-0.003572`, Class-aware Chi² 100k `+0.000873`, Gradient × Input 100k `+0.000085`.
- None of the three methods improved macro-F1 on more than one of the three development splits. S06's large Class-aware advantage did not replicate. The main reliable engineering observation was a reduction in `model.pt` from approximately 410 MB to 103 MB and markedly faster MLP training.
- S07 development data is already explored; its three splits are **not independent test samples**. The original S06 `sealed_test` has not been used for inference, quality scoring or model selection. This is a previously unused *internal ODS holdout*, not an external industrial sample.
- Roughly 200/293 L3 have only 2–4 validation examples in S07; macro-F1 uncertainty at the per-class level is substantial. A small difference between point estimates alone does not establish non-inferiority.
- Previous work established strict artifact provenance/fail-closed handling and table-only regeneration of self-contained HTML. Preserve the S03–S07 contracts and outputs.

## Scope and decisions versus open gates

### Proposed comparison (research hypothesis, NOT a signed-off test decision)

| Role | Configuration | Status |
|---|---|---|
| Reference | Full TF-IDF, 400k (300k char / 100k word) | Fixed reference |
| **Proposed primary candidate** | Class-aware Chi², 100k (75k char / 25k word) | **Requires explicit user confirmation before holdout use** |
| Secondary descriptive comparator | Global Chi², 100k (75k char / 25k word) | Pre-specified, no confirmatory non-inferiority claim |
| Gradient × Input | 100k | Out of primary scope; S07 showed no convincing benefit for the extra attribution step |

The primary candidate is proposed based on S07 development evidence and its positive, although unstable, mean paired difference. Do not choose a different primary candidate based on the sealed test. A different candidate set is possible only by **updating and re-freezing the pre-registration before any holdout prediction or label access**.

### Non-inferiority definition

- Primary endpoint: raw top-1 L3 **macro-F1**, fixed 293-L3 label universe; no reject threshold. It is a model-comparison endpoint, **not** a directly validated measure of price-index error.
- Define `Delta = mean_{seed in {42,43,44}} [macro_F1(candidate, seed) - macro_F1(full, seed)]` on the **same sealed holdout rows**. The three seed pairs are a predefined training-replicate ensemble *for the estimand*, not three statistically independent holdouts. Use unrounded predictions and metrics.
- Let `delta_NI > 0` denote the maximum acceptable absolute **macro-F1 decrease** (e.g., `0.01` means one F1 percentage point). **A provisional planning example is `delta_NI = 0.01`, not an approved scientific or business threshold.** Do not silently adopt it as a default. Its practical rationale, owner, exact value and approval must be frozen before test access. This sprint cannot independently determine what loss is acceptable for downstream price dynamics.
- Statistical decision (subject to feasibility and approval): use a **paired, class-stratified holdout bootstrap** on the same sampled rows across the Full and candidate predictions **and across all fixed model-seed pairs**. Preserve the observed support of each true L3 in each bootstrap replicate; compute the 293-class macro-F1 afresh (never bootstrap pre-computed F1 values). Prefer group-aware resampling if distinct rows share a leakage-group identity; if the chosen grouping prevents a valid class-stratified resampling design, **stop and document the incompatibility rather than silently switching methods**.
- Pre-register bootstrap RNG seed, replicate count (proposed `B >= 5000`), deterministic percentile CI implementation and all treatment of ties, zero-division and missing predictions. A one-sided test with `alpha = 0.025` uses the **lower 97.5% one-sided bound** (equivalently the lower bound of a two-sided 95% percentile interval). Declare **non-inferiority only if `lower_bound(Delta) > -delta_NI`**. Equality or an interval crossing `-delta_NI` is **not established/inconclusive**, not proof of inferiority. Do not convert the number of model seeds into an inflated effective sample size.
- Bootstrap uncertainty is **conditional on this finite ODS holdout, its class composition, and the three fixed model seeds**; it does not account for dataset shift or fully characterize training stochasticity. Provide separate descriptive seed-wise results and support-bin sensitivity checks, not unsupported claims of population-level significance.
- **No superiority claims**, unless separately pre-specified and multiplicity controlled; the secondary Global comparison is descriptive and must not be reported as a second confirmatory NI test. Do not modify margin, primary method, confidence level or bootstrap design after seeing sealed results.

### A. Independent holdout isolation and preflight

1. Verify the integrated S07 code and the exact S06 sealed-test manifest, row IDs and hashes. Construct the final *labeled development* set as the union of the original S06 `outer_train` and `comparison_val`; exclude the original sealed test by stable row identity and duplicate-group key. **Do not re-filter categories or change label maps.** All 293 classes must remain defined.
2. Verify group disjointness and structural metadata of the holdout **without loading holdout label/text columns for quality analysis or performing inference**. Preserve the original sealed files read-only. If leakages exist, fail closed and report; do not silently reassign or clean rows.
3. Assess feasibility using **development data and already authorized structural counts only**: class-support histogram, potential CI precision/width from S07 development results or simulated paired binary/categorical predictions, constraints introduced by macro-F1 and classes supported by two holdout examples. Such projections are exploratory, not holdout performance estimates. Record whether the proposed `delta_NI` appears resolvable at all; do not choose the margin to force a favorable result.
4. Draft/freeze a machine-readable `protocol.json` and human-readable `pre_registration.md` with the primary comparator, margin **as an explicit `UNAPPROVED` value until approved**, confidence level, inference method, exact model-training configuration, seeds, data lineage, all hashes, and a deterministic experiment plan. Record provenance and an immutable protocol hash before any test evaluation.
5. **Mandatory user authorization gate:** before opening sealed holdout for inference, obtain explicit user approval of (i) `delta_NI` and its practical rationale, (ii) the primary candidate, and (iii) the one-time use of the S06 sealed test. An existing request to create sprint artifacts or to run automated training does **not** supply that approval. If approval is absent, stop after preparation and report `awaiting_user_approval`; do not read test labels or generate test predictions.

### B. Final training without test leakage

1. Use the exact verified full S02 TF-IDF and unlabeled-corpus snapshot, retaining the historically documented partially transductive overlap. Use S07/S06 fixed MLP architecture, training hyperparameters, optimizer, and final epoch rule. No new architecture/feature engineering/retuning on S07 validation.
2. Recompute Global and Class-aware Chi² **using only the final development labels**. Build a separate 100k fixed-vocabulary TF-IDF for each, refitted on the identical original unlabeled-corpus row sample; preserve char/word identity, normalization, IDF and provenance. No gradient attribution needed.
3. Train Full, primary Class-aware 100k, and secondary Global 100k afresh for the same seeds `42,43,44`, on the full authorized development pool. **Nine final MLP runs** (3 configurations × 3 seeds); no test-based checkpoint selection, early stopping, or epoch tuning. If training the secondary method exceeds a documented resource gate, stop or mark it incomplete; do not compromise the primary pairing. Save/reload all checkpoints and verify input vocabulary identities.
4. Freeze exact trained model hashes, feature artifacts, prediction-interface settings and all decision rules **before** any sealed inference. If the underlying protocol must change, start a new versioned experiment before test access; never reuse a contaminated test as fresh evidence.

### C. One-time holdout evaluation (only after authorization)

1. Verify signed-off pre-registration and all model/feature/data fingerprints, then make **one planned inference pass** on exactly the same sealed-test rows for each frozen model. Compute/stash immutable predictions with row IDs, true labels, predicted labels and model hashes, with no test-driven changes thereafter. Do not load test scores between model runs to adapt later runs.
2. Recompute macro-F1 per seed and paired `Delta`; compute one pre-registered confidence bound for the **primary** candidate. Show result, margin, interval, decision and caveats. Accuracy, per-L3 F1/recall, support bins, confusion pairs, model-training cost and dictionary size are secondary descriptive results.
3. Distinguish `non_inferior`, `not_established`, `evaluation_blocked` and `not_run_awaiting_approval`. In particular, a wide CI is **inconclusive**; never silently loosen `delta_NI`, reduce the label universe, discard challenging classes or rerun a new split to obtain a better result.
4. Keep the test sealed for any *subsequent model selection*. Once this one-time evaluation has occurred, **future adaptive iterations require genuinely new labeled holdout data** for another confirmatory claim.

## Required output artifacts

Use a fresh, non-destructive path `outputs/S08/<experiment_id>/`, with snapshots/hashes of all inputs and an explicit stop-gate state:

```text
outputs/S08/<experiment_id>/
├── protocol.json              # frozen configuration, candidate, margin, alpha, CI method, status and hashes
├── pre_registration.md        # rationale, signed-off/unsigned decisions, date and provenance
├── feasibility.json           # support / uncertainty study without sealed quality data
├── plan.json                  # three methods × three seeds, artifact dependencies and evaluation step
├── shared/                    # validated full/selected TF-IDF, ranked vocabularies, provenance
├── runs/                      # frozen train results, checkpoints, hashes; no user data in Git
└── report/
    ├── report.html            # offline self-contained report, including blocked/partial status
    ├── manifest.json          # gate state, plan, approval, all hashes and limitations
    ├── runs.csv               # 9 training statuses; validation/test metric fields nullable
    ├── stages.csv             # train/rank/refit/inference cost, shared stages once
    ├── ni_decision.json       # pending or final margin, CI, estimator and decision
    ├── paired_test.csv        # ONLY after approval: per-seed paired sealed-test metrics
    ├── per_class.parquet      # ONLY after approval: per-L3 paired metrics
    ├── support_bins.csv       # ONLY after approval: grouped descriptive sensitivity
    ├── predictions_test.parquet # ONLY after approval: frozen predictions, protected locally
    └── confusion_pairs.csv    # ONLY after approval: diagnostic errors
```

If approval is pending, the five test-dependent files **must not be fabricated** and `ni_decision.json` must explicitly state `not_run_awaiting_approval`. If approved, include all planned reports and preserve explicit failed/skipped run statuses. Make HTML rebuildable from reporting tables and `ni_decision.json` alone, without loading models, running inference or accessing `runs/` or `shared/`. If storing bootstrap replicates is useful for auditing, include a compact Parquet/CSV with seed, replicate ID and `Delta` or a versioned deterministic regeneration recipe.

## Out of scope

- Using a new unlabeled corpus, changing L3 taxonomy or `support >= 12` policy, changing the MLP architecture, trying new selection strategies, Gradient × Input, new feature budgets or an adaptive hyperparameter search.
- Reusing S07 development validation as though it were fresh independent test evidence; estimating industrial non-inferiority, throughput on 300m names or downstream price-index impact.
- Choosing or revising the NI margin based on observed sealed-test metrics; more than one adaptive attempt at the holdout; selecting a winner post hoc.
- Commit, push, merge, deleting user files, clearing previous outputs, modifying `artefacts/SPEC_v1.0.md` without explicit authorization.

## Definition of Done

- [ ] **D01 — Isolation and roles.** Confirm integrated S07 and current `AGENTS.md` roles, inspect working tree; create a dedicated S08 branch without changing user files.
- [ ] **D02 — Data/provenance.** Reconstruct fixed 293-L3 development pool, verify sealed-test exclusion and original S06 manifest/split/group provenance, confirm immutable full TF-IDF and unlabeled corpus.
- [ ] **D03 — Explicit NI contract.** Establish the primary candidate, `delta_NI`, practical rationale, confidence bound, fixed macro-F1 universe, bootstrap units/seed/B and decision inequality; write pre-registration **without silently approving a proposed margin**.
- [ ] **D04 — Feasibility.** Document class-support limitations and expected precision using only development/structural information. Clearly distinguish a poorly powered design from evidence of equivalence.
- [ ] **D05 — Frozen plan.** Before test access, freeze hashed `protocol.json`, `plan.json`, all dependencies, seeds, output schema, stop gates and audit state; mismatches fail closed.
- [ ] **D06 — Final selection/refit.** Produce selected Global and Class-aware 100k vocabularies from development-only labels; refit on original corpus snapshot; verify IDF, normalization, hashes, reloading and full-artifact immutability.
- [ ] **D07 — Final models.** Train 9 final models with matching fixed seeds/hyperparameters and no test leakage; reload checkpoints and verify interface/provenance. Any missing model is explicit `failed/skipped`, not success.
- [ ] **D08 — Authorization gate.** Before any holdout inference, require documented user approval of margin, primary candidate and one-time test use. Without approval, stop safely at `awaiting_user_approval`, with no predictions/metrics from test.
- [ ] **D09 — Valid NI evaluation (conditional).** Only if D08 is approved, produce one frozen set of paired holdout predictions, recompute fixed-label macro-F1 and pre-registered paired CI, make the correct NI/not-established decision without pseudoreplication.
- [ ] **D10 — Honest uncertainty.** Sensitivity to support, model seeds and class imbalance reported; no unwarranted claim of external generalization or inferiority from an inconclusive CI.
- [ ] **D11 — Resource measurement.** Report model bytes, ranking/refit/training/inference timings, RSS/VRAM when available; distinguish amortized shared from per-seed costs.
- [ ] **D12 — Report reproducibility.** Create all artifacts applicable to the approved/pending state; test table-only HTML regeneration, pending-state null fields, hashes, failed statuses and no hidden test access.
- [ ] **D13 — Tests and independent review.** Dedicated Codex coding session and separate Codex testing session; max two English read-only Kimi reviews, independent test coverage of leakage/margin/CI/resume failure modes; full pytest and `git diff --check`; late fixes reported without claiming independent re-review.
- [ ] **D14 — Honest closure.** `artefacts/S08/result.md` identifies every D01–D14 as done/partial/blocked; report exact commands/verified results, authorization status and a recommendation. Update `STATUS.md` and append-only `DECISION_LOG.md` only for actual decisions; do not claim S08 confirmatory completion if sealed holdout was not approved/evaluated.

## Acceptance boundary and stop conditions

The study is **gated by an open stakeholder decision**. Complete implementation and training can be prepared autonomously, but the one-time sealed-test evaluation is separately authorized. A safe `awaiting_user_approval` outcome is **not a failed implementation**, but D09 remains blocked and the final NI claim is pending. If holdout support proves too weak for a meaningful interval, recommend collecting additional independent annotations instead of broadening a margin post hoc.


ORIGINAL ORCHESTRATOR TASK:

# S08 — Orchestrator prompt (English; final response in Russian only)

You are the **Codex CLI orchestrator** for sprint S08 of `kkt_mlp_classifier`.

**Read `artefacts/S08/value_dod.md` as the authoritative sprint specification.** Also read the *current* `AGENTS.md`, `artefacts/STATUS.md`, append-only `artefacts/DECISION_LOG.md`, and the integrated S06/S07 protocols, manifests and final results. Do not alter this prompt or the original S08 Value/DoD. Verify paths/configurations against the actual repository and local artifacts instead of assuming that examples still exist.

**Language:** Every prompt addressed to subordinate Codex or Kimi sessions must be in English. Working logs, technical artifacts and intermediate communication may be in English. **Write ONLY the final user-facing orchestrator summary in Russian.** Never translate technical identifiers, paths, hashes or file names.

## 0. Verify roles and Git isolation

- **This Codex CLI session is the orchestrator:** plan, delegate, inspect, verify and report; **do not personally implement code or tests**.
- A separate `codex exec` session implements and fixes code; a different `codex exec` session creates independent tests.
- A separate **read-only `kimi -p` session** provides independent code review, at most **two rounds** across this sprint. Do not swap these responsibilities or use the obsolete S01–S06 delegation rules.
- Save the exact full **English prompts before each invocation** under `artefacts/S08/prompts/` (e.g., `01_codex_implementation.md`, `02_codex_tests.md`, `03_kimi_review.md`, `04_codex_fixes.md`, `05_kimi_rereview.md`). The review prompt must include the original S08 requirements, DoD, tested diff boundaries and unresolved risks.
- Check `git status`, verify that S07 has been committed and integrated into local `dev`, preserve untracked and user-authored files, and create `sprint/S08-noninferiority` from the verified integrated branch. Do not force checkout or overwrite conflicts.
- **Standing task-scoped approval:** invoke `codex exec`, `kimi -p`, local tests, fit/refit, feature selection, model training and preflight checks autonomously without approval requests for each command. Follow real execution security restrictions. **This is NOT approval to unseal the test**, commit, merge, push or delete data.

## 1. Establish the statistical protocol and safety boundary

S08 is **not** another search for the highest dev macro-F1. The research question is whether 100k TF-IDF is non-inferior to Full 400k within an **explicitly approved practical loss margin**.

- Use S07 evidence only for designing this comparison, not as independent confirmatory evidence. Preserve the original 293-L3 class universe and original S06 sealed-test identity. Verify exact full TF-IDF snapshot/hash and dev/test group separation.
- Proposed primary method: Class-aware Chi² 100k. Secondary descriptive method: Global Chi² 100k. Full TF-IDF 400k is the reference. G×I is excluded unless the user explicitly authorizes a revised pre-registration **before test access**. Proposed primary status is not user confirmation.
- Proposed planning-only `delta_NI=0.01` absolute L3 macro-F1: **not authorized**. Do not default to 0.01 merely because the sprint document mentions it. The user must explicitly approve the margin and its practical rationale.
- Freeze the primary estimand: mean of three seed-matched `macro_F1(candidate) − macro_F1(full)` differences over the same S06 holdout. Seeds 42/43/44 are predefined replicates, not independent datasets. Primary L3 metric is raw top-1 macro-F1 over fixed 293 classes.
- Predefine paired resampling of the **same class-stratified holdout rows for all models and seeds**, from the underlying row-level predictions rather than aggregated F1. Include a group-aware alternative only with a pre-validated statistical design. Freeze bootstrap seed, `B >= 5000`, one-sided `alpha=0.025`, lower confidence-bound method and the rule: non-inferior iff `lower_bound(Delta) > -delta_NI`.
- Produce a feasibility study using only development results and permitted holdout structural metadata. Check whether 293 classes with tiny holdout supports can plausibly resolve the proposed margin; a failure to resolve the bound must be reported as **inconclusive**, not accommodated by broadening the margin after the fact.
- **Hard approval gate:** Before any sealed-test inference, test-label inspection, metrics or error analysis, request *one explicit user decision* approving the exact `delta_NI`, primary candidate and one-time holdout use. Do not treat this prompt or standing model-training approval as that decision. You may do the planned preparation and model training first, but if user approval is missing, stop after preparing the frozen artifacts with status `awaiting_user_approval`. Do not continue or simulate consent.

## 2. Delegate implementation and negative tests

Orchestrate separate Codex implementation and independent-test sessions. Preserve compatibility with S03–S07 artifacts and reporting. Do not change the MLP architecture, L3 taxonomy, TF-IDF grammar, preprocessing, or fixed feature-selection algorithms.

Required negative tests include at least:

- attempted test inference/label access before approval; unauthenticated or mismatched approval record;
- mismatched dataset/split hashes, conflicting group IDs, changed model weights, label maps or selected vocabularies;
- frozen protocol/plan tampering or unsafe artifact reuse; old or failed run falsely marked successful;
- CI recomputation from row-level paired predictions, use of identical bootstrap indices across seeds, fixed class universe and class counts, `Delta` sign and strict bound inequality;
- zero/division, missing predictions, ties, insufficient class support and group-resampling incompatibilities;
- blocked-state report generated without test-dependent files, with honest nulls/status;
- deterministic report rebuild from saved table outputs alone without any model, refit or inference;
- 100k selected feature dimension and refit provenance from the same unlabeled corpus snapshot.

Conduct at most two **read-only Kimi** code reviews with severity-ranked findings. Organize fixes through delegated Codex sessions and have independent Codex testing verify regression scenarios. Do not represent fixes after the final Kimi review as independently reviewed.

## 3. Pre-registration and autonomous training (without opening test)

1. Reconstruct the **development pool** from the original S06 `outer_train + comparison_val`, excluding every `sealed_test` row. Do not re-filter or alter 293 L3. Verify group-aware disjointness and provenance, inspecting holdout structural hashes/counts only.
2. Read the exact final training protocol frozen in S07/S06. Use it for all runs; no validation-driven retuning. Train matched seeds `42,43,44` on the full available development pool.
3. On this development pool only, compute the agreed Class-aware and Global Chi² rankings. Refit selected `75k char + 25k word` TF-IDF on the exact unlabeled corpus snapshot from S02, verify column identity, IDF/normalization, artifact size and round-trip. Reuse full TF-IDF unchanged.
4. Train a total of **nine** MLP models: Full, Class-aware 100k and Global 100k, each at all three seeds. Freeze model hashes, selected feature identities, prediction configuration, artifact lineage and evaluation rules **before any test inference**. Run smoke/regression tests on synthetic or dev data only.
5. Produce `pre_registration.md`, `protocol.json`, `plan.json`, `feasibility.json` and the pre-unseal `report/` as specified. Do not fabricate post-holdout metrics or output files when approval is missing.
6. If an integrity gate fails or a defensible confidence-bound method cannot be established, stop with an explicit blocker. Do not relax the margin, switch targets or quietly reuse previously evaluated dev validation as a new test.

## 4. The sealed holdout remains closed until a separate instruction

**Do not enter the following phase automatically.** The user must separately approve all three: `delta_NI`, primary Class-aware 100k candidate and one-time use of the original S06 `sealed_test`.

After approval, in this same sprint and only if the pre-registered hashes still match:

- In one scheduled, frozen evaluation stage, run paired inference for all fixed Full and selected checkpoints on the identical holdout rows; persist immutable row-ID predictions. No intermediate test-result-driven changes or model selection.
- Calculate seed-wise macro-F1 and primary mean paired `Delta`, the predetermined one-sided 97.5%-lower CI bound, and an explicit `non_inferior` / `not_established` / `evaluation_blocked` decision. Include Global as secondary descriptive comparison only.
- Report support-bin precision, per-class metrics, confusion pairs, accuracy, training/refit cost and model byte sizes as appropriate. The test may never be called fresh again after being inspected.
- Verify the aggregated HTML can be rebuilt using saved report tables only; record the exact command.

## 5. Verify and report the actual state

Run quick/full pytest, targeted negative tests, `git diff --check` and a small synthetic end-to-end smoke test. Confirm all expected row/feature/checkpoint/protocol hashes, resource measurements and run statuses. Preserve past outputs in fresh versioned directories; do not delete user files.

Update `artefacts/S08/result.md` with exact commands, results and honest D01–D14 statuses. Update `artefacts/STATUS.md` and append-only `artefacts/DECISION_LOG.md` only with confirmed factual progress and decisions. Do **not** modify `artefacts/SPEC_v1.0.md`, commit, merge or push.

If authorization has **not** been given, a successful result of this session is **a reproducible prepared study awaiting approval**, not an invented NI conclusion. Set the gate explicitly to `awaiting_user_approval` and stop there.

**ONLY the final user-facing response must be in Russian.** Include: implementation/tests/review results, whether nine final training runs completed, the feasibility limitations, the proposed but unapproved margin, primary-candidate status, exact artifact paths, holdout gate and next required user approval, and any unresolved risks. Never claim that the sealed holdout was evaluated unless it actually was.
