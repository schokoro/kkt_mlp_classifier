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
