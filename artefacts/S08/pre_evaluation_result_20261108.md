# S08 — reproducible prepared study awaiting user approval

> Latest continuation status (2026-11-08): stakeholder authorization received, but evaluation is blocked before test access by a 95% versus frozen97.5% one-sided confidence-level conflict. See the continuation section below. Preparation-stage awaiting_user_approval statements are historical.

## Actual outcome

**Gate: `awaiting_user_approval`.** Nine final development-only models completed successfully, with zero failed/skipped/pending/incomplete runs. Two selected TF-IDF refits passed. No original sealed labels were loaded; no original holdout inference, metrics, error analysis, predictions, approval record or consumption claim was generated. `report/ni_decision.json` is `not_run_awaiting_approval`, with null margin, Delta, bound and decision. **No non-inferiority conclusion exists.**

Primary Class-aware Chi²100k is **proposed and unapproved**. Global100k is secondary descriptive only. Full400k is the fixed reference; G×I is excluded. `delta_NI=0.01` is an illustrative planning example, **not an approved margin or default**. Its practical rationale and owner remain open. A single explicit stakeholder decision must approve the exact margin/rationale, primary candidate and one-time original S06 holdout use before evaluation. Training authorization did not supply that decision.

Branch `sprint/S08-noninferiority` was created from integrated local `dev` `dd7559a550333480f4392310990b2b94f82499b1`. No commit, merge, push, staging or deletion. Original source documents and unrelated user notebook preserved. Implementation/test roles used separate `codex exec` sessions; two read-only Kimi rounds completed, final PASS/no blockers. No code/tests/README changed after the final reviewed snapshot.

## Artifacts and frozen identities

Study root: `outputs/S08/noninferiority_20261107_v1/`.

- `pre_registration.md`: unsigned practical decisions and fixed statistical/training rules.
- `protocol.json`, `plan.json`, `feasibility.json`: frozen pre-test contract, all9 run dependencies and numerical exploratory projection.
- `shared/class_aware/` and `shared/global/`: development-only rankings/selections, fixed-vocabulary TF-IDF and refit receipts. Full unchanged reference remains `outputs/S02/tfidf_1m/`.
- `runs/<method>__seed_<seed>/checkpoint/`: nine independently initialized final models, features, label maps, training provenance and recursive receipts.
- `training_freeze.json`: all nine models and selected-feature file hashes, protocol/plan bindings and prediction interface.
- `report/report.html`, `manifest.json`, `runs.csv`, `stages.csv`, `ni_decision.json`: **only five files**, all applicable to pre-unseal state.
- Reviewed preflight: `outputs/S08/preflight_20261107_reviewed/`.
- Development documentation: `artefacts/S08/preflight.md`, `implementation_handoff.md`, `testing_handoff.md`, `review_fixes_handoff.md`, `review_regression_handoff.md`, `review1.md`, `review2.md`; exact English prompts `prompts/01` through `06`.

The five test-dependent files are absent everywhere in the real study: `paired_test.csv`, `per_class.parquet`, `support_bins.csv`, `predictions_test.parquet`, `confusion_pairs.csv`. `evaluation_claim.json` and `approval_decision.json` are also absent. Report training_complete=true, confirmatory_complete=false, success9, all other statuses0. Validation/test F1 columns are null for every run. Pending JSON `approval_authentication=not_provisioned` means no actual stakeholder decision/trusted anchor has been supplied; the future authentication/evaluation mechanism itself is implemented and reviewed.

| File | SHA256 |
| --- | --- |
| protocol.json | `8aacc5c9ba3306ec1fe4dc0e16f214d976abe2f57d93891f75f53de134718175` |
| plan.json | `3b51cfa1f6b8e3df3db2de0fe7de2a48a0fdb711935668a89c26d0c64bcfc953` |
| feasibility.json | `ec85fc3a3b3a11b1a53ba7bc1cbbcf0aa7cc435392249d5273674e12c01d6f6b` |
| pre_registration.md | `fd83bba473926587ab5f046ece4203148706569f128c034e9c96f9a3bd4ed88f` |
| training_freeze.json | `6d0f5725d1028f2a4a1f64f01603ce82fde7429705095e5a3d0fdb037caa8e59` |
| execution_finished.json | `adca6a5fb14ec940ad4380adaa99f3fe3ab785c842dfae8fdfd515fc50241418` |
| report/report.html | `754c9bb47dc73dd1ca98aee94f6d838e05a44533d02a7aa9da32fac2733b66d0` |

Full/corpus/original split/training protocol hashes are recorded in `preflight.md` and in machine-readable protocol/plan. Original sealed SHA256 remains `1842de401867e21a42c1dd342716548e1c40ef1ea57cb6109845ca8e308015f4`.

## Data, features and training

Exactly original S06 outer_train12,960 + comparison_val1,692 = **14,652 development rows**, all293 L3, minimum development support10, no new support filter. Development groups are singleton and disjoint from original1,637-row holdout. Parent/implementation used structural row_id/name projections, file hashes and existing audit metadata only for the holdout; no L3/text quality analysis. All five original partitions and inner union provenance were verified. Row IDs SHA256 `bff8231a1955cb83a2c86cc691198d1dbbef13d891bede8b0d958bbbecbb41ed`; development groups `aeee845e06ac915ecbfbff100719f96bd3495f302e7ef6d554c6c7fc6811e3fa`.

Both rankings use all final development labels only. Two new TF-IDF artifacts each have exactly75,000 char +25,000 word, independently refitted on the original981,850-row S02 corpus snapshot and identical actual fit sample. Full300k+100k is reused unchanged. Both refits: char/word IDF max absolute difference0; per-branch L2 maximum deviation1.1920928955078125e-7; vocabulary/index identity, serialization and refit provenance verified. No full dense feature matrix or new selection algorithm.

All9 models: seeds42/43/44, fixed14 epochs, hidden[256], dropout.3, ReLU/no norm, AdamW lr.001/weight_decay0/batch256, constant LR, final epoch weights, restore_best=False. Full development union is fitting input and explicitly in-sample diagnostic monitor required by the existing API; no held-out checkpoint or epoch selection. All checkpoints reloaded and checked. Prediction interface is fixed raw top1 L3/no reject, bias path for empty rows, pinned runtime and batch size; future inference uses frozen CPU checkpoint snapshots.

| Run | Epochs | model.pt bytes | model.pt SHA256 |
| --- | --- | --- | --- |
| class_aware__seed_42 | 14 | 102778289 | `3140feb8fb51d56bf902a8ec06228f3d0b399d9077be8c77b134985d72cff0c3` |
| class_aware__seed_43 | 14 | 102778289 | `8640c0dde6129f1619642aee10c6693fe15e13e5446561e0851cc9a391c074a7` |
| class_aware__seed_44 | 14 | 102778289 | `b1adb5cb83df8dfee6b214a5ade03c95411b8a15e4c776f61694db60f3460dc7` |
| global__seed_42 | 14 | 102778289 | `321fd3d0333d910385a8cf7efbc4dd60747bbf30a2196568928c59bf10298eeb` |
| global__seed_43 | 14 | 102778289 | `0578b6b6710963d3aa755821d7371601fb59ab6e283830cce487e1649714d349` |
| global__seed_44 | 14 | 102778289 | `2ddf250452328d3c356c6f19559c5c57bd6c50574151d6358e381e1893951f2d` |
| none__seed_42 | 14 | 409978289 | `920caa749b98744ddccd4bf65b7170d67d23fa403f9cf78666480ada92df0a7e` |
| none__seed_43 | 14 | 409978289 | `64b115dd41f285b96d92a0544e9468d736d46ef3cb1205eac455ebd5df88aa5e` |
| none__seed_44 | 14 | 409978289 | `d2a2580c9accb3760d8a59c4141dad88777328301b432e0d26ad9a97703a41b4` |

## Statistical contract and feasibility

Estimand is mean of the three same-seed candidate-minus-Full macro-F1 differences on the same fixed293-class holdout. Seeds are training replicates, not independent datasets. Frozen class-stratified paired row bootstrap: B5000, numpy.default_rng/PCG64 seed808, alpha.025, linear2.5th-percentile lower bound; same sampled indices for all models/seeds, class supports preserved, zero_division0. Non-singleton/missing groups, insufficient supports and missing/out-of-universe predictions fail closed. NI requires strictly `lower_bound > -delta_NI`; equality/crossing is not_established/inconclusive, not proof of inferiority. Global descriptive, no superiority claim.

Feasibility uses **explicitly simulated categorical predictions on a development-derived1,508-row projection**, not the sealed holdout or confirmatory S07 evidence. Projected200/293 classes have2–4 examples. Two hypothetical paired-disagreement scenarios:

| Simulated disagreement probability | Simulated Delta | Simulated lower bound | Percentile width | Delta minus lower |
| --- | --- | --- | --- | --- |
| .03 | -.01387037 | -.02331059 | .01801060 | .00944022 |
| .10 | -.04993623 | -.06700700 | .03199504 | .01707077 |

A clearly labelled exploratory centering of the simulated point to zero gives lower bounds -.00944022 and -.01707077. This is **not a fitted-null bootstrap, power estimate, estimated sealed effect or acceptable-loss decision**. It demonstrates how a one-point planning margin can be borderline or unresolved depending on unknown paired error patterns; it does not guarantee resolvability. No actual F1 from S07 is treated as independent confirmatory evidence.

Original authorized structural audit `outputs/S06/audit/class_support_split.csv` (SHA256 `dd0dc50765eb9bb503279022c88cc10137070f169d3963c88d46f289b9ae8ba4`) records114 holdout classes at2 examples and186 at2–4, summing1,637 rows. Its development support counts independently match the allowed development labels. These existing counts are permitted structural metadata; the parent did not inspect sealed labels. The final review's suggestion that all per-class support metadata is unavailable is stricter than the user's explicit allowance for existing structural counts. The actual count audit and the simulated projection have different compositions and are not conflated.

Rare-class uncertainty remains substantial; empirical resampling cannot reveal unobserved error modes or guarantee nominal population coverage. Inference is conditional on this ODS composition and three fixed seeds, with no dataset-shift/training-population guarantee. If the future approved bound is unresolved, report inconclusive and collect genuinely independent annotations; never broaden the margin after observing test results. ODS is not representative of the industrial corpus, and macro-F1 does not validate price-index impact.

## Costs (observed CPU, two threads)

Execution625.755814s wall; measured24 stage windows sum575.140250s, excluding integrity/I/O overhead. Peak measured stage RSS2,864,926,720bytes (~2.668GiB); sampling can miss brief peaks. VRAM=null; GPU untested. Inference cost=null/not run. Shared rankings/refits charged once per method, not three times.

| Method | train stages | Train seconds | Mean/train | Ranking selection seconds | Refit seconds | model.pt bytes |
| --- | --- | --- | --- | --- | --- | --- |
| Full | 3 | 324.552773 | 108.184258 | n/a | unchanged full | 409,978,289 |
| Class-aware100k | 3 | 83.787359 | 27.929120 | 6.836284 | 35.162947 | 102,778,289 |
| Global100k | 3 | 74.655681 | 24.885227 | 2.021077 | 41.340381 | 102,778,289 |

Ranking transforms: Class-aware.716808s, Global.711155s. Training transforms: Full2.209161s, Class-aware1.448860s, Global1.697763s. Full/selected model byte reduction is about4x. These are local engineering measurements, not a quality comparison or industrial throughput claim.

## Tests, review and verification

- Separate initial implementation and independent-test Codex sessions; separate fix and regression-test Codex sessions. Exact saved prompts01/02/04/05 precede each invocation. No code or test authored by the orchestrator.
- Final quick suite: **711 passed,1 skipped,1 deselected**,187.46s (`/tmp/s08-fixes-fast-final.log`).
- Final full socket-enabled suite: **712 passed,1 skipped**,192.94s (`/tmp/s08-final-full-pytest.log`), CUDA warning only. All133 S08 tests included. Earlier sandbox-only full runs failed notebook socket creation; no test was weakened; successful rerun used the permitted environment.
- Independent original81 S08 tests and final52 authorization regressions pass. Includes generated nine-run smoke, label-free negative source, separate generated labeled positive source, authentic-but-mismatched records, post-claim/source/weights mutation, O_EXCL concurrent complete callers (one inference pass), failure on fourth inference, consumed replay prohibition, persisted row alignment, independent sklearn/paired-bootstrap recomputation, null/failed/table-only reporting, IDF/index/norm/corpus checks, fake resealed statuses and metrics.
- Initial independent tests found NaN row/group identity acceptance; Codex fixed it, independent regressions pass. Later tests found resealed evaluated-status/metric inconsistencies; Codex fixed them, regressions pass. An overly broad test spy that blocked allowed DEVELOPMENT labels was corrected by its independent author; its historical failures remain logged. No production/sealed access occurred in these runs.
- Kimi round1 H1 required an executable approval path before final freeze; resolved through Codex. L2 row-ID pairing and L3 explicit status membership resolved. Kimi round2 **PASS/no blockers**. Optional redundant identity check and an early execution marker orphan remain unchanged; interrupted preparation can retain an honest pending/non-resumable root. No post-final-review code/test/README fixes or third review.
- Parent independently hashed every runs/shared file against training_freeze, checked all9 histories14 epochs/final weights/null quality fields, all9 model hashes, all24 stage records, exact reviewed-file fingerprints and absence of all test-dependent files/claim/approval. `/tmp/s08-final-independent-verification.json`.
- Isolated `/tmp/s08-final-tables-only-gr_3nan8/` contained only manifest,runs.csv,stages.csv,ni_decision.json. CLI rebuilt HTML **byte-identically** with no runs/shared/models/source data in the isolated directory. Tests additionally forbid source/model API access during rendering.
- All3,717 existing non-S08 output files retain sizes/mtimes; original S08 source documents, SPEC and unrelated notebook hashes unchanged. Preservation snapshot `/tmp/s08-before-formal-preservation.json`. Original sealed SHA unchanged. Git whitespace checks pass; no staged changes.

Actual principal commands (completed roots are single-use; do not replay execute):

```bash
git status --short
git branch --show-current
git log -6 --oneline
git switch -c sprint/S08-noninferiority dev
codex exec --ephemeral -s workspace-write -o /tmp/s08-implementation-last.md - < artefacts/S08/prompts/01_codex_implementation.md > /tmp/s08-implementation.log 2>&1
codex exec --ephemeral -s workspace-write -o /tmp/s08-tests-last.md - < artefacts/S08/prompts/02_codex_tests.md > /tmp/s08-tests.log 2>&1
kimi -p "$(cat artefacts/S08/prompts/03_kimi_review.md)" > /tmp/s08-kimi-review1.log 2>&1
codex exec --ephemeral -s workspace-write -o /tmp/s08-fixes-last.md - < artefacts/S08/prompts/04_codex_fixes.md > /tmp/s08-fixes.log 2>&1
codex exec --ephemeral -s workspace-write -o /tmp/s08-regression-tests-last.md - < artefacts/S08/prompts/05_codex_regression_tests.md > /tmp/s08-regression-tests.log 2>&1
kimi -p "$(cat artefacts/S08/prompts/06_kimi_rereview.md)" > /tmp/s08-kimi-review2.log 2>&1
.venv/bin/python -m pytest -q tests/test_s08_integrity_independent.py tests/test_s08_report_independent.py tests/test_s08_smoke_independent.py tests/test_s08_statistics_independent.py
.venv/bin/python -m pytest -q tests/test_s08_authorization_independent.py --tb=short
.venv/bin/python -m pytest -q -m 'not slow' --tb=short
.venv/bin/python -m pytest -q --tb=short > /tmp/s08-final-full-pytest.log 2>&1
.venv/bin/python scripts/run_s08.py preflight --output-root outputs/S08/preflight_20261107_reviewed --device cpu --num-threads 2
.venv/bin/python scripts/run_s08.py plan --output-root outputs/S08/noninferiority_20261107_v1 --device cpu --num-threads 2
.venv/bin/python scripts/run_s08.py execute --output-root outputs/S08/noninferiority_20261107_v1
.venv/bin/python scripts/run_s08.py report --output-root outputs/S08/noninferiority_20261107_v1
.venv/bin/python scripts/run_s08.py rebuild --report-dir /tmp/s08-final-tables-only-gr_3nan8 --output-html /tmp/s08-final-tables-only-gr_3nan8/rebuilt.html
sha256sum outputs/S08/noninferiority_20261107_v1/protocol.json outputs/S08/noninferiority_20261107_v1/plan.json outputs/S08/noninferiority_20261107_v1/feasibility.json outputs/S08/noninferiority_20261107_v1/pre_registration.md
git diff --check
```

Training/preflight/plan/report/rebuild stdout logs: `/tmp/s08-final-training.log`, `/tmp/s08-reviewed-preflight.log`, `/tmp/s08-final-plan.log`, `/tmp/s08-pre-unseal-report.log`, `/tmp/s08-table-only-rebuild.log`. Worker handoffs list intermediate exact test commands/results. Failed Codex/Kimi sandbox runtime startup and incompatible Kimi `--plan -p` attempts happened before active reviews; same saved prompts were then used with necessary runtime permissions. Exactly two actual Kimi reviews occurred. Parent verification-only snippets corrected a Series-index ordering comparison and a training-record diagnostic key lookup; final verification passed without file/code/data changes.

## D01–D14 actual acceptance status

| ID | Status | Evidence / remaining condition |
| --- | --- | --- |
| D01 | done | Integrated dev verified, dedicated branch, separate roles, user files preserved. |
| D02 | done | Exact development union293 L3, original five partitions, sealed row/group exclusion and corpus/full/protocol pins. |
| D03 | partial / awaiting user | Statistical contract frozen; exact margin, rationale/owner and primary confirmation intentionally unapproved. |
| D04 | done (exploratory) | Numerical simulated precision and existing structural supports; no power/NI guarantee. |
| D05 | done | Frozen protocol/plan/code/source dependencies before training; complete model freeze before any test access. |
| D06 | done | Two development-only100k rankings/refits; exact corpus/indices/IDF/L2/reload; Full immutable. |
| D07 | done | 9/9 fresh fixed14-epoch models, matched seeds, reloaded and hashed. |
| D08 | done gate; approval pending | Authentication/one-time negatives pass; no real consent/claim; actual gate awaiting_user_approval. |
| D09 | blocked / not run | Separate explicit stakeholder authorization absent; no real holdout predictions/CI/NI decision. |
| D10 | done for pre-unseal state | Sparse support/conditional uncertainty/inconclusive rules explicit; no external/industrial claim. |
| D11 | done for pre-unseal state | Measured CPU ranking/refit/training/bytes/RSS; inference and VRAM null/unrun. |
| D12 | done for pending state | Five required files/null metrics, no test files, byte-identical isolated table-only rebuild. |
| D13 | done | Separate Codex code/tests,133 S08 tests, final full712/quick711,2 Kimi reviews final PASS, whitespace clean. |
| D14 | done as gated preparation | Honest status/report/decision log; conditional scientific completion is NOT claimed. |

## Next required decision and residual risks

Stop here at **awaiting_user_approval**. Request **one explicit user decision** approving (1) exact positive delta_NI and its practical rationale, (2) Class-aware100k primary candidate, and (3) one-time original S06 sealed_test use for all nine frozen models. No production approval template has been filled or signed. A trusted orchestrator must record that actual decision separately, bind the unchanged artifacts using `approval_bindings`, supply its independently trusted raw digest and verify the freeze again. Never auto-derive an authorization anchor from an untrusted record or treat hashes as stakeholder consent. No evaluation command has been run on this study.

The local atomic claim assumes trusted operator/storage and cannot resist deliberate account-owner rollback/deletion; failed/partial access consumes use permanently and cannot be resumed. An abrupt interruption may leave only a durable claim. Runtime support/group/integrity gates can still block a future authorized evaluation; incompatible data or an unresolved bound must not trigger retries, relaxed margins or a new "fresh" claim on the same inspected test. Sparse ODS supports, conditional bootstrap limitations, historical partial transduction and CPU-only measurements remain. Further adaptive confirmatory work after actual unsealing requires genuinely new holdout annotations. No positive NI finding, industrial equivalence, superiority, price-index guarantee or sprint acceptance is asserted.


## 2026-11-08 continuation — authorized, blocked before sealed access

The user explicitly approved Class-aware100k primary versus Full400k, Global100k exploratory, delta_NI=.01, the practical ODS research rationale (one F1 point for approximately4x model-size reduction and training savings), and one-time original S06 sealed_test inference by all9 frozen checkpoints. This authorization is recorded in `evaluation_authorization_20261108.md`; it is NOT being requested again.

The same instruction specifies one-sided95% while requiring the reviewed/bootstrap procedure to remain unchanged and ordering a stop rather than protocol-invalidating changes. Frozen protocol/pre-registration/code require alpha=.025, **one-sided97.5%**, equivalently the lower endpoint of a two-sided95% percentile interval. One-sided95% means alpha=.05, a different quantile explicitly rejected by current code. A clarification of the intended confidence level is required before any irrevocable access. Do not silently use either interpretation, edit frozen protocol/code, or retrain.

Read-only `load_plan` / `verify_training_freeze` passed all9 model/source/feature/map/training checks. Protocol/plan/training-freeze hashes equal the preparation snapshot; original sealed SHA256 remains1842de401867e21a42c1dd342716548e1c40ef1ea57cb6109845ca8e308015f4. `/tmp/s08-20261108-integrity-precheck.json` records this check and frozen bootstrap B5000/seed808/alpha.025. No evaluation claim or executable detached approval record was created, no original sealed labels loaded, no original test inference. **0/9 test evaluations; all test F1, primary Delta, confidence bound and formal NI decision remain uncomputed.** Prior preparation report and model artifacts preserved. No new code/tests/training/refit/selection/review/commit/merge/push.

Current external gate: `evaluation_blocked_pending_confidence_level_clarification` (pre-consumption). The original preparation report's awaiting_user_approval status is a preserved historical snapshot, not a denial of the latest actual consent. No synthetic test predictions or final report are fabricated.

Fresh full-suite command `.venv/bin/python -m pytest -q --tb=short > /tmp/s08-20261108-full-pytest.log 2>&1`:711passed/1skipped/1failed in160.85s. Sole failure was notebook socket creation denied by sandbox. Rerun with permitted local sockets is underway in `/tmp/s08-20261108-full-pytest-sockets.log`; final result will be appended. `git diff --check` passed. The two completed Kimi reviews remain final; no additional review initiated.

| ID | Current continuation status |
| --- | --- |
| D01 | done; existing S08 branch/worktree preserved |
| D02 | done; source/feature/map/sealed fingerprints reverified, structural projection only |
| D03 | partial; margin/rationale/primary approved, conflicting confidence-level instruction unresolved |
| D04 | done exploratory; original power/sparse-support limitations unchanged |
| D05 | frozen artifacts unchanged; detached executable evaluation decision withheld pending statistical consistency |
| D06 | done previously; no new ranking/refit |
| D07 | done;9/9 checkpoints reverified, no retraining |
| D08 | user authorization received; confidence/protocol consistency gate blocks before claim/access |
| D09 | blocked;0/9 holdout evaluations and no NI answer |
| D10 | limitations retained; no actual test supports/metrics inspected |
| D11 | prior training/refit resources retained; test inference resources unmeasured |
| D12 | historical preparation report preserved; final evaluated report blocked, no test tables fabricated |
| D13 | done; existing133 S08 tests/two reviews retained; fresh full712passed/1skipped with permitted sockets, whitespace clean |
| D14 | honest pre-access blocker recorded; final evaluated acceptance not recommended yet |

Next action is only a statistical clarification: retain reviewed alpha=.025/one-sided97.5%, or insist on alpha=.05/one-sided95%, which cannot execute under this unchanged frozen protocol. Margin/candidate/one-time-use authorization persists. The test remains unused and eligible for the authorized once-only stage after a consistent decision.


### 2026-11-08 verification completed

Socket-enabled full command `.venv/bin/python -m pytest -q --tb=short > /tmp/s08-20261108-full-pytest-sockets.log 2>&1`: **712 passed,1 skipped,1 CUDA warning in187.68s**. No test/code changes; earlier notebook failure was solely the sandbox socket restriction. Final `git diff --check` passes. Protocol/training-freeze/preparation-HTML hashes remain8aacc5c9ba3306ec1fe4dc0e16f214d976abe2f57d93891f75f53de134718175 /6d0f5725d1028f2a4a1f64f01603ce82fde7429705095e5a3d0fdb037caa8e59 /754c9bb47dc73dd1ca98aee94f6d838e05a44533d02a7aa9da32fac2733b66d0. No evaluation claim, sealed labels/inference, metrics or test-dependent files. Confidence-level clarification has not been received; stop before access under the user fail-closed restriction. Recommendation: do not accept S08 as a completed NI evaluation yet; its prepared models and unused holdout remain preserved.
