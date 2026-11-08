# S08 — Accepted with limitations

**Final status: Accepted with limitations.** The user explicitly accepted S08 and its documented limitations and authorized the intended commit(s) and fast-forward-only integration into `dev`. Push, branch deletion and artifact cleanup are not authorized.

## Outcome and acceptance boundary

**9/9 original S06 holdout evaluations succeeded. Formal primary decision: `non_inferior`.** All nine previously trained and frozen models were used exactly once in the single reviewed evaluation stage, on identical1,637 rows and the fixed293-L3 universe. No model/feature/epoch/seed selection or tuning used test data. The evaluation introduced no retraining/refit/new selection, source/test-code change, third Kimi review or deletion. Commit/staging and fast-forward-only integration were separately authorized by the user during closure; no push is authorized.

Approved primary Class-aware Chi²100k versus Full400k:

- Mean raw-top1 fixed-universe L3 macro-F1: **Full0.7303545159195526; Class-aware0.7319562708352091**.
- Estimand: mean over matched seeds42/43/44 of candidate-minus-Full F1; **Delta=+0.001601754915656543**.
- Frozen **one-sided97.5% lower confidence bound=-0.009842370585305337**.
- Approved protocol: **delta_NI=0.01**, **alpha=0.025**, **one-sided 97.5% lower confidence bound**. Strict `-0.009842370585305337 > -0.01` holds.
- Clearance above threshold only **0.000157629414694663** (0.01576294 F1 percentage points). This is a narrow conditional pass, not a robust-equivalence/superiority/industrial guarantee.
- Formal decision: **non-inferiority established**, within this pre-approved ODS PoC protocol. Upper descriptive bootstrap bound0.013467470248304170; interval crosses zero. **Superiority was not established.**

Global100k is exploratory only: mean macro-F10.721844764241538; paired mean Delta=-0.008509751678014467. No second confirmatory NI decision, alternative primary selection or inferiority conclusion.

Rounded acceptance figures: Full400k macro-F1 **0.730355**, Class-aware100k **0.731956**, Global100k exploratory **0.721845**; primary paired difference **+0.00160175**, lower bound **-0.00984237**, margin clearance only **0.00015763**. Unrounded values above drive the decision.

**S08 is accepted with documented limitations as a completed, auditable ODS PoC NI study.** Intended commits and fast-forward-only integration are explicitly authorized, subject to fresh closure verification. This is not approval for production deployment, industrial equivalence, price-index accuracy or future adaptive reuse of this test. The original holdout is now **used/consumed permanently**; no retry/resume or fresh-test claim. Further adaptive confirmatory experiments require genuinely new holdout annotations.

## Authorization and unchanged statistical protocol

`artefacts/S08/evaluation_authorization_20261108.md` records the explicit user approval of margin.01, primary Class-aware100k, Full400k reference, Global exploratory and one-time original S06 sealed use. Practical rationale: accept at most one absolute F1 percentage point for approximately4x smaller models and substantial training savings, as an ODS research tolerance, not a production quality threshold.

The initial continuation's mistaken one-sided95% wording was identified **before any label access or claim**. The user explicitly superseded it, confirming original alpha=.025/one-sided97.5%, the lower endpoint of a two-sided95% percentile interval. No frozen procedure, aggregation or fingerprint was changed. Earlier blocked/preparation records are preserved in `pre_evaluation_result_20261108.md`; their0/9/awaiting-approval statements are historical.

Mandatory gate repeated before evaluation: all9 checkpoints/feature contracts/maps/training/source fingerprints passed; original sealed identity unchanged. Approved immutable detached artifacts:

- `outputs/S08/authorization_20261108_v1/evaluation_protocol.json`: approved decisions, original bootstrap and exact frozen bindings; SHA256`2735ce0649233efc33e183e0507a79a544af5920f0c1572cab3bcb829c83a55d`.
- `outputs/S08/authorization_20261108_v1/decision.json`: explicit consent/rationale, exact bindings, caller-trusted raw SHA256`0ffd78bfb7e92746fe1ccd6f55df44fb64559d04dfae839b38388080467fb757`.
- `outputs/S08/noninferiority_20261107_v1/evaluation_claim.json`: O_EXCL single-use claim before label access; consumed=true, retry_or_resume=false; SHA256`bd1c77fb7fb83c70abb7d4f50a232ac971f7976fbc52d5040bccaa2a6576c2e5`.

Authentication assumes trusted caller/operator/storage; integrity hashes alone are not consent and deliberate account-owner ledger rollback is outside this boundary. No replay was attempted. Operator trace and claim/immutable prediction files document one execution; saved outputs alone cannot independently prove absence of undocumented external invocations.

Frozen bootstrap remains B5000, PCG64 seed808, alpha.025, linear percentile, true-L3 class-stratified row sampling. Every replicate uses the same sampled row indices across all three matched seed pairs, preserves class counts, recomputes fixed293-class F1 from row-level predictions, and averages seed differences. Seeds are fixed training replicates, not three independent datasets. Zero_division0, rawtop1/no reject; incompatible grouping/support/missing predictions fail closed. Decision uses unrounded values and strict `>`; equality/crossing would be not_established/inconclusive, not proof of inferiority.

## All nine results and descriptive comparisons

| Method | Seed | Holdout macro-F1 | Accuracy | Inference status |
| --- | --- | --- | --- | --- |
| none | 42 | 0.732680969858788 | 0.792913866829566 | success |
| none | 43 | 0.729816118465586 | 0.794135613927917 | success |
| none | 44 | 0.728566459434284 | 0.794135613927917 | success |
| class_aware | 42 | 0.728773676202439 | 0.786805131337813 | success |
| class_aware | 43 | 0.737126841531094 | 0.792302993280391 | success |
| class_aware | 44 | 0.729968294772094 | 0.789859499083690 | success |
| global | 42 | 0.710644713440658 | 0.784361637141112 | success |
| global | 43 | 0.731155341753417 | 0.793524740378742 | success |
| global | 44 | 0.723734237530539 | 0.791081246182040 | success |

| Method | Seed | Candidate-minus-Full macro-F1 | Role |
| --- | --- | --- | --- |
| class_aware | 42 | -0.003907293656348 | primary |
| class_aware | 43 | +0.007310723065508 | primary |
| class_aware | 44 | +0.001401835337810 | primary |
| global | 42 | -0.022036256418129 | descriptive_only |
| global | 43 | +0.001339223287830 | descriptive_only |
| global | 44 | -0.004832221903745 | descriptive_only |

Class-aware gains in2/3 fixed seed pairs; seed42 loses. Its mean accuracy0.7896558745672979 is lower than Full0.7937283648951334 by0.0040724903278355; Global has the same mean accuracy0.7896558745672979. Accuracy is descriptive; no test-driven endpoint switch. Three observed seeds do not characterize broad training stochasticity.

## Support, per-class and confusion diagnostics

Observed support:293 classes/1,637 rows, minimum2/maximum72. **114 classes have2 examples;186 have2–4**. No class removed or re-filtered. All9 models share exact row IDs/truth/supports. Descriptive bin means average class F1 over the same three seeds; each class is counted once for support:

| Support bin | Classes | Examples | Full mean F1 | Class-aware mean F1 | Class-aware Delta | Global mean F1 | Global Delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2–4 |186|471|.683077317|.690079365|+.007002048|.675649997|-.007427320|
| 5–9 |69|443|.797303331|.787122869|-.010180462|.784642138|-.012661193|
| 10+ |38|723|.840199009|.836761775|-.003437234|.833929182|-.006269827|

Class-aware's positive mean is concentrated in the sparse2–4 bin, with negative differences in both larger-support bins. This is descriptive sensitivity, not stable rare-class superiority. Full2,637 per-class precision/recall/F1/support rows,27 support-bin rows and2,751 confusion-pair rows are saved. Class-aware's most frequent error pair across the three model replicates is `Готовые супы → Другая готовая еда` (12); `БАД: все → Лекарства: все`, `Книги → Лекарства: все` and `Мясные полуфабрикаты → Другая готовая еда` each9. Counts sum repeated model predictions, not independent unique people/rows. Full error tables preserve method/seed.

## Data/features/models and resources

Final training already completed on original outer_train12,960 + comparison_val1,692 =14,652 development rows/all293 classes, excluding original sealed row/group identities. Full300k char+100k word reused unchanged. Class-aware/Global each ranked from development labels only, refitted75k char+25k word on the identical981,850-row S02 unlabeled corpus snapshot; IDF max difference0, per-branch L2 max deviation1.1920928955078125e-7, vocabulary/index/refit/reload provenance verified. Historical fit is partly transductive; no held-out labels used for ranking/refit/training.

All nine fresh final models: matched42/43/44, fixed14 epochs, hidden[256]/dropout.3/ReLU/no norm, AdamW.001/weight_decay0/batch256, constant LR/final epoch weights/restore_best=False. No held-out epoch/checkpoint selection. Inference used individually verified immutable CPU checkpoint byte snapshots and persisted each model's predictions before the next model; the final stash was re-read and rowID/truth/model alignment proven before statistics.

| Method | model.pt bytes | Train sum3seeds(s) | Mean train(s) | Ranking selection(s) | Shared refit(s) | Inference sum3seeds(s) |
| --- | --- | --- | --- | --- | --- | --- |
| Full |409,978,289|324.552773|108.184258|n/a|unchanged|.282508104|
| Class-aware100k |102,778,289|83.787359|27.929120|6.836284|35.162947|.192011735|
| Global100k |102,778,289|74.655681|24.885227|2.021077|41.340381|.216605986|

Training CPU/two threads:625.755814s execution wall,575.140250s in24 measured stage windows; max observed stageRSS2,864,926,720bytes. Model.pt size ratio~3.99x, Class-aware mean training~3.87x lower on this machine. Six shared ranking/transform/refit stages charged once, not per seed. Full/Class-aware/Global train transforms2.209161/1.448860/1.697763s; selected ranking transforms.716808/.711155s.

Nine inference windows total **.691125825s**, including transform+predict; peak measured inferenceRSS1,430,568,960bytes. Complete33-stage table retains24 training/shared plus9 inference stages, charge_count1. Inference windows exclude integrity verification, checkpoint snapshotting, other I/O, bootstrap and HTML; **end-to-end evaluation wall time was not separately measured**. RSS sampling may miss short peaks; peaks are not additive. VRAM=null, GPU untested. These local CPU observations do not establish industrial throughput on300m names.

## Final artifacts, immutability and rebuild

Study: `outputs/S08/noninferiority_20261107_v1/`.

- Original `protocol.json`, `plan.json`, `pre_registration.md`, `feasibility.json`, `shared/`, `runs/`, `training_freeze.json` unchanged.
- Original preparation report `report/` preserved byte-for-byte (HTML SHA256754c9bb47dc73dd1ca98aee94f6d838e05a44533d02a7aa9da32fac2733b66d0).
- **Final report: `report_evaluated_20261108_v1/report.html`**.
- Final reporting inputs: `manifest.json`, `ni_decision.json`, `approval_decision.json`, `runs.csv`, `stages.csv`, `paired_test.csv`, `per_class.parquet`, `support_bins.csv`, `predictions_test.parquet`, `confusion_pairs.csv`.
- `prediction_parts/`: nine immutable model-specific row prediction files; combined stash14,733 rows. `checkpoint_snapshots/`: protected verified copies used for the single inference stage, not needed for report rebuild.
- Independent audit/rebuild bundle: `outputs/S08/verification_20261108_v1/` (`audit_script.py`, `audit_results.json`, `audit_log.txt`, isolated `tables/`, byte-identical `rebuilt.html`). Audit code authored by the separate Codex verification worker; no inference/model deserialization.
- Independent written verification: `artefacts/S08/evaluation_verification.md`; exact saved English prompt `prompts/07_codex_evaluation_verification.md`.

| Artifact | SHA256 |
| --- | --- |
| Original protocol.json |8aacc5c9ba3306ec1fe4dc0e16f214d976abe2f57d93891f75f53de134718175|
| Original plan.json |3b51cfa1f6b8e3df3db2de0fe7de2a48a0fdb711935668a89c26d0c64bcfc953|
| Original training_freeze.json |6d0f5725d1028f2a4a1f64f01603ce82fde7429705095e5a3d0fdb037caa8e59|
| Final manifest.json |685af7031a26bf6970d4ee508d4541c14476e89367dfea939767fce644f1f82c|
| Final ni_decision.json |128b29c5deccd8ed379ff306ebbfe58868edbd067ea86b76b909e565d1286cdd|
| Combined predictions_test.parquet |85aa5f5d00131a7361dfc33091516f071413e2d34c8fcec14de84386975836d0|
| Final report.html / isolated rebuilt HTML |8547ef3b502e5188e235444ef0fd308b34c1c3c1de8466909f2963954c53a0a2|

Original sealed SHA2561842de401867e21a42c1dd342716548e1c40ef1ea57cb6109845ca8e308015f4 unchanged. All3,717 old non-S08 output files retain sizes/mtimes. Original S08 Value/DoD/orchestrator prompt, SPEC_v1.0 and unrelated notebook hashes unchanged. All source/test fingerprints match the final Kimi-reviewed snapshot; no late code fix. README and acceptance documents are updated during closure to reflect the accepted outcome; runtime protocol/code fingerprints remain unchanged.

Independent table-only HTML rebuild copied only manifest+nine listed reporting inputs, with no checkpoint_snapshots/prediction_parts/runs/shared/models/source files. Rebuilt HTML is **byte-identical**; original destinations were not overwritten. Exact successful command:

```bash
/home/ustyantsev/workspace/KKT/kkt_mlp_classifier/.venv/bin/python /home/ustyantsev/workspace/KKT/kkt_mlp_classifier/scripts/run_s08.py rebuild --report-dir /tmp/s08-independent-table-only-f9tal1js/tables --output-html /tmp/s08-independent-table-only-f9tal1js/rebuilt.html
```

The retained `outputs/S08/verification_20261108_v1/tables/` permits the same CLI rebuild with any fresh output filename, without model/refit/inference.

## Verification, exact principal commands and review status

- Latest full suite on the same unchanged source: **712passed/1skipped/1CUDA warning**,187.68s, `/tmp/s08-20261108-full-pytest-sockets.log`. Includes133 S08 tests. The immediately preceding sandbox run711passed/1skipped/1failed was solely the Jupyter socket permission; permitted-socket rerun passed without test edits. Earlier final quick suite711passed/1skipped/1deselected.
- Separate Codex implementation and independent test/fix/regression workers, original133 S08 tests. Two Kimi rounds completed, final PASS; **no third review**. Optional redundant identity check/early execution-marker orphan remain unchanged and nonblocking. No implementation change required by actual evaluation.
- Separate independent Codex numerical worker verified saved outputs only: **2,843 assertions passed**, all9 F1/accuracies, all6 seed deltas, exact bootstrap lower/upper/std/recipe, all per-class/bin/confusion rows,33 stage records, hashes/approvals/claim/row alignment and isolated HTML. Largest numerical discrepancy8.24e-17 from CSV parsing; primary point/bounds/std exactly reproduced. Audit never imported/called production statistics or opened original datasets/checkpoint model contents.
- Independent bootstrap oracle uses all saved model row predictions, same class-stratified sampled indices across seeds/models, frozen B5000/seed808 and linear.025/.975 quantiles; no new inference or alternate CI chosen. Re-running saved-row arithmetic for audit does not reopen or rerun the test models.
- `git diff --check` passed throughout; final whitespace/preservation checks recorded below. No source/test changes after the reviewed snapshot; closure updates README and acceptance documentation only.

```bash
git status --short
git branch --show-current
.venv/bin/python -m pytest -q --tb=short > /tmp/s08-20261108-full-pytest-sockets.log 2>&1
.venv/bin/python scripts/run_s08.py evaluate --output-root outputs/S08/noninferiority_20261107_v1 --approval-record outputs/S08/authorization_20261108_v1/decision.json --trusted-approval-sha256 0ffd78bfb7e92746fe1ccd6f55df44fb64559d04dfae839b38388080467fb757 --report-dir outputs/S08/noninferiority_20261107_v1/report_evaluated_20261108_v1 > /tmp/s08-20261108-authorized-evaluation.log 2>&1
codex exec --ephemeral -s workspace-write -o /tmp/s08-20261108-numerical-verification-last.md - < artefacts/S08/prompts/07_codex_evaluation_verification.md > /tmp/s08-20261108-numerical-verification.log 2>&1
.venv/bin/python /tmp/s08_independent_numerical.py > /tmp/s08-20261108-independent-numerical.log 2>&1
git diff --check
```

**Do not replay the evaluate command: use is consumed permanently.** Approved preflight evidence `/tmp/s08-20261108-approved-preflight.json`; pre-existing initial clarification precheck `/tmp/s08-20261108-integrity-precheck.json`. Exact saved-output audit code/results/log retained outside artefacts in the verification bundle. The independent Codex runtime initially failed read-only app-server storage, then launched with the same saved prompt using required runtime permission; no model invocation occurred in that failed startup.

## Final D01–D14 status

| ID | Status | Evidence |
| --- | --- | --- |
| D01 | done | Existing isolated S08 branch, integrated dev origin and separate roles; user work preserved. |
| D02 | done | Exact293-class development pool/source lineage; original holdout/feature/map/protocol fingerprints repeated before access. |
| D03 | done | Explicit margin/rationale/primary/reference/Global role and alpha.025/97.5% clarified and detached immutable decision bound before test access. |
| D04 | done (exploratory) | Pre-unseal simulated feasibility and authorized structural counts preserved; precision/power limits explicit. |
| D05 | done | Original frozen plan/protocol/source/code/checkpoints unchanged, authenticated detached approval, atomic one-time claim. |
| D06 | done previously | Two development-only100k selections/exact corpus refits; no new fit/selection after test approval. |
| D07 | done | All9 frozen14-epoch checkpoints, matched seeds, reload/interface/provenance verified; no retraining. |
| D08 | done | Actual user consent plus confidence clarification; gate passed, claim consumed before label access. |
| D09 | done |9/9 same-row evaluations, saved predictions, fixed-label paired bootstrap, correct strict NI decision. |
| D10 | done | Sparse supports, limited3 fixed seeds/power, prior dev exposure/ODS/internal-test limitations; no superiority/production guarantee. |
| D11 | done within measured scope | Training/refit/model bytes and9 inference CPU windows/RSS; VRAM/end-to-end-eval wall unavailable, not invented. |
| D12 | done | Complete final tables/manifest/immutable row predictions; original preparation report preserved; byte-identical table-only rebuild. |
| D13 | done | Separate code/tests,133 S08 tests/full712passed1skipped,2 Kimi reviews final PASS, independent2,843-assertion saved-output audit, whitespace clean. |
| D14 | done; Accepted with limitations | User acceptance recorded; final history/commands/hashes/limitations consolidated; authorized commit and fast-forward integration gated by fresh closure checks. |

## Accepted limitations and closure

The formal primary NI criterion is met **narrowly**, under the pre-approved finite bootstrap procedure only. B5000 quantiles have finite Monte Carlo resolution; no alternative bootstrap seed/count/CI or margin was tried to improve the result. Rare classes have only2–4 observations; empirical conditional resampling cannot reveal unseen error modes or guarantee nominal population coverage. Passing NI is not a general power guarantee. Development validation was examined in S06/S07; original S06 holdout was unused before this authorized stage but is now consumed. Three seeds are fixed replicates and do not estimate broad training-seed uncertainty. Unlabeled TF-IDF historical transduction, unrepresentative ODS distribution, CPU-only engineering measurements, unvalidated downstream price-index impact and no production guarantees remain.

The user accepted the auditable PoC result and resource trade-off with these limits; this does not certify industrial performance or promote Global to a confirmatory comparator. Retain Full baseline and all frozen artifacts. Future adaptive model selection/confirmation needs new independent labeled holdout data. Commit and fast-forward-only merge into dev are authorized after fresh checks; push is not authorized. No new experiments or review rounds are part of closure.


## Final acceptance and Git closure verification

The user accepted S08 with the documented limitations and explicitly authorized commit(s) on `sprint/S08-noninferiority` followed by fast-forward-only integration into `dev`; no push. Preparation, authorization, confidence clarification, evaluated report and independent verification history remain preserved in this file, `pre_evaluation_result_20261108.md`, `evaluation_authorization_20261108.md`, `evaluation_verification.md`, original prompts/reviews and append-only decision log. The one-time nine-checkpoint evaluation is complete and the original sealed test is consumed; closure does not rerun original inference, bootstrap, model training, selection or refit.

Fresh closure fullpytest passed **712 passed, 1 skipped** in **173.97s** (one CUDA warning), `/tmp/s08-closure-full-pytest.log`. Final report/manifest/approved protocol/model and feature tree verification passed without original inference/training/bootstrap (`/tmp/s08-closure-report-verification.json`). All **3,987** existing output files retain exact sizes/mtimes, with no new output files; protected user ZIP/notebook/SPEC/original S08 source documents/AGENTS hashes unchanged (`/tmp/s08-closure-preservation-check.json`). Both original staged and unstaged sets were inspected; index was empty at entry. Exact **34-file** S08 allowlist in `/tmp/s08-closure-stage-allowlist.json` excludes outputs/data/checkpoints/predictions/user files. Git whitespace checks for tracked and added files passed. Staged-file equality and staged whitespace checks are mandatory immediately before commit. Local dev and origin/dev initially equal `dd7559a550333480f4392310990b2b94f82499b1`; read-only `git ls-remote origin refs/heads/dev` confirmed the same remote SHA. Final commit/dev SHA and merge result are reported after execution. No push. Unrelated user files `S08_review_light.zip` and `notebooks/make_data.ipynb`, all datasets/models/predictions/experiment outputs, SPEC_v1.0 and original source prompts remain preserved. Integration must stop on a blocking regression or non-fast-forward condition.


Closure verification commands (no original scientific recomputation):

```bash
.venv/bin/python -m pytest -q --tb=short > /tmp/s08-closure-full-pytest.log 2>&1
git ls-remote origin refs/heads/dev
git diff --check
git diff --cached --check
git diff --cached --name-only
git merge-base --is-ancestor dev sprint/S08-noninferiority
```

The accepted34-file change set will be committed on the sprint branch and integrated using `git merge --ff-only sprint/S08-noninferiority`; stop on any failure. No history rewrite, push, sprint branch deletion or output cleanup.
