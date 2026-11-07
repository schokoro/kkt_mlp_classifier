# S07 — Accepted with limitations

The final study completed **36/36 successful runs**, with zero failed, skipped or pending runs, and **nine verified selected TF-IDF refits**. The single aggregated report is `outputs/S07/stability_20261107_v2/report/report.html`. All ten required report files exist. The user accepted S07 with the documented limitations and explicitly authorized commit and fast-forward-only integration into `dev`; push is not authorized.

**Observation:** S06's +0.013905 Class-aware Chi² gain did not reproduce as a stable positive effect. S07 Class-aware mean paired Δ macro-F1 is only +0.000873, positive in 1/3 split means and 5/9 seed pairs. G×I is +0.000085, positive in 1/3 split means and 6/9 seed pairs. These nine pairs are correlated development resamples, not nine independent holdouts. No confidence interval is asserted.

**Research recommendation:** continue testing 100k as a resource-saving PoC configuration on larger, independently held-out data; do not select it on a claim of improved macro-F1 from S07. Class-aware is a reasonable simple candidate alongside full baseline, not an established winner. G×I has no convincing macro-F1 advantage over Class-aware here to justify its additional pipeline cost. The ODS dataset is not representative of the future corpus; no industrial accuracy or throughput claim follows.

## Provenance, splits and training

Branch `sprint/S07-stability-validation` was created from integrated `dev` `c7df0d4d47f83a380ca2c3e91f8956bc7dacbd5f`. Revised working AGENTS roles were verified before delegation; all initial user changes/untracked files were preserved. Source hashes and preflight are in `preflight.md` and `implementation_handoff.md`.

The development pool is exactly S06 outer_train + comparison_val: **14,652 rows, 293 L3**, no re-filtering. There are 14,652 distinct normalized groups, with no multi-L3 groups. Original sealed test has 1,637 rows and remains excluded; only structural name/row-ID projection and byte fingerprints were used, no test labels for training, ranking, inference, diagnostics or quality metrics. Final sealed-file SHA256 remains `1842de401867e21a42c1dd342716548e1c40ef1ea57cb6109845ca8e308015f4`.

Split seeds **101/202/303** each give outer_train **13,144**, comparison_val **1,508**, train_fit **11,766**, train_monitor **1,378**. All class minima are 6/2/2; seed-specific row/group identities, support tables and hashes are frozen. Deterministic assignments are constructive feasibility witnesses. Model seeds **42/43/44** are shared across all four methods. Every split used its own new rankings, selected vocabularies, refits and model checkpoints. G×I uses that split's predetermined full seed42 checkpoint, never a validation winner.

Final plan file SHA256: `cafcabcdf5e33688d11cc2dfd31ce70743d8a6d09683efda3dc505524fff8deb`. All 36 keys, nine dependencies, inputs, corpus selection, versions, code fingerprints and settings were frozen before v2 validation inference. Pinned S02 features are 300k char + 100k word; selected artifacts are 75k + 25k. Every selected vectorizer was refitted on exactly the original 981,850-row corpus snapshot, with recomputed IDF and separate branch normalization; no matrix slicing or S06 supervised artifact reuse.

The final accepted S06 protocol was verified: **14 final epochs**, hidden [256], dropout [.3], ReLU/no norm, AdamW lr .001, weight_decay 0, batch256, scheduler none, final epoch weights. CPU/two threads; no S07 validation tuning. Inner monitor is included in outer_train during final training and is diagnostic only. Scoring is fixed-universe raw top-1 L3, no rejection, retaining S06's bias path for empty feature rows. Existing S03–S06 implementation/API files are unchanged.

## Metrics and paired effects

Unrounded values drive calculations; displayed values below are rounded only for presentation. Aggregate means give each split equal weight (each has three model seeds).

| method | macro_F1 | accuracy | paired_delta_F1 | paired_delta_accuracy |
| --- | --- | --- | --- | --- |
| none | 0.724313 | 0.780799 | 0.000000 | 0.000000 |
| global | 0.720741 | 0.778588 | -0.003572 | -0.002210 |
| class_aware | 0.725186 | 0.778809 | 0.000873 | -0.001989 |
| gradient_x_input | 0.724398 | 0.784262 | 0.000085 | 0.003463 |

Within each split, `within_split_metric_std` is population SD of model-seed macro-F1; `within_split_seed_std` is population SD of the same-seed paired differences. Gains/losses below count **seed pairs**.

| split_id | method | mean_macro_f1 | mean_delta_macro_f1 | within_split_metric_std | within_split_seed_std | gains | losses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| split_101 | none | 0.719538 | 0.000000 | 0.002615 | 0.000000 | 0 | 0 |
| split_202 | none | 0.717820 | 0.000000 | 0.004309 | 0.000000 | 0 | 0 |
| split_303 | none | 0.735581 | 0.000000 | 0.007929 | 0.000000 | 0 | 0 |
| split_101 | global | 0.723184 | 0.003645 | 0.003135 | 0.003644 | 2 | 1 |
| split_202 | global | 0.708477 | -0.009343 | 0.002428 | 0.003027 | 0 | 3 |
| split_303 | global | 0.730561 | -0.005020 | 0.002359 | 0.005737 | 1 | 2 |
| split_101 | class_aware | 0.719145 | -0.000394 | 0.001849 | 0.000886 | 2 | 1 |
| split_202 | class_aware | 0.717472 | -0.000348 | 0.000722 | 0.004156 | 1 | 2 |
| split_303 | class_aware | 0.738943 | 0.003362 | 0.003128 | 0.007461 | 2 | 1 |
| split_101 | gradient_x_input | 0.721091 | 0.001553 | 0.002120 | 0.004451 | 2 | 1 |
| split_202 | gradient_x_input | 0.717652 | -0.000168 | 0.004464 | 0.004998 | 2 | 1 |
| split_303 | gradient_x_input | 0.734450 | -0.001131 | 0.001100 | 0.008741 | 2 | 1 |

Across-split SD is computed from the three complete split means, separately from within-split seed variation. Gains/losses here count **split means**.

| method | split_count | between_split_metric_mean_std | between_split_mean_std | gains | losses |
| --- | --- | --- | --- | --- | --- |
| none | 3 | 0.007998 | 0.000000 | 0 | 0 |
| global | 3 | 0.009180 | 0.005400 | 1 | 2 |
| class_aware | 3 | 0.009751 | 0.001760 | 1 | 2 |
| gradient_x_input | 3 | 0.007245 | 0.001110 | 1 | 2 |

G×I minus Class-aware mean macro-F1 is -0.000788; G×I's mean accuracy is higher, but accuracy is secondary and does not establish a justified attribution premium. `paired_summary.csv` and the HTML retain split-local comparisons, paired counts and variability; no unsupported intervals or independence claims.

## Rare classes and errors

**200 of 293 classes have only 2–4 validation examples** (512 examples total) in every split. Remaining bins: 59 classes/385 examples (5–9), 24/319 (10–19), 10/292 (>=20). Each class is counted once per bin; repeated seed/split predictions do not increase its independent support.

| method | bin | class_count | example_support | mean_F1 | mean_paired_delta |
| --- | --- | --- | --- | --- | --- |
| none | 2-4 | 200 | 512 | 0.687475 | 0.000000 |
| none | 5-9 | 59 | 385 | 0.788094 | 0.000000 |
| none | 10-19 | 24 | 319 | 0.864452 | 0.000000 |
| none | >=20 | 10 | 292 | 0.748431 | 0.000000 |
| global | 2-4 | 200 | 512 | 0.684426 | -0.003049 |
| global | 5-9 | 59 | 385 | 0.785827 | -0.002267 |
| global | 10-19 | 24 | 319 | 0.853030 | -0.011422 |
| global | >=20 | 10 | 292 | 0.745531 | -0.002900 |
| class_aware | 2-4 | 200 | 512 | 0.690925 | 0.003450 |
| class_aware | 5-9 | 59 | 385 | 0.784743 | -0.003351 |
| class_aware | 10-19 | 24 | 319 | 0.855527 | -0.008925 |
| class_aware | >=20 | 10 | 292 | 0.746209 | -0.002222 |
| gradient_x_input | 2-4 | 200 | 512 | 0.687523 | 0.000048 |
| gradient_x_input | 5-9 | 59 | 385 | 0.788797 | 0.000703 |
| gradient_x_input | 10-19 | 24 | 319 | 0.859728 | -0.004725 |
| gradient_x_input | >=20 | 10 | 292 | 0.757152 | 0.008721 |

Class-aware's small aggregate gain is concentrated in the sparse 2–4 bin (+0.003450), with negative mean effects in all other bins. This is a fragile observation, not evidence of general rare-class improvement. Across the nine paired resamples, its largest mean per-class gains include `Серебряные ювелирные изделия` (+0.296296), `Скатерти` and `Ключи` (+0.259259 each); losses include `Зеленый чай` (-0.259259), `Говядина` and `Холсты, крупноформатная бумага` (-0.222222 each). All these classes have only **2 validation examples per split**. Their large changes cannot support reliable category-level conclusions.

For Class-aware, recurring confusion `Мясные полуфабрикаты → Другая готовая еда` totals 24 errors versus 19 in matched full runs; `Мясные деликатесы → Другая готовая еда` totals 21 versus19. These totals sum repeated runs and overlapping development resamples, not unique independent examples. Full per-run precision/recall/F1, paired changes and confusion counts are in `per_class.parquet`, `support_bins.csv`, `confusion_pairs.csv` and HTML.

## Vocabulary stability

All 18 pair/branch comparisons were independently recomputed from stable feature **names**, not local column IDs. Exact pairwise intersections/Jaccards are in `feature_stability.csv`.

| method | branch | min_Jaccard | max_Jaccard | min_intersection | max_intersection |
| --- | --- | --- | --- | --- | --- |
| class_aware | char | 0.764021 | 0.769035 | 64967 | 65208 |
| class_aware | word | 0.910293 | 0.914095 | 23826 | 23878 |
| global | char | 0.808951 | 0.816486 | 67079 | 67423 |
| global | word | 0.910293 | 0.914095 | 23826 | 23878 |
| gradient_x_input | char | 0.729705 | 0.733403 | 63280 | 63465 |
| gradient_x_input | word | 0.909126 | 0.913509 | 23810 | 23870 |

Char overlap is lower for G×I than for either Chi² method. High word overlap is specific to this dataset/budget. Class-aware word selection required global fallback of 4,249 / 4,288 / 4,286 features in splits101/202/303; G×I required remainder fill of 4,275 / 4,311 / 4,304. Char fallback/remainder counts were zero. Do not interpret every selected word as positive class-aware/attribution evidence.

## Resource accounting

Final execution: `2026-11-07T14:58:19.525238+00:00` to `2026-11-07T15:35:15.137285+00:00`, **2,215.612 s wall time**. Stage windows sum to **1,992.146 s**, excluding unmeasured integrity/I/O overhead; they are not a substitute for end-to-end wall time or additive memory peaks. Report generation time is outside execution wall time. Measured max stage RSS **3,322,728,448 bytes** (~3.095 GiB), sampled and potentially missing brief peaks. VRAM is **null**, CUDA untested.

| method | stage | count | seconds |
| --- | --- | --- | --- |
| class_aware | inference | 9 | 0.512 |
| class_aware | ranking_selection | 3 | 20.242 |
| class_aware | ranking_transform | 3 | 1.913 |
| class_aware | refit | 3 | 109.995 |
| class_aware | train | 9 | 228.802 |
| class_aware | train_transform | 9 | 4.276 |
| global | inference | 9 | 0.544 |
| global | ranking_selection | 3 | 6.071 |
| global | ranking_transform | 3 | 1.899 |
| global | refit | 3 | 123.143 |
| global | train | 9 | 238.642 |
| global | train_transform | 9 | 4.546 |
| gradient_x_input | attribution_ranking_selection | 3 | 11.218 |
| gradient_x_input | inference | 9 | 0.565 |
| gradient_x_input | ranking_transform | 3 | 1.880 |
| gradient_x_input | refit | 3 | 137.984 |
| gradient_x_input | train | 9 | 234.661 |
| gradient_x_input | train_transform | 9 | 4.648 |
| none | inference | 9 | 0.766 |
| none | train | 9 | 853.591 |
| none | train_transform | 9 | 6.247 |

Nine full training stages total **853.591 s** (94.843 s/run); 27 selected stages total **702.105 s** (26.004 s/run). Nine shared refits total **371.122 s**, two Chi² methods' six ranking/selection stages **26.313 s**, three G×I attribution/ranking/selection stages **11.218 s**. Ranking transforms, downstream transforms and inference remain separately listed above.

Serialized `model.pt` is **409,978,289 bytes** full versus **102,778,289 bytes** selected (~4x smaller). Mean training: global26.516 s, class-aware25.422 s, G×I26.073 s. Measurements are local CPU observations, not production throughput.

Shared stages are charged once per split/method. Method totals including shared refits/rankings/transforms/downstream/inference: full860.604 s; global374.846 s; class-aware365.740 s; G×I390.956 s. Full seed42 checkpoint training/input-transform cost **278.657 s** across three splits is already counted once in full comparator costs. G×I's incremental checkpoint training cost within this joint comparison is zero; a standalone G×I scenario allocates that cost explicitly, total **669.613 s** (390.956 +278.657). This allocation is not an additional measured training run.

## Failed first attempt and independent-review limitation

Preserved `outputs/S07/stability_20261107_final/`: **0 success / 1 failed / 0 skipped / 35 pending**, truthful partial report. It stopped after first model training but **before validation inference** because strict JSON rejected an uncomputed constant-LR `val_loss_subset=NaN` diagnostic. The eight saved checkpoint tensors were finite, but unsaved epoch losses/metrics cannot be retrospectively verified. Its measured transform/training cost was **97.043 s** (0.811 +96.232), additional to v2 costs, and is not hidden as a success or reused.

A separately delegated S07-only fix maps only that explicitly uncomputed diagnostic to null; infinity, measured diagnostic NaN, nonfinite losses/metrics and model tensors remain rejected. No global permissive JSON, training-setting change or weakened gate. New independent tests exercise >=10 real optimizer steps and adversarial nonfinite inputs. V2 uses a fresh root/plan and retrains everything.

Kimi rounds1/2 are preserved in `review1.md`/`review2.md`, final verdict PASS. M1/M2 provenance findings and L1–L5 reporting ambiguities were repaired and regression-tested before round2. **The later LR serialization/finite-model fix is independently unreviewed**, because the agreed two-round budget was exhausted; no third review. This residual risk remains despite 53 extra regression tests and green full suite. No critical open code-review finding remains from round2. Optional notes include the unused caller `parts` parameter, frozen run-ID suffix role matching, and partial G×I contrast counts; none were used to relax gates.

## Verification and reproduction

- Independent S07 tests: **143 passed** (90 original/review regressions +53 serialization cases); tiny synthetic 36-run/nine-refit smoke verifies mechanics only.
- Full `.venv/bin/python -m pytest -q`: **579 passed, 1 skipped**, 74.18 s; CUDA initialization warning. Notebook execution requires local sockets; earlier sandbox failures were resolved by the authorized environment escalation, without modifying tests.
- Final output verification: all36 F1/accuracy values, 27 paired deltas, 10,548 class rows, 54,288 validation-prediction rows, support bins/confusion totals, 18 stable-name overlaps, nine unique shared refits and single-charge stage events. Metric/delta differences from independent recomputation <=1.71e-16; per-class differences exactly0. Evidence `/tmp/s07-final-numerical-verification.json`.
- All ten report files exist; no failed/pending prediction is presented as success. Report build verified artifact reload/provenance, and finite-model checks run before save/after load.
- Isolated rebuild copied only manifest +eight report tables to `/tmp/s07-final-tables-only-bk1nx2qf/`; no models, source data, splits, runs or shared artifacts copied. `rebuilt.html` is **byte-identical** to original HTML. Unit tests additionally prohibit access outside report tables during rebuild.
- `git diff --check` and separate `git diff --no-index --check /dev/null <new-file>` checks passed. Sealed-file and final-plan hashes remain unchanged.

Actual principal commands (use fresh roots for a new study; existing execution is single-use):

```bash
.venv/bin/python -m pytest -q tests/test_s07*.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_s07.py preflight --output-root outputs/S07/preflight_implementation_20261107_final
.venv/bin/python scripts/run_s07.py plan --output-root outputs/S07/stability_20261107_v2 --device cpu --num-threads 2
.venv/bin/python scripts/run_s07.py execute --output-root outputs/S07/stability_20261107_v2
.venv/bin/python scripts/run_s07.py report --output-root outputs/S07/stability_20261107_v2
.venv/bin/python scripts/run_s07.py rebuild --report-dir /tmp/s07-final-tables-only-bk1nx2qf --output-html /tmp/s07-final-tables-only-bk1nx2qf/rebuilt.html
sha256sum outputs/S06/splits/sealed_test.parquet outputs/S07/stability_20261107_v2/plan.json
git diff --check
```

Complete English subordinate prompts (including failed launch drafts) are preserved in `prompts/`. Runtime logs are outside artefacts: `/tmp/s07-formal-v2-execution.log`, `/tmp/s07-full-pytest-final.log`, `/tmp/s07-kimi-review1.log`, `/tmp/s07-kimi-review2.log`, delegated Codex logs/results. Failed review launches due to argument size/incompatible CLI flags/read-only session storage did not consume extra independent review rounds; exactly two reviews ran. No read-only review is counted as testing.

## Final D01–D14 assessment

| ID | Status | Evidence / limitation |
|---|---|---|
| D01 | done | Isolated branch from integrated dev; revised roles confirmed; user changes preserved. |
| D02 | done | Original sealed identities/groups excluded; source fingerprint unchanged; no sealed quality use. |
| D03 | done | Three deterministic split witnesses, fixed 293-class universe, minima6/2/2, saved identities/supports/hashes. |
| D04 | done | Frozen 36-row v2 plan and nine dependency identities before v2 scoring; tamper/resume rejects. |
| D05 | done | Nine fresh full baselines, all14 final epochs, verified protocol/checkpoints. |
| D06 | done | Nine split-local rankings; new fixed seed42 checkpoint per split; train-only inputs checked. |
| D07 | done | Nine 75k/25k fixed-vocabulary refits on exact pinned corpus, IDF/norm/reload checks. |
| D08 | done | Final36/36 success; first failed attempt separately preserved/reported. |
| D09 | done | All27 unrounded paired deltas verified; separate split/seed variability; no invalid CI. |
| D10 | done | Verified per-class/support-bin/confusion analysis, rare counts prominent. |
| D11 | done | All18 branch/split-pair overlaps verified using feature names. |
| D12 | done | Ten-file self-contained report; byte-identical isolated table-only rebuild; single shared charging. |
| D13 | done with limitation | Separate Codex implementation/tests, exactly2 Kimi rounds, full579/skip1; later serialization fix independently unreviewed. |
| D14 | done | Results, STATUS, append-only decisions, README commands and conditional recommendation updated; user acceptance recorded. |

No current execution blocker remains. GPU, independent external/industrial validity and industrial corpus scale are untested. The unsupervised corpus fit is partially transductive (historical two matching ODS texts); new development assignments do not create an untouched holdout. Source S07 Value/DoD, original orchestrator prompt, SPEC_v1.0 and prior sprint results are preserved. The user has accepted S07 with these limitations. Earlier attempt histories and their partial statuses remain preserved.


## User acceptance and closure verification — 2026-11-07

**Accepted with limitations.** The user explicitly accepted the complete S07
study and its documented limitations, including the tested but independently
unreviewed final LR-diagnostics serialization fix. The two-round review budget
is exhausted; no new review, feature work, research experiment or S08 work is
part of closure. Commit and fast-forward-only merge into `dev` are authorized;
no push, branch deletion, history rewrite or experiment cleanup is authorized.

Final research record remains: 36/36 success, splits101/202/303, model
seeds42/43/44, nine TF-IDF refits; report and complete manifest at
`outputs/S07/stability_20261107_v2/report/`. Full400k macro-F1=0.724313;
Global100k pairedΔ=-0.003572, Class-aware100k +0.000873, G×I100k +0.000085.
No selected method improved consistently across three splits; the apparent
Class-aware S06 advantage was not reproduced. Vocabulary reduction cuts model
size from about410 MB to103 MB and substantially reduces CPU training time.

Accepted limitations remain explicit: many L3 classes have extremely small
validation support; development data and validation results were examined in
previous experiments; three development splits are not independent external
validation; execution was CPU-only; final LR-diagnostic serialization fix was
tested without another independent code review; industrial classification
quality and scalability are not established. The earlier failed experiment,
its report/checkpoint and all other outputs remain preserved.

Closure verification passed before commit:

- Full `.venv/bin/python -m pytest -q`: **579 passed, 1 skipped**, 77.99 s;
  one CUDA initialization warning. Log `/tmp/s07-acceptance-pytest.log`.
- `load_plan` and `load_report_tables` verified the frozen successful study,
  complete manifest, 36 successes, split/model seeds and nine refit receipts.
- `git diff --check` and `git diff --cached --check` both passed without
  whitespace exceptions. The 28-file staging allowlist excludes user data,
  models, outputs, ZIP archives and the unrelated notebook/S08 documents.
- All 3,717 existing experiment output files retained their sizes/mtimes,
  including the earlier failed attempt. Protected source-document/notebook
  hashes are unchanged; SPEC_v1.0 and previous decisions were not rewritten.
- Concurrent worktree changes were observed: the two original ZIP archives
  became absent and untracked S08 source documents appeared during verification.
  This closure performed no deletion, restoration or S08 work; the unrelated
  current worktree state is preserved and excluded from staging.
- Integration is restricted to the authorized commit and `git merge --ff-only`;
  no push or sprint-branch deletion. Final commit/dev hashes and integration
  result are reported to the user after execution.
