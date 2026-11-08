# S08 independent numerical verification — 2026-11-08

**PASS for the saved evaluated report:** `outputs/S08/noninferiority_20261107_v1/report_evaluated_20261108_v1`. No numerical, table-integrity or binding defects found (2,843 audit assertions passed). This is an independent saved-output numerical verification, not an additional code review. Both Kimi rounds remain exhausted; none was invoked.

Read `AGENTS.md`, `value_dod.md`, `result.md`, `evaluation_authorization_20261108.md` and `review2.md`. Latest explicit worker restrictions govern this audit. Existing dirty worktree on `sprint/S08-noninferiority` was inspected and preserved. Only this requested repository document was authored. All audit code/logs/results are in `/tmp`. No original datasets/sealed file were opened or hashed, no checkpoint/model was deserialized, and no inference, training, refit, selection, repository code/test edit, commit, merge, push or deletion occurred. CLI/serialization/report schemas were inspected to drive the audit; production statistics functions were neither imported nor called by the independent oracle.

## Independently reproduced numbers

Sklearn raw-top1 fixed-293-label macro-F1 (`zero_division=0`) and accuracy from saved prediction rows:

| Method | Seed | Macro-F1 | Accuracy |
| --- | --- | --- | --- |
| none | 42 | 0.732680969858788 | 0.792913866829566 |
| none | 43 | 0.729816118465586 | 0.794135613927917 |
| none | 44 | 0.728566459434284 | 0.794135613927917 |
| class_aware | 42 | 0.728773676202439 | 0.786805131337813 |
| class_aware | 43 | 0.737126841531094 | 0.792302993280391 |
| class_aware | 44 | 0.729968294772094 | 0.789859499083690 |
| global | 42 | 0.710644713440658 | 0.784361637141112 |
| global | 43 | 0.731155341753417 | 0.793524740378742 |
| global | 44 | 0.723734237530539 | 0.791081246182040 |

Class-aware minus Full differences for seeds42/43/44: `[-0.003907293656348121, 0.007310723065507840, 0.001401835337809909]`.
Global descriptive differences: `[-0.022036256418129120, 0.001339223287830538, -0.004832221903744882]`. All six paired-table entries and primary/descriptive roles match.

The independent bootstrap explicitly encodes saved row predictions, draws each true-class stratum with PCG64 seed808 for B5000 replicates, reuses each replicate's indices across all nine models, and rebuilds confusion-cell counts and fixed-universe F1 for every model. Thus all six primary-pair F1 values are recomputed per replicate. It never resamples F1 values or treats seeds as holdouts. Linear quantiles .025/.975 and population bootstrap standard deviation (`ddof=0`) reproduce:

| Primary quantity | Independent value |
| --- | --- |
| Mean paired Delta | 0.001601754915656543 |
| One-sided97.5% lower bound | -0.009842370585305337 |
| Upper descriptive bound | 0.013467470248304170 |
| Bootstrap standard deviation | 0.005955043260347070 |
| Lower bound above -0.01 | 0.000157629414694663 |

The strict rule `lower_bound > -0.01` holds; saved `non_inferior` status is correct under the approved frozen procedure. This clearance is small; the margin/confidence/procedure were not changed or selected after observation. Global remains descriptive only. Point, seed deltas, lower/upper and bootstrap std agree exactly as floating-point values. Largest discrepancy in all numerical comparisons is `8.24e-17` (CSV parsing precision only).

The complete bootstrap recipe matches, including versions, seed order, label hash and encoded input SHA256 `7cb9be2dcc9c66ffce0175c81d191d299301c14bb7edbfa179f8d752efbf4716`.

## Integrity and reporting checks

- All nine successful training/inference statuses; exactly1,637 unique rows per run with identical ordered row IDs/truth (14,733 saved prediction rows). All predictions are in the frozen293-label universe. Supports sum1,637, minimum2; 186 classes have2–4 rows (114 have2).
- All nine model SHA256 values and model sizes match frozen training metadata and approval bindings. Only frozen training/result/receipt JSON metadata was hashed/read; model contents were not opened. Full model409,978,289bytes; each selected model102,778,289bytes.
- All2,637 per-class precision/recall/F1/support rows, all27 support-bin rows, unique class counting per method/seed, and all2,751 error confusion-pair counts match independent reconstruction. Confusion totals equal each run's misclassified row count.
- All33 stage rows are distinct and charge_count=1; 24 original training/shared rows match frozen execution/stage metadata. Six shared ranking/transform/refit costs are charged once, not per seed. Nine inference windows have positive saved elapsed time/RSS and null CPU VRAM. Saved inference seconds summed over three seeds: Full0.282508104, Class-aware0.192011735, Global0.216605986. Saved train sums:324.552773436 /83.787359093 /74.655681101seconds respectively. These are consistency checks of local measurements, not fresh benchmarks. A pandas FutureWarning concerns equivalent CSV NaN versus JSON null in unavailable VRAM fields; no comparison failed.
- Protocol `8aacc5c9ba3306ec1fe4dc0e16f214d976abe2f57d93891f75f53de134718175`, plan `3b51cfa1f6b8e3df3db2de0fe7de2a48a0fdb711935668a89c26d0c64bcfc953`, training_freeze `6d0f5725d1028f2a4a1f64f01603ce82fde7429705095e5a3d0fdb037caa8e59` match preparation and approval. Internal JSON seals, manifest/table fingerprints, frozen execution and metadata fingerprints pass.
- Detached approval raw trusted anchor `0ffd78bfb7e92746fe1ccd6f55df44fb64559d04dfae839b38388080467fb757` matches original decision, report copy, manifest and claim. Immutable approved evaluation protocol matches `2735ce0649233efc33e183e0507a79a544af5920f0c1572cab3bcb829c83a55d`. Full bindings agree, including fixed reference/primary/Global role, margin.01 and alpha.025/one-sided97.5% clarification. Claim binds this report and approval, records consumed=true/retry_or_resume=false, and matches the manifest.
- Post-audit original report tables, manifest/HTML, claim, approved files and protocol/plan/freeze hashes are unchanged.

## Isolated table-only rebuild and commands

Executed independent oracle twice while strengthening preservation assertions; both succeeded. This repeats saved-row calculations only, not inference. Final command:

```sh
.venv/bin/python /tmp/s08_independent_numerical.py > /tmp/s08-20261108-independent-numerical.log 2>&1
```

Fresh directory `/tmp/s08-independent-table-only-f9tal1js/tables` contains **only** `manifest.json` and its nine listed table/decision files; no checkpoints, prediction_parts, run/shared/source/model artifacts were copied. Executed existing CLI:

```sh
/home/ustyantsev/workspace/KKT/kkt_mlp_classifier/.venv/bin/python /home/ustyantsev/workspace/KKT/kkt_mlp_classifier/scripts/run_s08.py rebuild --report-dir /tmp/s08-independent-table-only-f9tal1js/tables --output-html /tmp/s08-independent-table-only-f9tal1js/rebuilt.html
```

Exit0; generated HTML is byte-identical to the original final report. HTML SHA256 `8547ef3b502e5188e235444ef0fd308b34c1c3c1de8466909f2963954c53a0a2`. The original evaluation/rebuild destinations were never used for writes.

Audit evidence: `/tmp/s08-20261108-independent-numerical.json` (checks, errors, metrics, hashes, recipe, costs, exact rebuild command and limitations), `/tmp/s08_independent_numerical.py`, `/tmp/s08-20261108-independent-numerical.log`. `/tmp` evidence is local and may be ephemeral.

## Discrepancies and limits

No evaluated-output numerical/integrity defect found. `artefacts/S08/result.md` still has a pre-evaluation header/outcome and historical0/9 statements: an orchestrator continuation/closure entry is needed to reconcile it with this evaluated report. This worker leaves that document unchanged and does not certify all sprint DoD items closed.

Saved claim/approval/predictions demonstrate consistent one-time-use bookkeeping; they cannot independently prove the historical absence of extra model invocations or label access. Atomic-claim guarantees assume trusted local operator/storage and cannot resist deliberate rollback/deletion. No replay attempt was made.

The uncertainty is conditional on the finite internal ODS holdout, observed class composition/errors and only three fixed seeds. Sparse-class empirical bootstrap cannot reveal unseen errors or guarantee population coverage; no formal power guarantee follows from passing NI. Development data was already examined; historical TF-IDF fitting was partially transductive. ODS is unrepresentative of the industrial corpus. No external/industrial non-inferiority, superiority, downstream price-index accuracy, broader training-seed variability or GPU/industrial throughput claim is supported. Future adaptive confirmatory work requires genuinely new holdout annotations.
