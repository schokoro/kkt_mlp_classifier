# S08 implementation handoff (work in progress)

Scope: implementation only; independent tester owns all tests. No formal production workload or sealed inference is authorized at this stage.

Planned public API (import from `kkt_clf.s08`): `S08Config` (same source/fixture fields as S07Config), `verify_sources(cfg)`, `preflight(cfg, output_root)`, `write_plan(cfg, output_root)`, `load_plan(output_root)`, `execute(output_root)`, `verify_training_freeze(output_root)`, `evaluate(output_root, approval=None)`. All output roots must be fresh; execution is single-use. `execute` is dev-only (nine final models); do not run formal workload until tests/review pass.

Statistics (`kkt_clf.s08_stats`): `paired_bootstrap(y_true, full_predictions, candidate_predictions, *, labels, row_ids=None, groups=None, B=5000, rng_seed=808, alpha=.025, fixture=False)` takes predictions shaped `(3, n_rows)` in fixed seed order 42/43/44, and returns point estimate, seed differences, lower/upper percentile bounds and deterministic regeneration recipe. Production requires 293 distinct labels, all supported. Fixture=True only relaxes universe size, never CI pairing/design. Shared groups containing more than one row fail closed. `ni_decision(result, margin)` uses strict `lower_bound > -margin`; null margin yields pending, no approved decision.

Reports (`kkt_clf.s08_report`): `build_report(output_root, report_dir=None)` (fresh directory), `render_report(report_dir, output_html)` (new file; table-only). Pending reports have 9 rows and null quality fields. No test files are created.

CLI `scripts/run_s08.py`: preflight/plan/execute/report/rebuild/evaluate. No approval-generating CLI. Protocol always freezes `delta_NI: null`, margin_status/primary_status `unapproved`. Planning .01 is illustrative only.

Approval security: a local JSON boolean or self-hashed record cannot establish stakeholder consent. No trusted external authentication anchor has been provisioned. `evaluate` is deliberately closed even if supplied a purported approval; it never opens sealed label columns or runs holdout inference. Future integration requires orchestrator-recorded explicit user decision authenticated outside this worker's writable experiment tree, binding exact margin, rationale, candidate, one-time use, protocol/plan/training/model/feature hashes, plus an external atomic consumption ledger. Synthetic approvals cannot unlock production.

This handoff will be finalized with implemented signatures, smoke commands and limits.

## Implemented API and current boundary

The planned signatures above are implemented. Additional public integrity helpers:
`verify_selected(root, plan, method, full)`, `verify_checkpoint(root, plan, run,
builder, feature_id, device=None)`. `load_plan(output_root, verify_inputs=True)`
validates semantic contracts, pinned sources, runtime/code, full development union,
original five partition provenance, protocol and dependency plan. `verify_sources`
uses only `row_id/name` projection for sealed data; its dev pool has all development
labels. Production source pins are imported unchanged from S07. `METHODS` is
`('none', 'class_aware', 'global')`; there is no GxI execution path. Each selected
artifact is checked by replaying the frozen Chi2 ranking, feature-name/index order,
IDF equality, per-branch L2 and exact corpus source/fit metadata. All model
checkpoints are recursively frozen, reloaded and checked against label maps,
architecture, feature contract and training/run provenance.

Fixture contract: all five original parts are required (`outer_train`,
`comparison_val`, `sealed_test`, `train_fit`, `train_monitor`), including
`n_rows`, `n_groups`, bytes/hash; inner parts must exactly cover outer_train.
Sealed fixture may deliberately have only row_id/name columns. No synthetic
approval path exists. Fixture source pins cover manifest/protocol/corpus/features;
all remaining S07 fixture settings are retained. Fixed model seeds and pairing
remain unchanged even for fixtures.

The implementation intentionally does **not** provide post-approval holdout
inference or test-dependent reporting. `evaluate(output_root, approval=None)`
always rejects before labels/inference, including if passed a purported approval
record. A trusted external consent issuer/atomic ledger must first be specified,
provisioned, and independently reviewed; the current code cannot authenticate
that decision. This is a documented conditional D09 implementation limitation,
not a completed confirmatory study. Hash seals provide mutation/staleness evidence,
not cryptographic authentication of consent. Future authorized integration must
bind the exact freeze, verify every checkpoint/feature before label reads, persist
an external single-use claim before evaluation, and never permit retry/resume
following partial access. No arbitrary approved boolean is accepted.

Training uses the full development union as both fitting input and explicitly
in-sample diagnostic monitor required by the existing training API; patience equals
fixed epochs and restore_best=False. No held-out metric can select epochs or a
checkpoint. Pending report quality columns remain null. No formal9 workload has
been launched by this worker. There is no configurable production resource budget;
execution stops at the first dependency/training/integrity failure and records all
remaining planned rows as pending (or skipped for a failed shared dependency).
An orchestrator-defined secondary resource gate, if required, must be frozen
before production execution; it must not change primary seed pairing.

## Worker verification (implementation only)

- `python -m compileall -q` on all four implementation files succeeded.
- `scripts/run_s08.py --help` succeeded.
- Inline synthetic smoke adapted the pre-existing S07 fixture with five original
  partitions, 32 full/8 selected features and one frozen epoch: nine fresh models,
  two independent Chi2 ranking/refits, save/reload, checkpoint/training freeze,
  pending and completed-training blocked reports all succeeded. Rebuilt HTML was
  byte-identical; purported `{'approved': True}` was rejected; sealed fixture
  remained only `row_id/name`. Output: `/tmp/s08-implementation-smoke-1_0zxh9t/`.
  This snapshot predates the later identity/ranking replay hardening and is a
  preserved historical smoke, not an artifact eligible for reuse after code edits.
- Production **preflight only**:
  `.venv/bin/python scripts/run_s08.py preflight --output-root /tmp/s08-production-preflight-implementation-v1`
  passed (zero production runs, no sealed labels/inference). It verified development
  14,652/293, all original partition hashes, fixed14 epochs, Full400k and original
  corpus981,850. Its numerical feasibility scenario uses dev-derived 1,508-row
  projected support: 126 classes at2,36 at3,38 at4 (200/293 total at2–4).
  Simulated disagreement .03: delta=-.0138704, lower=-.0233106, upper=-.00529998,
  downward uncertainty=.00944022. Simulated .10: delta=-.0499362,
  lower=-.0670070, upper=-.0350120, downward uncertainty=.0170708.
  These are hypothetical categorical prediction scenarios, **not sealed scores**,
  power estimates, or approval of .01. Current feasibility output additionally
  records interval width and an explicitly labelled point-centered precision
  projection. The preflight snapshot predates that additive reporting field.
- The first run of independently authored S08 tests reported 61 passed/2 failed;
  both failures found NaN row/group identity accepted as singleton. Implementation
  now rejects all pandas-null identities; the independent statistical module rerun
  passed29. No test file was authored or modified by this worker.
- Further independent tests were rerun after feature/input contract hardening;
  final command/results are appended below when complete.
- `git diff --check` passed. No commit/merge/push/delete; original S08 documents,
  SPEC, prior outputs and unrelated notebook were preserved. Tests appearing in
  the working tree belong to the independent tester.

Final worker reruns after input/ranking hardening:

```bash
.venv/bin/python -m pytest -q tests/test_s08_integrity_independent.py tests/test_s08_statistics_independent.py tests/test_s08_smoke_independent.py > /tmp/s08-implementation-independent-tests.log 2>&1
# 64 passed in33.05s
.venv/bin/python -m pytest -q tests/test_s08_report_independent.py > /tmp/s08-implementation-report-tests.log 2>&1
# 14 passed in13.28s
```

These are runs of the **independent tester's** files, not worker-authored tests.
The passing synthetic smoke in that rerun includes current source verification,
feature ranking replay and checkpoint checks. Full-project pytest and Kimi review
remain orchestrator-owned and are not claimed here. The implementation's own
historical smoke is explicitly distinguished above. `git diff --check` and
`git diff --no-index --check /dev/null` for each new implementation file passed.

## Append-only H1 resolution (2026-11-07; supersedes disabled-path limitation)

The historical disabled-evaluation statements above describe the first review
snapshot and are retained as facts about that snapshot. H1 correctly identified
that adding evaluation after formal code freeze would invalidate frozen models.
The implementation now includes a complete explicit authorized path **before
production training/freeze**. No external issuer/service or external ledger is
required. A trusted caller-supplied SHA256 anchor, obtained by the orchestrator
ONLY after actual user consent and never automatically from untrusted payload,
authenticates a separate detached decision. Hash integrity itself is not consent.
A local O_EXCL claim consumes one-time use permanently before label access.

See `review_fixes_handoff.md` for exact API/CLI/schema, identity bindings,
checkpoint byte snapshots, rowID alignment, evaluated/failed-consumed reporting,
synthetic verification and limits. Original frozen protocol stays unapproved;
detached consent binds it without modifying unsigned files. All earlier smoke
outputs remain preserved historical snapshots and cannot be reused after code
fingerprints change. Actual S06 holdout stays CLOSED, actual delta_NI=null and
primary unapproved. No production decision record, production nine-model workload
or sealed quality evaluation was generated by this worker. No tests were edited.

H1 final worker verification is recorded in `review_fixes_handoff.md`: final fast
regression711 passed/1 skipped/1 deselected; full711 passed/1 skipped with only the
known notebook socket PermissionError failure. Initial independent authorization
44 tests passed, and the final fast run includes the subsequently expanded tests.
Synthetic evaluated/replay/offline mechanics passed. No production holdout use or
real consent occurred. Remaining independent review is orchestrator-owned.
