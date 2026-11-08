# S08 Kimi read-only review round 1

Exact raw session log: `/tmp/s08-kimi-review1.log`. Review snapshot `/tmp/s08-review1.diff`. Some line references refer to concatenated diff positions; inspect named functions in source. No edits/tests/training by reviewer. Initial `--plan -p` CLI conflict and sandbox runtime-storage failure happened before review; exactly one review ran.

• Review complete. I read the exact diff (SHA256 verified), the S07 helpers it binds to (`verify_sources`, `S07Config`, sealed/write/tree utilities, `_training_record`, `_Measurement`), the vectorizer/selection refit paths, the S08 manifests/handoffs, and the S06 split manifest structure. I did not edit files, run code, or open sealed label/text data.

  # S08 Round-1 Independent Review — findings

  ## HIGH

  **H1. The authorized evaluation path is structurally unimplemented, and the stated justification invents infrastructure the spec does not require — this must be resolved before production `execute`/formal freeze.**
  - Location: `kkt_clf/s08.py:567-577` (`evaluate`), reinforced at `kkt_clf/s08.py:172-173` (`protocol_contract['approval']`), `kkt_clf/s08.py:276-277` (`load_plan` gate string), README.md:48-55, `artefacts/S08/implementation_handoff.md:40-50`.
  - Trigger: `evaluate()` unconditionally rejects — `require(approval is None, ...)` (s08.py:574) makes it impossible by construction, then it raises `evaluation_blocked`. The justification claims a trusted **external** consent issuer **and** an external atomic one-time consumption ledger are prerequisites that "do not exist".
  - Violated requirement: Orchestrator task §2/§4 and D09's framing — "explicit orchestrator-recorded stakeholder approval with authenticated/mismatched-record rejection, **no external service requirement**". The spec's own negative tests ("unauthenticated or mismatched approval record") presuppose an approval-record format that can be *validated*; here no validation path exists at all, so `tests/test_s08_integrity_independent.py:1150-1165` exercises only the trivial `approval is None` branch and cannot distinguish an authentic record from a fabricated one.
  - Why it matters mechanically: `load_plan` binds `implementation_files()` (every `kkt_clf/*.py` fingerprint) into the frozen plan (s08.py:262), checkpoints bind `plan_sha256` via `_provenance` (s08.py:321-324, 439), and `verify_training_freeze` re-checks all of it (s08.py:534-564). Therefore any future code addition that enables evaluation — even a pure, protocol-neutral gate — invalidates the plan, the protocol binding, and all nine checkpoints. The "prepared study awaiting approval" as frozen can **never** be evaluated; a post-approval evaluation would force a new versioned experiment **including retraining all nine models**. That is not what "a reproducible prepared study awaiting approval" (orchestrator prompt §5) promises.
  - Assessment per the review brief: refusing local booleans/self-generated digests is correct and required (honest conditional-D09 part). The overreach is the claim that an *external* issuer and *external* ledger are the only possible safe design. A trusted caller-supplied approval record — digest binding margin, rationale, primary candidate, one-time use, and the exact protocol/plan/training/model/feature fingerprints, anchored outside the worker-writable experiment tree — plus a single-use local consumption claim via the existing atomic `write_new` (O_EXCL) and full `verify_training_freeze` before any label read, implements the authorized path fail-closed with no external service. The current design should be fixed *before* formal freeze precisely because no production `execute` has occurred yet; after it occurs, the fix becomes exponentially more expensive.
  - Repair: implement `evaluate(root, approval_record, trusted_anchor)` that (1) rejects any record failing digest/authentication against the caller-supplied anchor outside the experiment payload; (2) verifies plan, protocol, training freeze, all 9 checkpoints and both selected TF-IDF artifacts *before* opening sealed labels; (3) atomically writes a single-use consumption claim (fail if present); (4) runs the one planned paired inference + `paired_bootstrap` + `ni_decision`. Keep it gated (no real approval exists; nothing runs), and state in `result.md` that D09 remains blocked pending stakeholder approval. Independent tests must cover authenticated-but-mismatched, authentic-but-replayed (second call), and post-claim-mutation cases.

  ## LOW

  **L2. Prediction/label pairing in `paired_bootstrap` is positional only.**
  - `kkt_clf/s08_stats.py:640-643`. Inputs are validated for shape `(3, n)` and identity uniqueness, but nothing verifies that column *j* of `full_predictions`, `candidate_predictions`, `y_true`, and `row_ids` refer to the same row; a misaligned stash would silently produce a paired CI on scrambled pairs.
  - Violates: D09 "same sealed holdout rows" pairing intent.
  - Repair: in the future evaluation path, stash and reload predictions keyed by `row_id` and assert set/order equality against the sealed manifest row identity before calling `paired_bootstrap` (the function may keep its positional contract, but the caller must prove alignment).

  **L3. Vacuous validation in report rendering.**
  - `kkt_clf/s08_report.py:841` — `require(row['status'] in manifest['statuses'], ...)` is dict-key membership over a fixed-key dict; it can never fail. Harmless but reads as a check that isn't one.
  - Repair: compare against the literal status set `{'success','failed','skipped','pending','incomplete'}`.

  ## Verified as correct (spot-checked against requirements)

  - Fixed 293/14,652 dev pool: inherited S07 production pins (`kkt_clf/s07.py:288-294`) plus S08 five-part provenance, inner-partition union identity, row/group disjointness, singleton-group fail-closed (`kkt_clf/s08.py:128-155`); production preflight reportedly passed on the real pins.
  - Exact 981,850-row refit: `refit_selected_vectorizer` enforces corpus sha/bytes/row counters; `_verify_refit` checks 75k/25k branch dimensions, vocabulary name/order identity, IDF equality (rtol 1e-6), per-branch L2 (`kkt_clf/s08.py:327-341`).
  - Nine final models, seeds 42/43/44, fixed epochs: `run_matrix` + `training_contract` override epochs/patience to `fixed_epochs`, `restore_best=False`, in-sample-only monitor, explicit `len(history)==epochs` check (`kkt_clf/s08.py:183-190, 425-438`).
  - Immutable protocol/plan/weights freeze: semantic contract re-verification, source fingerprints, `implementation_files()`/`environment_versions()` binding, recursive `training_freeze.json`, per-checkpoint receipt + reload verification (`kkt_clf/s08.py:255-291, 448-473, 521-564`).
  - Statistics: class-stratified resampling with per-class size preservation, one shared index set per replicate across all three seed pairs, F1 recomputed from row-level predictions (never resampled F1 values), B≥5000 enforced, alpha fixed at .025, linear percentile, strict `lower_bound > -delta_NI` with equality → `not_established`, fixed-universe zero_division=0 macro-F1 matching sklearn semantics, conditional-interpretation caveats (`kkt_clf/s08_stats.py:593-674`). Independent oracle cross-check in `tests/test_s08_statistics_independent.py` confirms the arithmetic.
  - Feasibility: explicitly simulated scenarios on dev-derived support projection, no sealed quality data, honest conclusion that the illustrative .01 margin may be unresolved (`kkt_clf/s08_stats.py:677-714`).
  - Blocked-state honesty and isolated rebuild: pending report has 9 rows with null quality fields, five test-dependent files rejected if present, byte-identical HTML rebuilt from only `manifest.json`/`runs.csv`/`stages.csv`/`ni_decision.json` with outside reads forbidden; table tampering and resealed semantic forgery (fake approval/fake NI/fake metric/fake status) rejected.
  - Resource accounting: shared ranking/refit measured once per method with `charge_count=1`, per-run train stages, nullable RSS/VRAM, model bytes verified against checkpoint (`kkt_clf/s08.py:302-319`, `kkt_clf/s08_report.py:797-798`).
  - Sealed data handling: only `row_id`/`name` projected (for group-key disjointness), labels never read, no inference path exists; manifest confirms sealed `n_groups == n_rows` (singleton), consistent with the preregistered row-bootstrap design.

  ## Unverified / not checkable in this review

  - No formal 9-model production workload, production-scale refit, or sealed access (correctly absent); full-suite pytest and `git diff --check` are reported as green by the tester, not re-run here.
  - Whether the true sealed holdout gives every one of the 293 classes ≥2 rows (the ≥2 support requirement will fail closed otherwise — spec-compliant, but the feasibility projection cannot prove it).

  ## Verdict

  **PASS with one required design fix (H1) before production `execute` and formal freeze.** The implementation and independent tests are otherwise sound, faithful to the S08 statistical contract, and honest about the blocked state. H1 is not a safety defect — nothing unsafe can run today — but it converts a conditional D09 limitation into a permanent lockout that would invalidate the frozen study the moment the user approves. Whether to implement the approval-gate/evaluation path now (recommended, while retraining is still avoidable) or formally accept a new-versioned-experiment retrain after approval is a user/orchestrator decision; the current code and docs overstate the impossibility by requiring external infrastructure the specification never asked for. I remain available for round 2 after H1 (and optionally L2/L3) are addressed.

To resume this session: kimi -r session_ca263c7f-0c7f-411b-b615-7dbac6b8bc45
