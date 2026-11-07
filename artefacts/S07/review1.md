• # S07 Independent Review — Round 1

  **Verdict: PASS with findings — no critical integrity blockers.** The five-file diff is internally consistent, faithful to the authoritative spec, and correctly reuses S03–S06 contracts. I verified the pinned sources myself (hashes of `outputs/S06/splits/manifest.json`, `outputs/S06_final_cpu_20261107/shared/training_protocol.json`, `data/corpus.pqt` all match `SOURCE_PINS`; the features pin matches the protocol's `full_tfidf_id`), and confirmed pool arithmetic (12,960 + 1,692 = 14,652; 293 L3), the frozen protocol (14 epochs, scheduler `none`, hidden [256], adamw/lr 1e-3/batch 256), and every imported signature (`train_model` early-stop counter cannot truncate history when `patience == epochs == 14`; `predict` top-1 is never `-1`, so the null-prediction sentinel paths are defensive only; attribution margin/GxI ratings, `assign_groups`/`search_feasible_assignment`, `refit_selected_vectorizer`, selection artifact save/load, `save_artifact` meta/config round-trip all match usage).

  ## Confirmed sound (focus areas that hold)

  - **Frozen 14-epoch protocol**: `patience` is overridden to `fixed_epochs`, so early break is arithmetically impossible before epoch 14; `len(history) == epochs` and `restored_best == False` are asserted per run and re-verified at report time.
  - **Fail-closed plan/dependencies**: 36-row matrix rebuilt from config and compared; dependency DAG re-derived and compared; re-sealed semantically-invalid plans rejected (tested); source pins cannot be overridden in production; implementation fingerprint frozen; `verify_frozen_runtime` re-checked before every train and inference.
  - **Sealed test**: only `['row_id','name']` projected (test-asserted), identity and group-key overlap checked, labels never read; structural pins verified.
  - **Split-local selection**: ranking inputs monkeypatch-checked to equal the frozen `outer_train` transform and labels; GxI reference is predeclared `none/seed_42`, cross-split reference and seed-44 mutations rejected; changed partitions fail before ranking.
  - **Nine refits**: same corpus snapshot (`meta.source`/`meta.fit` equality), stable feature-name mapping, IDF invariant vs full vectorizer, per-branch L2 renormalization, recorded refit checks — all verified and independently tested including tamper-restore cycles.
  - **Paired statistics**: deltas paired by `(split, seed)` from unrounded values; across-split rows require all three seeds (`split_count` explicit); within-split SD and between-split mean SD reported separately; no independence-based CI anywhere; incomplete pairs stay NaN and out of means.
  - **Costs**: one immutable event per measurement, `charge_count == 1` enforced, duplicate-event detection, shared ranking/refit charged once per split, GxI checkpoint allocation disclosed, VRAM null on CPU.
  - **Report integrity**: table-only rebuild guarded by a `Path.open` monkeypatch test; per-table and manifest fingerprints; prediction row/label identities re-derived against frozen split digests (`row_l3_sha256`); statuses complete and truthful (`PARTIAL` warning, non-success rows carry no metrics).

  ## Findings

  **M1 — Medium (fail-closed hardening).** `kkt_clf/s07.py:736-738` — `_evaluate_one` calls `load_split(root, plan, run['split_id'])` but discards the result, then scores on the caller-supplied `parts['comparison_val']`. Names are not stored in `predictions_val.parquet`, so scoring on unverified names would be invisible downstream (row ids/labels would still match at report time). The only production caller passes verified parts, but the function's own comment promises revalidation and tests call it directly. *Repair:* `parts = load_split(root, plan, run['split_id'])` and score on its `comparison_val`.

  **M2 — Medium-low (integrity).** `kkt_clf/s07_report.py:148` — for non-success selected-method runs without verifiable dependencies, `feature_id` is taken from `result['provenance']['feature_id']` itself, making the provenance equality check self-referential for skipped/failed runs. With no execution ledger (crash before `execution_finished.json`), a re-sealed skipped result passes with arbitrary provenance. No metrics are at stake and the ledger catches status flips when present, but the check verifies nothing. *Repair:* compare against `run_provenance(plan, run, 'unavailable')` (the constant `execute` writes) instead of the file's own content.

  **M3 — Risk flag (verify before the workload, not a confirmed defect).** `kkt_clf/s07.py:185` — hard `require` that every normalized-name group maps to exactly one L3. This is stricter than the spec (which forbids splitting groups, not multi-label groups) and fails preflight with a generic error rather than per-class proven-infeasible diagnostics. S06 manifests show zero cross-part key overlap, so the production pool very likely satisfies it — but if any ODS name normalizes to one key under two L3 labels, the entire sprint blocks here. Confirm against the real pool during preflight; if it triggers, report it as structured infeasibility, not a generic gate error.

  **L1 — Low (methodology/presentation).** `kkt_clf/s07_report.py:473-476` — `gxi_incremental_measured_seconds` sums *all* GxI stages (shared attribution/ranking/refit + 3 seeds × train/inference). It is not an incremental cost vs Chi² despite the name, so spec question 2 is only answerable by manual subtraction. Rename (e.g. `gxi_total_measured_seconds`) or add the matched `class_aware` subtraction.

  **L2 — Low.** `kkt_clf/s07_report.py:78,103-105` — within-split rows fill `split_count` with `1`/`0` (conflicting with its across-split meaning of 3), and across-split `gains/losses/ties` count split means while sharing column names with within-split seed-pair counts. Ambiguous for table consumers.

  **L3 — Low.** `kkt_clf/s07_report.py:457-460` — per-class extremes sort includes NaN `delta_f1` rows when the baseline is incomplete; `head(30)` "largest losses" can be padded with NaNs in exactly the partial experiments the report must render truthfully.

  **L4 — Low.** Checkpoint-cost rule is defined twice: `cost_role`/`checkpoint_incremental_training_seconds` in `_record_stage` (`kkt_clf/s07.py:509-512`, `train` stage only) vs the render-side filter `stage.isin(['train','train_transform'])` + run-id suffix (`kkt_clf/s07_report.py:471`). Consistent today; can silently drift on stage renaming.

  **L5 — Low.** `scripts/run_s07.py:46` — `report` exits 0 even when `complete` is False (36-pending report indistinguishable by exit code from a complete one). The JSON carries the truth; document or fold `complete` into the exit code for automation.

  ## Unresolved limitations

  - Code review only; I executed nothing and claim no test results. Synthetic results (68 tests, 36 tiny runs) are reported by the orchestrator, not verified here.
  - Production split feasibility (group minima 6/2/2 across all 293 L3 under seeds 101/202/303) and the M3 multi-label-group question are unverified on real data.
  - Final full pytest and `git diff --check` were pending at review time; D13/D14 remain open items.
  - No S03–S06 implementation files are touched by this diff; compatibility is preserved by construction (frozen protocol/TF-IDF reused only as pinned references, no supervised S06 artifacts consumed).

  **Recommendation:** proceed with M1/M2 repairs plus regression tests from the independent test author, address L1–L5 at implementer discretion, treat M3 as a mandatory pre-workload check, and keep acceptance gates unchanged. Round-2 review remains available; any fixes landing after it must be flagged as not independently reviewed per D13.

To resume this session: kimi -r session_ee190bfd-f946-4e1b-bbf5-d2a21df548fc
