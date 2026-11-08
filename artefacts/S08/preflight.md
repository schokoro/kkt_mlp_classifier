# S08 initial isolation

Created `sprint/S08-noninferiority` from local `dev` at integrated S07 commit `dd7559a`. Initial tree contained only untracked `artefacts/S08/` supplied documents and `notebooks/make_data.ipynb`; preserved. Original S08 specification and orchestrator prompt are unchanged. S06/S07 results and source paths inspected. Frozen final schedule is 14 epochs, not historical 15. Holdout approval absent; no sealed labels or inference authorized. Implementation and independent test prompts saved before separate Codex invocation. Two read-only Kimi review rounds maximum.

Verified source byte fingerprints before implementation:

| Input | SHA256 |
| --- | --- |
| original S08 value_dod.md | 496d74548c34f760287a80bc81618c4569be9addc5da57035e6f322f7c76e7ac |
| original S08 orchestrator_prompt.md | bb5ab9ba2c71a154690daa20f8728ea2262fba6e88aaee48fb4eb7846f344963 |
| S06 manifest | ff44250867e864af23c72da94c38ed4d5a2796af9a658b83787193432ee2cd5d |
| final S06 training protocol | caffae8c2609d9f9cff28c47cf368cedddf7c38e7c4a4c8801bb553802c5cb0d |
| full S02 features.joblib | 7ff42885729d0bcfb2b6e9e0893b1be4ddf59562705e3ed048614183cd952052 |
| unlabeled corpus | f9a057a00db10ed115dd5f47e035f3a3c56c710d9463f47d38812d4cceef3dbc |
| sealed test bytes (no labels read) | 1842de401867e21a42c1dd342716548e1c40ef1ea57cb6109845ca8e308015f4 |
| final S07 plan | cafcabcdf5e33688d11cc2dfd31ce70743d8a6d09683efda3dc505524fff8deb |

Both sandboxed Codex launches failed before startup because app-server initialization encountered a read-only runtime. Authorized escalated launches succeeded with the same saved prompts; delegated workspace sandbox remains enabled. Logs: `/tmp/s08-implementation.log`, `/tmp/s08-tests.log`.

Parent verification used existing `s07.verify_sources(S07Config(device='cpu', num_threads=2))` and the original `outputs/S06/audit/class_support_split.csv` only. Confirmed 14,652 development rows, 293 L3, minimum development support10 and14,652 singleton normalized groups disjoint from holdout. Development row IDs SHA256 `bff8231a1955cb83a2c86cc691198d1dbbef13d891bede8b0d958bbbecbb41ed`; group SHA256 `aeee845e06ac915ecbfbff100719f96bd3495f302e7ef6d554c6c7fc6811e3fa`. Structural holdout count1,637, group SHA256 `7416a245f3064d663aa58004afbdd9afd33c7def3154ae732eec46bdb8a8a10a`. Structural audit supports sum1,637;114 classes have2 examples,186 classes have2–4 examples. Audit development supports independently match source development labels. No sealed labels loaded, no holdout inference or quality metrics. `/tmp/s08-orchestrator-source-verification.json` contains exact counts. An initial verification-only Series comparison required sorting matching label indices; rerun succeeded, no data changes.


## Reviewed production plan

Two read-only Kimi rounds completed, final PASS/no blockers. H1 executable detached approval path, row pairing and report status checks fixed through Codex; independent tester found and verified NaN identities and evaluated-table semantic regressions. Final source/test/README hashes exactly match second-review snapshot `/tmp/s08-review2-source-hashes.json`; no post-review code changes. Optional redundant identity check and an early-start-marker orphan possibility remain accepted nonblocking notes; no extra review.

Fresh reviewed preflight `outputs/S08/preflight_20261107_reviewed/` passed. Final quick suite711 passed/1 skipped/1 deselected (187.46s); final socket-enabled full suite712 passed/1 skipped (192.94s), CUDA warning. Expanded S08 suite133 tests, with genuine simultaneous-call and partial-inference tests on generated labeled fixtures only.

Formal production plan `outputs/S08/noninferiority_20261107_v1/` frozen before execution: protocol SHA256 `8aacc5c9ba3306ec1fe4dc0e16f214d976abe2f57d93891f75f53de134718175`, plan `3b51cfa1f6b8e3df3db2de0fe7de2a48a0fdb711935668a89c26d0c64bcfc953`, feasibility `ec85fc3a3b3a11b1a53ba7bc1cbbcf0aa7cc435392249d5273674e12c01d6f6b`, unsigned pre-registration `fd83bba473926587ab5f046ece4203148706569f128c034e9c96f9a3bd4ed88f`. CPU/two threads, nine fresh models, all development union, fixed14 epochs. About9.6 GiB available RAM/205 GiB disk before execution; no formal GPU measurement. No production approval is issued; gate `awaiting_user_approval`.
