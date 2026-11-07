# S07 preflight

Branch `sprint/S07-stability-validation` created from integrated `dev` at `c7df0d4d47f83a380ca2c3e91f8956bc7dacbd5f`. Working AGENTS.md contains the revised roles; its user modification is preserved. Pre-existing `S07_docs.zip`, `notebooks/make_data.ipynb`, and source S07 documents remain untouched. Initial sandbox branch creation failed because `.git` is read-only; the authorized escalated `git switch` succeeded.

Sources verified by actual SHA-256:

| Source | SHA-256 |
|---|---|
| S06 split manifest | ff44250867e864af23c72da94c38ed4d5a2796af9a658b83787193432ee2cd5d |
| Full S02 features | 7ff42885729d0bcfb2b6e9e0893b1be4ddf59562705e3ed048614183cd952052 |
| Fixed corpus | f9a057a00db10ed115dd5f47e035f3a3c56c710d9463f47d38812d4cceef3dbc |
| Final S06 training protocol | caffae8c2609d9f9cff28c47cf368cedddf7c38e7c4a4c8801bb553802c5cb0d |

All five original partition file hashes match the S06 manifest. The parent inspected only development labels; sealed file bytes were hashed, without reading sealed labels. The delegate projected sealed `name`/`row_id` only to validate group and identity exclusion.

The development pool is exactly 14,652 rows (12,960 + 1,692), 293 L3, minimum support 10, and 14,652 unique normalized-name groups. No new support filter. Saved diagnostic: `outputs/S07/preflight_20261107T135054_cba50c95/preflight.json`. Feasible fresh assignments for seeds 101/202/303 each have train_fit 11,766, train_monitor 1,378 and comparison_val 1,508, with minima 6/2/2. This feasibility diagnostic is not the formal frozen experiment plan.

The final accepted S06 protocol specifies **14 final epochs**. Earlier 15-epoch attempts are historical and are not reused. Model [256], dropout .3, relu, no norm; AdamW .001, weight_decay 0, constant LR, batch 256, final epoch weights. Actual corpus fit was all 981,850 rows, deterministic head selection with no truncation or subsampling. Full artifact reload verified 300k char/100k word. No specification deviation identified at preflight.

Implementation delegated with exact saved English prompt `prompts/01_codex_implementation.md`; independent test authorship uses a separate session with `prompts/02_codex_tests.md`. Initial sandboxed Codex startup failed (read-only runtime); escalated startup succeeded. No production training, validation inference or selected refits authorized to delegates at this stage. Two Kimi review rounds maximum.
