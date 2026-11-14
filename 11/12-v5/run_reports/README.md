# v5.1 run reports (branch v51-run-20261114)

Checkout: `667a3034e225727660a3a378c68702f796916b27` (codex/v5.1-eb-sibling-experts), seed 42.

## Result: both components failed their pre-registered gates -> no build, no submission

- EB: base BA 65.913, candidate 65.598, delta **-0.315**, paired prompt-bootstrap 95% CI [-0.770, +0.126]; lower bound not > 0.
- gemini: within-group native 65.56 vs expert 59.33 (delta -6.22); SAGE family-conditional delta +0.128, CI [-0.345, +0.597].
- glm_claude: within-group native 50.33 vs expert 40.76 (delta -9.58); SAGE family delta -0.106, CI [-0.417, +0.205].
- gpt_kimi: within-group native 69.47 vs expert 62.13 (delta -7.33); SAGE family delta +0.013, CI [-0.264, +0.286].
- `decision.json.enabled = {eb: false, experts: [], build: false}`; `negative_result_action = do_not_build_or_submit`.

All metrics are official-train prompt OOF **validation**, native-alone scope, 10 grouping draws, 4000 prompt-bootstrap draws. Hidden test was not evaluated; public leaderboard was not used for any selection.

## Files

- `run_manifest.json` – SHA, deadline, data audit
- `status_snapshot.json` – final launcher status `VALIDATED`
- `decision.json` – full CPU gate output, artifacts and hashes
- `final_summary.json` – compact gate/metric summary
- `evaluation.log` – full v51_evaluate output
- `progress.json` – mid-run progress snapshot
- `v51_evaluate_parity_tolerance.patch` – only code change: native OOF parity tolerance 0.05 -> 0.20 + argmax >= 0.99 (Ada fp16; pinned public OOF replaces official-row values before EB fitting)

## Data

- v15 pool files verified by SHA256; cleaned 203,219 rows / 35,310 prompts; gemini-3.7-flash 3,535 cleaned rows.
- 10,838 unverified rows remain quarantined: local q14v5/ll8v5 bases+adapters were not present, so the required 3 independent votes could not be produced; no votes were fabricated.

## Training

- 3 expert groups on GPUs 0-2, EB collector on GPU 3; external 1 epoch + official 2 epochs per exclusion, 5 single + 10 pair exclusions per group, all 15 stages scored and validated; EB uses 62,716 rows (2,699 official + 60,017 external).
- Wall time 06:10-10:08 UTC on 2026-11-14; no checkpoint search; fixed last epoch.
