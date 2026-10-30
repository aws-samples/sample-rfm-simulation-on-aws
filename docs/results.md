# Results

Back to the [README](../README.md) · [Documentation index](README.md)

- [Downloading results](#downloading-results) and comparing runs with `vlasim results`
- [Post-run judge](#post-run-judge-judgejson): Flora-Rewarder scores next to a run
- [Expected Results](#expected-results), including the [same-stack baseline](#same-stack-baseline-official-stack-50-episodes-per-suite) and [per-target smoke results](#per-target-smoke-results)

## Downloading results

When simulation finishes you receive an SNS email with download instructions:

```bash
# GR00T results
aws s3 sync s3://vla-sim-results-gr00t-demo-us-east-1-<ACCOUNT>/RUN_ID/ ./results/ --region us-east-1

# π0.5 results
aws s3 sync s3://vla-sim-results-pi-demo-us-east-1-<ACCOUNT>/RUN_ID/ ./results/ --region us-east-1
```

Each `task-N/` folder contains:
- `videos/rollout_<task_name>_success.mp4` — successful episodes
- `videos/rollout_<task_name>_failure.mp4` — failed episodes
- `videos/<name>.txt` — per-video metadata
- `summary.txt` — `success_rate`, `suite`, `num_episodes_per_task`
- `compose.log` / `userdata.log` — execution logs

To compare runs without downloading them, `vlasim results` reads every `vla-sim-results-*` bucket in the account (read-only) and prints one row per stack, run, suite and condition with k/n, the rate and a 95% Wilson interval, followed by the stack fingerprint recorded in `summary.txt`:

```bash
python vlasim.py results                                  # latest run of every bucket
python vlasim.py results --vla openvla-oft --all-runs     # every OFT run
python vlasim.py results --since 2026-10-19 --csv results.csv
```

```text
stack                region          run              suite           condition    k_n    rate  ci95       tasks  not_done
openvla-oft-spatial  ap-northeast-2  2026-10-19-0558  libero_spatial  smoke-knobs  10/10  1.00  0.72-1.00  1      0
```

A task with no `summary.txt` (still running, or the run died) is counted under `not_done`, never as a 0.

## Post-run judge (judge.json)

`python vlasim.py judge --vla <target> [--run RUN_ID]` ([targets](targets.md#post-run-judge-flora-rewarder)) scores every rollout video of a finished run with NVIDIA Flora-Rewarder and writes, next to the run:

- `<RUN_ID>/judge.json`: one entry per video (`id`, `task_idx`, `video_key`, `instruction` + `instruction_source`, `sim_success` + `sim_success_source`, `flora_score` 1..5, `flora_success`, `flora_expected_score`, `rationale`, `skipped_reason`), a `summary` (`agreement`, `n_agree` / `n_compared`, `confusion` with the four sim x Flora cells, `mean_score`, `mean_score_sim_success`, `mean_score_sim_failure`, `sim_success_rate`, `flora_success_rate`, `score_histogram`) and the run facts (`model`, `model_revision`, `license`, `instance_type`, `tensor_parallel`, `gpu_driver`, `stack_versions`, `scoring_time_s`, `wall_time_s`).
- `<RUN_ID>/judge.txt`: the same as a readable table; `judge-flora-terminal.jsonl` is NVIDIA's raw `terminal_reward` output (with `P(1..5)`).

`flora_success = flora_score >= 4` (`success_threshold` in `judge.json`): 1 is complete failure, 5 complete success, and the midpoint 3 counts as failure. Agreement is computed only over episodes that have both a Flora score and a simulator success flag. `vlasim results` prints it below the table:

```text
post-run judge (judge.json):
  smolvla-demo@ap-northeast-2 2026-10-20-0130: agreement 0.80 (8/10), scored 10/10, mean score sim-success 4.6 / sim-failure 1.8 [nvidia/Flora-Rewarder@031299e4, success = score >= 4]
```

_The numbers above show the format only; no judged run is published yet._ Flora is a learned judge: the model card reports MAE 0.40 on the 1-5 scale (RoboRewardBench) and asks to validate it on held-out episodes of your own robot before using it for training or data selection. Treat disagreements as episodes to look at, not as a correction of the simulator.

---

## Expected Results

> **A success rate belongs to the stack that produced it.** The companion study [*Same Policy, Different Numbers*](https://github.com/byongwu/pai-same-policy-different-numbers) (AMLC 2026 workshop) ran seven LIBERO policy families through this harness and changed one stack setting at a time: executing one action per predicted chunk instead of the official chunk moved OpenVLA-OFT from 1.00 to 0.00 and FastWAM LIBERO-10 from 0.96 to 0.54, while π0.5 and GR00T stayed within ±0.10. Compare numbers below only when episode count, init states, step cap and chunk execution match — the rows here mix 2 tasks × 5 episodes and 10 tasks × 1 trial.
>
> To run such a condition on the shared LeRobot template, add per-task fields in `models/<id>.yaml`: `condition` (a label carried into the results) and `extra_args` (`lerobot-eval` flags appended last, replacing any earlier flag with the same name), e.g. `condition: S3b` + `extra_args: ["--policy.n_action_steps=1"]`. Each `summary.txt` records `n_success`, `n_episodes`, `condition`, `extra_args` and a stack fingerprint (`stack_lerobot_commit`, `stack_mujoco`, `stack_eval_args`).
>
> `openvla-oft`, `pi` and `gr00t` take the same idea as per-task fields (absent = the upstream default): `condition`, `replan` (actions executed per predicted chunk: OFT `--num_open_loop_steps`, π0.5 `--args.replan-steps`, GR00T `--n-action-steps`), `seed`, and for OFT / π0.5 `step_scale` (multiplies the per-suite step cap) and `random_init: true` (random initial states instead of the fixed public ones). OFT also takes `replan_head: true` — **upstream OpenVLA-OFT executes the *last* `num_open_loop_steps` actions of the 8-action chunk when that number is below 8**; this executes the first ones instead. `models/pi.yaml` `model.mujoco_pin` (e.g. `mujoco==3.1.6`) swaps the LIBERO client's physics version. `assets/stack-knobs/stackpatch.py` applies the eval-script edits from a pristine copy before every task and stops the run if a patch site is missing. `cosmos-policy` takes run flags at model level (`model.eval_args`).

### Same-stack baseline (official stack, 50 episodes per suite)

From the companion study, every family on its own official evaluation stack (condition `S0`): 10 tasks × 5 trials per suite = 50 episodes, the public fixed init states, the official step cap and chunk execution. These are the numbers to compare against when you run your own condition with the knobs above — use the same episode count. Brackets are 95% Wilson intervals.

| Family (`--vla`) | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-10 |
|---|---|---|---|---|
| OpenVLA-OFT (`openvla-oft`) | 50/50 [0.93–1.00] | 50/50 [0.93–1.00] | 48/50 [0.87–0.99] | 49/50 [0.90–1.00] |
| π0.5 (`pi`) | 48/50 [0.87–0.99] | 48/50 [0.87–0.99] | 48/50 [0.87–0.99] | 47/50 [0.84–0.98] |
| GR00T N1.7 (`gr00t`) | 50/50 [0.93–1.00] | 50/50 [0.93–1.00] | 47/50 [0.84–0.98] | 46/50 [0.81–0.97] |
| Cosmos Policy (`cosmos-policy`) | 50/50 [0.93–1.00] | 50/50 [0.93–1.00] | 50/50 [0.93–1.00] | 48/50 [0.87–0.99] |
| FastWAM (`fastwam`) | 44/50 [0.76–0.94] | 49/50 [0.90–1.00] | 49/50 [0.90–1.00] | 48/50 [0.87–0.99] |
| LingBot-VA (`lingbot-va`) | — | — | — | 49/50 [0.90–1.00] |
| DreamZero (study harness only) | 41/50 [0.69–0.90] | 46/50 [0.81–0.97] | 29/50 [0.44–0.71] | 28/50 [0.42–0.69] |

Source: [`results/cells.csv`](https://github.com/byongwu/pai-same-policy-different-numbers/blob/main/results/cells.csv) of the study (149 cells). LingBot-VA publishes a LIBERO-10 checkpoint only. DreamZero ran on the study's own RLinf stack (`dreamzero/`), not on a target in this repo.

### Per-target smoke results

| `--vla` | Task | Expected Success Rate | Source |
|---------|------|-----------------------|--------|
| `gr00t` | KITCHEN_SCENE3 (stove + moka pot) | ~90–100% | validated 2026-04-27 |
| `gr00t` | KITCHEN_SCENE4 (black bowl in drawer) | ~90–100% | validated 2026-04-27 |
| `gr00t-gr1` | PosttrainPnPNovelFromPlateToBowlSplitA (GR1) | ~80% | validated 2026-04-12 |
| `gr00t-gr1` | PnPCanToDrawerClose (GR1) | 0% (pre-trained N1.6 not supported) | validated 2026-04-12 |
| `gr00t-g1` | LMPnPAppleToPlateDC (G1 loco-manip, community re-finetune) | 30% (3/10) | validated 2026-06-15 — `nvidia/...` checkpoint scores 0%; `cloudwalk-research` re-finetune used |
| `pi` | libero_object | ~80–94% | validated 2026-04-27 |
| `pi` | libero_spatial | ~85–95% | validated 2026-04-27 |
| `openvla-oft --libero-suite spatial` | libero_spatial | 97.6% (paper Table I) | validated 2026-06-01 — 0.92 (46/50, 5 trials/task × 10 tasks, g6.xlarge); within paper ±5%p band at n=50 |
| `openvla-oft --libero-suite object` | libero_object | 98.4% (paper Table I) | pending smoke test |
| `openvla-oft --libero-suite goal` | libero_goal | 97.9% (paper Table I) | pending smoke test |
| `openvla-oft --libero-suite 10` | libero_10 (long-horizon) | 94.5% (paper Table I) | validated 2026-05-04 |
| `lap` | libero_spatial | ~85-95% (paper Table III, LIBERO fine-tuned) | validated 2026-05-17 — 0.98 @ 5 trials/task |
| `molmoact2` | libero_goal (2 tasks × 5 ep) | 97.2% LIBERO avg (paper) | validated 2026-06-28 — 1.000 (10/10 ep, fp32 in-process, g6e.2xlarge L40S 48GB, full `deploy.py` 1-shot smoke, us-west-2) |
| `xvla` | libero_spatial (2 tasks × 5 ep) | 98.1% LIBERO avg (paper) · 93% (LeRobot card) | validated 2026-10-14 — 0.90 (9/10 ep, g6.2xlarge L4, ap-northeast-2) |
| `smolvla` | libero_spatial (2 tasks × 5 ep) | — (not stated on the model card) | validated 2026-10-14 — 0.80 (8/10 ep, g6.xlarge L4, ap-northeast-2) |
| `lingbot-va` | libero_10 (2 tasks × 3 ep) | 98.5% LIBERO-Long (LeRobot docs) | validated 2026-10-14 — 1.00 (6/6 ep, g6e.2xlarge L40S, ap-northeast-2; ~20 min/episode) |
| `vla-adapter` | libero_spatial (10 tasks × 1 trial) | 99.6% LIBERO-Spatial-Pro (inference log on the model card) | validated 2026-10-14 — 1.00 (10/10, g6.xlarge L4, ap-northeast-2) |
| `fastwam` | libero_spatial (2 tasks × 5 ep) | 97.6% LIBERO-Spatial (LeRobot docs, fp32) | validated 2026-10-15 — 0.90 (9/10 ep, bf16, g6e.2xlarge L40S, ap-northeast-2) |
| `cosmos-policy` | libero_spatial (10 tasks × 1 trial) | 98.1% LIBERO-Spatial (paper / LIBERO.md) | validated 2026-10-17 — 1.00 (10/10, g6e.xlarge L40S, ap-northeast-2; ~55 min end to end incl. Docker build) |
| `aspire` | libero_spatial (2 tasks × 5 ep, saved init states 0-4) | — (ASPIRE reports LIBERO-Pro perturbation suites, not this split) | validated 2026-10-17 — 1.00 (10/10, g6e.2xlarge L40S, ap-northeast-2; ~41 min end to end; agent 1,544 s / US$3.21 Bedrock for both tasks) |
| `rldx` | libero_spatial (single task) | 97.8% LIBERO avg (paper) | validated 2026-06-20 — 1.0 (5/5 ep, eager, g6.xlarge L4 sm_89, full `deploy.py` smoke) |
| `rldx-simpler` | simpler_env_google/google_robot_pick_coke_can (Google-VM Fractal) | 81.5% Google-VM (paper README) | validated 2026-06-24 — 1.0 (5/5 ep, eager, g6.2xlarge L4, full `deploy.py` 1-shot smoke) |
| `rldx-simpler` | simpler_env_google/google_robot_move_near (Google-VM Fractal) | 81.5% Google-VM (paper README) | validated 2026-06-24 — 1.0 (5/5 ep, eager, g6.2xlarge L4, full `deploy.py` 1-shot smoke) |
| `rldx-gr1` | gr1_unified/PnPCupToDrawerClose (GR-1 bimanual + waist) | 58.7% 24-task avg (paper README) | validated 2026-06-25 — 0.80 (4/5 ep, eager, g5.2xlarge A10G 24GB, full `deploy.py` 1-shot smoke) |
| `rldx-gr1` | gr1_unified/PnPMilkToMicrowaveClose (GR-1 bimanual + waist) | 58.7% 24-task avg (paper README) | validated 2026-06-25 — 0.20 (1/5 ep, eager, g5.2xlarge A10G 24GB, full `deploy.py` 1-shot smoke) |
| `rldx-kitchen` | robocasa_panda_omron/OpenSingleDoor (PandaOmron mobile manip) | 70.6% 24-task avg (paper README) | 0.80 (4/5), validated 2026-06-26 (g6.2xlarge L4) |
| `rldx-kitchen` | robocasa_panda_omron/OpenDrawer (PandaOmron mobile manip) | 70.6% 24-task avg (paper README) | 0.80 (4/5), validated 2026-06-26 (g6.2xlarge L4) |
| `smolvla-robotwin` | RoboTwin2/clean, 5 tasks × 5 ep | — (not stated on the model card) | not run yet (STEP B) |
| `lingbot-vla2` | RoboTwin2/clean, 5 tasks × 5 ep | 93.52% clean / 92.80% randomized, 50-task avg (upstream README) | not run yet (STEP B) |

**Validated results (us-east-1, g6.12xlarge / g5.xlarge / g6.xlarge):**
- GR00T N1.7: KITCHEN_SCENE3 = 1.0 (5/5), KITCHEN_SCENE4 = 1.0 (3/3)
- GR00T N1.6 + GR1: PosttrainPnP = 0.8 (4/5), PnPCanToDrawer = 0.0 (pre-trained model limitation)
- GR00T N1.6 + Unitree G1 (whole-body loco-manip): LMPnPAppleToPlateDC = 0.3 (3/10, `n_envs=1`, g6.12xlarge, us-west-2) — `cloudwalk-research` community re-finetune; the `nvidia/GR00T-N1.6-G1-PnPAppleToPlate` checkpoint scores 0/10 under the same command (matches open Isaac-GR00T issues, no upstream fix)
- π0.5: libero_object = 0.94 (47/50)
- OpenVLA-OFT: libero_10 = 1.0 (10/10 at 1 trial/task, g6.xlarge)
- OpenVLA-OFT: libero_spatial = 0.92 (46/50, 5 trials/task × 10 tasks, g6.xlarge) — paper Table I = 97.6%; the gap (5.6%p) is within sampling noise at n=50 (SE ≈ 3.8%p). 8/10 tasks at 5/5; misses concentrated on `on_the_ramekin` (2/5) and `next_to_the_plate` (4/5)
- LAP-3B: libero_spatial = 0.98 (49/50, 5 trials/task × 10 tasks, g6.xlarge) — exceeds paper Table III range (85–95%); requires upstream `scripts/libero/main.py` vertical-flip patch (see `templates/lap-userdata.sh.j2`)
- MolmoAct2 (Ai2): libero_goal = 1.000 (10/10, 2 tasks × 5 ep, g6e.2xlarge L40S 48GB, us-west-2) — paper LIBERO average = 97.2%. Apache-2.0 5B Action Reasoning Model run **in-process** on the sim GPU (no policy server) from the `allenai/lerobot` fork (`molmoact2-policy@a4f15bf3`, not upstream); official fp32 LIBERO recipe (`model_dtype=float32`, `use_amp=false`, `norm_tag=libero`, continuous). The ~20 GB fp32 checkpoint forces a ≥40 GB single GPU **and** ≥64 GiB host RAM (`g6e.xlarge`'s 32 GiB host RAM thrashes the load → excluded); checkpoint `allenai/MolmoAct2-LIBERO@0d24a92b`, Python pinned 3.12. Full `deploy.py` 1-shot smoke; pipe-proof cap was 2 tasks × 5 ep (drop `task_ids` + `n_episodes: 50` for the full 10-task suite)
- RLDX-1 (RLWRLD): libero_10 long-horizon, two tasks (eager, g6.xlarge L4 sm_89) — `STUDY_SCENE1` book→caddy = 1.0 (5/5, tight 20–23 steps), `KITCHEN_SCENE6` mug→microwave+close = 0.8 (4/5; successes span 32–47 steps and the one failure is a 90-step-cap timeout — marginal even when it succeeds, see the showcase caveat). Full `deploy.py` deploy; ZeroMQ policy server + sim client, two-venv (main + `libero_uv`). Default `n_envs=1` (5 recommended on g6.2xlarge+); checkpoint `RLWRLD/RLDX-1-FT-LIBERO@989037c6`, repo `RLWRLD/RLDX-1@ecbfaf80`
- RLDX-1 (RLWRLD) GR-1 Tabletop: two articulated tasks (eager, g5.2xlarge A10G 24GB) — `PnPCupToDrawerClose` = 0.80 (4/5, clean 13–19 steps), `PnPMilkToMicrowaveClose` = 0.20 (1/5; the single success at 24 steps, the four failures all hit the 45-step cap as timeouts). Combined 5/10 ≈ paper 24-task avg 58.7%. The only target on a **bimanual humanoid + waist** (`GR1ArmsAndWaistFourierHands`); MuJoCo/robosuite headless EGL (no Vulkan). Source-verified needed **no SimplerEnv-style code patch** — confirmed live: the `[5/8]` gr1-venv verify (dispatch + obs/action contract + import-time env registration) passed without FIX 4/6/7, and `setup_gr1.sh` exited 0 (committed-as-files submodule → `git submodule update` is a no-op, no FIX 5). Server cold-load fast (~210s). Checkpoint `RLWRLD/RLDX-1-FT-GR1@d0277c5c`, repo `RLWRLD/RLDX-1@ecbfaf80`


### RoboTwin 2.0 (`smolvla-robotwin`, `lingbot-vla2`)

`vlasim results` prints one row per run and setting: `suite` = `RoboTwin2/clean` or `RoboTwin2/randomized`, `condition` = the protocol (`subset`, `clean`, `randomized`, `full`), k/n pooled over the tasks of that run. Shards of the official protocol are separate runs; add `--all-runs` to see them all.

| `--vla` | Setting | Published (50 tasks × 100 episodes) | Source | Measured on AWS |
|---|---|---|---|---|
| `lingbot-vla2` | clean | 93.52% | [Robbyant/lingbot-vla-v2 README](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/README.md) (Muon post-training on clean + randomized data, upstream RoboTwin launcher, FP32), quoted in LeRobot `docs/source/lingbot_vla_v2.mdx` | pending |
| `lingbot-vla2` | randomized | 92.80% | same | pending |
| `smolvla-robotwin` | clean / randomized | none published (model card has no evaluation) | [`lerobot/smolvla_robotwin`](https://huggingface.co/lerobot/smolvla_robotwin) @eb55d3a7 | pending |

For context, the same upstream table lists π0.5 at 82.74% (clean) / 76.76% (randomized) and LingBot-VLA 1.0 at 88.56% / 86.68%. The published numbers use RoboTwin's own evaluation script, which keeps only seeds the scripted expert solves; the LeRobot wrapper used here does not filter seeds, and the default run is a 5-task subset, so expect lower and noisier rates (5 tasks × 5 episodes = 25 episodes, a 95% interval about ±0.2 wide).

---
