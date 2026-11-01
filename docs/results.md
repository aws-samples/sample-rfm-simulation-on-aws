# Results

Back to the [README](../README.md) · [Documentation index](README.md)

- [Downloading results](#downloading-results) and comparing runs with `vlasim results`
- [Cross-model matrix](#cross-model-matrix-leaderboard): target x benchmark with cost and latency (`vlasim results --matrix`)
- [Post-run judge](#post-run-judge-judgejson): Flora-Rewarder scores next to a run
- [LIBERO-Plus robustness](#libero-plus-robustness): per-dimension drop vs the LIBERO baseline (`vlasim results --robustness`)
- [LIBERO-R2S](#libero-r2s-reconstructed-objects): public vs reconstructed-object scenes (`--libero-objects real2sim`)
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

## Cross-model matrix (leaderboard)

`vlasim results --matrix` turns the same buckets into one table: rows are model targets, columns are benchmarks, and each cell is the latest run of that target on that benchmark under a baseline condition, with k/n and a 95% Wilson interval. A second table estimates cost and time per cell. Every region that has a `vla-sim-results-*` bucket is read (the bucket name carries the region; `--region` restricts it). Read-only.

```bash
python vlasim.py results --matrix                          # all regions, baseline conditions
python vlasim.py results --matrix --all-conditions         # plus one row per stack condition (S2a, P0, ...)
python vlasim.py results --matrix --condition S0 --csv m.csv --json m.json
python vlasim.py results --write-docs                      # refresh the block below from your buckets
```

**Columns.** `suite` (or the part of `env_name` before `/`) is normalised; a `condition` of the form `<Benchmark>/<setting>` names a benchmark variant and becomes the column itself:

| `summary.txt` | Column |
|---|---|
| `suite: libero_spatial` / `libero_object` / `libero_goal` / `libero_10` | `LIBERO-Spatial` / `LIBERO-Object` / `LIBERO-Goal` / `LIBERO-10` |
| `suite: libero_spatial` + `condition: LIBERO-Plus/camera` | `LIBERO-Plus/camera (Spatial)` |
| `condition: RoboLab/subset12` | `RoboLab/subset12` |
| `condition: RoboTwin2/demo_clean` | `RoboTwin2/demo_clean` |
| `suite: simpler_bridge`, `env_name: simpler_env_widowx/...` | `SimplerEnv-Bridge` |
| `env_name: libero_sim/<scene>` (GR00T) | `LIBERO scenes (GR00T)` |
| `env_name: robocasa_panda_omron/...`, `gr1_unified/...`, `gr00tlocomanip_g1_sim/...` | `RoboCasa-Kitchen`, `GR1-Tabletop`, `G1-LocoManip` |
| anything else | the suite as written |

**Rows and comparability.** The row is the summary's `model:` when it is a `--vla` target, otherwise the target the stack name starts with. Baseline conditions are `S0`, `-` (not recorded) and the `<Benchmark>/<setting>` labels; other conditions are left out unless `--all-conditions` or `--condition` is given. Shards of one evaluation run as separate stacks (same condition, checkpoint and stack fingerprint, disjoint `task_ids`, started within `--pool-hours`, default 6) are pooled into one cell; reruns of the same tasks are not. A cell is marked † when a fingerprint key (`stack_*`, `step_scale`, `random_init`, `replan_head`, `max_episode_steps`) differs from another cell in the same column (listed under the table), and ‡ when the summary records no fingerprint or no condition, so its stack is unknown.

**Cost and time (estimates).** Wall time per run = RUN_ID (`date '+%Y-%m-%d-%H%M'` at the top of the user data, UTC = instance boot) to the last object the run uploaded, ignoring `judge*` outputs and video captions (the judge re-uploads captions later). A `run_wall_s` field in `summary.txt` takes precedence. Instance type comes from `instance_type:` / `instance:` in `summary.txt`, else the `Instance: <id> (<type>)` line of a video caption, else `userdata.log`. USD per hour is a pinned On-Demand Linux table per region in `leaderboard.py` (AWS Price List API, `aws pricing get-products --service-code AmazonEC2`, checked 2026-10-30); USD/episode divides by every episode of the run. Stack create/delete, NAT gateway, EBS and data transfer are not included ([cost](cost.md)).

**Latency.** s/episode is read from `episode_wall_s` in `summary.txt`, else LeRobot's `eval_info.json` (`overall.eval_ep_s`), else the GR00T rollout log (`Collecting N episodes took X seconds`). No template records per-step policy latency today, so ms/step stays empty until a template writes it. Optional `summary.txt` fields any template can add (one `key: value` line each; all are read if present and ignored otherwise):

| Field | Meaning |
|---|---|
| `instance_type` | EC2 instance type, e.g. `g6.xlarge` |
| `run_wall_s` | seconds from instance boot to the moment this summary was written (the largest value of a run is used as its wall time) |
| `episode_wall_s` | mean wall seconds per episode (reset + rollout + video write) |
| `policy_ms_per_step` | mean policy inference time per control step, in milliseconds |

For example, in a user-data template: `BOOT_TS=$(date +%s)` next to `RUN_ID=...`, then `echo "run_wall_s: $(( $(date +%s) - BOOT_TS ))"` and `echo "instance_type: ${INSTANCE_TYPE}"` inside the `summary.txt` block.

<!-- leaderboard:begin -->
_Generated 2026-10-31 by `python vlasim.py results --write-docs` from the results buckets in this account; estimates are labelled as such._

| Target | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-10 | LIBERO-Plus/background (10) | LIBERO-Plus/background (Goal) | LIBERO-Plus/background (Spatial) | LIBERO-Plus/camera (10) | LIBERO-Plus/camera (Goal) | LIBERO-Plus/camera (Spatial) | LIBERO-Plus/language (10) | LIBERO-Plus/language (Goal) | LIBERO-Plus/language (Spatial) | LIBERO-Plus/layout (10) | LIBERO-Plus/layout (Goal) | LIBERO-Plus/layout (Spatial) | LIBERO-Plus/light (10) | LIBERO-Plus/light (Goal) | LIBERO-Plus/light (Spatial) | LIBERO-Plus/noise (10) | LIBERO-Plus/noise (Goal) | LIBERO-Plus/noise (Spatial) | LIBERO-Plus/robot (10) | LIBERO-Plus/robot (Goal) | LIBERO-Plus/robot (Spatial) | LIBERO-R2S/collision (Object) | LIBERO-R2S/full (Object) | LIBERO-R2S/public (Object) | LIBERO scenes (GR00T) | SimplerEnv-Bridge | RoboCasa-Kitchen | GR1-Tabletop | G1-LocoManip | RoboLab | RoboTwin2/clean (subset) | gr1_tabletop | hope-video-scene-0000 | simpler_google_vm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `cosmos-policy` | 1.00 (100/100) [0.96-1.00]‡ | 1.00 (100/100) [0.96-1.00]‡ | 0.94 (94/100) [0.88-0.97]‡ | 0.98 (98/100) [0.93-0.99]‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `fastwam` | 0.88 (44/50) [0.76-0.94]‡ | 0.98 (49/50) [0.90-1.00]‡ | 0.98 (49/50) [0.90-1.00]‡ | 0.96 (48/50) [0.87-0.99]‡ |  |  | 0.90 (9/10) [0.60-0.98]† |  |  | 0.10 (1/10) [0.02-0.40]† |  |  | 0.70 (7/10) [0.40-0.89]† |  |  | 0.50 (5/10) [0.24-0.76]† |  |  | 0.90 (9/10) [0.60-0.98]† |  |  | 0.40 (4/10) [0.17-0.69]† |  |  | 0.40 (4/10) [0.17-0.69]† |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `gr00t` |  |  |  |  |  |  | 1.00 (10/10) [0.72-1.00]† |  |  | 0.60 (6/10) [0.31-0.83]† |  |  | 0.80 (8/10) [0.49-0.94]† |  |  | 0.90 (9/10) [0.60-0.98]† |  |  | 0.80 (8/10) [0.49-0.94]† |  |  | 1.00 (10/10) [0.72-1.00]† |  |  | 0.40 (4/10) [0.17-0.69]† |  |  |  | 1.00 (50/50) [0.93-1.00]‡ |  |  |  |  |  |  |  |  |  |
| `gr00t-bridge` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.55 (22/40) [0.40-0.69] |  |  |  |  |  |  |  |  |
| `gr00t-g1` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.40 (4/10) [0.17-0.69]‡ |  |  |  |  |  |
| `gr00t-gr1` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.25 (2/8) [0.07-0.59]‡ |  |  |  |  |  |  |
| `lap` | 0.92 (rate only)‡ |  |  |  |  |  | 0.90 (9/10) [0.60-0.98] |  |  | 0.50 (5/10) [0.24-0.76] |  |  | 0.50 (5/10) [0.24-0.76] |  |  | 0.90 (9/10) [0.60-0.98] |  |  | 0.80 (8/10) [0.49-0.94] |  |  | 0.60 (6/10) [0.31-0.83] |  |  | 0.70 (7/10) [0.40-0.89] |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `lingbot-va` |  |  |  | 0.98 (49/50) [0.90-1.00]‡ | 1.00 (2/2) [0.34-1.00] |  |  | 0.50 (1/2) [0.09-0.91] |  |  | 1.00 (2/2) [0.34-1.00] |  |  | 1.00 (2/2) [0.34-1.00] |  |  | 0.50 (1/2) [0.09-0.91] |  |  | 0.00 (0/2) [0.00-0.66] |  |  | 1.00 (2/2) [0.34-1.00] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `lingbot-vla2` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.80 (20/25) [0.61-0.91] |  |  |  |
| `molmoact2` |  |  | 1.00 (rate only)‡ |  |  | 1.00 (10/10) [0.72-1.00] |  |  | 1.00 (10/10) [0.72-1.00] |  |  | 0.80 (8/10) [0.49-0.94] |  |  | 0.60 (6/10) [0.31-0.83] |  |  | 0.90 (9/10) [0.60-0.98] |  |  | 0.90 (9/10) [0.60-0.98] |  |  | 0.80 (8/10) [0.49-0.94] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oftb3e1g` |  |  | 0.96 (96/100) [0.90-0.98]‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oftb3e1l` |  |  |  | 0.99 (99/100) [0.95-1.00]‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `openvla-oft` | 0.96 (96/100) [0.90-0.98]‡ | 0.98 (49/50) [0.90-1.00]‡ | 0.98 (49/50) [0.90-1.00]‡ | 0.96 (48/50) [0.87-0.99]‡ |  |  | 1.00 (10/10) [0.72-1.00]† |  |  | 1.00 (10/10) [0.72-1.00]† |  |  | 0.90 (9/10) [0.60-0.98]† |  |  | 0.90 (9/10) [0.60-0.98]† |  |  | 1.00 (10/10) [0.72-1.00]† |  |  | 1.00 (10/10) [0.72-1.00]† |  |  | 0.30 (3/10) [0.11-0.60]† | 0.88 (35/40) [0.74-0.95]† | 0.95 (38/40) [0.83-0.99]† | 1.00 (40/40) [0.91-1.00]† |  |  |  |  |  |  |  |  |  |  |
| `pi` | 0.99 (rate only)‡ | 0.98 (rate only)‡ | 0.96 (rate only)‡ | 0.94 (rate only)‡ |  |  | 1.00 (10/10) [0.72-1.00]† |  |  | 0.70 (7/10) [0.40-0.89]† |  |  | 0.80 (8/10) [0.49-0.94]† |  |  | 1.00 (10/10) [0.72-1.00]† |  |  | 0.90 (9/10) [0.60-0.98]† |  |  | 1.00 (10/10) [0.72-1.00]† |  |  | 0.80 (8/10) [0.49-0.94]† | 0.95 (38/40) [0.83-0.99]† | 0.72 (29/40) [0.57-0.84]† | 0.97 (39/40) [0.87-1.00]† |  |  |  |  |  |  |  |  |  |  |
| `pib3e1` |  |  | 0.96 (rate only)‡ | 0.91 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s01` | 0.99 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s02` | 1.00 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s03` | 0.99 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s04` | 1.00 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s05` | 1.00 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s06` | 0.90 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s07` | 0.85 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s08` | 1.00 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s09` | 0.97 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s10` | 0.97 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s11` | 0.97 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s12` | 0.97 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s13` | 1.00 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s14` | 1.00 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s15` | 1.00 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s16` | 1.00 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s17` | 1.00 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s18` | 1.00 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s19` | 1.00 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s20` | 1.00 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s22` | 1.00 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s23` | 1.00 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s27` | 1.00 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s29` | 1.00 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s30` | 1.00 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s31` | 1.00 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s32` | 1.00 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s33` | 1.00 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pib3e3s34` | 1.00 (rate only)‡ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `real2sim` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.89 (8/9) [0.56-0.98]‡ |  |
| `rldx` | 0.98 (rate only)‡ | 1.00 (rate only)‡ | 0.97 (rate only)‡ | 1.00 (rate only)‡ |  |  | 0.90 (9/10) [0.60-0.98]† |  |  | 0.80 (8/10) [0.49-0.94]† |  |  | 0.90 (9/10) [0.60-0.98]† |  |  | 1.00 (10/10) [0.72-1.00]† |  |  | 1.00 (10/10) [0.72-1.00]† |  |  | 0.90 (9/10) [0.60-0.98]† |  |  | 1.00 (10/10) [0.72-1.00]† |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `rldx-bridge` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.65 (rate only)‡ |  |  |  |  |  |  |  |  |
| `rldx-gr1` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.50 (rate only)‡ |  |  |
| `rldx-kitchen` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.70 (rate only)‡ |  |  |  |  |  |  |  |
| `rldx-simpler` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1.00 (rate only)‡ |
| `rlinf-oft-libero` |  |  | 0.40 (8/20) [0.22-0.61] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `robolab-cosmos3` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.40 (48/120) [0.32-0.49]‡ |  |  |  |  |
| `robolab-pi05` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.28 (34/120) [0.21-0.37]‡ |  |  |  |  |
| `smolvla` | 0.80 (8/10) [0.49-0.94] |  |  |  |  |  | 0.50 (5/10) [0.24-0.76]† |  |  | 0.20 (2/10) [0.06-0.51]† |  |  | 0.10 (1/10) [0.02-0.40]† |  |  | 0.20 (2/10) [0.06-0.51]† |  |  | 0.70 (7/10) [0.40-0.89]† |  |  | 0.00 (0/10) [0.00-0.28]† |  |  | 0.00 (0/10) [0.00-0.28]† |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `smolvla-robotwin` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.48 (12/25) [0.30-0.67] |  |  |  |
| `vla-adapter` | 1.00 (10/10) [0.72-1.00]‡ |  |  |  |  |  | 1.00 (10/10) [0.72-1.00] |  |  | 0.70 (7/10) [0.40-0.89] |  |  | 1.00 (10/10) [0.72-1.00] |  |  | 0.80 (8/10) [0.49-0.94] |  |  | 1.00 (10/10) [0.72-1.00] |  |  | 1.00 (10/10) [0.72-1.00] |  |  | 0.30 (3/10) [0.11-0.60] |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `xvla` | 0.90 (rate only)‡ |  |  |  |  |  | 0.90 (9/10) [0.60-0.98]† |  |  | 0.00 (0/10) [0.00-0.28]† |  |  | 0.80 (8/10) [0.49-0.94]† |  |  | 0.80 (8/10) [0.49-0.94]† |  |  | 1.00 (10/10) [0.72-1.00]† |  |  | 0.90 (9/10) [0.60-0.98]† |  |  | 0.60 (6/10) [0.31-0.83]† |  |  |  |  |  |  |  |  |  |  |  |  |  |

Cell = latest run under a baseline condition: rate (k/n) [95% Wilson interval]. † = a stack fingerprint key differs from another cell in the same column (see below); ‡ = no stack fingerprint or no condition recorded (older summary; comparability unknown). Compare cells only when episode count, init states, step cap and chunk execution match.

Stack differences per column:

- LIBERO-Plus/camera (Spatial): `stack_mujoco`: `fastwam`=3.7.0, `gr00t`=3.3.1, `openvla-oft`=3.3.1, `pi`=3.2.3, `rldx`=3.3.1, `smolvla`=3.7.0, `xvla`=3.7.0
- LIBERO-Plus/robot (Spatial): `stack_mujoco`: `fastwam`=3.7.0, `gr00t`=3.3.1, `openvla-oft`=3.3.1, `pi`=3.2.3, `rldx`=3.3.1, `smolvla`=3.7.0, `xvla`=3.7.0
- LIBERO-Plus/language (Spatial): `stack_mujoco`: `fastwam`=3.7.0, `gr00t`=3.3.1, `openvla-oft`=3.3.1, `pi`=3.2.3, `rldx`=3.3.1, `smolvla`=3.7.0, `xvla`=3.7.0
- LIBERO-Plus/light (Spatial): `stack_mujoco`: `fastwam`=3.7.0, `gr00t`=3.3.1, `openvla-oft`=3.3.1, `pi`=3.2.3, `rldx`=3.3.1, `smolvla`=3.7.0, `xvla`=3.7.0
- LIBERO-Plus/background (Spatial): `stack_mujoco`: `fastwam`=3.7.0, `gr00t`=3.3.1, `openvla-oft`=3.3.1, `pi`=3.2.3, `rldx`=3.3.1, `smolvla`=3.7.0, `xvla`=3.7.0
- LIBERO-Plus/noise (Spatial): `stack_mujoco`: `fastwam`=3.7.0, `gr00t`=3.3.1, `openvla-oft`=3.3.1, `pi`=3.2.3, `rldx`=3.3.1, `smolvla`=3.7.0, `xvla`=3.7.0
- LIBERO-Plus/layout (Spatial): `stack_mujoco`: `fastwam`=3.7.0, `gr00t`=3.3.1, `openvla-oft`=3.3.1, `pi`=3.2.3, `rldx`=3.3.1, `smolvla`=3.7.0, `xvla`=3.7.0
- LIBERO-R2S/public (Object): `stack_mujoco`: `openvla-oft`=3.3.1, `pi`=3.2.3
- LIBERO-R2S/collision (Object): `stack_mujoco`: `openvla-oft`=3.3.1, `pi`=3.2.3
- LIBERO-R2S/full (Object): `stack_mujoco`: `openvla-oft`=3.3.1, `pi`=3.2.3

Cost and time per cell (**estimates**: EC2 On-Demand only, prices pinned 2026-10-30 from the AWS Price List API):

| Target | Benchmark | Runs | Instance | Instance-min (est.) | USD/run (est.) | USD/episode (est.) | s/episode | ms/step |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `cosmos-policy` | LIBERO-Spatial | 1 | g6e.xlarge | 65 | 2.49 | 0.012 | - | - |
| `cosmos-policy` | LIBERO-Object | 1 | g6e.xlarge | 65 | 2.49 | 0.012 | - | - |
| `cosmos-policy` | LIBERO-Goal | 1 | g6e.xlarge | 84 | 3.20 | 0.016 | - | - |
| `cosmos-policy` | LIBERO-10 | 1 | g6e.xlarge | 84 | 3.20 | 0.016 | - | - |
| `fastwam` | LIBERO-Spatial | 1 | g6e.2xlarge | 155 | 8.42 | 0.034 | 13.4 | - |
| `fastwam` | LIBERO-Object | 1 | g6e.2xlarge | 360 | 19.53 | 0.078 | 13.4 | - |
| `fastwam` | LIBERO-Goal | 1 | g6e.2xlarge | 146 | 6.72 | 0.027 | 11.3 | - |
| `fastwam` | LIBERO-10 | 1 | g6e.2xlarge | 331 | 17.93 | 0.072 | 23.4 | - |
| `fastwam` | LIBERO-Plus/background (Spatial) | 1 | g6e.2xlarge | 88 | 4.06 | 0.058 | 16.4 | - |
| `fastwam` | LIBERO-Plus/camera (Spatial) | 1 | g6e.2xlarge | 88 | 4.06 | 0.058 | 33.6 | - |
| `fastwam` | LIBERO-Plus/language (Spatial) | 1 | g6e.2xlarge | 88 | 4.06 | 0.058 | 18.0 | - |
| `fastwam` | LIBERO-Plus/layout (Spatial) | 1 | g6e.2xlarge | 88 | 4.06 | 0.058 | 25.1 | - |
| `fastwam` | LIBERO-Plus/light (Spatial) | 1 | g6e.2xlarge | 88 | 4.06 | 0.058 | 13.1 | - |
| `fastwam` | LIBERO-Plus/noise (Spatial) | 1 | g6e.2xlarge | 88 | 4.06 | 0.058 | 133.2 | - |
| `fastwam` | LIBERO-Plus/robot (Spatial) | 1 | g6e.2xlarge | 88 | 4.06 | 0.058 | 22.3 | - |
| `gr00t` | LIBERO-Plus/background (Spatial) | 1 | g6.12xlarge | 76 | 7.15 | 0.102 | 25.2 | - |
| `gr00t` | LIBERO-Plus/camera (Spatial) | 1 | g6.12xlarge | 76 | 7.15 | 0.102 | 36.9 | - |
| `gr00t` | LIBERO-Plus/language (Spatial) | 1 | g6.12xlarge | 76 | 7.15 | 0.102 | 31.2 | - |
| `gr00t` | LIBERO-Plus/layout (Spatial) | 1 | g6.12xlarge | 76 | 7.15 | 0.102 | 33.7 | - |
| `gr00t` | LIBERO-Plus/light (Spatial) | 1 | g6.12xlarge | 76 | 7.15 | 0.102 | 30.5 | - |
| `gr00t` | LIBERO-Plus/noise (Spatial) | 1 | g6.12xlarge | 76 | 7.15 | 0.102 | 72.5 | - |
| `gr00t` | LIBERO-Plus/robot (Spatial) | 1 | g6.12xlarge | 76 | 7.15 | 0.102 | 40.3 | - |
| `gr00t` | LIBERO scenes (GR00T) | 1 | g6.2xlarge | 73 | 1.19 | 0.005 | 7.9 | - |
| `gr00t-bridge` | SimplerEnv-Bridge | 1 | g6.2xlarge | 28 | 0.56 | 0.014 | 16.0 | - |
| `gr00t-g1` | G1-LocoManip | 1 | g6.12xlarge | 32 | 3.02 | 0.302 | 46.3 | - |
| `gr00t-gr1` | GR1-Tabletop | 1 | g6.12xlarge | 25 | 2.36 | 0.295 | 21.9 | - |
| `lap` | LIBERO-Spatial | 1 | g5.xlarge | 32 | - | - | - | - |
| `lap` | LIBERO-Plus/background (Spatial) | 1 | g6.xlarge | 52 | 0.85 | 0.012 | - | - |
| `lap` | LIBERO-Plus/camera (Spatial) | 1 | g6.xlarge | 52 | 0.85 | 0.012 | - | - |
| `lap` | LIBERO-Plus/language (Spatial) | 1 | g6.xlarge | 52 | 0.85 | 0.012 | - | - |
| `lap` | LIBERO-Plus/layout (Spatial) | 1 | g6.xlarge | 52 | 0.85 | 0.012 | - | - |
| `lap` | LIBERO-Plus/light (Spatial) | 1 | g6.xlarge | 52 | 0.85 | 0.012 | - | - |
| `lap` | LIBERO-Plus/noise (Spatial) | 1 | g6.xlarge | 52 | 0.85 | 0.012 | - | - |
| `lap` | LIBERO-Plus/robot (Spatial) | 1 | g6.xlarge | 52 | 0.85 | 0.012 | - | - |
| `lingbot-va` | LIBERO-10 | 10 | g6e.2xlarge | 2985 | 13.71 | 0.914 | 1213.5 | - |
| `lingbot-va` | LIBERO-Plus/background (10) | 1 | g6e.2xlarge | 284 | 13.06 | 0.933 | 1100.8 | - |
| `lingbot-va` | LIBERO-Plus/camera (10) | 1 | g6e.2xlarge | 284 | 13.06 | 0.933 | 1318.1 | - |
| `lingbot-va` | LIBERO-Plus/language (10) | 1 | g6e.2xlarge | 284 | 13.06 | 0.933 | 1076.5 | - |
| `lingbot-va` | LIBERO-Plus/layout (10) | 1 | g6e.2xlarge | 284 | 13.06 | 0.933 | 1105.5 | - |
| `lingbot-va` | LIBERO-Plus/light (10) | 1 | g6e.2xlarge | 284 | 13.06 | 0.933 | 1142.6 | - |
| `lingbot-va` | LIBERO-Plus/noise (10) | 1 | g6e.2xlarge | 284 | 13.06 | 0.933 | 1214.6 | - |
| `lingbot-va` | LIBERO-Plus/robot (10) | 1 | g6e.2xlarge | 284 | 13.06 | 0.933 | 1117.1 | - |
| `lingbot-vla2` | RoboTwin2/clean (subset) | 1 | g6e.2xlarge | 48 | 2.19 | 0.088 | 39.8 | - |
| `molmoact2` | LIBERO-Goal | 1 | g6e.2xlarge | 10 | 0.47 | 0.047 | 13.5 | - |
| `molmoact2` | LIBERO-Plus/background (Goal) | 1 | g6e.2xlarge | 46 | 2.11 | 0.030 | 11.2 | - |
| `molmoact2` | LIBERO-Plus/camera (Goal) | 1 | g6e.2xlarge | 46 | 2.11 | 0.030 | 18.9 | - |
| `molmoact2` | LIBERO-Plus/language (Goal) | 1 | g6e.2xlarge | 46 | 2.11 | 0.030 | 14.3 | - |
| `molmoact2` | LIBERO-Plus/layout (Goal) | 1 | g6e.2xlarge | 46 | 2.11 | 0.030 | 19.7 | - |
| `molmoact2` | LIBERO-Plus/light (Goal) | 1 | g6e.2xlarge | 46 | 2.11 | 0.030 | 16.2 | - |
| `molmoact2` | LIBERO-Plus/noise (Goal) | 1 | g6e.2xlarge | 46 | 2.11 | 0.030 | 68.4 | - |
| `molmoact2` | LIBERO-Plus/robot (Goal) | 1 | g6e.2xlarge | 46 | 2.11 | 0.030 | 15.9 | - |
| `oftb3e1g` | LIBERO-Goal | 1 | g6.xlarge | 32 | 0.53 | 0.005 | - | - |
| `oftb3e1l` | LIBERO-10 | 1 | g6.xlarge | 46 | 0.75 | 0.008 | - | - |
| `openvla-oft` | LIBERO-Spatial | 1 | g6.xlarge | 60 | 0.80 | 0.008 | - | - |
| `openvla-oft` | LIBERO-Object | 1 | g6.xlarge | 28 | 0.37 | 0.007 | - | - |
| `openvla-oft` | LIBERO-Goal | 1 | g5.xlarge | 29 | 0.49 | 0.010 | - | - |
| `openvla-oft` | LIBERO-10 | 1 | g6.xlarge | 33 | 0.65 | 0.013 | - | - |
| `openvla-oft` | LIBERO-Plus/background (Spatial) | 1 | g6.xlarge | 55 | 0.91 | 0.013 | - | - |
| `openvla-oft` | LIBERO-Plus/camera (Spatial) | 1 | g6.xlarge | 55 | 0.91 | 0.013 | - | - |
| `openvla-oft` | LIBERO-Plus/language (Spatial) | 1 | g6.xlarge | 55 | 0.91 | 0.013 | - | - |
| `openvla-oft` | LIBERO-Plus/layout (Spatial) | 1 | g6.xlarge | 55 | 0.91 | 0.013 | - | - |
| `openvla-oft` | LIBERO-Plus/light (Spatial) | 1 | g6.xlarge | 55 | 0.91 | 0.013 | - | - |
| `openvla-oft` | LIBERO-Plus/noise (Spatial) | 1 | g6.xlarge | 55 | 0.91 | 0.013 | - | - |
| `openvla-oft` | LIBERO-Plus/robot (Spatial) | 1 | g6.xlarge | 55 | 0.91 | 0.013 | - | - |
| `openvla-oft` | LIBERO-R2S/collision (Object) | 1 | g6.xlarge | 41 | 0.68 | 0.006 | - | - |
| `openvla-oft` | LIBERO-R2S/full (Object) | 1 | g6.xlarge | 41 | 0.68 | 0.006 | - | - |
| `openvla-oft` | LIBERO-R2S/public (Object) | 1 | g6.xlarge | 41 | 0.68 | 0.006 | - | - |
| `pi` | LIBERO-Spatial | 1 | g5.2xlarge | 37 | 0.76 | 0.008 | - | - |
| `pi` | LIBERO-Object | 1 | g5.xlarge | 86 | 1.77 | 0.009 | - | - |
| `pi` | LIBERO-Goal | 1 | g5.xlarge | 86 | 1.77 | 0.009 | - | - |
| `pi` | LIBERO-10 | 1 | g5.xlarge | 86 | 1.77 | 0.009 | - | - |
| `pi` | LIBERO-Plus/background (Spatial) | 1 | g5.xlarge | 75 | 1.55 | 0.022 | - | - |
| `pi` | LIBERO-Plus/camera (Spatial) | 1 | g5.xlarge | 75 | 1.55 | 0.022 | - | - |
| `pi` | LIBERO-Plus/language (Spatial) | 1 | g5.xlarge | 75 | 1.55 | 0.022 | - | - |
| `pi` | LIBERO-Plus/layout (Spatial) | 1 | g5.xlarge | 75 | 1.55 | 0.022 | - | - |
| `pi` | LIBERO-Plus/light (Spatial) | 1 | g5.xlarge | 75 | 1.55 | 0.022 | - | - |
| `pi` | LIBERO-Plus/noise (Spatial) | 1 | g5.xlarge | 75 | 1.55 | 0.022 | - | - |
| `pi` | LIBERO-Plus/robot (Spatial) | 1 | g5.xlarge | 75 | 1.55 | 0.022 | - | - |
| `pi` | LIBERO-R2S/collision (Object) | 1 | g5.xlarge | 57 | 1.17 | 0.010 | - | - |
| `pi` | LIBERO-R2S/full (Object) | 1 | g5.xlarge | 57 | 1.17 | 0.010 | - | - |
| `pi` | LIBERO-R2S/public (Object) | 1 | g5.xlarge | 57 | 1.17 | 0.010 | - | - |
| `pib3e1` | LIBERO-Goal | 1 | g5.xlarge | 86 | 1.78 | 0.009 | - | - |
| `pib3e1` | LIBERO-10 | 1 | g5.xlarge | 86 | 1.78 | 0.009 | - | - |
| `pib3e3s01` | LIBERO-Spatial | 1 | g5.xlarge | 31 | 0.64 | 0.008 | - | - |
| `pib3e3s02` | LIBERO-Spatial | 1 | g5.xlarge | 38 | 0.78 | 0.010 | - | - |
| `pib3e3s03` | LIBERO-Spatial | 1 | g5.xlarge | 54 | 0.90 | 0.006 | - | - |
| `pib3e3s04` | LIBERO-Spatial | 1 | g5.xlarge | 35 | 0.73 | 0.036 | - | - |
| `pib3e3s05` | LIBERO-Spatial | 1 | g5.xlarge | 31 | 0.52 | 0.026 | - | - |
| `pib3e3s06` | LIBERO-Spatial | 1 | g5.xlarge | 25 | 0.41 | 0.021 | - | - |
| `pib3e3s07` | LIBERO-Spatial | 1 | g5.xlarge | 33 | 0.55 | 0.027 | - | - |
| `pib3e3s08` | LIBERO-Spatial | 1 | g5.xlarge | 31 | 0.52 | 0.026 | - | - |
| `pib3e3s09` | LIBERO-Spatial | 1 | g5.xlarge | 33 | 0.55 | 0.014 | - | - |
| `pib3e3s10` | LIBERO-Spatial | 1 | g5.xlarge | 33 | 0.55 | 0.014 | - | - |
| `pib3e3s11` | LIBERO-Spatial | 1 | g5.xlarge | 33 | 0.55 | 0.014 | - | - |
| `pib3e3s12` | LIBERO-Spatial | 1 | g5.xlarge | 33 | 0.55 | 0.014 | - | - |
| `pib3e3s13` | LIBERO-Spatial | 1 | g5.xlarge | 21 | 0.36 | 0.036 | - | - |
| `pib3e3s14` | LIBERO-Spatial | 1 | g5.xlarge | 21 | 0.35 | 0.035 | - | - |
| `pib3e3s15` | LIBERO-Spatial | 1 | g5.xlarge | 22 | 0.37 | 0.037 | - | - |
| `pib3e3s16` | LIBERO-Spatial | 1 | g5.xlarge | 22 | 0.37 | 0.037 | - | - |
| `pib3e3s17` | LIBERO-Spatial | 1 | g5.xlarge | 24 | 0.41 | 0.041 | - | - |
| `pib3e3s18` | LIBERO-Spatial | 1 | g5.xlarge | 23 | 0.39 | 0.039 | - | - |
| `pib3e3s19` | LIBERO-Spatial | 1 | g5.xlarge | 23 | 0.39 | 0.039 | - | - |
| `pib3e3s20` | LIBERO-Spatial | 1 | g5.xlarge | 23 | 0.38 | 0.038 | - | - |
| `pib3e3s22` | LIBERO-Spatial | 1 | g5.xlarge | 24 | 0.50 | 0.050 | - | - |
| `pib3e3s23` | LIBERO-Spatial | 1 | g5.xlarge | 26 | 0.53 | 0.053 | - | - |
| `pib3e3s27` | LIBERO-Spatial | 1 | g5.xlarge | 26 | 0.53 | 0.053 | - | - |
| `pib3e3s29` | LIBERO-Spatial | 1 | g5.xlarge | 23 | 0.39 | 0.039 | - | - |
| `pib3e3s30` | LIBERO-Spatial | 1 | g5.xlarge | 21 | 0.35 | 0.035 | - | - |
| `pib3e3s31` | LIBERO-Spatial | 1 | g5.xlarge | 21 | 0.36 | 0.036 | - | - |
| `pib3e3s32` | LIBERO-Spatial | 1 | g5.xlarge | 21 | 0.35 | 0.035 | - | - |
| `pib3e3s33` | LIBERO-Spatial | 1 | g5.xlarge | 21 | 0.36 | 0.036 | - | - |
| `pib3e3s34` | LIBERO-Spatial | 1 | g5.xlarge | 27 | 0.45 | 0.022 | - | - |
| `real2sim` | hope-video-scene-0000 | 1 | g5.2xlarge | 12 | 0.30 | 0.034 | - | - |
| `rldx` | LIBERO-Spatial | 1 | g6.xlarge | 56 | 0.92 | 0.004 | - | - |
| `rldx` | LIBERO-Object | 1 | g6.xlarge | 56 | 1.08 | 0.004 | - | - |
| `rldx` | LIBERO-Goal | 1 | g5.xlarge | 97 | 1.62 | 0.006 | - | - |
| `rldx` | LIBERO-10 | 1 | g6.xlarge | 18 | 0.30 | 0.030 | - | - |
| `rldx` | LIBERO-Plus/background (Spatial) | 1 | g6.xlarge | 94 | 1.54 | 0.022 | - | - |
| `rldx` | LIBERO-Plus/camera (Spatial) | 1 | g6.xlarge | 94 | 1.54 | 0.022 | - | - |
| `rldx` | LIBERO-Plus/language (Spatial) | 1 | g6.xlarge | 94 | 1.54 | 0.022 | - | - |
| `rldx` | LIBERO-Plus/layout (Spatial) | 1 | g6.xlarge | 94 | 1.54 | 0.022 | - | - |
| `rldx` | LIBERO-Plus/light (Spatial) | 1 | g6.xlarge | 94 | 1.54 | 0.022 | - | - |
| `rldx` | LIBERO-Plus/noise (Spatial) | 1 | g6.xlarge | 94 | 1.54 | 0.022 | - | - |
| `rldx` | LIBERO-Plus/robot (Spatial) | 1 | g6.xlarge | 94 | 1.54 | 0.022 | - | - |
| `rldx-bridge` | SimplerEnv-Bridge | 1 | g6.2xlarge | 38 | 0.77 | 0.019 | - | - |
| `rldx-gr1` | gr1_tabletop | 1 | g5.xlarge | 30 | 0.51 | 0.051 | - | - |
| `rldx-kitchen` | RoboCasa-Kitchen | 1 | g6.2xlarge | 26 | 0.52 | 0.052 | - | - |
| `rldx-simpler` | simpler_google_vm | 1 | g5.2xlarge | 27 | 0.54 | 0.054 | - | - |
| `rlinf-oft-libero` | LIBERO-Goal | 1 | g6e.8xlarge | 62 | - | - | - | - |
| `robolab-cosmos3` | RoboLab | 1 | g6e.2xlarge | 175 | 9.47 | 0.079 | - | - |
| `robolab-pi05` | RoboLab | 1 | g6e.4xlarge | 164 | 8.23 | 0.069 | - | - |
| `smolvla` | LIBERO-Spatial | 1 | g6.xlarge | 9 | 0.15 | 0.015 | 14.6 | - |
| `smolvla` | LIBERO-Plus/background (Spatial) | 1 | g6.xlarge | 39 | 0.64 | 0.013 | 8.8 | - |
| `smolvla` | LIBERO-Plus/camera (Spatial) | 1 | g6.xlarge | 39 | 0.64 | 0.013 | 16.9 | - |
| `smolvla` | LIBERO-Plus/language (Spatial) | 1 | g6.xlarge | 39 | 0.64 | 0.013 | 10.9 | - |
| `smolvla` | LIBERO-Plus/layout (Spatial) | 1 | g6.xlarge | 53 | 0.87 | 0.043 | 15.8 | - |
| `smolvla` | LIBERO-Plus/light (Spatial) | 1 | g6.xlarge | 39 | 0.64 | 0.013 | 7.7 | - |
| `smolvla` | LIBERO-Plus/noise (Spatial) | 1 | g6.xlarge | 53 | 0.87 | 0.043 | 223.4 | - |
| `smolvla` | LIBERO-Plus/robot (Spatial) | 1 | g6.xlarge | 39 | 0.64 | 0.013 | 11.5 | - |
| `smolvla-robotwin` | RoboTwin2/clean (subset) | 1 | g6.2xlarge | 54 | 1.07 | 0.043 | 94.2 | - |
| `vla-adapter` | LIBERO-Spatial | 1 | g6.xlarge | 10 | 0.17 | 0.017 | - | - |
| `vla-adapter` | LIBERO-Plus/background (Spatial) | 1 | g6.xlarge | 37 | 0.62 | 0.009 | - | - |
| `vla-adapter` | LIBERO-Plus/camera (Spatial) | 1 | g6.xlarge | 37 | 0.62 | 0.009 | - | - |
| `vla-adapter` | LIBERO-Plus/language (Spatial) | 1 | g6.xlarge | 37 | 0.62 | 0.009 | - | - |
| `vla-adapter` | LIBERO-Plus/layout (Spatial) | 1 | g6.xlarge | 37 | 0.62 | 0.009 | - | - |
| `vla-adapter` | LIBERO-Plus/light (Spatial) | 1 | g6.xlarge | 37 | 0.62 | 0.009 | - | - |
| `vla-adapter` | LIBERO-Plus/noise (Spatial) | 1 | g6.xlarge | 37 | 0.62 | 0.009 | - | - |
| `vla-adapter` | LIBERO-Plus/robot (Spatial) | 1 | g6.xlarge | 37 | 0.62 | 0.009 | - | - |
| `xvla` | LIBERO-Spatial | 1 | g6.2xlarge | 8 | 0.17 | 0.017 | 12.8 | - |
| `xvla` | LIBERO-Plus/background (Spatial) | 1 | g6.2xlarge | 45 | 0.89 | 0.013 | 8.9 | - |
| `xvla` | LIBERO-Plus/camera (Spatial) | 1 | g6.2xlarge | 45 | 0.89 | 0.013 | 30.7 | - |
| `xvla` | LIBERO-Plus/language (Spatial) | 1 | g6.2xlarge | 45 | 0.89 | 0.013 | 11.4 | - |
| `xvla` | LIBERO-Plus/layout (Spatial) | 1 | g6.2xlarge | 45 | 0.89 | 0.013 | 16.2 | - |
| `xvla` | LIBERO-Plus/light (Spatial) | 1 | g6.2xlarge | 45 | 0.89 | 0.013 | 6.5 | - |
| `xvla` | LIBERO-Plus/noise (Spatial) | 1 | g6.2xlarge | 45 | 0.89 | 0.013 | 92.3 | - |
| `xvla` | LIBERO-Plus/robot (Spatial) | 1 | g6.2xlarge | 45 | 0.89 | 0.013 | 15.3 | - |

Instance-min = sum over the pooled runs of RUN_ID (instance boot) to the last object the run uploaded, or `run_wall_s` from `summary.txt`; USD/episode divides by every episode of those runs (all suites of a run share its instance); stack create/delete, NAT gateway, EBS and data transfer are not included. USD/run is per pooled run. s/episode comes from `episode_wall_s`, LeRobot `eval_info.json` (`eval_ep_s`) or the GR00T rollout log; ms/step only from `policy_ms_per_step`.

Sources: `cosmos-policy` LIBERO-Spatial: cosmos-t1-a@ap-northeast-2 2026-10-31-0211; `cosmos-policy` LIBERO-Object: cosmos-t1-a@ap-northeast-2 2026-10-31-0211; `cosmos-policy` LIBERO-Goal: cosmos-t1-b@ap-northeast-2 2026-10-31-0213; `cosmos-policy` LIBERO-10: cosmos-t1-b@ap-northeast-2 2026-10-31-0213; `fastwam` LIBERO-Spatial: fastwam-s@ap-northeast-1 2026-10-15-1356; `fastwam` LIBERO-Object: fastwam-o@ap-northeast-1 2026-10-15-1356; `fastwam` LIBERO-Goal: fastwam-g@ap-northeast-2 2026-10-15-1428; `fastwam` LIBERO-10: fastwam-l@ap-northeast-1 2026-10-15-1355; `fastwam` LIBERO-Plus/background (Spatial): fastwam-demo@ap-northeast-2 2026-10-31-1103; `fastwam` LIBERO-Plus/camera (Spatial): fastwam-demo@ap-northeast-2 2026-10-31-1103; `fastwam` LIBERO-Plus/language (Spatial): fastwam-demo@ap-northeast-2 2026-10-31-1103; `fastwam` LIBERO-Plus/layout (Spatial): fastwam-demo@ap-northeast-2 2026-10-31-1103; `fastwam` LIBERO-Plus/light (Spatial): fastwam-demo@ap-northeast-2 2026-10-31-1103; `fastwam` LIBERO-Plus/noise (Spatial): fastwam-demo@ap-northeast-2 2026-10-31-1103; `fastwam` LIBERO-Plus/robot (Spatial): fastwam-demo@ap-northeast-2 2026-10-31-1103; `gr00t` LIBERO-Plus/background (Spatial): gr00t-demo@ap-northeast-2 2026-10-31-1330; `gr00t` LIBERO-Plus/camera (Spatial): gr00t-demo@ap-northeast-2 2026-10-31-1330; `gr00t` LIBERO-Plus/language (Spatial): gr00t-demo@ap-northeast-2 2026-10-31-1330; `gr00t` LIBERO-Plus/layout (Spatial): gr00t-demo@ap-northeast-2 2026-10-31-1330; `gr00t` LIBERO-Plus/light (Spatial): gr00t-demo@ap-northeast-2 2026-10-31-1330; `gr00t` LIBERO-Plus/noise (Spatial): gr00t-demo@ap-northeast-2 2026-10-31-1330; `gr00t` LIBERO-Plus/robot (Spatial): gr00t-demo@ap-northeast-2 2026-10-31-1330; `gr00t` LIBERO scenes (GR00T): gr00t-demo@us-east-1 2026-10-15-0847; `gr00t-bridge` SimplerEnv-Bridge: gr00t-bridge-demo@ap-northeast-2 2026-10-30-1519; `gr00t-g1` G1-LocoManip: gr00t-g1-demo@ap-northeast-2 2026-10-09-1435; `gr00t-gr1` GR1-Tabletop: gr00t-gr1-demo@ap-northeast-2 2026-10-09-1339; `lap` LIBERO-Spatial: lap-demo@us-west-2 2026-10-09-1444; `lap` LIBERO-Plus/background (Spatial): lap-demo@ap-northeast-2 2026-10-31-1238; `lap` LIBERO-Plus/camera (Spatial): lap-demo@ap-northeast-2 2026-10-31-1238; `lap` LIBERO-Plus/language (Spatial): lap-demo@ap-northeast-2 2026-10-31-1238; `lap` LIBERO-Plus/layout (Spatial): lap-demo@ap-northeast-2 2026-10-31-1238; `lap` LIBERO-Plus/light (Spatial): lap-demo@ap-northeast-2 2026-10-31-1238; `lap` LIBERO-Plus/noise (Spatial): lap-demo@ap-northeast-2 2026-10-31-1238; `lap` LIBERO-Plus/robot (Spatial): lap-demo@ap-northeast-2 2026-10-31-1238; `lingbot-va` LIBERO-10: lingbot-t9@ap-northeast-2 2026-10-15-1707, lingbot-t8@ap-northeast-2 2026-10-15-1706, lingbot-t7@ap-northeast-2 2026-10-15-1705, lingbot-t6@ap-northeast-2 2026-10-15-1704, lingbot-t5@ap-northeast-2 2026-10-15-1703, lingbot-t4@ap-northeast-2 2026-10-15-1703, lingbot-t3@ap-northeast-2 2026-10-15-1702, lingbot-t2@ap-northeast-2 2026-10-15-1701, lingbot-t1@ap-northeast-2 2026-10-15-1700, lingbot-t0@ap-northeast-2 2026-10-15-1659; `lingbot-va` LIBERO-Plus/background (10): lingbot-va-demo@ap-northeast-2 2026-10-31-1454; `lingbot-va` LIBERO-Plus/camera (10): lingbot-va-demo@ap-northeast-2 2026-10-31-1454; `lingbot-va` LIBERO-Plus/language (10): lingbot-va-demo@ap-northeast-2 2026-10-31-1454; `lingbot-va` LIBERO-Plus/layout (10): lingbot-va-demo@ap-northeast-2 2026-10-31-1454; `lingbot-va` LIBERO-Plus/light (10): lingbot-va-demo@ap-northeast-2 2026-10-31-1454; `lingbot-va` LIBERO-Plus/noise (10): lingbot-va-demo@ap-northeast-2 2026-10-31-1454; `lingbot-va` LIBERO-Plus/robot (10): lingbot-va-demo@ap-northeast-2 2026-10-31-1454; `lingbot-vla2` RoboTwin2/clean (subset): lingbot-vla2-demo@ap-northeast-2 2026-10-30-1728; `molmoact2` LIBERO-Goal: molmoact2-demo@ap-northeast-2 2026-10-09-1348; `molmoact2` LIBERO-Plus/background (Goal): molmoact2-demo@ap-northeast-2 2026-10-31-1238; `molmoact2` LIBERO-Plus/camera (Goal): molmoact2-demo@ap-northeast-2 2026-10-31-1238; `molmoact2` LIBERO-Plus/language (Goal): molmoact2-demo@ap-northeast-2 2026-10-31-1238; `molmoact2` LIBERO-Plus/layout (Goal): molmoact2-demo@ap-northeast-2 2026-10-31-1238; `molmoact2` LIBERO-Plus/light (Goal): molmoact2-demo@ap-northeast-2 2026-10-31-1238; `molmoact2` LIBERO-Plus/noise (Goal): molmoact2-demo@ap-northeast-2 2026-10-31-1238; `molmoact2` LIBERO-Plus/robot (Goal): molmoact2-demo@ap-northeast-2 2026-10-31-1238; `oftb3e1g` LIBERO-Goal: oftb3e1g@ap-northeast-2 2026-10-30-1818; `oftb3e1l` LIBERO-10: oftb3e1l@ap-northeast-2 2026-10-30-1818; `openvla-oft` LIBERO-Spatial: oft-spatial-h@us-east-1 2026-10-15-1149; `openvla-oft` LIBERO-Object: openvla-oft-object-demo@us-east-1 2026-10-15-1147; `openvla-oft` LIBERO-Goal: openvla-oft-goal-demo@us-east-1 2026-10-15-0854; `openvla-oft` LIBERO-10: openvla-oft-demo@ap-northeast-1 2026-10-15-0947; `openvla-oft` LIBERO-Plus/background (Spatial): openvla-oft-spatial@ap-northeast-2 2026-10-30-1516; `openvla-oft` LIBERO-Plus/camera (Spatial): openvla-oft-spatial@ap-northeast-2 2026-10-30-1516; `openvla-oft` LIBERO-Plus/language (Spatial): openvla-oft-spatial@ap-northeast-2 2026-10-30-1516; `openvla-oft` LIBERO-Plus/layout (Spatial): openvla-oft-spatial@ap-northeast-2 2026-10-30-1516; `openvla-oft` LIBERO-Plus/light (Spatial): openvla-oft-spatial@ap-northeast-2 2026-10-30-1516; `openvla-oft` LIBERO-Plus/noise (Spatial): openvla-oft-spatial@ap-northeast-2 2026-10-30-1516; `openvla-oft` LIBERO-Plus/robot (Spatial): openvla-oft-spatial@ap-northeast-2 2026-10-30-1516; `openvla-oft` LIBERO-R2S/collision (Object): openvla-oft-object@ap-northeast-2 2026-10-31-0725; `openvla-oft` LIBERO-R2S/full (Object): openvla-oft-object@ap-northeast-2 2026-10-31-0725; `openvla-oft` LIBERO-R2S/public (Object): openvla-oft-object@ap-northeast-2 2026-10-31-0725; `pi` LIBERO-Spatial: pi-demo@us-east-1 2026-10-26-1911; `pi` LIBERO-Object: pi-demo@ap-northeast-2 2026-10-15-0948; `pi` LIBERO-Goal: pi-demo@ap-northeast-2 2026-10-15-0948; `pi` LIBERO-10: pi-demo@ap-northeast-2 2026-10-15-0948; `pi` LIBERO-Plus/background (Spatial): pi-demo@ap-northeast-2 2026-10-31-1336; `pi` LIBERO-Plus/camera (Spatial): pi-demo@ap-northeast-2 2026-10-31-1336; `pi` LIBERO-Plus/language (Spatial): pi-demo@ap-northeast-2 2026-10-31-1336; `pi` LIBERO-Plus/layout (Spatial): pi-demo@ap-northeast-2 2026-10-31-1336; `pi` LIBERO-Plus/light (Spatial): pi-demo@ap-northeast-2 2026-10-31-1336; `pi` LIBERO-Plus/noise (Spatial): pi-demo@ap-northeast-2 2026-10-31-1336; `pi` LIBERO-Plus/robot (Spatial): pi-demo@ap-northeast-2 2026-10-31-1336; `pi` LIBERO-R2S/collision (Object): pi-demo@ap-northeast-2 2026-10-31-0812; `pi` LIBERO-R2S/full (Object): pi-demo@ap-northeast-2 2026-10-31-0812; `pi` LIBERO-R2S/public (Object): pi-demo@ap-northeast-2 2026-10-31-0812; `pib3e1` LIBERO-Goal: pib3e1@ap-northeast-2 2026-10-30-1812; `pib3e1` LIBERO-10: pib3e1@ap-northeast-2 2026-10-30-1812; `pib3e3s01` LIBERO-Spatial: pib3e3s01@ap-northeast-2 2026-10-30-1829; `pib3e3s02` LIBERO-Spatial: pib3e3s02@ap-northeast-2 2026-10-30-1829; `pib3e3s03` LIBERO-Spatial: pib3e3s03@us-west-2 2026-10-30-1818; `pib3e3s04` LIBERO-Spatial: pib3e3s04@ap-northeast-2 2026-10-30-1844; `pib3e3s05` LIBERO-Spatial: pib3e3s05@us-east-1 2026-10-30-1816; `pib3e3s06` LIBERO-Spatial: pib3e3s06@us-east-1 2026-10-30-1815; `pib3e3s07` LIBERO-Spatial: pib3e3s07@us-east-1 2026-10-30-1816; `pib3e3s08` LIBERO-Spatial: pib3e3s08@us-east-1 2026-10-30-1816; `pib3e3s09` LIBERO-Spatial: pib3e3s09@us-west-2 2026-10-30-1829; `pib3e3s10` LIBERO-Spatial: pib3e3s10@us-west-2 2026-10-30-1829; `pib3e3s11` LIBERO-Spatial: pib3e3s11@us-west-2 2026-10-30-1829; `pib3e3s12` LIBERO-Spatial: pib3e3s12@us-west-2 2026-10-30-1829; `pib3e3s13` LIBERO-Spatial: pib3e3s13@us-east-1 2026-10-30-1907; `pib3e3s14` LIBERO-Spatial: pib3e3s14@us-east-1 2026-10-30-1907; `pib3e3s15` LIBERO-Spatial: pib3e3s15@us-east-1 2026-10-30-1906; `pib3e3s16` LIBERO-Spatial: pib3e3s16@us-east-1 2026-10-30-1906; `pib3e3s17` LIBERO-Spatial: pib3e3s17@us-east-1 2026-10-30-1903; `pib3e3s18` LIBERO-Spatial: pib3e3s18@us-east-1 2026-10-30-1903; `pib3e3s19` LIBERO-Spatial: pib3e3s19@us-east-1 2026-10-30-1903; `pib3e3s20` LIBERO-Spatial: pib3e3s20@us-east-1 2026-10-30-1903; `pib3e3s22` LIBERO-Spatial: pib3e3s22@ap-northeast-2 2026-10-30-1903; `pib3e3s23` LIBERO-Spatial: pib3e3s23@ap-northeast-2 2026-10-30-1903; `pib3e3s27` LIBERO-Spatial: pib3e3s27@ap-northeast-2 2026-10-30-1903; `pib3e3s29` LIBERO-Spatial: pib3e3s29@us-east-1 2026-10-30-1906; `pib3e3s30` LIBERO-Spatial: pib3e3s30@us-east-1 2026-10-30-1908; `pib3e3s31` LIBERO-Spatial: pib3e3s31@us-east-1 2026-10-30-1906; `pib3e3s32` LIBERO-Spatial: pib3e3s32@us-east-1 2026-10-30-1907; `pib3e3s33` LIBERO-Spatial: pib3e3s33@us-east-1 2026-10-30-1906; `pib3e3s34` LIBERO-Spatial: pib3e3s34@us-east-1 2026-10-30-1921; `real2sim` hope-video-scene-0000: real2sim-demo@ap-northeast-2 2026-10-29-0142; `rldx` LIBERO-Spatial: rldx-demo@ap-northeast-2 2026-08-19-0959; `rldx` LIBERO-Object: rldx-demo@ap-northeast-1 2026-08-19-0855; `rldx` LIBERO-Goal: rldx-demo@us-east-1 2026-08-19-0855; `rldx` LIBERO-10: rldx-demo@ap-northeast-2 2026-10-09-1346; `rldx` LIBERO-Plus/background (Spatial): rldx-demo@ap-northeast-2 2026-10-31-0923; `rldx` LIBERO-Plus/camera (Spatial): rldx-demo@ap-northeast-2 2026-10-31-0923; `rldx` LIBERO-Plus/language (Spatial): rldx-demo@ap-northeast-2 2026-10-31-0923; `rldx` LIBERO-Plus/layout (Spatial): rldx-demo@ap-northeast-2 2026-10-31-0923; `rldx` LIBERO-Plus/light (Spatial): rldx-demo@ap-northeast-2 2026-10-31-0923; `rldx` LIBERO-Plus/noise (Spatial): rldx-demo@ap-northeast-2 2026-10-31-0923; `rldx` LIBERO-Plus/robot (Spatial): rldx-demo@ap-northeast-2 2026-10-31-0923; `rldx-bridge` SimplerEnv-Bridge: rldx-bridge-demo@ap-northeast-2 2026-10-30-1332; `rldx-gr1` gr1_tabletop: rldx-gr1-demo@us-west-2 2026-10-09-1333; `rldx-kitchen` RoboCasa-Kitchen: rldx-kitchen-demo@ap-northeast-2 2026-10-09-1343; `rldx-simpler` simpler_google_vm: rldx-simpler-demo@us-east-1 2026-10-09-1327; `rlinf-oft-libero` LIBERO-Goal: rlinf-oft-libero@ap-northeast-2 2026-10-31-1828; `robolab-cosmos3` RoboLab: robolab-cosmos3@ap-northeast-1 2026-10-31-1058; `robolab-pi05` RoboLab: robolab-pi05-demo@us-west-2 2026-10-31-1532; `smolvla` LIBERO-Spatial: smolvla-demo@ap-northeast-2 2026-10-27-0514; `smolvla` LIBERO-Plus/background (Spatial): smolvla-demo@ap-northeast-2 2026-10-30-1516; `smolvla` LIBERO-Plus/camera (Spatial): smolvla-demo@ap-northeast-2 2026-10-30-1516; `smolvla` LIBERO-Plus/language (Spatial): smolvla-demo@ap-northeast-2 2026-10-30-1516; `smolvla` LIBERO-Plus/layout (Spatial): smolvla-demo@ap-northeast-2 2026-10-30-1610; `smolvla` LIBERO-Plus/light (Spatial): smolvla-demo@ap-northeast-2 2026-10-30-1516; `smolvla` LIBERO-Plus/noise (Spatial): smolvla-demo@ap-northeast-2 2026-10-30-1610; `smolvla` LIBERO-Plus/robot (Spatial): smolvla-demo@ap-northeast-2 2026-10-30-1516; `smolvla-robotwin` RoboTwin2/clean (subset): smolvla-robotwin@ap-northeast-2 2026-10-30-1728; `vla-adapter` LIBERO-Spatial: vla-adapter-demo@ap-northeast-2 2026-10-14-1337; `vla-adapter` LIBERO-Plus/background (Spatial): vla-adapter-demo@ap-northeast-2 2026-10-31-0817; `vla-adapter` LIBERO-Plus/camera (Spatial): vla-adapter-demo@ap-northeast-2 2026-10-31-0817; `vla-adapter` LIBERO-Plus/language (Spatial): vla-adapter-demo@ap-northeast-2 2026-10-31-0817; `vla-adapter` LIBERO-Plus/layout (Spatial): vla-adapter-demo@ap-northeast-2 2026-10-31-0817; `vla-adapter` LIBERO-Plus/light (Spatial): vla-adapter-demo@ap-northeast-2 2026-10-31-0817; `vla-adapter` LIBERO-Plus/noise (Spatial): vla-adapter-demo@ap-northeast-2 2026-10-31-0817; `vla-adapter` LIBERO-Plus/robot (Spatial): vla-adapter-demo@ap-northeast-2 2026-10-31-0817; `xvla` LIBERO-Spatial: xvla-demo@ap-northeast-2 2026-10-14-0642; `xvla` LIBERO-Plus/background (Spatial): xvla-demo@ap-northeast-2 2026-10-31-0726; `xvla` LIBERO-Plus/camera (Spatial): xvla-demo@ap-northeast-2 2026-10-31-0726; `xvla` LIBERO-Plus/language (Spatial): xvla-demo@ap-northeast-2 2026-10-31-0726; `xvla` LIBERO-Plus/layout (Spatial): xvla-demo@ap-northeast-2 2026-10-31-0726; `xvla` LIBERO-Plus/light (Spatial): xvla-demo@ap-northeast-2 2026-10-31-0726; `xvla` LIBERO-Plus/noise (Spatial): xvla-demo@ap-northeast-2 2026-10-31-0726; `xvla` LIBERO-Plus/robot (Spatial): xvla-demo@ap-northeast-2 2026-10-31-0726.
<!-- leaderboard:end -->

## Comparing two policies (`--compare`)

`python vlasim.py results --compare A B` answers "is A better than B, or is it a tie?" one row per suite and
evaluation setting. A side is `<stack>[:<condition>]`: without a condition both sides are matched on (suite,
condition); `openvla-oft:S0 openvla-oft:S3b` compares one policy under two settings, matched on suite.

Each row shows both k/n, a two-sided Fisher exact p and a Benjamini-Hochberg q over all rows of the comparison;
the verdict is `A>B`, `B>A` or `tie` at q < 0.05 (`--alpha`, `--raw-p` decides on p). When both sides store
per-episode outcomes (video names `episode=<i>--success=<bool>--task=<t>`), episodes are paired by (task, k-th
occurrence of that task) — the start-state index in LIBERO — and an exact McNemar p is printed as well.
With `--all-runs`, runs of one stack with the same suite, condition and stack fingerprint are compared with the
same test and reported as the **re-run null**: if re-running a stack produces verdicts of its own, a verdict
between two policies at that sample size means little.

```
$ python vlasim.py results --compare fastwam:S0 fastwam:S3b --from-csv cells.csv --episodes episodes.json
suite           condition  A=fastwam:S0  B=fastwam:S3b  fisher_p  bh_q     verdict  paired (only A / only B, p)
libero_10       S0 vs S3b  48/50         27/50          1.1e-06   4.6e-06  A>B      50 (22/1, p=5.7e-06)
...
```

`--from-csv` reads a table with columns `model,suite,condition,successes,episodes` (A and B are model names) and
needs no AWS access; `--episodes` adds per-episode outcomes for the paired test. The statistics are pure Python
and are tested against SciPy / statsmodels on the public AMLC 2026 study table (`tests/fixtures/study_cells.csv`).
`--csv` / `--json` write the rows.

## Protocol card (`protocol.yaml`)

Every task that writes `summary.txt` also writes `protocol.yaml` next to it (`templates/_protocol-card.sh.j2`,
27 of the 28 targets; `openarm-lift-act` writes no summary). It records what a reader needs to decide whether
two numbers are comparable:

| field | source |
|---|---|
| `target`, `checkpoints` (repo, revision, suite), `code_pins`, `tool_commit` | model YAML and this repository at render time |
| `suite`, `condition`, `seed`, `start_states`, `chunk_rule`, `step_cap`, `instruction`, `k`, `n` | this task's `summary.txt` (the key that was found is kept, e.g. `replan=1`) |
| `instance_type`, `gpu`, `region`, `written_utc` | instance metadata and `nvidia-smi` at run time |

A field the target does not record is written as `unknown`, never guessed. `--matrix` reads the cards: a cell
whose tasks all wrote one is no longer marked ‡ (comparability unknown); the CSV / JSON output has a
`protocol_card` column.

## Shared seed (`--seed`)

A comparison is only reproducible if the randomness behind each number is known and the same for every
policy. `deploy --seed` sets it for one run:

- `--seed auto` draws one seed when the test starts (`secrets.randbits(31)`) and prints it, for example
  `[seed] drew 1460571231 at 2026-10-31T05:20:00Z — deploy every other target of this comparison with --seed 1460571231`.
- `--seed <n>` gives that value to every task of the run. Use the same `<n>` for every target you compare.
- `--seed model` (the default) keeps the per-task seeds of `models/<target>.yaml`, i.e. the settings behind
  the numbers already published in this file.

A seed has two jobs: the **env seed** decides start states, object placement and resets; the **policy seed**
decides sampling noise (diffusion / flow noise, token sampling). On LIBERO the start states come from the
benchmark's init-state files, so they are already shared between policies (`fixed` below) and the seed
reaches the policy side. Comparing two policies on the same start states is what the paired McNemar test of
`--compare` uses; a shared policy seed does not correlate two different networks, it makes each run
repeatable.

`protocol.yaml` records the plan under `seed_plan` (`mode` drawn / given / model, `value`, `drawn_utc`,
`status` applied / model / not-controllable, `env_seed`, `policy_seed`, `via`, `not_controllable_reason`).
The tool writes the seed for 14 of the 28 targets. For the others it does not
pretend: the card says `not-controllable` and why.

| target | where the tool writes it | env seed | policy seed | how / why not |
|---|---|---|---|---|
| `gr00t` | not set by the tool | no | no | rollout_policy.py at Isaac-GR00T 4b1dca9 has no seed and calls env.reset() without one (random resets); upstream 65cc4a1 adds --seed |
| `gr00t-bridge` | every task's `seed` | no | yes | rollout_policy.py --seed (seeds Python / NumPy / torch) |
| `gr00t-gr1` | not set by the tool | no | no | the template passes no seed to the RoboCasa GR-1 rollout (random resets) |
| `gr00t-g1` | not set by the tool | no | no | rollout_policy.py at Isaac-GR00T 7786639 has a seedless env.reset() |
| `pi` | every task's `seed` | fixed | yes | openpi examples/libero main.py --args.seed |
| `openvla-oft` | every task's `seed` | fixed | yes | run_libero_eval.py --seed |
| `lap` | not set by the tool | no | no | the template passes no seed to the LAP evaluation client |
| `rldx` | not set by the tool | no | no | the template passes no seed to the RLDX evaluation entry point |
| `rldx-simpler` | not set by the tool | no | no | the template passes no seed to the RLDX evaluation entry point |
| `rldx-bridge` | not set by the tool | no | no | the template passes no seed to the RLDX evaluation entry point |
| `rldx-gr1` | not set by the tool | no | no | the template passes no seed to the RLDX evaluation entry point |
| `rldx-kitchen` | not set by the tool | no | no | the template passes no seed to the RLDX evaluation entry point |
| `rlinf-oft-libero` | not set by the tool | no | no | the template passes no seed to RLinf evaluations/run_eval.sh (the eval config's own seed applies) |
| `openarm-isaac` | every task's `seed` | yes | yes | run_eval.py --seed (Isaac Lab env + policy) |
| `openarm-lift-act` | not set by the tool | no | no | the template passes no seed to the ACT play script; it writes no summary.txt |
| `molmoact2` | every task's `seed` | fixed | yes | lerobot-eval --seed |
| `xvla` | every task's `seed` | fixed | yes | lerobot-eval --seed |
| `smolvla` | every task's `seed` | fixed | yes | lerobot-eval --seed |
| `lingbot-va` | every task's `seed` | fixed | yes | lerobot-eval --seed |
| `vla-adapter` | not set by the tool | fixed | no | the template passes no seed to the upstream eval script |
| `fastwam` | every task's `seed` | fixed | yes | lerobot-eval --seed |
| `cosmos-policy` | every task's `seed` | fixed | yes | run_libero_eval --seed (with --randomize_seed False) |
| `aspire` | not set by the tool | fixed | no | scored on LIBERO-Pro saved init states 0..4 (eval_seeds index them); the frozen program has no sampling |
| `real2sim` | not set by the tool | no | no | reconstruction pipeline, not a policy evaluation (segment_args fix its own seed) |
| `robolab-cosmos3` | `eval.seed` | yes | yes | RoboLab eval --seed (one seed per run) |
| `robolab-pi05` | `eval.seed` | yes | yes | RoboLab eval --seed (one seed per run) |
| `smolvla-robotwin` | every task's `seed` | yes | yes | lerobot-eval --seed (RoboTwin setup_demo seed) |
| `lingbot-vla2` | every task's `seed` | yes | yes | lerobot-eval --seed (RoboTwin setup_demo seed) |

## Re-run null per simulator

A verdict is only as good as the noise under it. On LIBERO (MuJoCo, CPU physics) an identical re-run gives the
same episodes back: with the same seed, SmolVLA and X-VLA repeated 50 of 50 LIBERO-Spatial episodes, also on
another machine and region. On GPU physics it does not: an identical RoboLab (Isaac Lab) re-run changed 25 of 120
pi0.5 episodes and 19 of 120 Cosmos3 episodes and turned a tie (p = 0.69) into a win (p = 0.023); in Isaac Lab
Arena the same seed on the same machine changed 9 to 46 of 100 episodes between two runs. A shared seed
(`--seed`) is therefore necessary but not sufficient there: the comparison also needs repeated runs and their spread.

`repeatability.py` records this per simulator (`yes` / `no` / `unmeasured`, with the evidence), and
`results --compare` applies it:

| simulator class | one finished run per side | two or more finished runs (same stack, suite, condition, fingerprint) |
|---|---|---|
| `yes` (LIBERO) | verdict as before | verdict as before; the k range is printed |
| `no` (RoboLab, Isaac Lab) | `needs re-run`; the one-run verdict is kept as `single_run_verdict` | the re-run spread (k range, episodes that changed between runs) is printed and the verdict is tested on every pair of runs (one run of A x one run of B, two-sided Fisher at `--alpha`); if the pairs disagree the verdict is `unstable` |
| `unmeasured` (SimplerEnv, RoboCasa, RoboTwin, other MuJoCo) | verdict as before, with a note | as `yes` |

On S3, runs of one bucket are only read with `--all-runs`; runs of the same target in two regions are two buckets
and both count. Offline, a `run` column in the `--from-csv` table marks several runs of one cell and a `simulator`
column overrides the target table (`--episodes` entries may carry `run`). `--assume-repeatable` turns the rule off.

```
$ python vlasim.py results --compare robolab-pi05 robolab-cosmos3 --from-csv tests/fixtures/rerun_cells.csv \
      --episodes tests/fixtures/rerun_episodes.json
suite       condition  A=robolab-pi05  B=robolab-cosmos3  ...  verdict       re-run
robolab-12  C0         54/120          36/120             ...  unstable      A 45-54/120 over 2 runs, 25 of 120 episodes changed (21%); B 36-41/120 over 2 runs, 19 of 120 episodes changed (16%); run pairs A>B 1 · tie 3
robolab-12  C1         55/120          36/120             ...  needs re-run  one run only (AB); one-run verdict A>B
robolab-12  Cs         27/120          51/120             ...  needs re-run  one run only (AB); one-run verdict B>A
```

The protocol card records the simulator class next to every result (`rerun_null`: `simulator`, `repeatable`,
`evidence`, `runs_for_a_verdict` = 2 where the simulator does not repeat), so a reader of one result file knows
whether one run of it can carry a verdict.

### Two runs per deploy where re-runs do not repeat (`--repeats`)

So that the second run is not forgotten, `deploy.py` runs the whole evaluation twice by default on the targets
whose simulator class is `no` and whose UserData supports it (RoboLab: `robolab-pi05`, `robolab-cosmos3`): one
deploy, the same instance, servers, seed and start states, a new `RUN_ID` per run. Before the deploy it prints the
plan and the instance hours expected from measured runs, and it refuses a plan that would not fit the variant's
deploy limit (`instance.creationpolicy_timeout`):

```
$ python deploy.py --vla robolab-pi05
[runs] 2 runs in this deploy: default for RoboLab (Isaac Lab, GPU physics): an identical re-run does not repeat, so one run cannot call a winner
[deploy] Runs:   2 run(s) x 60 episodes = 120 episodes, ~2.1-2.2 instance-h (scaled from measured 120-episode runs of 2.08-2.25 h; set-up is shared, so this is an upper bound)
```

`--repeats 1` turns it off (with a warning: `results --compare` will then say `needs re-run` for that side);
`--repeats N` (up to 3) asks for more. Targets that run once per deploy (every other template, including
`openarm-isaac`) refuse `--repeats 2` and are re-run by deploying again. Both runs land in the same results bucket,
so read them with `results --compare A B --all-runs`. The protocol card of each run records `repeats`,
`repeat_index` (`1/2`, `2/2`) and the reason. A repeat that fails after the first run is reported in the completion
e-mail and does not fail the deploy. The name is `--repeats`, not `num_runs`: RoboLab already uses `num_runs` for
sequential episodes per task inside one run.

Why the default is two (measured with this tool, RoboLab "specific" instructions, 12 tasks x 10 episodes,
pi0.5-DROID vs Cosmos3, `tests/fixtures/robolab_cs_two_runs.*`): the first run per policy said Cosmos3 better,
51 vs 27 of 120 (Fisher p = 0.001). The second runs gave 48 and 39 (28 of 120 pi0.5 episodes changed), and
`results --compare` on both runs prints `unstable` (run pairs: B>A 2, tie 2).

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

## LIBERO-Plus robustness

A `--libero-variant plus` run ([targets](targets.md#libero-plus-robustness-sweep)) writes one task per perturbation dimension with `condition: LIBERO-Plus/<dimension>`, plus `benchmark: libero_plus`, `perturbation`, `perturbation_category`, `variants` and the fork / asset pins (`stack_libero_plus_commit`, `stack_libero_plus_assets`) in `summary.txt`, and the evaluated variant ids in `task-N/libero_plus_variants.json`. `vlasim results` shows each dimension as a row. `--robustness` puts the latest sweep of each stack next to the latest LIBERO run of the same stack and suite with the default condition (`S0`) and prints the drop in percentage points:

```text
$ python vlasim.py results --robustness --vla smolvla
stack         region     suite           perturbation  run  k_n  rate  ci95  baseline  drop_pp
smolvla-demo  us-east-1  libero_spatial  camera        ...
...
smolvla-demo  us-east-1  libero_spatial  all           ...
```

With 10 variants per dimension the 95% interval of one dimension is roughly ±0.3 wide; use `--perturb-n 50` (350 episodes) when the differences between dimensions matter.

### Measured on this harness

| `--vla` | Suite | Camera | Robot | Language | Light | Background | Noise | Layout | All | LIBERO baseline |
|---|---|---|---|---|---|---|---|---|---|---|
| `smolvla` | libero_spatial, 10 variants x 1 ep (2026-10-30, `g6.xlarge`) | 0.20 | 0.00 | 0.10 | 0.70 | 0.50 | 0.00 | 0.20 | 0.24 (17/70) | 0.80 (8/10, 2 tasks x 5 ep, 2026-10-14) |
| `openvla-oft --libero-suite spatial` | libero_spatial, 10 variants x 1 ep (2026-10-30, `g6.xlarge`) | 1.00 | 0.30 | 0.90 | 1.00 | 1.00 | 1.00 | 0.90 | 0.87 (61/70) | 0.92 (46/50, 2026-06-01) |
| `pi` (pi0.5) | libero_spatial, 10 variants x 1 ep (2026-10-31) | 0.70 | 0.80 | 0.80 | 0.90 | 1.00 | 1.00 | 1.00 | 0.89 (62/70) | LIBERO-Spatial rate-only row |
| `rldx` | libero_spatial, 10 variants x 1 ep (2026-10-31) | 0.80 | 1.00 | 0.90 | 1.00 | 0.90 | 0.90 | 1.00 | 0.93 (65/70) | LIBERO-Spatial rate-only row |
| `molmoact2` | libero_goal, 10 variants x 1 ep (2026-10-31) | 1.00 | 0.80 | 0.80 | 0.90 | 1.00 | 0.90 | 0.60 | 0.86 (60/70) | LIBERO-Goal smoke row (10/10) |
| `vla-adapter` | libero_spatial, 10 variants x 1 ep (2026-10-31) | 0.70 | 0.30 | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 | 0.83 (58/70) | 1.00 (10/10, smoke row) |
| `gr00t` | libero_spatial, 10 variants x 1 ep (2026-10-31) | 0.60 | 0.40 | 0.80 | 0.80 | 1.00 | 1.00 | 0.90 | 0.79 (55/70) | none on Spatial |
| `xvla` | libero_spatial, 10 variants x 1 ep (2026-10-31) | 0.00 | 0.60 | 0.80 | 1.00 | 0.90 | 0.90 | 0.80 | 0.71 (50/70) | LIBERO-Spatial smoke row |
| `lap` | libero_spatial, 10 variants x 1 ep (2026-10-31) | 0.50 | 0.70 | 0.50 | 0.80 | 0.90 | 0.60 | 0.90 | 0.70 (49/70) | LIBERO-Spatial rate-only row |
| `fastwam` | libero_spatial, 10 variants x 1 ep (2026-10-31) | 0.10 | 0.40 | 0.70 | 0.90 | 0.90 | 0.40 | 0.50 | 0.56 (39/70) | LIBERO-Spatial row |
| `lingbot-va` | libero_10, 2 variants x 1 ep (2026-10-31, ~4.7 h: the model generates video with its actions) | 0.50 | 1.00 | 1.00 | 0.50 | 1.00 | 0.00 | 1.00 | 0.71 (10/14) | 1.00 (6/6, 2 tasks x 3 ep, 2026-10-14) |

All sweeps ran in ap-northeast-2 (2026-10-30 and 2026-10-31) with the default plan (seeded draw of 10 variants per dimension, seed 7, so every LIBERO-Spatial row saw the same 70 variants), each on the first instance type of its target's list. The baseline column points to the LIBERO table below where a same-stack run exists; it is not a 70-episode run on the same states. With 10 variants per dimension, a single cell is a coarse signal; the pattern is what to read: OpenVLA-OFT keeps its LIBERO-Spatial skill under every dimension except the robot initial state, while SmolVLA drops on almost all of them. Across the ten targets, background and light change little; camera viewpoint and robot initial state separate the policies most (X-VLA 0/10 and FastWAM 1/10 on camera; OpenVLA-OFT and VLA-Adapter 3/10 on robot), which matches the shape of the paper's leaderboard below. The `lingbot-va` row is 2 episodes per dimension (one cell is 0, 0.5 or 1), so read only its total. For `rldx`, RLDX-1 publishes 84.3% total over all variants of the four suites with the same checkpoint ([RLDX-1 `run_scripts/eval/libero_plus/README.md`](https://github.com/RLWRLD/RLDX-1/blob/ecbfaf80cd031dcc892186ed30465de3591047e6/run_scripts/eval/libero_plus/README.md)); the 0.93 here is the LIBERO-Spatial 70-episode default, not that protocol. The sensor-noise dimension is CPU-bound on the LeRobot target (the fork recomputes its fog map in Python on every frame at 360 px): about 1.8 s per step, ~80 min for 10 episodes, against ~3 min for the other dimensions.

`cosmos-policy` and `aspire` are not wired (reasons in [targets](targets.md#libero-plus-robustness-sweep)).

### Reported in the LIBERO-Plus paper

Success rate (%) per dimension from the leaderboard of the [LIBERO-Plus README](https://github.com/sylvestf/LIBERO-plus/blob/4976dc30028e805ff8094b55501d532c48fec182/README.md) (pinned commit; Fei et al., [arXiv 2510.13626](https://arxiv.org/abs/2510.13626)). The paper protocol evaluates every variant of all four suites (10,030 episodes, 1 each) and "Total" is over all of them, so these numbers are a reference for the shape of the drop, not a same-stack baseline for the 70-episode LIBERO-Spatial default here.

| Model | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| π0 | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π0-FAST | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |

SmolVLA is not in the paper. The `lerobot/smolvla_libero_plus` card (a SmolVLA fine-tuned on LIBERO-Plus data, not the `smolvla` checkpoint) reports 69.0% (400 episodes, task ids 0-9 of four suites); those ids are all Background Textures variants.

---

## LIBERO-R2S (reconstructed objects)

A `--libero-objects real2sim` run ([targets](targets.md#libero-r2s-reconstructed-objects-in-libero)) writes one task per condition with `condition: LIBERO-R2S/<public|collision|full>`, so `vlasim results` shows one row per condition and `--matrix` puts them in the columns `LIBERO-R2S/<condition> (<Suite>)`. `public` is the stock scene evaluated in the same run with the same task ids, initial states and seed, so the drop from `public` to `collision` / `full` is the effect of the objects alone. `<run>/libero-r2s/summary.txt` has the comparison in one place:

```text
suite: libero_object
model: pi05_libero (openpi)
benchmark: libero_r2s
public: k/n = rate [lo, hi]
collision: k/n = rate [lo, hi] (drop +x.x pp)
full: k/n = rate [lo, hi] (drop +x.x pp)
per_object:
  pick up the alphabet soup and place it in the basket: public k/n; collision k/n; full k/n
  ...
objects (cluster, scale, extent ratio vs stock):
  ...
  kept stock (unstable in the physics check): milk
```

With 10 trials per task and 4 tasks (40 episodes per condition) the 95% interval of one condition is roughly ±0.1-0.15 wide near a 0.9 success rate; raise `--r2s-trials` when a drop of a few points matters.

### Measured rows (HOPE-Video scene 0 stage-1 run, `libero_object`)

Both runs used the stage-1 run `2026-10-29-0142` (HOPE-Video scene 0), `scale: fit`, `mass: match`, `libero_object` tasks 0, 1, 5, 9 × 10 trials, seed 7, the public fixed initial states, upstream chunk execution; `ap-northeast-2`, 2026-10-31. Brackets are 95% Wilson intervals; the drop is in percentage points vs `public` of the same run.

| `--vla` (run) | `LIBERO-R2S/public` | `LIBERO-R2S/collision` | `LIBERO-R2S/full` |
|---|---|---|---|
| `openvla-oft --libero-suite object` (`2026-10-31-0725`, `moojink/openvla-7b-oft-finetuned-libero-object`, MuJoCo 3.3.1) | 40/40 = 1.00 [0.91-1.00] | 35/40 = 0.88 [0.74-0.95] (-12.5) | 38/40 = 0.95 [0.83-0.99] (-5.0) |
| `pi` (`2026-10-31-0812`, `pi05_libero`, openpi `650c5b0`, MuJoCo 3.2.3) | 39/40 = 0.97 [0.87-1.00] | 38/40 = 0.95 [0.83-0.99] (-2.5) | 29/40 = 0.72 [0.57-0.84] (-25.0) |

Per object (10 trials each, public / collision / full):

| Object of interest (task id) | OpenVLA-OFT | π0.5 |
|---|---|---|
| alphabet soup (0) | 10 / 9 / 10 | 9 / 9 / 9 |
| cream cheese (1) | 10 / 10 / 8 | 10 / 10 / 8 |
| tomato sauce (5) | 10 / 10 / 10 | 10 / 10 / 10 |
| orange juice (9) | 10 / 6 / 10 | 10 / 9 / **2** |

Physics check on the instance: milk kept stock (as in the local check), the recheck passed for every replaced object in the 4 selected scenes (48 object checks per run, worst resting-pose drift vs the stock object 2.6 mm, tilt ≤ 0.4°).

Cost and time (On-Demand EC2 only, `ap-northeast-2` prices pinned in `leaderboard.py`; wall time = RUN_ID to the last uploaded object):

| `--vla` | Instance | Wall time | Episodes | USD/run (est.) | USD/episode (est.) |
|---|---|---:|---:|---:|---:|
| `openvla-oft --libero-suite object` | `g6.xlarge` (US$0.9896/h) | 41 min | 120 | 0.68 | 0.006 |
| `pi` | `g5.xlarge` (US$1.237/h) | 57 min | 120 | 1.17 | 0.010 |

`python vlasim.py results --matrix --region ap-northeast-2` shows these runs in the columns `LIBERO-R2S/collision (Object)`, `LIBERO-R2S/full (Object)` and `LIBERO-R2S/public (Object)` (marked †: 40 episodes, a subset of the suite) with the same cost rows; the generated leaderboard block above is refreshed separately with `--write-docs`.

**Reading the numbers.** The two swaps hit the two policies in opposite places. With the reconstructed collision geometry but the stock textured mesh (`collision`), OpenVLA-OFT loses 12.5 points and π0.5 2.5; almost all of OFT's loss is the orange juice carton (6/10), whose reconstructed hull has extent ratios 1.06 / 0.88 / 1.07 against the stock box; the likely cause is that the policy sees the stock carton and closes the gripper for its shape, while the contact shape is different. When the cameras also see the reconstructed hull in a single mean 3DGS colour (`full`), OFT recovers to 38/40 (its picture and its contact shape agree again), while π0.5 drops 25 points, almost entirely on orange juice (2/10): the untextured carton no longer looks like the object named in the instruction. Cream cheese loses 2 of 10 in `full` for both policies; tomato sauce is unaffected everywhere. With 40 episodes per condition the OFT intervals overlap for every pair, so OFT's drops are indicative only; π0.5's `full` interval [0.57-0.84] does not overlap its `public` interval [0.87-1.00]. A textured reconstruction (baking the splat colours instead of one mean colour) is the obvious next condition to separate appearance from shape.

Reference for the `public` column: the same-stack baseline below (LIBERO-Object, 10 tasks × 5 trials) is 50/50 for OpenVLA-OFT and 48/50 for π0.5; `public` here covers 4 of those 10 tasks. Milk (task 7) is kept stock by the physics check on this capture (the reconstructed carton topples), so it has no swapped rows.

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
| `gr00t-bridge` | simpler_bridge: widowx_spoon_on_towel / carrot_on_plate / stack_cube / put_eggplant_in_basket (4 tasks × 10 ep) | 78.0% / 58.0% / 48.0% / 53.0% (4-task mean 59.3%; 7-task Bridge average 62.3%, 100 ep per task) — [Isaac-GR00T examples/SimplerEnv README](https://github.com/NVIDIA/Isaac-GR00T/blob/65cc4a192e6d084650d97747308b6a8deb790722/examples/SimplerEnv/README.md) | measured 2026-10-30 (`g6.2xlarge`, ap-northeast-2, 10 ep per task): 80% / 50% / 50% / 40% = 4-task mean 55.0% (22/40) |
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
| `rldx-bridge` | simpler_bridge: the same 4 WidowX tasks (4 tasks × 10 ep) | 71.9% SIMPLER WidowX average ([RLDX-1 README](https://github.com/RLWRLD/RLDX-1) and model card; per-task numbers not published) | measured 2026-10-30 (`g6.2xlarge`, ap-northeast-2, 10 ep per task): 80% / 80% / 80% / 20% = 4-task mean 65.0% (26/40) |
| `rldx-gr1` | gr1_unified/PnPCupToDrawerClose (GR-1 bimanual + waist) | 58.7% 24-task avg (paper README) | validated 2026-06-25 — 0.80 (4/5 ep, eager, g5.2xlarge A10G 24GB, full `deploy.py` 1-shot smoke) |
| `rldx-gr1` | gr1_unified/PnPMilkToMicrowaveClose (GR-1 bimanual + waist) | 58.7% 24-task avg (paper README) | validated 2026-06-25 — 0.20 (1/5 ep, eager, g5.2xlarge A10G 24GB, full `deploy.py` 1-shot smoke) |
| `rldx-kitchen` | robocasa_panda_omron/OpenSingleDoor (PandaOmron mobile manip) | 70.6% 24-task avg (paper README) | 0.80 (4/5), validated 2026-06-26 (g6.2xlarge L4) |
| `rldx-kitchen` | robocasa_panda_omron/OpenDrawer (PandaOmron mobile manip) | 70.6% 24-task avg (paper README) | 0.80 (4/5), validated 2026-06-26 (g6.2xlarge L4) |
| `smolvla-robotwin` | RoboTwin2/clean, 5 tasks × 5 ep | — (not stated on the model card) | measured 2026-10-30 (`g6.2xlarge`, ap-northeast-2): **48.0% (12/25)**, CI 30-67%; beat_block_hammer 4/5, click_bell 5/5, handover_block 0/5, lift_pot 2/5, stamp_seal 1/5 |
| `lingbot-vla2` | RoboTwin2/clean, 5 tasks × 5 ep | 93.52% clean / 92.80% randomized, 50-task avg (upstream README) | measured 2026-10-30 (`g6e.2xlarge`, ap-northeast-2): **80.0% (20/25)**, CI 61-91%; beat_block_hammer 3/5, click_bell 5/5, handover_block 5/5, lift_pot 3/5, stamp_seal 4/5 |

### RoboLab (`robolab-cosmos3`, `robolab-pi05`)

Reference numbers are from the public [RoboLab leaderboard](https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html) (`static/data/leaderboard.json` v76, generated 2026-10-23): RoboLab-120, 10 episodes per task, default instructions, success rate with a 95% Clopper-Pearson interval, Isaac Lab 2.2.0 / Isaac Sim 5.0.0. The measured column is filled by the first AWS run (STEP B). The default 12-task subset is **not** comparable to these rows; only the `edge-full` / `full` variants run the same protocol.

| `--vla` (variant) | Protocol | Leaderboard (RoboLab-120) | Measured on AWS |
|---|---|---|---|
| `robolab-cosmos3` (`edge`, `edge-full`) | subset12 / RoboLab-120 | Cosmos3-Edge-Policy 22.9% (275/1200, CI 20.6-25.4; L40, 9.2 GB; RoboLab `454152dd`, 2026-07-11) | subset12, `edge`: 31.7% (19/60), 2026-10-30, `g6e.2xlarge`, RoboLab `9cce00ca` |
| `robolab-cosmos3` (`nano`) | subset12 | Cosmos3-Nano-Policy 36.8% (441/1200, CI 34.0-39.5; L40, 40 GB; RoboLab `88976795`, 2026-05-22) | not run yet |
| `robolab-pi05` (`subset`, `full`) | subset12 / RoboLab-120 | π0.5 28.0% (336/1200, CI 25.5-30.6; L40S; RoboLab `09545754`, 2026-05-09) | subset12: 40.0% (24/60), 2026-10-30, `g6e.2xlarge`, RoboLab `9cce00ca` |

The 12-task subset is not the leaderboard protocol, so read the two measured rows against each other rather than against RoboLab-120: on the same subset, instance and RoboLab pin, π0.5-DROID (40.0%) leads Cosmos3-Edge-Policy (31.7%), the same order as on the leaderboard (28.0% vs 22.9%). With 60 episodes each the two 95% intervals overlap.

The leaderboard rows were produced with RoboLab v0.1.0-v0.2.0 builds; these targets pin `9cce00ca` (v0.3.2), and RoboLab notes that contact dynamics differ across simulator versions, so expect small shifts even on the full protocol.

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
| `lingbot-vla2` | clean | 93.52% | [Robbyant/lingbot-vla-v2 README](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/README.md) (Muon post-training on clean + randomized data, upstream RoboTwin launcher, FP32), quoted in LeRobot `docs/source/lingbot_vla_v2.mdx` | 5-task clean subset only: 80.0% (20/25, 2026-10-30, run 2026-10-30-1728); not the 50-task protocol |
| `lingbot-vla2` | randomized | 92.80% | same | pending |
| `smolvla-robotwin` | clean / randomized | none published (model card has no evaluation) | [`lerobot/smolvla_robotwin`](https://huggingface.co/lerobot/smolvla_robotwin) @eb55d3a7 | 5-task clean subset only: 48.0% (12/25, 2026-10-30, run 2026-10-30-1728) |

For context, the same upstream table lists π0.5 at 82.74% (clean) / 76.76% (randomized) and LingBot-VLA 1.0 at 88.56% / 86.68%. The published numbers use RoboTwin's own evaluation script, which keeps only seeds the scripted expert solves; the LeRobot wrapper used here does not filter seeds, and the default run is a 5-task subset, so expect lower and noisier rates (5 tasks × 5 episodes = 25 episodes, a 95% interval about ±0.2 wide).

### RL post-training (`rlinf-oft-libero`)

One run writes two rows: `task-0` = the SFT checkpoint (condition `baseline`) and `task-1` = the RL checkpoint (condition `rl-grpo-<N>steps`), evaluated with the same RLinf eval config, the same fixed init states and the same overrides. `results.json` at the run root has both rows, the difference in points and whether the two 95% Wilson intervals overlap; `rl/training_curve.csv` has reward and training success per GRPO step. `vlasim results --matrix` puts `rlinf-oft-libero` and `rlinf-oft-libero [rl-grpo-<N>steps]` in the LIBERO-Goal column.

Published by RLinf (model cards `RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-*`, e.g. [`-goal` @222f7df4](https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-goal); 500 episodes per suite, OpenVLA-OFT from the one-demo SFT checkpoints `Haozhan72/Openvla-oft-SFT-libero-*-traj1`):

| Suite | SFT (one demo per task) | RLinf GRPO | SimpleVLA-RL, same SFT checkpoints ([arXiv 2509.09674](https://arxiv.org/abs/2509.09674)) | Measured on AWS |
|---|---|---|---|---|
| LIBERO-Goal (`model.suite: goal`, default) | 49.40% | 93.96% | 59.6% -> 98.8% | `smoke`: 45.0% (9/20) -> 50.0% (10/20) after 5 LoRA steps, intervals overlap (2026-10-31, `g6e.8xlarge`, ap-northeast-2); `full`: running |
| LIBERO-Spatial | 52.22% | 94.76% | 63.6% -> 98.2% | not wired (no OFT eval config upstream at v0.4) |
| LIBERO-Object (`model.suite: object`) | 28.83% | 97.68% | 54.9% -> 98.7% | pending |
| LIBERO-10 (`model.suite: "10"`) | 14.92% | 90.93% | 17.3% -> 91.7% | pending |

Measured `smoke` run (`2026-10-31-1959`, ap-northeast-2, `g6e.8xlarge`, 1 × L40S 48 GB, 20 eval envs): SFT baseline 45.0% (9/20, 95% CI 25.8-65.8%), RL checkpoint after 5 GRPO steps 50.0% (10/20, 29.9-70.1%), +5.0 points with overlapping intervals, as expected for 5 steps. Training success per step (16 trajectories): 0.44, 0.38, 0.88, 0.88, 0.69. One GRPO step took 355-521 s (median 358 s); the whole deploy (setup, two evals, training, upload of the 31 GB checkpoint) took 56 min (`run_wall_s` 3366), about US$5.2 of EC2.

Read these with three caveats. (1) The card's SFT row is greedy decoding; the RL row and this target's evals sample at temperature 1.6 (the upstream eval config), so the baseline measured here is not expected to equal 49.40%. (2) The two projects measure the same SFT checkpoints 5-26 points apart (MuJoCo version and eval protocol; an RLinf user measured 25.78% on Object); compare the two rows of one run, not a run with a card. (3) The published RL numbers come from the full recipe trained to convergence; the `smoke` profile (5 LoRA steps on one GPU) checks the pipeline and is not expected to show a gain, and how many `full` steps reach the published number is not published. RLinf also reports π0.5 few-shot SFT -> PPO on LIBERO-Goal at 84.6% -> 98.8% (not wired here).

---
