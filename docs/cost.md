# Cost estimate

Back to the [README](../README.md) · [Documentation index](README.md)

Prices are **us-east-1 Linux On-Demand, checked 2026-10-22** with the AWS Price List API (`aws pricing get-products --region us-east-1 --service-code AmazonEC2`, shared tenancy, no pre-installed software). Other regions differ; check the [Amazon EC2 On-Demand pricing page](https://aws.amazon.com/ec2/pricing/on-demand/) for the region you deploy to. `deploy.py` launches On-Demand instances only; there is no Spot option.

## Hourly price per instance type

| Instance | GPU | vCPU | Memory | On-Demand USD/h |
|---|---|---|---|---:|
| `g6.xlarge` | 1 × L4 24 GB | 4 | 16 GiB | 0.8048 |
| `g6.2xlarge` | 1 × L4 24 GB | 8 | 32 GiB | 0.9776 |
| `g6.4xlarge` | 1 × L4 24 GB | 16 | 64 GiB | 1.3232 |
| `g6.12xlarge` | 4 × L4 24 GB | 48 | 192 GiB | 4.6016 |
| `g5.xlarge` | 1 × A10G 24 GB | 4 | 16 GiB | 1.006 |
| `g5.2xlarge` | 1 × A10G 24 GB | 8 | 32 GiB | 1.212 |
| `g5.4xlarge` | 1 × A10G 24 GB | 16 | 64 GiB | 1.624 |
| `g5.12xlarge` | 4 × A10G 24 GB | 48 | 192 GiB | 5.672 |
| `g6e.xlarge` | 1 × L40S 48 GB | 4 | 32 GiB | 1.861 |
| `g6e.2xlarge` | 1 × L40S 48 GB | 8 | 64 GiB | 2.24208 |
| `g6e.4xlarge` | 1 × L40S 48 GB | 16 | 128 GiB | 3.00424 |
| `g6e.12xlarge` | 4 × L40S 48 GB | 48 | 384 GiB | 10.49264 |
| `g6e.24xlarge` | 4 × L40S 48 GB | 96 | 768 GiB | 15.06559 |
| `g6e.48xlarge` | 8 × L40S 48 GB | 192 | 1536 GiB | 30.13118 |
| `g7e.2xlarge` | 1 × RTX PRO 6000 Blackwell 96 GB | 8 | 64 GiB | 3.36312 |
| `g7e.4xlarge` | 1 × RTX PRO 6000 Blackwell 96 GB | 16 | 128 GiB | 3.99816 |
| `p5.4xlarge` | 1 × H100 80 GB | 16 | 256 GiB | 6.88 |
| `g6e.8xlarge` | 1 × L40S 48 GB | 32 | 256 GiB | 4.52856 |
| `g6e.16xlarge` | 1 × L40S 48 GB | 64 | 512 GiB | 7.57719 |
| `p4d.24xlarge` | 8 × A100 40 GB | 96 | 1152 GiB | 21.957642 |
| `p4de.24xlarge` | 8 × A100 80 GB | 96 | 1152 GiB | 27.44705 |
| `p5.48xlarge` | 8 × H100 80 GB | 192 | 2048 GiB | 55.04 |
| `p5en.48xlarge` | 8 × H200 141 GB | 192 | 2048 GiB | 63.296 |

The `g6e.12xlarge`, `g7e` and `p5.4xlarge` rows were checked on 2026-10-28 (same API); they are used by the post-run judge and the RoboLab `nano` variant. `g6e.24xlarge` / `g6e.48xlarge` were checked on 2026-10-30 (RoboLab variants). In `ap-northeast-2`, `g7e.2xlarge` is US$4.13478, `g6e.2xlarge` US$2.75652, `g6e.12xlarge` US$12.90014 and `g6e.48xlarge` US$37.04468 per hour.

## Per-target estimate

Instance = the first entry of `instance.preferred` (the AZ selector may fall back to a later, differently priced type; see [targets.md](targets.md#gpu-instances-per-target)). Runtime = the typical end-to-end time noted for the deploy command in [targets.md](targets.md#deploy-commands). Cost = hourly price × runtime, EC2 only, rounded.

| `--vla` | Instance | Typical runtime | Approx. EC2 cost (USD) |
|---|---|---|---:|
| `gr00t` | `g6.12xlarge` | ~120 min | ~9.20 |
| `gr00t-gr1` | `g6.12xlarge` | ~90 min | ~6.90 |
| `gr00t-g1` | `g6.12xlarge` | ~20-30 min | ~1.53-2.30 |
| `pi` | `g5.xlarge` | ~4 h | ~4.02 |
| `openvla-oft` (suite `10`, default) | `g6.xlarge` | ~3 h | ~2.41 |
| `openvla-oft --libero-suite spatial` | `g6.xlarge` | ~1.5 h | ~1.21 |
| `lap` | `g6.xlarge` | ~1.5-2.5 h | ~1.21-2.01 |
| `molmoact2` | `g6e.2xlarge` | ~40-60 min | ~1.49-2.24 (validated in us-west-2) |
| `gr00t-bridge` | `g6.2xlarge` | ~90-150 min (estimate, not yet run) | ~1.47-2.44 |
| `xvla` | `g6.2xlarge` | ~15-30 min | ~0.24-0.49 |
| `smolvla` | `g6.xlarge` | ~15-30 min | ~0.20-0.40 |
| `lingbot-va` | `g6e.2xlarge` | ~2.5 h (6 episodes) | ~5.61 |
| `vla-adapter` | `g6.xlarge` | not recorded | 0.8048 per hour |
| `fastwam` | `g6e.2xlarge` | ~30-40 min | ~1.12-1.49 |
| `cosmos-policy` | `g6e.xlarge` | ~60-90 min | ~1.86-2.79 |
| `aspire` | `g6e.2xlarge` | ~45 min | ~1.68, plus ~3 Amazon Bedrock (agent, 2 tasks) |
| `real2sim` | `g5.2xlarge` | ~12 min measured (2026-10-29, `ap-northeast-2`, HOPE-Video scene 0: instance boot to done 11 min 53 s, of which packages, both venvs and the HOPE-Video download ~5 min, splatfacto training 272 s; stack create / delete not included) | ~0.24; ~0.30 in `ap-northeast-2` (US$1.4903 per hour there, Price List API 2026-10-29) |
| `rldx` | `g6.xlarge` | ~80-120 min | ~1.07-1.61 |
| `rldx-simpler` | `g6.2xlarge` | ~35-45 min | ~0.57-0.73 |
| `rldx-bridge` | `g6.2xlarge` | ~100-160 min (estimate, not yet run) | ~1.63-2.61 |
| `rldx-gr1` | `g6.2xlarge` | ~25-35 min | ~0.41-0.57 |
| `rldx-kitchen` | `g6.2xlarge` | ~25-35 min | ~0.41-0.57 |
| `openarm-lift-act` | `g6.12xlarge` | not recorded | 4.6016 per hour |
| `openarm-isaac` | `g6.12xlarge` | not recorded (creation timeout 240 min) | 4.6016 per hour |
| `smolvla --libero-variant plus` (70 episodes) | `g6.xlarge` | measured ~2 h in total (2026-10-30; the sensor-noise dimension alone ~80 min, CPU-bound fog at 360 px) | ~0.94-1.21, plus ~0.29 NAT processing for the 6.4 GB asset pack |
| `openvla-oft --libero-suite spatial --libero-variant plus` (70 episodes) | `g6.xlarge` | measured ~1 h (2026-10-30) | ~1.61-2.01, plus ~0.29 NAT processing for the 6.4 GB asset pack |
| `robolab-cosmos3` (`edge`, default) | `g6e.2xlarge` | measured ~1.9 h (2026-10-30, ap-northeast-2: ~35 min setup incl. a 30 GB server image build, ~75 min eval) | ~9-13 |
| `robolab-cosmos3` (`nano`) | `g6e.12xlarge` | estimate ~5-8 h (2 shards; Nano ~4 s per inference call on the leaderboard) | ~52-84 |
| `robolab-cosmos3` (`edge-full`) / `robolab-pi05` (`full`) | `g6e.48xlarge` | estimate ~6-9 h (8 shards over RoboLab-120 × 10; RoboLab quotes ~40 GPU hours for 120 tasks) | ~180-270 |
| `robolab-pi05` (`subset`, default) | `g6e.2xlarge` | measured ~1.4 h (2026-10-30, ap-northeast-2) | ~9-12 |
| `smolvla-robotwin` (default: 5 tasks × 5 ep, clean) | `g6.2xlarge` | measured ~55 min instance time (2026-10-30, ap-northeast-2: ~13 min setup incl. CuRobo build and 4 GB assets, ~41 min for 25 episodes = ~98 s per episode) | ~1.10 |
| `lingbot-vla2` (default: 5 tasks × 5 ep, clean) | `g6e.2xlarge` | measured ~49 min instance time (2026-10-30, ap-northeast-2: ~23 min setup incl. the 25.5 GB fp32 checkpoint, ~25 min for 25 episodes = ~59 s per episode) | ~2.26 |
| `smolvla-robotwin --robotwin-protocol clean` (49 tasks × 100 ep, 19 shards) | `g6.2xlarge` | estimate up to ~126 h of episode loop + ~1.25 h setup per shard ≈ 150 h (upper bound: every episode at its step limit) | up to ~147 |
| `smolvla-robotwin --robotwin-protocol full` (clean + randomized, 47 shards) | `g6.2xlarge` | estimate up to ~252 h + ~1.5 h per shard ≈ 323 h | up to ~315 |
| `lingbot-vla2 --robotwin-protocol full` (47 shards) | `g6e.2xlarge` | about the same hours as above (the simulator, not the policy, sets the pace) | up to ~723 |
| `rlinf-oft-libero` (`smoke`, default / `--rl-profile full`) | `g6e.8xlarge` / `p5.48xlarge` | estimate ~2-4 h / ~12-28 h (capped at 6 h / 40 h) | ~9-18 / ~660-1,540; see [RL post-training](#rl-post-training-rlinf-oft-libero) |
| `flora-judge` (post-run judge, `vlasim judge`) | `g7e.2xlarge` (measured on the `g6e.12xlarge` fallback, US$12.90/h in `ap-northeast-2`) | 13-15 min instance time measured for 10-47 episodes (+ ~10 min stack create/delete) | ~0.85-1.10 on `g7e.2xlarge`; ~2.80-3.25 measured on `g6e.12xlarge` |

RoboTwin rows: prices checked 2026-10-30 (same API; `g5.4xlarge` and `g6.4xlarge` added). In `ap-northeast-2` the RoboTwin instances are US$1.20208 (`g6.2xlarge`), US$1.4903 (`g5.2xlarge`), US$1.62704 (`g6.4xlarge`), US$1.99691 (`g5.4xlarge`), US$2.75652 (`g6e.2xlarge`) and US$3.69355 (`g6e.4xlarge`) per hour. The 0.15 s per simulated step behind the episode-loop figures of the official-protocol rows is a planning value; `deploy.py` uses it to size the shards. The first subset runs (2026-10-30) took ~59-98 s per episode including policy inference, which is below the step-limit upper bound the shard plan assumes, so the shard counts stay conservative. Measured rows cover the stack instance only (stack create/delete and NAT not included).

## RL post-training (`rlinf-oft-libero`)

Prices: AWS Price List API, Linux On-Demand, shared tenancy, checked **2026-10-31** (the six rows added above). Runtimes are **estimates** except the measured `smoke` row: RLinf publishes no time per GRPO step for this recipe. The `smoke` run (2026-10-31) measured 355-521 s per LoRA step on one L40S with 16 trajectories; the first `full` run measures the 8-GPU step. Assumptions: setup (17.6 GB image, 15 GB SFT checkpoint, verify) ~30-45 min; one eval of 100 episodes ~20-40 min and of 500 episodes ~45-90 min; `full` training ~6-15 min per step on 8 × H100, extrapolated from an RLinf user report (4 × A100 80 GB, 2 rollout epochs: ~90 h projected for 1000 steps) to 16 rollout epochs on twice the GPUs; L40S and A100 assumed ~2-3 × slower than H100.

| Profile | Instance | us-east-1 USD/h | ap-northeast-2 USD/h | Estimated runtime | Approx. EC2 cost us-east-1 / ap-northeast-2 (USD) | At the wall-clock cap |
|---|---|---:|---:|---|---:|---:|
| `smoke` (1 GPU, 5 LoRA steps) | `g6e.8xlarge` | 4.52856 | 5.56762 | **56 min measured** (2026-10-31, ap-northeast-2) | ~4.2 / **~5.2 measured** | 6 h: ~27 / ~33 |
| `smoke` fallback | `g6e.4xlarge` (16 vCPU: 8 envs fit) | 3.00424 | 3.69355 | ~2-4 h | ~6-12 / ~7-15 | 6 h: ~18 / ~22 |
| `full` (8 GPU, 100 steps) | `p5.48xlarge` | 55.04 | 75.9552 | ~12-28 h | ~660-1,540 / ~910-2,130 | 40 h: ~2,200 / ~3,040 |
| `full` fallback | `p4de.24xlarge` | 27.44705 | not offered | likely reaches the cap | ~1,100 / — | 40 h: ~1,100 / — |
| `full` fallback | `p5en.48xlarge` | 63.296 | 87.34848 | ~10-25 h | ~630-1,580 / ~870-2,180 | 40 h: ~2,530 / ~3,490 |

The most one `full` deploy can cost is bounded by the wall-clock cap (`profiles.full.max_wall_hours`, 40 h; the instance terminates itself) plus setup: about US$3,500 on the most expensive fallback in `ap-northeast-2`. `--rl-steps` and `max_wall_hours` scale it linearly; `--rl-eval-run` costs two evals only (~2-3 h).

**Capacity** (EC2 `DescribeInstanceTypeOfferings`, 2026-10-31). `ap-northeast-2`: `g6e.4xlarge` / `8xlarge` / `16xlarge` / `48xlarge` in AZs a and b; `p5.48xlarge` and `p5en.48xlarge` in AZ a only; `p4d.24xlarge` in b and d; `p4de.24xlarge` and `p5.4xlarge` not offered. `us-east-1`: all of them. `p5` / `p5en` On-Demand capacity is often unavailable without an On-Demand Capacity Reservation or EC2 Capacity Blocks for ML, which `deploy.py` does not use; the AZ selector then falls through to the next type and fails if none has capacity (`g6e.48xlarge` was dropped from the list after the smoke runs ran a 44 GB L40S out of memory at 40 eval envs), so plan `full` in `us-east-1`, where `p4de.24xlarge` is offered. P instances count against the separate quota *Running On-Demand P instances* (`vlasim doctor` checks the G and VR quota only); `full` needs 192 vCPU of either. `p4d.24xlarge` (40 GB GPUs) is not in the list: RLinf users report out-of-memory on 40 GB GPUs for smaller configs.

**Not in the table:** EBS 1500 GB gp3 for `full` (~US$0.16 per hour), NAT processing for ~33 GB of downloads plus the checkpoint upload (LoRA for `smoke`; the full-parameter DCP shards with optimizer state for `full`, on the order of 100 GB, about US$5 at US$0.045 per GB), and S3 storage of the saved checkpoints (delete `<RUN_ID>/rl/checkpoints/` when you no longer need them).

## Not included

- **NAT gateway** of the stack VPC (local mode): US$0.045 per hour plus US$0.045 per GB processed in us-east-1. Model and dataset downloads go through it, so a multi-GB checkpoint adds data-processing charges. The NAT gateway is billed until you run `vlasim.py destroy`.
- **EBS** root volume (gp3, US$0.08 per GB-month in us-east-1), sized by `instance.ebs_gb` (120-500 GB per target).
- **S3** storage for videos and logs in the retained results bucket, CloudWatch Logs, SNS and the AZ selector Lambda (usually small).
- **Amazon Bedrock** usage for `aspire` (reported at US$0.91-2.30 per task in the [showcase](showcase.md)).

With `deployment.auto_terminate: true` (the default) the instance terminates after the run; the rest of the stack stays until you destroy it.
