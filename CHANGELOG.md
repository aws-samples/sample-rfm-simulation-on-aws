# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- SimplerEnv WidowX/Bridge targets `gr00t-bridge` and `rldx-bridge`: the four standard Bridge visual-matching tasks (spoon on towel, carrot on plate, stack green block on yellow, eggplant in basket; 10 episodes each by default), results labelled `suite: simpler_bridge` for `vlasim results`, videos in the gallery. `gr00t-bridge` runs `nvidia/GR00T-N1.7-SimplerEnv-Bridge` @940134b3 (NVIDIA Open Model License) on Isaac-GR00T @65cc4a19 with the upstream Bridge recipe (`SIMPLER_ENV_WIDOWX`, 4 actions per chunk, 300 steps; reported 62.3% Bridge average over 7 tasks); `templates/gr00t-userdata.sh.j2` is now branched by `sim_id` (Vulkan loader, `setup_SimplerEnv.sh`, missing rollout-client packages, a rollout import + `gym.make` check, checkpoint download without the 28.8 GB of optimizer states) and still renders `gr00t` byte-identically. `rldx-bridge` runs `RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX` @e1314689 (RLWRLD Model License, **non-commercial**; reported 71.9% WidowX average) through the shared rldx template with a WidowX branch of the SimplerEnv observation/action adapter (`model.simpler_robot: widowx`); `rldx-simpler` renders byte-identically. Both on `g6.2xlarge` first, local mode only; rendered and synthesized, not yet run on a GPU. See [docs/targets.md](docs/targets.md#note-13-simplerenv-bridge-gr00t-bridge-rldx-bridge).
- LIBERO-Plus robustness sweep: `deploy.py --libero-variant plus [--perturb camera,light,... ] [--perturb-n N]` runs an existing LIBERO target on [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) (arXiv 2510.13626, fork pinned @4976dc30, asset pack `Sylvest/LIBERO-plus` @dd2bd61b, 6.4 GB, sha256-checked) instead of its LIBERO tasks: one task per perturbation dimension (camera, robot, language, light, background, noise, layout) with condition `LIBERO-Plus/<dimension>`, N variants per dimension picked on the instance from the fork's `task_classification.json` with a fixed seed (`assets/libero-plus/select_variants.py`, ids saved as `libero_plus_variants.json`), 1 episode per variant as in the paper. Available for `smolvla` (LeRobot `--env.type=libero_plus`) and `openvla-oft` (the fork replaces LIBERO in the conda env and `run_libero_eval.py`'s task loop is restricted to the selected ids); settings in the `libero_plus` block of `models/<id>.yaml`, which also raises the creation timeout and EBS size (CDK context `libero_variant=plus`). Same stack and results bucket as the LIBERO run. `vlasim results --robustness` prints per dimension k/n, rate, Wilson interval, the latest LIBERO baseline of the stack and the drop in points. Off by default: every rendered UserData and synthesized template is unchanged without the flag. The fork's code declares no license (research / evaluation use); nothing is vendored. See [docs/targets.md](docs/targets.md#libero-plus-robustness-sweep).
- Targets `robolab-cosmos3` and `robolab-pi05`: [NVIDIA RoboLab](https://github.com/NVlabs/RoboLab) (`@9cce00ca`, Isaac Lab 2.2.0 / Isaac Sim 5.0 NGC image pinned by digest, DROID Franka) with its policy server on the same instance — NVIDIA cosmos-framework `action_policy_server_robolab` serving `nvidia/Cosmos3-Edge-Policy-DROID` (default) or `-Nano-` (OpenMDW-1.1), and openpi `pi05_droid_jointpos` (the checkpoint RoboLab evaluated). One shared template `templates/robolab-userdata.sh.j2`: ECC-off reboot, RoboLab Git LFS assets, RoboLab's own Docker image, one server + headless Isaac Sim client pair per GPU ("shards"), round-robin task split. `model.variant` selects checkpoint, protocol and instance list (the CDK stack and `vlasim doctor` read the variant's `instance:`); defaults run a 12-task RoboLab-120 subset × 5 episodes on `g6e.2xlarge`, `edge-full` / `full` run the leaderboard protocol (120 tasks × 10) on `g6e.48xlarge`. `assets/robolab/robolab_report.py` writes `summary.txt` (suite `RoboLab/<subset>`), `robolab_results.json` (per task / difficulty, Wilson intervals, subtask score) and gallery videos with `success=` names. RoboLab's bundled assets are mostly **CC BY-NC-SA 4.0 (non-commercial)**. Leaderboard references in [docs/results.md](docs/results.md#robolab-robolab-cosmos3-robolab-pi05); not run on AWS yet. See [docs/targets.md](docs/targets.md#note-14-robolab).
- Targets `smolvla-robotwin` (SmolVLA-450M, `lerobot/smolvla_robotwin`) and `lingbot-vla2` (LingBot-VLA 2.0, `lerobot/lingbot_vla_v2_robotwin`, Apache-2.0) on **RoboTwin 2.0** (50 dual-arm tasks, SAPIEN, Aloha-AgileX) through upstream LeRobot `lerobot-eval --env.type=robotwin` (new shared template `lerobot-robotwin-userdata.sh.j2`; LeRobot @552777b7, RoboTwin @0aeea2d6, assets @981c92aa, CuRobo v0.7.8 built on the instance, Vulkan loader for SAPIEN). Default run: 5 tasks × 5 episodes, clean setting, official per-task step limits; `--robotwin-protocol clean|randomized|full` + `--robotwin-shard K/N` run the official 49 tasks × 100 episodes in 12-hour shards. `vlasim results` labels the rows `RoboTwin2/<setting>`. Checks before the checkpoint download: imports of the whole eval chain and one env reset on the GPU, then a 1-episode smoke eval. RoboTwin builds NVIDIA CuRobo, which is **non-commercial** (research / evaluation only). Not run on AWS yet. See [docs/targets.md](docs/targets.md#note-15-robotwin-20-smolvla-robotwin-lingbot-vla2). Measured on 2026-10-30 (5-task clean subset × 5 episodes, ap-northeast-2): `smolvla-robotwin` 48.0% (12/25), `lingbot-vla2` 80.0% (20/25).

- Target `real2sim` (Real2Sim, stage 1): a multi-view tabletop capture becomes a 3D Gaussian Splatting model (nerfstudio 1.1.5 splatfacto, gsplat 1.4.0), is cut into a table plane + per-object collision meshes (RANSAC + DBSCAN + convex hull, UsdPhysics `.usda`), and one scripted top-down grasp per object is replayed in MuJoCo 3.16 (MuJoCo Menagerie UR5e + Robotiq 2F-85, pinned) against the reconstructed objects, a single scene hull and the ground truth. Default input: HOPE-Video scene 0, downloaded at run time and MD5-checked (**CC BY-NC-SA 4.0, non-commercial**); your own capture via `input.capture_s3_uri` in `models/real2sim.yaml` (the stack grants read on that one object). Results: PSNR / SSIM / LPIPS, fidelity vs ground truth, grasp success per world in `summary.txt`, videos in the gallery. `g5.2xlarge` first. Stage scripts in `assets/real2sim/`, tests in `tests/test_real2sim.py` (synthetic scenes; CI installs the geometry deps for them). End-to-end validated 2026-10-29 on `g5.2xlarge` in `ap-northeast-2` (HOPE-Video scene 0, ~12 min instance time): PSNR 23.998 / SSIM 0.855 / LPIPS 0.192, 10 of 10 objects matched (2 over-split), collision volume 1.25 × ground truth, grasps recon / scene hull / ground truth = 8/9, 0/9, 6/9 (floating 2F-85) and 4/9, 0/9, 4/9 (UR5e). The per-object `volume_ratio` in `segment.json` is now filled: the HOPE-Video meshes are welded along their texture seams before the closed-mesh check (it was `null` for every object while the scene-level ratio was set). See [docs/targets.md](docs/targets.md#note-12-real2sim).

- Post-run judge `flora-judge`: `python vlasim.py judge --vla <target> [--run RUN_ID]` (= `deploy.py --judge`) scores every rollout video of a finished run with NVIDIA Flora-Rewarder (`nvidia/Flora-Rewarder` @031299e4, Qwen3.5-27B, outcome score 1-5, success = score >= 4) on a `g7e.2xlarge` (fallbacks `g7e.4xlarge`, `g6e.12xlarge` with tensor parallel 4, `p5.4xlarge`) and writes `judge.json` / `judge.txt` next to the run, then e-mails a summary and terminates. Scoring uses NVIDIA's own `batch_utils.terminal_reward` from the pinned model revision (downloaded at run time). Own CDK app `cdk/bin/judge.ts` (stack `FloraJudge-<target stack>`, Deep Learning Base OSS Nvidia Driver AMI for Blackwell), `destroy.py --judge`. `vlasim results` prints the judge's agreement with the simulator. Flora-Rewarder is under the NVIDIA OneWay Noncommercial License (research only). See [docs/targets.md](docs/targets.md#post-run-judge-flora-rewarder).

- Video history gallery: the completion e-mail links to a password-protected Amazon CloudFront page with every video of every run under the notification e-mail, grouped by target, newest run first (shared `RFM-Gallery` stack: private S3 bucket + OAC, CloudFront Function + KeyValueStore Basic auth; per-target `GalleryPublisher` Lambda). `vlasim.py gallery [--reset-password]` prints the URL and password; `deploy.py --no-gallery` / `deployment.gallery: false` opt out. Videos already in a kept results bucket are added on the next deploy (backfill). The `aws s3 sync` download line stays in the e-mail. See [docs/gallery.md](docs/gallery.md).

### Changed

- Video history: a video whose caption carries a judge score shows a `Flora N/5` badge, and a re-uploaded caption is read again instead of reusing the cached text.
- `destroy.py` asks what to do with the data: `--keep-data` (default, today's behaviour: results bucket and gallery are kept) or `--delete-all` (also deletes the results bucket, the stack's Lambda log groups and this target's gallery videos, and the whole gallery when no other target stack is left in the region). A terminal gets a numbered prompt; without one, destroy keeps the data.
- `requirements.txt` installs `boto3[crt]` (the CloudFront KeyValueStore API needs SigV4A).

### Fixed

- Shared LeRobot x LIBERO template: videos are uploaded as `<suite>_<task_id>--eval_episode_N.mp4`. A task entry with several `task_ids` used to upload only the last task id's videos, because every task id restarts at `eval_episode_0.mp4`.

## [1.0.0] - 2026-10-22

First tagged release. One-click AWS CDK deployment that launches a GPU instance, runs a robot
foundation model evaluation in simulation, writes results to Amazon S3 and notifies by e-mail.
20 `--vla` targets (23 stacks counting the OpenVLA-OFT LIBERO suites) across LIBERO, SimplerEnv,
RoboCasa and Isaac Lab, driven by the `vlasim.py` CLI (`doctor`, `init`, `deploy`, `results`, `destroy`).

### Added

- Initial VLA Simulator on AWS: 1-click VLA evaluation in Isaac Lab / LIBERO, with GR00T and pi0.5 targets.
- OpenVLA-OFT LIBERO suite selection and LAP-3B bridge mode.
- OpenArm Isaac Lab targets, including the unimanual Lift-Cube ACT demo-collection target.
- GR00T Unitree G1 whole-body loco-manipulation (`gr00t-g1`), its trajectory-collection mode and the optional GR00T N1.7 adapter rollout server.
- RLDX-1 targets: LIBERO-10 (`rldx`), SimplerEnv Google Fractal (`rldx-simpler`), RoboCasa GR-1 Tabletop (`rldx-gr1`) and RoboCasa Kitchen 24-task (`rldx-kitchen`).
- MolmoAct2 (Ai2) LIBERO target.
- `vlasim.py` CLI with `doctor`, `init`, `deploy` and `destroy` subcommands.
- Shared LeRobot x LIBERO template with X-VLA, SmolVLA and LingBot-VA targets (#29).
- VLA-Adapter-Pro (`vla-adapter`) target on LIBERO (#31).
- FastWAM (`fastwam`) and Cosmos Policy (`cosmos-policy`) world action model targets on LIBERO (#33, #35).
- ASPIRE (`aspire`) agentic skill-discovery target on the LIBERO-Pro fork (#36).
- Per-task `condition` / `extra_args` and a stack fingerprint in `summary.txt` (#37).
- Stack knobs for OpenVLA-OFT, pi and GR00T (#38).
- `vlasim results` collector with 95% Wilson intervals and a 50-episode same-stack baseline (#39).
- CI on every pull request, grouped Dependabot updates and Dependabot auto-merge (#20).
- CI coverage of all 20 targets (render, `bash -n`, `cdk synth`) and a target-list parity test (#43).
- Community files: issue forms, pull request template, CODEOWNERS, SECURITY.md, CITATION.cff and this changelog.

### Changed

- Renamed the repository from `sample-vla-simulator-on-aws` to `sample-rfm-simulation-on-aws`; internal identifiers (`vlasim.py`, `/vla-simulator/*` SSM keys, stack file names) are unchanged (#40).
- Public-repo hygiene: removed private references, neutral wording, license notes and English-only strings (#41).
- Split the README into `docs/` pages and rewrote the README (#44).
- Lint and test tooling: ruff, pre-commit and pytest configuration (#45).
- Moved cdk-nag suppressions to `Validations.of().acknowledge()` for cdk-nag 3 (#30).
- Dependency updates for the CDK app and GitHub Actions (aws-cdk-lib, TypeScript 6, js-yaml 5, actions v7, and others).

### Security

- Least-privilege IAM: the AZ-selector Lambda role is scoped to its own describe/run calls, `iam:PassRole` is limited to the instance role and `ec2.amazonaws.com`, and unused instance grants (`s3:PutObjectAcl`, `cloudformation:SignalResource`) are removed (#47).
- IMDSv2 is required on every instance (`HttpTokens=required`, hop limit 2 for containers) (#47).

### Fixed

- Pinned MuJoCo below 3.10 and `tensorflow-metadata` for GR00T, OpenVLA-OFT and RLDX LIBERO targets; read the eval success rate from the plain-text log.
- Bounded stalled Hugging Face Xet checkpoint downloads so a dead CDN connection cannot hold a GPU until the CreationPolicy expires (#8).
- `vlasim` bootstrap on a fresh checkout (#11).
- Issues found by a full 13-target deploy (#17).
- apt boot hardening on the DLAMI (#37).
- pi boot script now retries `apt-get` on lock contention like the other targets (#47).

[Unreleased]: https://github.com/aws-samples/sample-rfm-simulation-on-aws/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/aws-samples/sample-rfm-simulation-on-aws/releases/tag/v1.0.0
