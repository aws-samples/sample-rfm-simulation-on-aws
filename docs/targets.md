# Supported targets

Back to the [README](../README.md) · [Documentation index](README.md)

Each target is one `--vla <target>` value for `python vlasim.py deploy` / `destroy` (and `deploy.py` / `destroy.py`). The table lists the model, simulator, robot and CloudFormation stack name; numbered notes below the table carry the license, pinning and validation detail for each target. Sample clips are in [showcase.md](showcase.md) and measured success rates in [results.md](results.md).

## Target table

| `--vla` | `--libero-suite` | Model | Sim Environment | Robot | Stack |
|---------|------------------|-------|----------------|-------|-------|
| `gr00t` | — | GR00T N1.7-LIBERO | LIBERO-10 kitchen tasks | Franka Panda (7-DOF) | `GR00T-Demo` |
| `gr00t-bridge` | — | GR00T N1.7-SimplerEnv-Bridge[¹³](#note-13-simplerenv-bridge-gr00t-bridge-rldx-bridge) | SimplerEnv Bridge (visual matching): spoon on towel, carrot on plate, stack cube, eggplant in basket | WidowX 250 (`SIMPLER_ENV_WIDOWX`) | `GR00T-Bridge-Demo` |
| `gr00t-gr1` | — | GR00T N1.6-3B | RoboCasa GR1 tabletop tasks | Fourier GR1 humanoid (22-DOF) | `GR00T-GR1-Demo` |
| `gr00t-g1` | — | GR00T N1.6-G1 (community re-finetune) | GR00T-WholeBodyControl loco-manip (GEAR WBC + MuJoCo) | Unitree G1 humanoid (whole-body) | `GR00T-G1-Demo` |
| `pi` | — | π0.5 (pi05_libero) | LIBERO spatial/object | Franka Panda (7-DOF) | `Pi-Demo` |
| `openvla-oft` | `spatial` | OpenVLA-OFT-7B (LIBERO-Spatial fine-tune) | LIBERO-Spatial | Franka Panda (7-DOF) | `OpenVLA-OFT-Spatial-Demo` |
| `openvla-oft` | `object` | OpenVLA-OFT-7B (LIBERO-Object fine-tune) | LIBERO-Object | Franka Panda (7-DOF) | `OpenVLA-OFT-Object-Demo` |
| `openvla-oft` | `goal` | OpenVLA-OFT-7B (LIBERO-Goal fine-tune) | LIBERO-Goal | Franka Panda (7-DOF) | `OpenVLA-OFT-Goal-Demo` |
| `openvla-oft` | `10` (default, alias `long`) | OpenVLA-OFT-7B (LIBERO-10 fine-tune) | LIBERO-10 long-horizon | Franka Panda (7-DOF) | `OpenVLA-OFT-Demo` |
| `lap` | — | LAP-3B (PaliGemma-3B + Flow Matching, JAX) | LIBERO-Spatial | Franka Panda (7-DOF) | `LAP-Demo` |
| `molmoact2` | — | MolmoAct2 (Ai2 Action Reasoning Model, 5B; Molmo2-ER VLM + flow-matching action expert)[⁵](#note-5-molmoact2) | LIBERO-Goal | Franka Panda (7-DOF) | `MolmoAct2-Demo` |
| `xvla` | — | X-VLA-0.9B (2toINF / Tsinghua AIR, soft-prompted cross-embodiment VLA)[⁶](#note-6-shared-lerobot-libero-template-xvla-smolvla-lingbot-va) | LIBERO-Spatial | Franka Panda (7-DOF) | `XVLA-Demo` |
| `smolvla` | — | SmolVLA-450M (Hugging Face)[⁶](#note-6-shared-lerobot-libero-template-xvla-smolvla-lingbot-va) | LIBERO-Spatial | Franka Panda (7-DOF) | `SmolVLA-Demo` |
| `lingbot-va` | — | LingBot-VA (Robbyant world action model: autoregressive video + action, Wan2.2)[⁶](#note-6-shared-lerobot-libero-template-xvla-smolvla-lingbot-va) | LIBERO-10 long-horizon | Franka Panda (7-DOF) | `LingBot-VA-Demo` |
| `vla-adapter` | — | VLA-Adapter-Pro (OpenHelix, Qwen2.5-0.5B Prismatic VLM + bridge-attention action head)[⁷](#note-7-vla-adapter) | LIBERO-Spatial | Franka Panda (7-DOF) | `VLA-Adapter-Demo` |
| `fastwam` | — | FastWAM (world action model: Wan2.2 video DiT + Mixture-of-Transformers action expert)[⁸](#note-8-fastwam) | LIBERO-Spatial | Franka Panda (7-DOF) | `FastWAM-Demo` |
| `cosmos-policy` | — | Cosmos Policy (NVIDIA world action model: Cosmos-Predict2-2B video model fine-tuned to predict actions, future frames and value)[⁹](#note-9-cosmos-policy) | LIBERO-Spatial | Franka Panda (7-DOF) | `Cosmos-Policy-Demo` |
| `aspire` | — | ASPIRE (NVlabs agentic skill discovery: Claude on Amazon Bedrock writes a per-task robot program against SAM3 / Contact-GraspNet / PyRoki tools — no trained policy)[¹⁰](#note-10-aspire) | LIBERO-Spatial (LIBERO-Pro fork) | Franka Panda (7-DOF) | `ASPIRE-Demo` |
| `real2sim` | — | Real2Sim — not a policy: capture → 3D Gaussian Splatting (nerfstudio splatfacto) → per-object collision meshes → scripted grasp replay[¹²](#note-12-real2sim) | MuJoCo, 3 collision worlds (reconstructed / scene hull / ground truth) of a real tabletop (HOPE-Video, **NC** data) or your capture | UR5e + Robotiq 2F-85 (and a floating 2F-85) | `Real2Sim-Demo` |
| `rldx` | — | RLDX-1 (RLWRLD MSAT / Qwen3-VL-8B, eager)[¹](#note-1-rldx-1-weights-license) | LIBERO-10 long-horizon | Franka Panda (7-DOF) | `RLDX-Demo` |
| `rldx-simpler` | — | RLDX-1 (RLWRLD MSAT / Qwen3-VL-8B, eager)[¹](#note-1-rldx-1-weights-license) [²](#note-2-rldx-simpler) | SimplerEnv Google-VM (Fractal) | Google Robot (OXE_FRACTAL real-robot embodiment) | `RLDX-Simpler-Demo` |
| `rldx-bridge` | — | RLDX-1 (RLWRLD MSAT / Qwen3-VL-8B, eager)[¹](#note-1-rldx-1-weights-license) [¹³](#note-13-simplerenv-bridge-gr00t-bridge-rldx-bridge) | SimplerEnv Bridge (visual matching), same 4 tasks | WidowX 250 (`OXE_BRIDGE_ORIG`) | `RLDX-Bridge-Demo` |
| `rldx-gr1` | — | RLDX-1 (RLWRLD MSAT / Qwen3-VL-8B, eager)[¹](#note-1-rldx-1-weights-license) [³](#note-3-rldx-gr1) | RoboCasa GR-1 Tabletop (24-task) | GR-1 humanoid (bimanual + waist) | `RLDX-GR1-Demo` |
| `openarm-isaac` | — | π0.5 (LeRobot pi05 `folding_latest`, 16-DOF bimanual), pipe-proof only[¹¹](#note-11-openarm-isaac-experimental) | Isaac Lab bimanual Reach (empty scene) | OpenArm (bimanual, 2 × 7-DOF + grippers) | `OpenArm-Isaac-Demo` |
| `robolab-cosmos3` | — | NVIDIA Cosmos3-Edge-Policy-DROID (4B world action model; Cosmos3-Nano-Policy-DROID 16B via `model.variant: nano`), cosmos-framework policy server[¹⁴](#note-14-robolab) | NVIDIA RoboLab (Isaac Lab 2.2 / Isaac Sim 5.0): 12-task subset of RoboLab-120 (default) or the full 120 tasks | DROID Franka Panda + Robotiq 2F-85 | `RoboLab-Cosmos3-Demo` |
| `robolab-pi05` | — | π0.5-DROID (`pi05_droid_jointpos`, openpi policy server)[¹⁴](#note-14-robolab) | NVIDIA RoboLab, same protocol as `robolab-cosmos3` | DROID Franka Panda + Robotiq 2F-85 | `RoboLab-Pi05-Demo` |
| `smolvla-robotwin` | — | SmolVLA-450M (`lerobot/smolvla_robotwin`)[¹⁵](#note-15-robotwin-20-smolvla-robotwin-lingbot-vla2) | RoboTwin 2.0: 5-task clean subset (default) or the official 49 tasks × 100 episodes, clean / randomized | Aloha-AgileX dual arm (2 × 6-DOF + grippers, 14-D joint actions) | `SmolVLA-RoboTwin-Demo` |
| `lingbot-vla2` | — | LingBot-VLA 2.0 (Robbyant, Qwen3-VL-4B + sparse-MoE action expert, 6B, fp32)[¹⁵](#note-15-robotwin-20-smolvla-robotwin-lingbot-vla2) | RoboTwin 2.0, same protocol as `smolvla-robotwin` | Aloha-AgileX dual arm | `LingBot-VLA2-Demo` |
| `rlinf-oft-libero` | — | OpenVLA-OFT-7B one-demo SFT (`Haozhan72/Openvla-oft-SFT-libero-goal-traj1`) **improved with RL** (RLinf GRPO)[¹⁶](#note-16-rl-post-training-with-rlinf-rlinf-oft-libero) | LIBERO-Goal: eval of the SFT checkpoint, GRPO training, eval of the RL checkpoint (same protocol) | Franka Panda (7-DOF) | `RLinf-OFT-LIBERO-Demo` |
| `lerobot-gym` | — | LeRobot reference policies, one checkpoint per task: ACT (`lerobot/act_aloha_sim_insertion_human`, `lerobot/act_aloha_sim_transfer_cube_human`) and Diffusion Policy (`lerobot/diffusion_pusht`)[¹⁷](#note-17-lerobot-gym-act-on-aloha-diffusion-policy-on-push-t) | gym-aloha `AlohaInsertion-v0` (bimanual peg insertion), `AlohaTransferCube-v0` (bimanual cube transfer); gym-pusht `PushT-v0` (planar pushing) | ALOHA dual arm (2 × 6-DOF + grippers, MuJoCo) / 2-D point pusher | `LeRobot-Gym-Demo` |
| `multitask-dit` | — | LBM-style Multi-task DiT (LeRobot `multi_task_dit` port; community LIBERO-10 checkpoint, **not TRI weights**)[¹⁸](#note-18-multitask-dit-lbm-style-multi-task-dit-not-tri-weights) | LIBERO-10: stove + moka pot, bowl into drawer + close | Franka Panda (7-DOF) | `MultiTask-DiT-Demo` |
| `spatialvla` | — | SpatialVLA-4B zero-shot (`IPEC-COMMUNITY/spatialvla-4b-224-pt`, MIT), the authors' SimplerEnv eval[¹⁹](#note-19-simplerenv-with-the-authors-own-eval-spatialvla-magma) | SimplerEnv visual matching: Google pick coke can, move near (`--task-set actions`: open / close drawer); WidowX carrot on plate, eggplant in basket | Google Robot (Fractal) / WidowX 250 | `SpatialVLA-Demo` |
| `magma` | — | Magma-8B zero-shot (`microsoft/Magma-8B`, MIT), the authors' SimplerEnv eval[¹⁹](#note-19-simplerenv-with-the-authors-own-eval-spatialvla-magma) | SimplerEnv: WidowX carrot on plate, eggplant in basket; Google move near (`--task-set actions`: open the drawer and put the apple in) | WidowX 250 / Google Robot (Fractal) | `Magma-Demo` |
| `univla` | — | UniVLA (BAAI, Emu3-8.5B world model + FAST action tokens; `Yuqi1997/UniVLA` LIBERO img-sft checkpoint, Apache-2.0), the authors' LIBERO loop[²⁰](#note-20-libero-with-the-authors-own-loop-univla-ge-act) | LIBERO-Goal: bowl on stove, cream cheese in bowl, bowl on plate (`--task-set actions`: open drawer, push plate, turn on stove) | Franka Panda (7-DOF) | `UniVLA-Demo` |
| `ge-act` | — | GE-Act (AgiBot Genie Envisioner: LTX-Video 2B world model + action expert; per-suite LIBERO checkpoints, **CC BY-NC-SA 4.0, non-commercial**), the authors' LIBERO loop[²⁰](#note-20-libero-with-the-authors-own-loop-univla-ge-act) | same LIBERO-Goal tasks as `univla` | Franka Panda (7-DOF) | `GE-Act-Demo` |
| `dino-wm` | — | DINO-WM (NYU / Meta, frozen DINOv2 ViT-S/14 features + ViT predictor, MIT): **world-model planner, goal-image conditioned, not language** — MPC + CEM toward a goal image, the authors' `plan.py` eval[²¹](#note-21-latent-world-model-planners-dino-wm-jepa-wms) | Push-T (push the T block to a goal pose) and Wall (go through a door to a goal position) | 2D pusher / point agent | `DINO-WM-Demo` |
| `jepa-wms` | — | JEPA-WM (Meta FAIR, DINOv2 ViT-S/14 + AdaLN predictor; **CC BY-NC 4.0, non-commercial**): **world-model planner, goal-image conditioned, not language** — CEM toward a goal image, the authors' `evals.main` eval; `--task-set dino-wm-baseline` / `compare` add their DINO-WM baseline[²¹](#note-21-latent-world-model-planners-dino-wm-jepa-wms) | same Push-T and Wall | 2D pusher / point agent | `JEPA-WMs-Demo` |
| `vla-jepa` | — | VLA-JEPA (Qwen3-VL-2B + DiT-B flow-matching action head, trained with a V-JEPA 2 latent world-model objective; Apache-2.0), the LeRobot port `lerobot/VLA-JEPA-LIBERO` through upstream `lerobot-eval`[²²](#note-22-vla-jepa-lerobot-port-on-libero) | LIBERO-Goal: bowl on stove, cream cheese in bowl, bowl on plate (`--task-set actions`: open drawer, push plate, turn on stove; `spatial` / `object` / `long`: two tasks of each other suite) | Franka Panda (7-DOF) | `VLA-JEPA-Demo` |
| `vlaeval` | — | the harness's own model servers and benchmark images, run unmodified ([allenai/vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness) `vla-eval` 0.9.0, Apache-2.0); default model server LeRobot π0.5 `lerobot/pi05_libero_finetuned`[²³](#note-23-vlaeval-vla-evaluation-harness-run-unmodified) | LIBERO-Spatial, 10 tasks × 5 episodes (`--task-set object` / `goal` / `long` / `probe`; batch A: `libero-mem` / `calvin` / `kinetix` / `batch-a`) | Franka Panda (7-DOF) | `VLAEval-Demo` |

## Notes

### Note 1: RLDX-1 weights license

RLDX-1 weights are under the **RLWRLD Model License v1.0 (non-commercial)**; "simulation benchmarking" is an explicit intended use and commercial deployment is out of scope, so use these targets for research / simulation benchmarking only. Weights are downloaded from Hugging Face at runtime (not vendored).

### Note 2: rldx-simpler

`rldx-simpler` is the same RLDX-1 model on SimplerEnv (Google Fractal real-robot embodiment, `OXE_FRACTAL`) instead of LIBERO — the one target here that exercises a real-robot OXE embodiment rather than a simulated Panda/humanoid. Setup clones SimplerEnv + ManiSkill2_real2sim at runtime (MIT). End-to-end validated 2026-06-24 (`g6.2xlarge` L4, full `deploy.py` 1-shot smoke); checkpoint pinned at `@f59c79e1`. See the [showcase](showcase.md#rldx-simpler--rldx-1-rlwrld-on-simplerenv-google-fractal-real-robot-embodiment) and [Expected Results](results.md#expected-results).

### Note 3: rldx-gr1

`rldx-gr1` is the same RLDX-1 model family on **RoboCasa GR-1 Tabletop** (RLWRLD's own `RLDX-1-FT-GR1` checkpoint) — the only target here on a **bimanual humanoid with a waist DOF** (GR-1, `GR1ArmsAndWaistFourierHands`), distinct from all single-arm LIBERO/SimplerEnv clips. Renders via MuJoCo/robosuite (no Vulkan); the RoboCasa GR-1 tabletop-tasks suite is cloned at runtime (MIT). End-to-end validated 2026-06-25 (`g5.2xlarge` A10G 24GB, full `deploy.py` 1-shot smoke); checkpoint pinned at `@d0277c5c`. As source-verified, GR-1 needed **no SimplerEnv-style code patch** (dispatch + obs/action routing already correct at the pin) — only the shared dependency-pin fixes. Measured 0.80 (PnPCupToDrawerClose) and 0.20 (PnPMilkToMicrowaveClose) at n=5; reported paper SR = 58.7% (24-task average). See [Expected Results](results.md#expected-results).

### Note 4: rldx-kitchen

`rldx-kitchen` is the same RLDX-1 model family on the **RoboCasa Kitchen** 24-task benchmark (RLWRLD's own `RLDX-1-FT-ROBOCASA` checkpoint) — a **single-arm mobile manipulator** (`PandaOmron`: a Franka Panda arm on an Omron LD mobile base) doing pick-and-place, doors, drawers and appliances across full kitchen scenes, distinct from the GR-1 humanoid and the LIBERO/SimplerEnv clips. Renders via MuJoCo/robosuite (no Vulkan); the RoboCasa kitchen suite (`squarefk/robocasa`, MIT) is cloned at runtime and its ~5 GB of scene assets download during setup. Source-verified clean like GR-1 (no dispatch/obs/action patch) with **three RoboCasa-Kitchen-specific fixes** found by reading source: (a) the cloned `download_kitchen_assets.py` has no `-y` argparse and prompts via `input()` → made non-interactive so it runs headless; (b) a dependency-pin reconciliation (the kitchen fork pins `numpy 1.23.3`/`numba 0.56.4`, but the import-reachable `albumentations` needs `numpy>=1.24.4`, so the GR-1-proven `numpy 1.26.4`/`numba 0.61.2` stack is protected); (c) the env emits each of the 3 cameras at two resolutions (256 + 512) and the video recorder stacks all of them into a six-panel strip — narrowed to the 256-px policy views so the saved video is a clean three-panel clip (cosmetic; obs/policy/SR untouched). Checkpoint pinned at `@e8279a38`. End-to-end validated 2026-06-26 (`g6.2xlarge` L4 24 GB, full `deploy.py` 1-shot smoke); fixes (a)+(b) confirmed live (the ~5 GB asset download ran headless and the `numpy`/`numba`/`albumentations` stack resolved); fix (c) observed in the Gate-3 output and applied to the committed clips (re-validates live on next deploy). Two articulated tasks at n=5: `OpenSingleDoor` `success_rate = 0.80` (4/5) and `OpenDrawer` `success_rate = 0.80` (4/5); reported paper SR = 70.6% (24-task average). See [Expected Results](results.md#expected-results).

### Note 5: molmoact2

`molmoact2` code is **Apache-2.0** (Ai2 releases the weights as open weights; the `allenai/MolmoAct2-LIBERO` model card has no license field, so check it before commercial use). Unlike the LeRobot-native targets, it is **not on upstream LeRobot** — it is served from the `allenai/lerobot` fork (`molmoact2-policy` branch, pinned `@a4f15bf3`) installed via `uv sync --locked` (Python 3.12), running the public `allenai/MolmoAct2-LIBERO` checkpoint **in-process** on the sim GPU (no separate policy server). The official LIBERO eval recipe is **float32** (`use_amp=false`) + `norm_tag=libero`, continuous action mode — so the ~20 GB fp32 checkpoint forces a ≥40 GB single GPU **and** ≥64 GiB host RAM (`g6e.2xlarge`+ L40S; `g6e.xlarge`'s 32 GiB host RAM thrashes the checkpoint load and is excluded). End-to-end validated 2026-06-28 (`g6e.2xlarge`, us-west-2; full `deploy.py` 1-shot). `libero_goal = 1.000` (2 tasks × 5 episodes); reported paper LIBERO average = 97.2%. See the [showcase](showcase.md#molmoact2--molmoact2-ai2-on-libero-goal-franka-panda-7-dof) and [Expected Results](results.md#expected-results).

### Note 6: shared lerobot-libero template (xvla, smolvla, lingbot-va)

`xvla`, `smolvla` and `lingbot-va` share one template (`templates/lerobot-libero-userdata.sh.j2`): upstream LeRobot is installed from source at a pinned commit (`uv sync --locked --extra <policy> --extra libero`, Python 3.12) and `lerobot-eval --policy.path=<checkpoint> --env.type=libero` runs the policy in-process on the sim GPU. Adding another LeRobot policy is a new `models/<id>.yaml` (checkpoint, extras, extra `--policy.*`/`--env.*` flags, instance list and VRAM/RAM floors) plus the name maps. All three are **Apache-2.0** (code and weights). Validated 2026-10-14 in ap-northeast-2: `xvla` libero_spatial **0.90** (2 tasks × 5 ep, g6.2xlarge), `smolvla` libero_spatial **0.80** (2 × 5, g6.xlarge), `lingbot-va` libero_10 **1.00** (2 × 3, g6e.2xlarge L40S; ~20 min per episode because the model generates video with the actions).

### Note 7: vla-adapter

`vla-adapter` is an OpenVLA-OFT fork, so `templates/vla-adapter-userdata.sh.j2` is a copy of the `openvla-oft` template (kept separate so the validated `openvla-oft` target cannot regress) with the repo, checkpoint and extra `run_libero_eval.py` flags taken from `models/vla-adapter.yaml`. **MIT** (code and weights). Validated 2026-10-14 in ap-northeast-2: libero_spatial **1.00** (10 tasks × 1 trial, g6.xlarge). The checkpoint is evaluated from its local snapshot directory: with a Hub repo id the eval code skips registering the Prismatic classes and `transformers` rejects the custom config.

### Note 8: fastwam

`fastwam` runs on the shared `lerobot-libero` template ([note 6](#note-6-shared-lerobot-libero-template-xvla-smolvla-lingbot-va)). **Code MIT (FastWAM) / Apache-2.0 (LeRobot); the checkpoint's model card states no license, so treat this target as research/evaluation only and check the card before any other use.** Weights are downloaded at runtime and never committed here. Two deviations from the LeRobot reproduce line, both found on live runs: (a) the checkpoint runs in its default **bf16** — the README's `--policy.dtype=float32` targets a 140 GB GPU, and in fp32 the DiT + umt5-xxl text encoder + Wan2.2 VAE run out of memory on a 48 GB L40S; (b) it uses the `lerobot/fastwam_libero_uncond_2cam224` copy rather than the original upload, whose `config.json` predates the LeRobot port (tokenizer path `"."`) and fails to build the tokenizer; the `lerobot/` config fetches the tokenizer from `google/umt5-xxl` and the text encoder + VAE from `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (both Apache-2.0). Validated 2026-10-15 in ap-northeast-2: libero_spatial **0.90** (2 tasks × 5 ep, g6e.2xlarge L40S).

### Note 9: cosmos-policy

`cosmos-policy` is **not on LeRobot**: `templates/cosmos-policy-userdata.sh.j2` clones [NVlabs/cosmos-policy](https://github.com/NVlabs/cosmos-policy) at a pinned commit (`@18a2acca`), builds the project's own CUDA 12.8 Docker image, runs `uv sync --extra cu128 --group libero` inside it and evaluates with the repo's `run_libero_eval.py` (LIBERO.md flags in `models/cosmos-policy.yaml`) on one GPU. **Code Apache-2.0; the checkpoint `nvidia/Cosmos-Policy-LIBERO-Predict2-2B` (`@cb689ec0`) is under the NVIDIA One-Way Noncommercial License (NSCLv1) — research/evaluation only.** It also loads the gated base model `nvidia/Cosmos-Predict2-2B-Video2World` at eval time, so the HF token in SSM (`/vla-simulator/hf-token`) must belong to an account that has accepted the NVIDIA Open Model License on that page (`vlasim.py doctor` checks that the token exists, not that the license was accepted — an unaccepted license shows up as a 401/403 at eval time). Three fixes found on live runs: LIBERO's first import prompts via `input()` (headless `EOFError`) → the config is seeded once in a persisted path; a host `systemctl daemon-reload` revokes GPU access inside the container on cgroup v2 → Docker uses the `cgroupfs` driver and the GPU device nodes are passed explicitly; the apt timers are masked. Validated 2026-10-17 in ap-northeast-2: libero_spatial **1.00** (10 tasks × 1 trial, `g6e.xlarge` L40S); reported LIBERO-Spatial 98.1%.

### Note 10: aspire

`aspire` is **not a VLA**: [NVlabs/ASPIRE](https://github.com/NVlabs/ASPIRE) (Apache-2.0, pinned `@fa00f0ef`) runs a coding agent that iterates on a Python `cap-harness` program (perception → grasp → IK motions) until it solves the task. `templates/aspire-userdata.sh.j2` bootstraps the LIBERO-Pro fork and the tool services on one GPU (profile `rtx4090`), then runs Claude Code headless on **Amazon Bedrock** with the instance role (`instance.bedrock: true` in `models/aspire.yaml` adds `bedrock:InvokeModel*`). Protocol, chosen so the number is comparable with the LeRobot targets: the agent may only run ASPIRE's development seeds (51-65); the frozen program is then scored on LIBERO saved init states 0-4 — the same states `xvla` / `smolvla` / `fastwam` are scored on. Tool licenses: **SAM3 is gated on Hugging Face (Meta SAM License — the HF token must have been granted access; the boot probes it and stops early if not)** and **Contact-GraspNet is NVIDIA non-commercial**, so treat this target as research/evaluation only. The boot patches one upstream bug: `adapter.py` tests a numpy init-state array for truthiness (`elif not init_states:`), which makes every `--init-mode saved` episode a harness error.

### Note 11: openarm-isaac (experimental)

`openarm-isaac` is a valid `--vla` choice (stack `OpenArm-Isaac-Demo`) but is **experimental and a pipe proof, not a task benchmark**. It drives the LeRobot π0.5 port `lerobot/folding_latest` (16-DOF bimanual, pinned `@a53a7287`; Apache-2.0) on the [enactic/openarm_isaac_lab](https://github.com/enactic/openarm_isaac_lab) bimanual Reach environment (pinned `@bad82e2`) inside the NGC `nvcr.io/nvidia/isaac-lab:2.3.0` container (Isaac Sim 5.1 + Isaac Lab 2.3 + Python 3.11), through a thin LeRobot rollout driver (`assets/openarm-isaac/run_eval.py`). `folding_latest` is a real-robot cloth-folding policy, so on an empty Reach scene it is out of distribution and the success rate is about 0 **by design**; the video only shows that π0.5 drives 16-joint OpenArm bimanual motion in Isaac Lab. Local mode only (no bridge): inference runs in-process on the sim GPU. It needs the Hugging Face token in SSM because it downloads the gated tokenizer `google/paligemma-3b-pt-224` (see [Hugging Face token](configuration.md#hugging-face-token-gated-models)). Configuration and the integration contract: [`models/openarm-isaac.yaml`](../models/openarm-isaac.yaml) and the headers of `assets/openarm-isaac/*.py`.

### Note 12: real2sim

`real2sim` is **not a VLA evaluation**: it measures whether a scene reconstructed from a multi-view capture is good enough to plan a grasp in simulation. `templates/real2sim-userdata.sh.j2` runs the stage scripts in `assets/real2sim/` on one GPU instance:

1. **Input** — default: one [HOPE-Video](https://github.com/swtyree/hope-dataset) scene (NVIDIA; 10 tabletop sequences from a robot-arm-mounted RealSense D415 with camera poses, a CascadeStereo point cloud and ground-truth object meshes + poses), downloaded at run time from Google Drive with `gdown` and checked against the MD5 sums of the dataset's `setup.json` (@`621d855f`). **HOPE-Video is CC BY-NC-SA 4.0 — research / non-commercial only**; nothing from it is stored in this repository and the derived videos / meshes in your results bucket inherit that license. Your own capture: set `input.source: capture` and `input.capture_s3_uri: s3://<bucket>/<key>.zip` in [`models/real2sim.yaml`](../models/real2sim.yaml) (a zip with `images/` and a nerfstudio `transforms.json`, camera→world in OpenGL axes, **metric** poses, optional `ply_file_path` init cloud; images up to ~1600 px). The stack then grants `s3:GetObject` on exactly that object. Video input (frames + COLMAP via `ns-process-data`) is not wired: COLMAP poses have no metric scale and every segmentation threshold is in metres.
2. **3DGS** — nerfstudio 1.1.5 `splatfacto` (gsplat 1.4.0 prebuilt wheel, torch 2.4.1 cu124, Python 3.10), 15,000 iterations, initialised from the scene cloud; `ns-eval` (PSNR / SSIM / LPIPS on the held-out views), `ns-export gaussian-splat` and an orbit video `splat_orbit.mp4`.
3. **Back to metres** — `ns_to_world.py` inverts nerfstudio's `dataparser_transforms.json` (orientation + centring + scale) and keeps the opaque gaussians; its `nn_to_reference_median_m` (distance to the input cloud) shows whether the frames agree.
4. **Segment** — `r2s_segment.py`: RANSAC table plane (normal towards the cameras), DBSCAN clusters above it, each cluster completed with its footprint on the table and turned into a convex hull (`--decomp coacd` for a convex decomposition); writes `obj_XX.obj`, `scene_collision.usda` (UsdPhysics rigid bodies + collision) and `segment.json` (with HOPE-Video: matched / merged objects, Chamfer, F-score@5 mm and collision-volume ratio against the ground truth, per object and for the scene). Touching objects merge into one cluster (v1 limitation).
5. **Grasp** — `r2s_grasp_mj.py` (floating Robotiq 2F-85) and `r2s_grasp_ur5e.py` (UR5e on a pedestal, damped-least-squares IK, gravity compensation): the same top-down pick per object, scored as reached (< 1 cm) + lifted (≥ 5 cm) + undisturbed (< 2 cm push), in three worlds — `recon` (per-object meshes, this pipeline), `scenehull` (one static hull around everything, the single-mesh baseline; nothing can be lifted there by construction) and `gt` (HOPE-Video ground-truth meshes). Robot models: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) @`f3b60136` (UR5e BSD-3-Clause, Robotiq 2F-85 BSD-2-Clause), MuJoCo 3.16.0.

`summary.txt`: `success_rate` = floating-hand grasps that succeeded in the `recon` world / graspable objects (objects wider than the 85 mm gripper on both axes are skipped), plus `hand_*` / `ur5e_*` per world, PSNR / SSIM / LPIPS and the segmentation numbers. Videos (`task-0/videos/`: the splat orbit + one clip per object and world for the first `max_video_targets` objects) go to the gallery like every other target; the full outputs (`results.json`, `segment/`, `grasp/*.json`, the splat, logs) to `<run>/real2sim/`. Reference (the same stage scripts on the HOPE-Video scene-0 CascadeStereo cloud, 2026-10-29, CPU): 6 clusters, 6 of 10 objects matched (7 involved in merges), collision volume 2.65 × ground truth; floating-hand grasps recon 4/4, scene hull 0/4, ground truth 4/4. End-to-end validated 2026-10-29 in `ap-northeast-2` (`g5.2xlarge`, A10G, HOPE-Video scene 0, 3DGS input; instance boot to done 11 min 53 s, splatfacto 15,000 iterations in 272 s, the instance terminated itself, 25 videos in the gallery with MuJoCo EGL rendering):

| Stage | Measured |
|---|---|
| 3DGS | PSNR 23.998 / SSIM 0.855 / LPIPS 0.192, 196,858 gaussians kept |
| Frame inverse (`frame.json`) | NN distance splat → input cloud: median 5.8 mm, p90 93 mm |
| Segment | 13 clusters; 10 of 10 ground-truth objects matched, 0 merged; collision volume 1.25 × ground truth (all clusters / all GT meshes; per object 0.81-3.19 ×, recomputed from this run's `segment.json` with the welded GT meshes); scene hull 0.0299 m³ |
| Grasp, floating 2F-85 | recon 8/9, scene hull 0/9, ground truth 6/9 |
| Grasp, UR5e + 2F-85 | recon 4/9, scene hull 0/9, ground truth 4/9 |

Two ground-truth objects are over-split (CreamCheese into 2 clusters, AlphabetSoup into 3); they have the largest per-object volume ratios (1.86 × and 3.19 ×) and Chamfer distances (24 and 39 mm, the other eight 3-6 mm). The UR5e lifts fewer objects than the floating hand (`recon` 4 vs 8, `gt` 4 vs 6), and on this run the floating hand lifted more objects in `recon` (8) than in `gt` (6). These are the numbers of one run on one scene.

### Note 13: SimplerEnv Bridge (gr00t-bridge, rldx-bridge)

`gr00t-bridge` and `rldx-bridge` run the **SimplerEnv WidowX/Bridge** visual-matching benchmark (WidowX 250 on BridgeData scenes, ManiSkill2_real2sim + SAPIEN) next to the Google Fractal one of `rldx-simpler`. Both run the four tasks every Bridge results table reports: `widowx_spoon_on_towel`, `widowx_carrot_on_plate`, `widowx_stack_cube` (green block on yellow block) and `widowx_put_eggplant_in_basket`, 10 episodes each by default (`tasks[].n_episodes` / `num_episodes` in the yaml; the upstream tables use 100 to 200). `summary.txt` carries `suite: simpler_bridge`, so `vlasim results` shows one SimplerEnv Bridge row per run. "Bridge" here is the benchmark (BridgeData V2), not [bridge mode](bridge-mode.md): both targets run local mode only.

- **`gr00t-bridge`** — `nvidia/GR00T-N1.7-SimplerEnv-Bridge` @`940134b3` (NVIDIA Open Model License, commercial use allowed) on Isaac-GR00T @`65cc4a19`, the commit that published the N1.7 Bridge table. It follows the Bridge recipe of Isaac-GR00T `examples/SimplerEnv/README.md`: `run_gr00t_server.py --embodiment-tag SIMPLER_ENV_WIDOWX --use-sim-policy-wrapper` and the SimplerEnv rollout client with `--n-action-steps 4`, 300 steps per episode (`seed: 42` per task; at this pin the WidowX env ignores reset seeds, so only the RNGs are fixed). Same template as `gr00t` (`templates/gr00t-userdata.sh.j2`, branched by `sim_id`; the LIBERO render is unchanged). The SimplerEnv branch installs the Vulkan loader SAPIEN needs, runs `setup_SimplerEnv.sh`, adds the rollout-client packages that script misses at this pin (the list upstream added later, Isaac-GR00T `9c7e746`) and checks the rollout import chain and `gym.make` of the first task before the model loads. The checkpoint download skips the optimizer and RNG states (35.7 GB in the repo, 6.9 GB downloaded). Needs the HF token for the gated `nvidia/Cosmos-Reason2-2B` backbone, like `gr00t`.
- **`rldx-bridge`** — `RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX` @`e1314689` (RLWRLD Model License v1.0, **non-commercial**) on the same RLDX-1 pin, server and SimplerEnv venv as `rldx-simpler`, with the `widowx` settings of RLWRLD's `eval_simpler.sh` (`OXE_BRIDGE_ORIG`, 2 actions per chunk, 300 steps). The WidowX env reports euler roll/pitch/yaw plus a pad entry and no quaternion, so the observation/action adapter of `rldx-simpler` gets a WidowX branch (`model.simpler_robot: widowx`); the other rldx targets render unchanged.

Measured on 2026-10-30 (`g6.2xlarge`, 4 tasks x 10 episodes): `gr00t-bridge` 55.0% (22/40), `rldx-bridge` 65.0% (26/40); see [Expected Results](results.md#expected-results).

## LIBERO-Plus robustness sweep

`--libero-variant plus` is a switch on existing LIBERO targets, not a new `--vla` target. It evaluates the **same checkpoint** on [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) (Fei et al., [arXiv 2510.13626](https://arxiv.org/abs/2510.13626)), a fork of LIBERO that installs under the same `libero` package and replaces each suite's 10 tasks with ~2,400-2,600 perturbed variants (10,030 in total), each tagged with one of seven perturbation dimensions in the fork's `libero/libero/benchmark/task_classification.json`:

| `--perturb` key | Dimension (classification category) | Variants in `libero_spatial` |
|---|---|---:|
| `camera` | Camera Viewpoints | 376 |
| `robot` | Robot Initial States | 350 |
| `language` | Language Instructions | 390 |
| `light` | Light Conditions | 292 |
| `background` | Background Textures | 258 |
| `noise` | Sensor Noise | 351 |
| `layout` | Objects Layout | 385 |

```bash
# SmolVLA: LIBERO-Spatial, 7 dimensions x 10 variants x 1 episode = 70 episodes (~70-90 min, estimate)
python vlasim.py deploy --vla smolvla --libero-variant plus
# OpenVLA-OFT on the LIBERO-Spatial checkpoint (the sweep runs on the --libero-suite you pick)
python vlasim.py deploy --vla openvla-oft --libero-suite spatial --libero-variant plus
# Every other LIBERO target takes the same switch (suite from its libero_plus block)
python vlasim.py deploy --vla pi --libero-variant plus           # also gr00t, xvla, fastwam, lingbot-va,
python vlasim.py deploy --vla rldx --libero-variant plus         # vla-adapter, molmoact2, lap
# Fewer dimensions / more variants (0 = every variant of each dimension, ~350 episodes per dimension)
python vlasim.py deploy --vla smolvla --libero-variant plus --perturb camera,light,noise --perturb-n 30
# Compare with the LIBERO baseline of the same stack (run `deploy --vla smolvla` once without the flag too)
python vlasim.py results --robustness --vla smolvla
python vlasim.py destroy --vla smolvla
```

How it works:

- **Plan.** generate.py replaces the target's `tasks` with one entry per dimension, `condition: LIBERO-Plus/<dimension>`. Settings come from the `libero_plus` block of `models/<id>.yaml` (`suite`, `perturb`, `variants_per_category`, `seed`, `pip_pins`, `creationpolicy_timeout`, `ebs_gb`); the shared pins live in `libero_plus.py`. `--perturb` / `--perturb-n` override the yaml.
- **Variants.** On the instance, `assets/libero-plus/select_variants.py` reads the fork's classification file and draws N variants of each dimension without replacement with a fixed seed (`seed` in the yaml, default 7), so a re-run evaluates the same variants. The selected ids, names and difficulty levels are uploaded as `task-N/libero_plus_variants.json`. One episode per variant, the LIBERO-Plus protocol (`num_trials_per_task` 1). The 0-based suite index is the classification `id` minus 1 (the fork keeps the suite order at `task_order_index` 0).
- **Install.** The fork is cloned at `4976dc30` (the SHA LeRobot pins in `docker/Dockerfile.benchmark.libero_plus`) and its asset pack (`Sylvest/LIBERO-plus` dataset @`dd2bd61b`, `assets.zip`, 6.4 GB) is downloaded, sha256-checked and extracted to `libero/libero/assets`. apt adds `libmagickwand-dev`, `libfontconfig1-dev`, `libexpat1` and `unzip`.
  - `smolvla` (LeRobot): the LeRobot recipe — `robosuite==1.4.1` (the fork needs `single_arm_env`, removed in 1.5) and the other runtime pins, the fork `--no-deps -e`, `hf-libero` removed, the clone first on `PYTHONPATH`, and `lerobot-eval --env.type=libero_plus` through `uv run --no-sync` (a synced `uv run` would reinstall `hf-libero` from `uv.lock`). The other LeRobot LIBERO targets (`xvla`, `lingbot-va`, `fastwam`) share the template and the LeRobot commit, so they only carry a `libero_plus` block. `molmoact2` has its own template but the same recipe: the allenai LeRobot fork it pins carries `LiberoPlusEnv` and the same `docker/Dockerfile.benchmark.libero_plus` pins; its lerobot-eval videos are prefixed with `{suite}_{task_id}` so variants do not overwrite each other.
  - `openvla-oft`: the fork is cloned in place of LIBERO and installed the same way (`editable_mode=compat`), plus `Wand` / `scikit-image` (its `extra_requirements.txt`) before the `numpy<2` re-pin. `run_libero_eval.py` iterates every task of the suite, so `assets/libero-plus/oft_task_ids_patch.py` rewrites that loop to the selected ids (after `stackpatch.py` restores the upstream file; the run stops if the loop is not found). `n_success` / `n_episodes` come from the eval log.
  - `vla-adapter`: an OpenVLA-OFT fork that keeps the same `run_libero_eval.py` task loop (checked at the pinned commit), so the `openvla-oft` recipe and patch apply as is. Its template has no `stackpatch.py`, so a pristine copy of the eval script is restored before each dimension's patch. The checkpoint is LIBERO-Spatial only, so the sweep runs on `libero_spatial`.
  - `pi` (openpi, Python 3.8 client image): the client container imports `libero` from the host checkout mounted at `/app/third_party/libero`, so the fork replaces that submodule on the host and the assets are unpacked there. The client Dockerfile installs a filtered copy of the fork's `requirements.txt` (robosuite 1.4.0, `usd-core` — no Python 3.8 wheel, training code only — and unpinned wand / scikit-image dropped; openpi already pins robosuite 1.4.1) plus Python 3.8 pins (`Wand==0.6.13 scikit-image==0.21.0` and its dependencies) with openpi's requirements as constraints. `assets/libero-plus/openpi_verify.py` builds, resets and steps one variant of each dimension inside the image and loads `main.py`'s imports before the sweep; `openpi_task_ids_patch.py` writes the selected ids into `main.py`'s task loop and prefixes the replay video name with the variant id.
  - `lap`: the same Python 3.8 client stack as `pi` (openpi's `third_party/libero` and an openpi-style `main.py`), installed in a host venv instead of an image. Same fork swap, requirement filter, pins, check and task-loop patch; LAP also writes a wrist video, whose name gets the variant id too.
  - `gr00t`: `setup_libero.sh` installs `libero` editable from `external_dependencies/LIBERO`, so the fork replaces that submodule before the script runs; the script's own `git submodule update` line is dropped, its closing `gym.make` sanity check gets a fork task name, the fork requirements are filtered as for `pi` and the Python 3.10 pins (`robosuite==1.4.1`, wand, scikit-image 0.25.2, numpy / mujoco restated) follow the script's mujoco pin. One `rollout_policy.py` call per variant (`libero_sim/<variant name>`) against one policy server per dimension; the checkpoint subfolder follows the sweep suite.
  - `rldx`: RLDX-1 ships its own LIBERO-Plus path at the pinned commit (`rldx/eval/sim/LIBERO_PLUS`, `libero_plus_sim/<variant>` gym ids, `run_scripts/eval/libero_plus/setup_libero_plus.sh`; its README reports 84.3% for the same `RLWRLD/RLDX-1-FT-LIBERO` checkpoint on the full set). The fork and asset pack are fetched pinned before that script runs (it clones the fork's HEAD and downloads the zip unpinned, and skips both when present); its requirements get robosuite 1.4.1, no `usd-core` and pinned wand / scikit-image, then the target's own venv fixes and the `libero_plus` pins apply to the `libero_plus_uv` venv. One rollout per variant; k/n come from each variant's `simulation_results.csv`, and a variant that crashes is logged (`variants_failed`) and not counted.
- **Stack.** Same stack name and results bucket as the LIBERO run, so `vlasim results --robustness` finds both. The CDK context `libero_variant=plus` (set by deploy.py) only takes `creationpolicy_timeout` and `ebs_gb` from the `libero_plus` block. `destroy` is unchanged.
- **Default off.** Without `--libero-variant plus` every rendered UserData script is byte-identical to before (checked for all targets), and so is the synthesized template.
- **Not wired:** `cosmos-policy` and `aspire`. `cosmos-policy` runs every step as a synced `uv run --group libero` inside its own image, which reinstalls the locked `libero` over the fork on each call, so the switch needs the fork baked into that image plus `--no-sync` on every call; the language dimension also changes instructions that are not in the shipped T5 embedding cache, so the eval would compute T5 embeddings at run time and write them back to the cache (not measured). `aspire` writes one skill program per task with a Bedrock agent and then freezes it; a sweep would need one program per variant (70 agent runs, about US$1.6 each at the measured cost), and ASPIRE reports the separate LIBERO-Pro benchmark instead.
- **Cost (estimate).** 70 episodes per target unless noted; hours = install + assets + eval; price = first preferred instance, us-east-1 On-Demand, plus ~US$0.3 of NAT processing for the 6.4 GB asset pack. Measured eval minutes for `smolvla` and `openvla-oft` are in [results.md](results.md#measured-on-this-harness).

  | Target | Suite | Instance | Hours | US$ |
  |---|---|---|---:|---:|
  | `smolvla` | Spatial | g6.xlarge | 1.2-1.5 | ~1.0-1.2 |
  | `openvla-oft` | `--libero-suite` | g6.xlarge | 2-2.5 | ~1.6-2.0 |
  | `vla-adapter` | Spatial | g6.xlarge | 1.3-1.8 | ~1.0-1.5 |
  | `xvla` | Spatial | g6.2xlarge | 1.2-1.6 | ~1.2-1.6 |
  | `pi` | Spatial | g5.xlarge | 2-2.5 | ~2.0-2.5 |
  | `lap` | Spatial | g6.xlarge | 1.8-2.5 | ~1.5-2.0 |
  | `rldx` | Spatial | g6.xlarge | 2.5-3.5 | ~2.0-2.8 |
  | `gr00t` | Spatial | g6.12xlarge (g6.2xlarge fallback) | 3-3.5 | ~14-16 (~3-3.5) |
  | `fastwam` | Spatial | g6e.2xlarge | 1.5-2 | ~3.4-4.5 |
  | `molmoact2` | Goal (its baseline suite) | g6e.2xlarge | 2.5-3.5 | ~5.6-7.8 |
  | `lingbot-va` | LIBERO-10, 2 variants per dimension = 14 episodes | g6e.2xlarge | 5.5-6.5 | ~12-15 |
- **Comparability.** A dimension row and the LIBERO baseline differ only in the scene when the checkpoint, eval seed and step cap are the same, which is what the switch keeps. `LIBERO-Plus/all` pools the dimensions with equal weight; the paper's "Total" weights by the number of variants per dimension. The LeRobot card of `lerobot/smolvla_libero_plus` reports 69.0% on task ids 0-9 of four suites; those ids are all Background Textures variants, so it is not a seven-dimension number.
- **License.** The fork's code repository has no LICENSE file and GitHub reports no license: treat it as research / evaluation use. The asset pack is MIT on its Hugging Face dataset card. Both are downloaded at run time; nothing from LIBERO-Plus is vendored in this repository.

## LIBERO-R2S: reconstructed objects in LIBERO

`--libero-objects real2sim` is a switch on `openvla-oft` and `pi`, not a new `--vla` target. It asks whether a policy that is strong on the public LIBERO scene still succeeds when LIBERO's objects are replaced by the objects a [real2sim](#note-12-real2sim) stage-1 run reconstructed from a real capture. The default stage-1 input, HOPE-Video, films the same HOPE grocery objects LIBERO-Object uses, so each reconstructed object replaces the LIBERO object of the same name and the task (and its language instruction) stays the same.

```bash
# Stage 1 once (keeps its results bucket); note the run id it prints
python vlasim.py deploy --vla real2sim
# OpenVLA-OFT on the LIBERO-Object checkpoint: public vs reconstructed objects, same tasks / init states / seed
python vlasim.py deploy --vla openvla-oft --libero-suite object --libero-objects real2sim \
    --real2sim-run s3://<real2sim results bucket>/<RUN_ID>
# pi0.5 (pi05_libero covers every suite; --r2s-suite picks another one)
python vlasim.py deploy --vla pi --libero-objects real2sim --real2sim-run s3://<real2sim results bucket>/<RUN_ID>
# Fewer conditions / trials
python vlasim.py deploy --vla pi --libero-objects real2sim --real2sim-run s3://... --r2s-conditions full --r2s-trials 5
python vlasim.py results --vla pi        # one row per condition: LIBERO-R2S/public|collision|full
python vlasim.py results --matrix        # columns LIBERO-R2S/<condition> (Object)
python vlasim.py destroy --vla pi
```

How it works:

- **Conditions.** generate.py replaces the target's tasks with one entry per condition, `condition: LIBERO-R2S/<name>`: `public` (stock LIBERO objects, always evaluated in the same run as the reference), `collision` (reconstructed collision geometry, stock textured mesh: isolates contact geometry from appearance) and `full` (the reconstructed hull is also what the cameras see, coloured with the mean colour of the 3DGS points inside it; no texture baking). Every condition runs the same task ids, the same fixed initial states and the same eval seed (7), so the objects are the only difference. Settings: `libero_r2s` in `models/<id>.yaml` (`suite`, `conditions`, `trials_per_task`, `scale`, `mass`, `select`, `objects`, `creationpolicy_timeout`, `ebs_gb`); constants and the plan in `libero_r2s.py`.
- **Objects.** On the instance, `assets/libero-r2s/build_r2s_objects.py` (stdlib only) reads `<run>/real2sim/segment/` (`segment.json` + one convex hull per cluster). Each LIBERO HOPE object with a matched cluster (the one with the most points if the object was over-split) gets a new `<name>.xml` in the sibling dirs `stable_hope_objects_r2s_{collision,full}/`; every other object is a relative link to the stock one, and the stock dir is not touched. The hull is expressed in the table frame, its axes are matched to the stock collision box by extent order (the real cookie box stands, LIBERO's lies flat), it is scaled to the stock bounding-box volume (`scale: fit`; `metric` keeps the measured size), its bottom face is put on the stock bottom face, its mass is set to the stock mass (`mass: match`) and friction / solref / solimp are copied.
- **Loader.** LIBERO registers HOPE objects in `libero/libero/envs/objects/hope_objects.py` with the literal path `assets/stable_hope_objects/<name>/<name>.xml`. `assets/libero-r2s/r2s_select.py patch-hope` changes that one path to read a module global (idempotent; no fork, no new object classes, the BDDL files are unchanged), and `use <dir>` sets the global before each condition. The same patch works for OpenVLA-OFT's conda env and for openpi, whose LIBERO client container imports the host checkout through the compose bind mount (`third_party/libero` @`f78abd6`: same `hope_objects.py` and suite map as LIBERO master).
- **Tasks.** `r2s_select.py tasks` picks the suite's tasks whose `(:obj_of_interest ...)` contains a replaced object (`select: any` = the object anywhere in the scene) and `loop` restricts the eval script's suite loop (`run_libero_eval.py`, openpi `examples/libero/main.py`) to those ids, after `stackpatch.py` has restored the upstream file.
- **Physics check.** Before the policy runs, `assets/libero-r2s/check_scene.py` loads each selected task's first initial state in every asset dir, holds the robot still for 40 control steps and compares where each replaced object comes to rest with where the stock object rests (horizontal 3 cm, vertical 1 cm, tilt 20°). An object that falls over or slides away would make the run measure the asset, not the policy, so it is kept stock, the assets are rebuilt, the tasks reselected and the check repeated once (a second failure stops the run); the kept objects are listed in `objects.json` and the report. On `openvla-oft` the check runs in the eval's conda env, on `pi` inside the LIBERO client image (Python 3.8).
- **Outputs.** Per condition the usual `task-N/summary.txt` (`n_success`, `n_episodes`, `benchmark: libero_r2s`, `r2s_condition`, `r2s_task_ids`, `r2s_objects`, `stack_r2s_*`) and captioned gallery videos; `<run>/libero-r2s/results.json` + `summary.txt` (k/n, Wilson 95% interval and drop vs `public` per condition, per-object success, object build manifest, physics check), plus `objects.json`, `tasks.json`, `check.json`.
- **Stack.** Same stack and results bucket as the LIBERO run. The CDK context `libero_objects=real2sim` + `real2sim_run=<uri>` (set by deploy.py) grants `s3:GetObject` on exactly `<run>/real2sim/segment/*` and `<run>/real2sim/splat_world.ply` and `s3:ListBucket` only under that `segment/` prefix, and takes the timeout / EBS size from `libero_r2s`. Not combinable with `--libero-variant plus`, `--bridge`, `--collect` or `--judge`.
- **Default off.** Without the flag every rendered UserData script is byte-identical to before (all targets) and the synthesized templates have no grant on the stage-1 bucket.

Measured locally (CPU, robosuite 1.4.1 + MuJoCo 3.3.1, no renderer) with the stage-1 run of HOPE-Video scene 0 (2026-10-29): 7 of the 10 reconstructed objects are LIBERO HOPE objects (alphabet soup, cookies, cream cheese, macaroni and cheese, milk, orange juice, tomato sauce; mustard, parmesan and pineapple are not in LIBERO), extent ratios vs the stock box 0.88-1.25. `libero_object` tasks 0, 1, 5, 7 and 9 have one of them as the object of interest. In those scenes the reconstructed alphabet soup, cream cheese, orange juice and tomato sauce rest within 3 mm of the stock pose; the reconstructed milk carton topples (75°) under LIBERO's initial-state drop and is kept stock, which leaves tasks 0, 1, 5 and 9 (alphabet soup, cream cheese, tomato sauce, orange juice): 4 tasks × 10 trials × 3 conditions = 120 episodes per target. `libero_10` is denser (objects packed on a table): there the reconstructed alphabet soup, orange juice and milk get knocked over or off the table, so `libero_object` is the default suite. Success rates on a GPU: [results](results.md#libero-r2s-reconstructed-objects).

Limits: one capture, one scene; the hulls are convex (the policy sees a box-like `full` object with one colour, not a textured scan); the shape is kept but the size is fitted to the stock object by default, so `metric` is the closer test of a real-size swap.

### Note 14: RoboLab

`robolab-cosmos3` and `robolab-pi05` run [NVIDIA RoboLab](https://github.com/NVlabs/RoboLab) (pinned `@9cce00ca`, 2026-10-27) the way its README describes it: a **policy server** and a **headless Isaac Sim client** talking the OpenPI msgpack WebSocket protocol, here both on one instance. One shared template, `templates/robolab-userdata.sh.j2`, does, in order: disable GPU ECC and reboot once (Isaac Sim on L40S, same as `openarm-isaac`); `git lfs pull` RoboLab's ~6.2 GB of assets; build RoboLab's own image with `docker/build_docker.sh` on `nvcr.io/nvidia/isaac-lab:2.2.0` pinned by digest (Isaac Lab 2.2.0 / Isaac Sim 5.0.0, the stack every leaderboard row except one was measured on); build the policy server image; download the checkpoint once; start one server per shard (port 8000+i) and run `policies/<backend>/run.py --headless --instruction-type default` per shard over a round-robin slice of the task list. Shards run in parallel, one per GPU (`gpus_per_shard: 2` gives the server its own GPU). `assets/robolab/robolab_report.py` merges the shards' `episode_results.jsonl` into `task-0/summary.txt` (suite `RoboLab/subset12` or `RoboLab/RoboLab-120` in `vlasim results`), `task-0/robolab_results.json` (per task, per difficulty, Wilson intervals, subtask score) and the gallery videos (`<task>_run<r>_env<e>_success=<bool>[_viewport].mp4`, at most `max_videos_per_task` per task and camera).

- **Policy servers.** `robolab-cosmos3`: [NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework) `@86e1ee52`, `python -m cosmos_framework.scripts.action_policy_server_robolab`, image from the framework's Dockerfile on the **CUDA 12.8** base (`cu128-train` + `policy-server` groups; the framework default CUDA 13.0 needs driver ≥ 580, newer than the project DLAMI), apex skipped. Edge is served with `--format-prompt-as-json True --guidance-interval 960 1001` (framework docs); Nano with the defaults. Checkpoints: `nvidia/Cosmos3-Edge-Policy-DROID@0110e64c` (9.2 GB) / `nvidia/Cosmos3-Nano-Policy-DROID@24aa5d14` (33 GB), public, not gated. `robolab-pi05`: the [xuningy/openpi](https://github.com/xuningy/openpi) fork RoboLab links to, `@aa642056` (the same commit RoboLab's image pins for `openpi-client`), `serve_policy.py policy:checkpoint --policy.config=pi05_droid_jointpos --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos` (public GCS, 12.4 GB) with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`.
- **Variants** (`model.variant` in the model yaml; the CDK stack reads the instance list, EBS and timeout of the selected variant): `robolab-cosmos3` = `edge` (default; 1 × L40S, 12 tasks × 5 episodes), `nano` (`g6e.12xlarge`, 2 shards of sim GPU + server GPU), `edge-full` (`g6e.48xlarge`, 8 shards, RoboLab-120 × 10 = the leaderboard protocol); `robolab-pi05` = `subset` (default) and `full` (`g6e.48xlarge`, 8 shards).
- **Subset.** 12 of the 120 tasks, 6 simple / 4 moderate / 2 complex, covering the visual, relational and procedural axes, × 5 episodes (`--num-envs 5`), default instructions: `BananaInBowlTask`, `RubiksCubeAndBananaTask` (the README's sanity pair), `PickUpGreenObjectTask`, `SmallerObjectButterInBinTask`, `MustardInLeftBinTask`, `BowlStackingLeftOnRightTask`, `BananasInBinThreeTotalTask`, `PickUpBluePitcherTask`, `RubiksCubeBehindBowlTask`, `Stack3RubiksCubeTask`, `BlockStackingSpecifiedOrderTask`, `ReorientAllMugsTask`. A subset rate is **not** a RoboLab-120 number; compare it only with another run of the same subset.
- **Licenses.** RoboLab code Apache-2.0, but most bundled scenes, objects and robot models are **CC BY-NC-SA 4.0 (non-commercial)** (RoboLab `THIRD_PARTY_NOTICES.md`); Isaac Sim runs under NVIDIA's license (the container EULA is accepted with `ACCEPT_EULA=Y`); Cosmos3 weights and cosmos-framework are OpenMDW-1.1 (commercial use allowed); openpi is Apache-2.0. See [licenses.md](licenses.md).
- **Status.** Default variants measured on 2026-10-30 (`g6e.2xlarge`, ap-northeast-2, subset12 × 5 episodes): `robolab-cosmos3` (`edge`) 31.7%, `robolab-pi05` 40.0%; `nano` and the full RoboLab-120 protocol are not run yet. Numbers and leaderboard references: [results.md](results.md#expected-results).
### Note 15: RoboTwin 2.0 (smolvla-robotwin, lingbot-vla2)

[RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) (MIT; Chen et al., [arXiv 2506.18088](https://arxiv.org/abs/2506.18088)) is a 50-task dual-arm benchmark on SAPIEN with the Aloha-AgileX robot, evaluated in a **clean** (Easy, `demo_clean`) and a **randomized** (Hard, `demo_randomized`: clutter, backgrounds, lighting, table height) setting; the [leaderboard](https://robotwin-platform.github.io/leaderboard) protocol is 100 episodes per task in both. Both targets share `templates/lerobot-robotwin-userdata.sh.j2` and run upstream LeRobot's `lerobot-eval --env.type=robotwin` in-process, one call per task. Choosing new targets over a `--benchmark` switch on `smolvla`: the RoboTwin run needs a different checkpoint, a different simulator stack (SAPIEN + Vulkan + a CuRobo build) and other instance sizes and timeouts, and a separate stack keeps its results bucket apart from the LIBERO runs.

- **Pins.** LeRobot `@552777b7` (`uv sync --locked`, Python 3.12); RoboTwin `@0aeea2d6` (the commit LeRobot's own RoboTwin benchmark image uses; RoboTwin `main` has since moved its configs and no longer matches the LeRobot wrapper); assets `TianxingChen/RoboTwin2.0 @981c92aa` (embodiments 220 MB + objects 3.7 GB; the 11 GB background textures only when the plan has the randomized setting); NVIDIA CuRobo `v0.7.8` (`@d64c4b00`), compiled on the instance for its GPU with an nvcc matching torch's CUDA; SAPIEN `3.0.0b1`, mplib `0.2.1`. Checkpoints `lerobot/smolvla_robotwin @eb55d3a7` and `lerobot/lingbot_vla_v2_robotwin @54f79a4b`; the repos their processors load (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, `Qwen/Qwen3-VL-4B-Instruct`) are downloaded at pinned revisions and the eval runs with `HF_HUB_OFFLINE=1` against them.
- **Protocol.** Default: `beat_block_hammer`, `click_bell`, `handover_block`, `lift_pot`, `stamp_seal` × 5 episodes, clean, seeds from 100000 (RoboTwin's own start seed), each task capped at RoboTwin's official step limit (`task_config/_eval_step_limit.yml`). `--robotwin-protocol clean|randomized|full` runs the official 49 tasks × 100 episodes (`open_laptop` is left out: its success check is broken for policy evaluation, per the LeRobot docs). One deploy has to finish inside the 12-hour CloudFormation WaitCondition, so `deploy.py` refuses a plan that cannot and prints the shard count: `--robotwin-shard K/N` runs every N-th (setting, task) unit; deploy K = 1..N one after the other (each shard is its own `RUN_ID`, read them together with `vlasim results --all-runs`).
- **Results.** Each task is one `task-N/` with `suite: RoboTwin2/<setting>`, `robotwin_task`, `condition: <protocol>` and a stack fingerprint (LeRobot, RoboTwin, assets, SAPIEN, CuRobo), so `vlasim results` prints one `RoboTwin2/clean` / `RoboTwin2/randomized` row per run. Up to 10 videos per task (head camera) go to the results bucket and the gallery as `<task>_0--eval_episode_N.mp4`.
- **Not the leaderboard protocol in two ways.** RoboTwin's own `script/eval_policy.py` first replays the scripted expert on every seed and keeps only seeds the expert solves; LeRobot's wrapper does not (episodes use consecutive seeds), so rates here can be lower than the leaderboard's. And seeds that RoboTwin rejects as unstable (a falling object at setup) are retried with seed + 100003 instead of being skipped (counted as `unstable_seeds_skipped` in `summary.txt`).
- **Two patches** of `lerobot/envs/robotwin.py` at the pin, each asserted to apply exactly once: the task config comes from `LEROBOT_ROBOTWIN_TASK_CONFIG` (default `demo_clean`, so the randomized setting can run), and the unstable-seed retry above. mplib and SAPIEN get the two source fixes RoboTwin's installer applies.
- **Install order.** mplib `0.2.1` declares `numpy<2` and SAPIEN `3.0.0b1` declares `opencv-python`; both are installed `--no-deps` and the rest of the RoboTwin stack against the frozen LeRobot lock, so no locked package moves (their binaries support numpy 2). Installed with its metadata, mplib downgrades numpy to 1.26 and `lerobot-eval` then dies at `import pyarrow`; that is how LeRobot's RoboTwin CI job ends on 2026-10-30 although the job is green. Before the checkpoint download the run imports the whole eval chain (including lazily imported modules) and resets one RoboTwin env on the GPU, then runs a 1-episode smoke eval with the exact command line.
- **Licenses.** Both checkpoints and LeRobot are Apache-2.0, RoboTwin and its assets MIT, SAPIEN and mplib MIT; **CuRobo is under the NVIDIA License (non-commercial: research / evaluation only)**, and RoboTwin builds a CuRobo planner per arm on every reset, so these targets are research / evaluation only.
- **Not done here.** X-VLA's RoboTwin checkpoint (`2toINF/X-VLA-RoboTwin2`) is not in LeRobot format and predicts end-effector poses; `lerobot/lingbot_va_robotwin` and `lerobot/fastwam_robotwin_uncond_3cam_384` are world action models that need `--env.action_mode=ee` and a larger GPU; `lerobot/hy_vla_robotwin` has no policy class in LeRobot at the pin; G0.5 and InternVLA RoboTwin checkpoints are non-commercial. Not run on AWS yet; expected numbers are in [results.md](results.md#robotwin-20-smolvla-robotwin-lingbot-vla2).

### Note 16: RL post-training with RLinf (rlinf-oft-libero)

This target is a workflow, not only an evaluation: **evaluate -> improve with RL -> re-evaluate** in one deploy. It runs [RLinf](https://github.com/RLinf/RLinf) (Apache-2.0; [arXiv 2509.15965](https://arxiv.org/abs/2509.15965), RLinf-VLA [arXiv 2510.06710](https://arxiv.org/abs/2510.06710)) with its OpenVLA-OFT GRPO recipe on LIBERO-Goal, starting from the one-demonstration-per-task SFT checkpoint the recipe uses (the published SFT baseline is 49.40%, so an RL gain is visible). It is its own target instead of an `--rl` switch on `openvla-oft`: the checkpoint differs (one-demo SFT, not the per-suite fine-tunes), the runtime is RLinf's own Docker image, the GPU count goes from 1 to 8, and a separate stack keeps the RL checkpoints in their own bucket.

- **Pins.** RLinf release `v0.4` = `@2348ece3`; image `rlinf/rlinf:agentic-rlinf0.4-maniskill_libero` pulled **by digest** `sha256:b5563d95...` (17.6 GB compressed, CUDA 12.8.1, Ubuntu 22.04; the digest is checked again after the pull) with the RLinf clone mounted at `/workspace/RLinf`, venv `openvla-oft`; SFT checkpoint `Haozhan72/Openvla-oft-SFT-libero-goal-traj1 @d20e1d44` (MIT, ~15 GB); `mujoco==3.3.2` inside the venv (3.3.3 changes LIBERO rendering and 3.4 breaks `libero_spatial` task 5 init states; the RLinf maintainers recommend 3.3.2). `model.suite` in `models/rlinf-oft-libero.yaml` switches to LIBERO-10 or LIBERO-Object (pinned SFT checkpoints and upstream configs for both).
- **Stages.** (1) GPU count, VRAM per GPU and host RAM against the profile floors; (2) image pull; (3) RLinf clone and file checks (train config, eval config, `run_eval.sh`); (4) **verify before any model download**: full-chain imports with `PYTHONUNBUFFERED=1` and `faulthandler` (torch CUDA, numpy < 2, the pinned MuJoCo, flash-attn, transformers, peft, ray, hydra, robosuite, LIBERO, prismatic and every `rlinf.*libero*` env module and `rlinf.*openvla_oft*` model module found by a package walk), one LIBERO `OffScreenRenderEnv` reset with a non-black EGL frame, then a Hydra `--cfg job` compose of the **exact** training command line so a bad override key fails here; (5) SFT checkpoint at the pinned revision; (6) **(a)** eval of the SFT checkpoint -> `task-0`; (7) **(b)** GRPO training for `--rl-steps` steps, TensorBoard scalars -> `rl/training_curve.csv` / `.json` (reward and `env/success_once` per step), the last `global_step_<N>` -> `s3://<bucket>/<RUN_ID>/rl/checkpoints/<step>/` + `rl/checkpoint.json`; (8) **(c)** eval of the RL checkpoint with the same eval config, env count and overrides as (a) -> `task-1`; (9) **(d)** `results.json` + `summary.txt` with before / after success, 95% Wilson intervals, the difference in points and whether the intervals overlap.
- **Profiles** (`--rl-profile`, CDK context `rl_profile`; each profile's `instance:` keys override the top-level ones):
  - `smoke` (default): 1 GPU, LoRA rank 32 on the 7B policy, 8 envs (one GRPO group), 2 rollout epochs, 5 steps, evals on 20 of the 500 fixed init states (each eval env renders on the one GPU next to the policy; 100 and 40 ran out of memory), wall-clock cap 6 h, WaitCondition 300 min. `g6e.8xlarge` -> `g6e.4xlarge` -> `g6e.16xlarge` -> `p5.4xlarge`, 500 GB EBS, floors 44 GB VRAM and 120 GB host RAM. It proves the pipe (install, verify, both evals, training, checkpoint, report); five LoRA steps are not expected to move the success rate.
  - `full`: 8 GPUs, the recipe defaults (full-parameter FSDP, 64 envs, group 8, 16 rollout epochs, lr 2e-5), 100 steps, evals on all 500 init states (the upstream protocol). The stack completes after the verify + baseline stage (WaitCondition 240 min) and training continues on the instance; the RL eval and the report follow by e-mail. After `max_wall_hours` (40 h) the instance publishes to SNS and terminates itself; training is stopped 1.5 h before that and the latest saved step (every 25) is evaluated. `p5.48xlarge` -> `p4de.24xlarge` -> `p5en.48xlarge` (80 GB GPUs only: 500 eval envs are 62.5 renderers per GPU next to a policy copy, and one 44 GB L40S ran out of memory at 40 in the smoke profile), 1500 GB EBS, floor 900 GB host RAM.
- **Reuse a checkpoint.** `--rl-eval-run RUN_ID` skips training and evaluates the RL checkpoint an earlier run of the same stack saved (it downloads only `model_state_dict/full_weights.pt`; the DCP shards with optimizer state stay in S3 for resuming). The run has to use the profile the checkpoint was trained with; the instance role can read only `*/rl/*` of its own bucket.
- **Results.** `task-0` has condition `baseline` and `task-1` condition `rl-grpo-<N>steps` (N = the step actually evaluated), both with the same stack fingerprint (RLinf commit, image digest, MuJoCo, SFT repo @ revision, eval config and episode count, driver). `vlasim results` prints both rows and `vlasim results --matrix` shows `rlinf-oft-libero` and `rlinf-oft-libero [rl-grpo-<N>steps]` side by side. Up to 20 eval videos per phase go to the gallery (`baseline_*.mp4`, `rl_*.mp4`).
- **The baseline here is sampled.** The upstream eval config samples (temperature 1.6) like the RL rollouts; the model card's SFT row (49.40%) was measured greedy. Compare (a) with (c) of the same run, not (a) with the card. Published numbers: [results.md](results.md#rl-post-training-rlinf-oft-libero).
- **Hardware is the open question.** RLinf documents the OpenVLA-OFT recipe on 8 × H100 80 GB; there is no published single-GPU config and no published time per step. LIBERO rollouts are CPU-bound (about one core per env) and need EGL from the NVIDIA driver (the container gets `NVIDIA_DRIVER_CAPABILITIES=all` so the NVIDIA container toolkit mounts the driver's EGL libraries, and the verify fails on a black frame). 48 GB and 40 GB GPUs are not in the `full` list (below the tested memory; the smoke runs confirmed the eval OOM on 44 GB); first runs measure time per step and memory, see [cost.md](cost.md#rl-post-training-rlinf-oft-libero).
- **Not done here.** π0.5 PPO on LIBERO (RLinf reports 84.6% -> 98.8% on Goal from a few-shot SFT) needs openpi weights and another venv of the same image; GR00T and LingBot-VLA RL recipes and ManiSkill / RoboTwin RL are not wired. [SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL) (MIT) trains the same SFT checkpoints and was not chosen because it has no tagged release or image.

### Note 17: lerobot-gym (ACT on ALOHA, Diffusion Policy on Push-T)

The single-task reference policies LeRobot publishes for its own gym simulators, as one target with **one checkpoint per task** (each `tasks:` entry names its `hf_repo` + revision, `env_type`, gym `task` and `policy_type`; `templates/lerobot-gym-userdata.sh.j2` runs one `lerobot-eval --env.type=<env_type> --env.task=<task>` per entry). One target instead of one per policy because the three share the LeRobot install, the instance size and the results bucket; `--task-set aloha` or `--task-set pusht` runs one family only.

```bash
python vlasim.py deploy --vla lerobot-gym                     # ACT insertion + ACT cube transfer + Diffusion Push-T, 10 episodes each
python vlasim.py deploy --vla lerobot-gym --task-set aloha    # ALOHA only (bimanual)
python vlasim.py deploy --vla lerobot-gym --task-set pusht    # Push-T only (push)
```

- **Pins.** LeRobot `@ae2b57f1` (0.6.2, the `smolvla` pin; `uv sync --locked --extra aloha --extra pusht --extra diffusion`, Python 3.12), which resolves gym-aloha 0.1.5, gym-pusht 0.1.8, MuJoCo 3.8.1, dm-control 1.0.41, pymunk 6.11.1, gymnasium 1.3.0. Checkpoints `lerobot/act_aloha_sim_insertion_human @33259aa8`, `lerobot/act_aloha_sim_transfer_cube_human @ba73b276`, `lerobot/diffusion_pusht @84a7c231` (all Apache-2.0).
- **Checkpoint format.** The three hub checkpoints predate LeRobot's processor pipeline (no `policy_preprocessor.json`; the normalisation statistics are buffers in `model.safetensors`), and `lerobot-eval` at the pin refuses them with `ProcessorMigrationError`. The UserData converts each one with the in-tree `src/lerobot/processor/migrate_policy_normalization.py` of the same commit into its own directory and evaluates that. The script at the pin builds the config from raw `config.json` without draccus decoding, so a JSON list stays a list in a tuple field (Diffusion `crop_shape: [84, 84]`) and saving fails with `Couldn't encode 84`; a 20-line wrapper in the template turns those lists into tuples first. The weights are unchanged.
- **Checked locally (CPU, 2026-11-02).** All three checkpoints convert with the wrapper; the converted ACT checkpoints load and return a 14-D action through their processors; `lerobot-eval --env.type=pusht` ran 2 Push-T episodes end to end on the converted Diffusion checkpoint (capped at 48 steps, so 0/2 success; `eval_info.json` + 2 videos written). gym-aloha was not run locally (no EGL on that host); on the GPU instance MuJoCo renders through EGL as in the LIBERO targets.
- **Results.** Each task is one `task-N/` with `suite: <env_type>`, `env_name: <gym task>`, `policy_type`, `action_type` (`bimanual` for both ALOHA tasks, `push` for Push-T) and a stack fingerprint (LeRobot commit, MuJoCo, gym package version). Videos are named `<gym task>--eval_episode_N-<success|failure>.mp4` (outcome from `eval_info.json`), up to 10 per task. Episodes run from `--seed` 1000 like the `eval_info.json` in each hub repo. The two ACT repos ship the same `eval_info.json` (byte-identical, 83.0% over 500 episodes), so it is not a per-task number; the ACT paper (Table I, human data) reports 20% insertion and 50% transfer, and the `diffusion_pusht` card 65.4% over 500 episodes.
- **Measured (ap-northeast-2, `g6.xlarge`, 2026-11-02, run `2026-11-02-1056`).** `AlohaInsertion-v0` 1/10, `AlohaTransferCube-v0` 7/10, `PushT-v0` 8/10, about 13 minutes from run start to the last result. See [results](results.md#lerobot-gym-and-multi-task-dit-lerobot-gym-multitask-dit) and the [showcase](showcase.md#bimanual).

### Note 18: multitask-dit (LBM-style Multi-task DiT, not TRI weights)

TRI's Large Behavior Model ([arXiv 2507.05331](https://arxiv.org/abs/2507.05331)) has no public weights for any simulator. LeRobot ships `multi_task_dit` (`src/lerobot/policies/multi_task_dit`, CLIP vision + text conditioning, diffusion or flow-matching DiT head), whose docs say the implementation is "based on the work described in TRI's LBM paper". This target runs that policy class through the shared lerobot-libero template ([Note 6](#note-6-shared-lerobot-libero-template-xvla-smolvla-lingbot-va)).

- **Checkpoint.** The only public `multi_task_dit` checkpoint for a simulator LeRobot evaluates (HF Hub search, 2026-11-02) is a community upload, `LuigiDu/multitask-dit-libero10-all-seed1000 @0f6ca5d8` (Apache-2.0 card, LeRobot 0.6.1 format with processor files, ~249M parameters). Its `train_config.json`: `HuggingFaceVLA/libero` episodes 0-378 (the 10 LIBERO-10 tasks), **15000 steps at batch size 2**, no evaluation on the card. LeRobot's own LIBERO result for this policy (`docs/source/multi_task_dit.mdx`: LIBERO-10 83.2%, 100k steps at batch size 320) has no public checkpoint. Expect a low success rate: the target shows the policy class runs, not what LBM-style training reaches.
- **Settings.** `--env.observation_height=256 --env.observation_width=256` (training images are 256 × 256 and the policy center-crops 224 at eval; LIBERO's default 360 would crop a narrower view). LIBERO-10 task ids 2 and 3 (stove + moka pot, bowl into drawer + close), 5 episodes each.
- **Checked locally (CPU, 2026-11-02).** The checkpoint loads at the LeRobot pin and returns a 7-D action through its processors.
- **Measured (ap-northeast-2, `g6.xlarge`, 2026-11-02, run `2026-11-02-0931`).** LIBERO-10 tasks 2 and 3 × 5 episodes = 0/10 (max reward 0 in every episode). The pipeline runs end to end (10 videos, `eval_info.json`, `summary.txt`); the score reflects the short community training run, not the policy class. See [results](results.md#lerobot-gym-and-multi-task-dit-lerobot-gym-multitask-dit).
- **Not added.** `TRI-ML/multitask-dit-robosuite-wrist-diffusion-064000` (TRI's verified Hub org, 95.6% over 90 robosuite episodes on its card) is trained on robosuite Lift / PickPlaceCan / NutAssemblySquare with a 32-D state from the dataset `TRI-ML/PretrainFinetune`; LeRobot has no robosuite env type at the pin, so it cannot be evaluated with `lerobot-eval`. A train-then-evaluate path (`lerobot-train` on LIBERO) does not fit the existing templates cheaply: the documented recipe is 100k steps at batch size 320.

### Note 19: SimplerEnv with the authors' own eval (spatialvla, magma)

Both targets run the SimplerEnv evaluation the model authors publish, not a re-implementation, through one shared template (`templates/simpler-userdata.sh.j2`): a SimplerEnv checkout at a pinned commit (with its `ManiSkill2_real2sim` submodule, checked after the clone), the authors' plugin copied over it where they ship one, one Python 3.10 venv installed from a lock file in [`assets/simpler/`](../assets/simpler/) (the `.in` next to it is the input; recompile with the `uv pip compile` line in its header), and one `main_inference` call per task variant. Each `tasks:` entry carries the arguments of the authors' script (named in its `source:` field) minus the five the template sets (`--policy-model`, `--ckpt-path`, `--policy-setup`, `--env-name`, `--logging-dir`). SimplerEnv writes one video per episode named `success_…` / `failure_…`; the template counts them, uploads them as `<task>--<success|failure>-NNN.mp4`, and fails the run if the count differs from the episodes the arguments define (`generate.simpler_episode_count`: robot-pose grid × object grid or episode range).

| | `spatialvla` | `magma` |
|---|---|---|
| Checkpoint | `IPEC-COMMUNITY/spatialvla-4b-224-pt @886c4557` (the one `scripts/run_spatialvla.sh` evaluates; README "SpatialVLA (zero-shot)" rows) | `microsoft/Magma-8B @4159b1de` (pretrained; card "SE-Google Robot 52.3 / SE-Bridge 35.4") |
| Harness | `DelinQu/SimplerEnv-OpenVLA @32143c81` (SpatialVLA README: "We release the SimplerEnv evaluation code for SpatialVLA based on DelinQu/SimplerEnv-OpenVLA"); ManiSkill2_real2sim `@cd45dd27` | `simpler-env/SimplerEnv @d55e1916` + `microsoft/Magma @e3737803` `tools/simplerenv-magma/simpler_env` copied over `simpler_env/` (Magma README "Evaluation with SimplerEnv"); ManiSkill2_real2sim `@ef7a4d4f` |
| Entry | `simpler_env/main_inference.py --policy-model spatialvla --action-ensemble-temp -0.8` | `simpler_env/main_inference_magma.py --policy-model magma` |
| venv | torch 2.5.1, transformers 4.47.0 (SpatialVLA `requirements.txt`), tensorflow 2.15.0, sapien 2.2.2 | torch 2.3.1, transformers fork `jwyang/transformers@1708d3ca` (`dev/jwyang-v4.48.2`; the Magma-8B card says the fork is required), flash-attn 2.6.3 wheel, open_clip_torch 2.30.0, tensorflow 2.15.0, sapien 2.2.2 |
| Default run | pick coke can (1 of the script's 12 configs, 25 ep), move near (episodes 0-19 of 60), carrot on plate (24), eggplant in basket (24) = 93 episodes | carrot on plate (24), eggplant in basket (24), move near (episodes 0-19 of 60) = 68 episodes |
| `--task-set bridge` | all four WidowX tasks of `scripts/bridge.sh`, 4 × 24 | all four of Magma's `bridge.sh`, 4 × 24 |
| `--task-set actions` | open / close top drawer, 9 robot positions each (`drawer_visual_matching.sh`, first URDF variant, ray tracing on) | open the top drawer and put the apple in, 3 positions × 9 apple spots (`put_drawer_visual_matching.sh`, first URDF variant) |
| Instance | `g6.xlarge` (8.09 GB bf16 weights, 8.5 GB inference per the README) | `g6.2xlarge` (17.8 GB bf16 weights exceed the 16 GB host RAM of an xlarge; the load also reads the 4.8 GB `laion/CLIP-convnext_xxlarge` tower, prefetched at `@51633fd7`) |

- **Harness pin for `spatialvla`.** The fork is pinned to `32143c81` (2025-03-24), not its `main` (`ccfe3809`): from `879f73c` (2025-05-19) the fork's `maniskill2_evaluator.py` passes `eef_pos=obs["agent"]["eef_pos"]` to the policy, a key only newer `allenzren/ManiSkill2_real2sim` commits provide, while the submodule gitlink stays at `cd45dd27`, so a recursive clone of `main` fails on the first step. `32143c81` → `ccfe3809` changes no SpatialVLA file and no task script, only that line.
- **Remote code.** Both checkpoints load with `trust_remote_code`. The checkpoint files come from the pinned revision; Magma's `config.json` names its code as `microsoft/Magma-8B--modeling_magma...`, so transformers fetches the modeling code from the Hub's `main` (the same commit as the pin on 2026-11-02).
- **Changes to the authors' commands.** None to the arguments. Magma's scripts give overlay paths as `./SimplerEnv/ManiSkill2_real2sim/...` (run from the parent of the checkout); the template runs from the checkout, so they start with `./ManiSkill2_real2sim/`. `gymnasium` is pinned to 0.29.1: the fork's ManiSkill2_real2sim `@cd45dd27` asks for `gymnasium>=0.28.1` and an unpinned resolve picks 1.x; upstream capped it below 1.0 in `ef7a4d4` ("fix gym version <1.0").
- **Subsets.** Defaults are cut so a run finishes in about an hour; every cut is written in the task's `source:` field and in `summary.txt`. Published numbers average over the full protocol (all URDF / orientation variants, all 60 move-near episodes), so a default run is a spot check, not a reproduction.
- **Randomness.** SimplerEnv has no seed flag: start states are the fixed grids / episode ids in the arguments (`seeds.py` records `not-controllable`). SpatialVLA decodes greedily; Magma samples its action tokens (temperature 0.7, no seed), so its runs can differ episode by episode.
- **Measured (ap-northeast-2, 2026-11-02).** `spatialvla` (`g6.xlarge`, run `2026-11-02-1355`): pick coke can 20/25, move near 16/20, carrot on plate 7/24, eggplant in basket 14/24, about 108 minutes; `--task-set actions` (run `2026-11-02-1556`): open top drawer 5/9, close top drawer 4/9, about 39 minutes. `magma` (`g6.2xlarge`, run `2026-11-02-1425`): carrot 8/24, eggplant 19/24, move near 11/20, about 83 minutes; `--task-set actions` (run `2026-11-02-1622`): put the apple in the closed top drawer 3/27, about 79 minutes. See [results](results.md#simplerenv-with-the-authors-own-eval-spatialvla-magma) (with the published numbers) and the [showcase](showcase.md#articulated-doors-drawers-knobs).
- **Dropped: CogACT.** `CogACT/CogACT-Base` (MIT) has an official SimplerEnv plugin (`microsoft/CogACT` `sim_cogact`, README "Evaluation in SIMPLER"), but loading it builds the Prismatic `llama2-7b-pure` backbone from `meta-llama/Llama-2-7b-hf` (config + tokenizer, `prismatic/models/backbones/llm/base_llm.py`), a gated repo under the Llama 2 license, so every user would need an approved HF token; and the authors' `main_inference.py` snippet runs the 30.5 GB fp32 `.pt` without `use_bf16`, which does not fit a 24 GB GPU. A follow-up can add it with an HF-token path (`g6e`, ≥ 64 GB host RAM).

### Note 20: LIBERO with the authors' own loop (univla, ge-act)

Both targets run the LIBERO evaluation loop the model authors publish, through one shared template (`templates/libero-own-userdata.sh.j2`) and one runner ([`assets/libero/own_eval_runner.py`](../assets/libero/own_eval_runner.py)): the authors' repository and LIBERO at pinned commits (checked after the clone), one Python 3.10 venv from a lock file in [`assets/libero/`](../assets/libero/) (recompile with the `uv pip compile` line in the `.in` header; LIBERO is installed `--no-deps`), and the weights at pinned revisions. The authors' scripts evaluate every task of a suite and save no per-episode video, so the runner calls their loop **unchanged** and adds three things around it: a task subset (the suite object the loop reads sees only the requested task ids), a recorder (the `get_libero_env` the loop calls is wrapped: every `env.step` frame is kept, every `env.reset` opens an episode) and outcome pairing (both loops write one `Success: True|False` line per episode to their log; the runner pairs those lines with the recorded episodes and fails on any count or task-name mismatch). Videos are uploaded as `<task>--episode_<k>-<success|failure>.mp4`; each LIBERO task gets its own `task-N/summary.txt` (`env_name` = LIBERO task name, `task_id`, `instruction`, `action_type`). `generate.py` checks every task id against LIBERO's task table at render time, the instance checks it again before the model loads, and the venv check builds one EGL-rendered LIBERO env.

| | `univla` | `ge-act` |
|---|---|---|
| Checkpoint | `Yuqi1997/UniVLA @df6c94ff`, folder `UNIVLA_LIBERO_IMG_BS192_8K` (17.0 GB bf16; one checkpoint for all four suites) | ModelScope `agibot_world/Genie-Envisioner @3d743077`, `ge_act_libero_<suite>.safetensors` (4.2 GB bf16 each, sha256-checked; not on the Hugging Face Hub); only the suites a run uses are downloaded |
| Also downloaded | `BAAI/Emu3-Stage1 @083e245a` tokenizer files (the same blobs as the checkpoint folder's), `BAAI/Emu3-VisionTokenizer @c81f916a` (1.08 GB, remote code) | `Lightricks/LTX-Video @8984fa25` `tokenizer/`, `text_encoder/` (T5-XXL, 19 GB fp32, loaded in bf16), `vae/`, `model_index.json` |
| Code | `baaivision/UniVLA @a91ee5f2` `reference/RoboVLMs/eval/libero/evaluate_libero_emu.py` `evaluate()` + `model_wrapper_emu.py` `EmuVLAModel` (docs/libero.md "Model Evaluation") | `AgibotTech/Genie-Envisioner @d41358f7` `experiments/eval_libero.py` `InferenceLibero.infer()` with `configs/ltx_model/libero/action_model_libero.yaml` (experiments/RUN.md "Evaluating on LIBERO") |
| Changes to the authors' code | two source patches at import, each required to match exactly once: the FAST tokenizer path hard-coded to the authors' cluster (`/share/project/...`) points at the checkout's `pretrain/fast`; three imports the evaluation never calls are removed (`robovlms` is not in the repository; `pytorch_lightning` only seeds — the runner seeds python / numpy / torch with 0 as `parser_args` does) | none; the runner fills the three `PATH` placeholders of a copy of the config and checks that every checkpoint tensor reached the model (the upstream loader is `strict=False`) |
| Policy | greedy decoding (`do_sample=False`), action chunk 10, wrapper default `video_mode=False` | 10 flow-matching steps per chunk, `noise_seed=42`, exec step 8, memory threshold 20 (goal / 10) or 30 (object / spatial), as `experiments/eval_libero.sh` |
| venv | torch 2.4.0, transformers 4.44.0, tiktoken 0.6.0, flash-attn 2.6.3 wheel (UniVLA `requirements.txt`; its flash-attn 2.5.7 has no torch 2.4 wheel), robosuite 1.4.0, mujoco 3.3.1, numpy 1.26.4 | torch 2.7.1, diffusers 0.32.0, transformers 4.51.3, accelerate 1.0.0 (Genie-Envisioner `requirements.txt`, inference subset), robosuite 1.4.0, mujoco 3.3.1, numpy 1.26.4 |
| Default run | LIBERO-Goal tasks 1, 6, 8 (bowl on stove, cream cheese in bowl, bowl on plate) × 10 episodes = 30 | same |
| `--task-set actions` | LIBERO-Goal tasks 0, 5, 7 (open the middle drawer, push the plate, turn on the stove) × 10 | same |
| Instance | `g6e.xlarge` (48 GB L40S; on a 24 GB L4 `g6.2xlarge` every episode ran out of CUDA memory, measured) | `g6.2xlarge` (2.1B transformer + T5-XXL in bf16 ≈ 15 GB VRAM; the encoder is read from 19 GB of fp32 shards) |

- **Checkpoint choice for `univla`.** The README lists an "img sft" (avg 94.8%) and a "video sft" (95.5%) LIBERO checkpoint. The authors' wrapper hard-codes `video_mode=False` (single-frame input), and the video checkpoint was trained with two interleaved frames (`--frames 2`, `max_position_embeddings` 3200 vs 1600), so the img-sft checkpoint is the one the shipped wrapper matches.
- **`ge-act` config.** `experiments/eval_libero.sh` names `configs/ltx_model_libero/action_model_libero_goal_pdstate.yaml`, which is not in the repository; RUN.md says to edit `configs/ltx_model/libero/action_model_libero.yaml`, which is used here. Its action expert (15 action + state channels, 16 × 32 heads, 28 blocks, caption channels 4096) matches the tensor names and shapes in the `ge_act_libero_goal` checkpoint header (1,429 bf16 tensors, 2.11B parameters). The later calls in that script pass an unset `$exec_step`, i.e. the default 8.
- **`torch.load` and LIBERO init states (`ge-act`).** LIBERO's `get_task_init_states` calls `torch.load` on its own pickled init files; torch ≥ 2.6 defaults to `weights_only=True` and refuses them. The template sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for this target (trusted files of the pinned LIBERO checkout); the venv check loads one init file.
- **Randomness.** Episode k uses LIBERO's fixed init state k; both loops call `env.seed(0)`. UniVLA decodes greedily and GE-Act fixes its noise seed, so `seeds.py` records `not-controllable` (there is no seed to vary).
- **Licenses.** `ge-act` is **non-commercial** (CC BY-NC-SA 4.0 for the Genie-Envisioner code and weights; the GE-Base weights it starts from derive from `ltx-video-2b-v0.9`, LTX-Video Open Weights License). The UniVLA repository has no `LICENSE` file at the pin; its weights card says Apache-2.0. See [licenses](licenses.md).
- **Checked locally (CPU, no GPU / EGL).** UniVLA: the patched `model_wrapper_emu.py` imports at the pin, the Emu3 tokenizer loads from the `BAAI/Emu3-Stage1` tokenizer files, the FAST processor loads from `pretrain/fast`, and `Emu3MoE` built from the checkpoint's `config.json` has the same 291 tensor names as its `model.safetensors.index.json`. GE-Act: `LTXVideoTransformer3DModel` built from the config has the same 1,429 tensor names and shapes as the `ge_act_libero_goal` header; `experiments/eval_libero.py` imports up to the renderer; the LIBERO task table in `generate.py` equals LIBERO's at `8f1084e3`; `get_task_init_states` fails on torch 2.7.1 without `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` and works with it. The episode loops themselves need a GPU and EGL and run first on AWS.
- **Measured (ap-northeast-2, 2026-11-02).** `univla` (`g6e.xlarge`, run `2026-11-02-1426`): 30/30, about 12 minutes; `--task-set actions` (run `2026-11-02-1519`): open drawer 8/10, push plate 9/10, turn on stove 10/10, about 14 minutes. The first `univla` run on `g6.2xlarge` (24 GB L4) scored 0/30 because every episode ran out of CUDA memory; the target now uses `g6e.xlarge` and fails a run whose episodes raised. `ge-act` (`g6.2xlarge`, run `2026-11-02-1355`): 28/30 (cream cheese 8/10), about 22 minutes; `--task-set actions` (run `2026-11-02-1430`): 30/30, about 19 minutes. See [results](results.md#libero-with-the-authors-own-loop-univla-ge-act) and the [showcase](showcase.md#articulated-doors-drawers-knobs).
- **Dropped: AgiBot GO-1.** `agibot-world/GO-1` and `GO-1-Air` are public (CC BY-NC-SA 4.0) but are pre-trained on AgiBot World only; the repository's LIBERO track (`evaluate/libero`, GO-1 94.8% avg) evaluates a checkpoint the authors fine-tuned themselves ("fine-tuned jointly on four task suites for 50k steps") and did not publish (the `agibot-world` Hub org has no LIBERO checkpoint). Adding it would mean a 50k-step fine-tune first.
- **Dropped for now: CogACT.** See [Note 19](#note-19-simplerenv-with-the-authors-own-eval-spatialvla-magma): its backbone config / tokenizer come from the gated `meta-llama/Llama-2-7b-hf`.

### Note 21: latent world-model planners (dino-wm, jepa-wms)

These two targets are **not VLAs**. An action-conditioned world model predicts the next frame's frozen DINOv2 ViT-S/14 patch features from the current ones and an action; a planner (CEM, inside MPC for DINO-WM) samples action sequences, rolls them out in the world model and keeps the ones whose predicted final features are closest to the encoding of a **goal image**. There is no language instruction: `summary.txt` says `model_kind: world-model planner, goal-image conditioned, not language` and `instruction: none (goal image)`. Success is the simulator's own state test written by the authors (Push-T: agent and block within 20 px and block angle within π/9 of the goal state; Wall: agent within 4.5 of the goal position).

Both run the authors' planning evaluation through one shared template (`templates/latent-wm-userdata.sh.j2`) and one runner ([`assets/latent-wm/wm_plan_runner.py`](../assets/latent-wm/wm_plan_runner.py)): the repository and the DINOv2 torch.hub code at pinned commits (checked after the clone; `torch.hub.load("facebookresearch/dinov2", ...)` in the authors' code would otherwise fetch the moving `main`), one venv from a lock file in [`assets/latent-wm/`](../assets/latent-wm/), and every checkpoint and dataset from a pinned URL, sha256-checked. The datasets are needed even though nothing is trained: the planners read the action / state normalisation and (Push-T) the start / goal states from the authors' validation split. The runner leaves the planning loop unchanged; it sets the paths, episode count and seed, records the per-episode outcome the authors' evaluator returns, and renames the rollout videos to `<task>--ep<k>-<success|failure>.mp4` (jepa-wms: `<task>--<model>--ep<k>-...`).

| | `dino-wm` | `jepa-wms` |
|---|---|---|
| Code | [`gaoyuezhou/dino_wm`](https://github.com/gaoyuezhou/dino_wm) @`0a9492fa` (MIT); `plan.py` `planning_main()` with `conf/plan_<env>.yaml` | [`facebookresearch/jepa-wms`](https://github.com/facebookresearch/jepa-wms) @`13cf1d9c` (CC BY-NC 4.0); `evals.main` (`simu_env_planning`) with the released `configs/evals/simu_env_planning/<env>/<model>/*.yaml` |
| Checkpoints | OSF project `bmw48` `checkpoints/outputs.zip` (953 MB, sha256 `425b0d8c…`): `outputs/pusht`, `outputs/wall_single` (+ `point_maze`, not run) | Hugging Face `facebook/jepa-wms` @`9b9c41ef` (not gated): `jepa_wm_{pusht,wall}.pth.tar`; baseline `dino_wm_{pusht,wall}.pth.tar` |
| Datasets | OSF `datasets/pusht_noise.zip` (2.8 GB) and `wall_single.zip` (1.7 GB, **59 GB unpacked**) | the same two files (the README says the Hugging Face copies are DINO-WM's "without modification"; the HF dataset is gated, OSF is not) |
| Planner | MPC + CEM: horizon 5, 300 samples, top 30, 30 (Push-T) / 10 (Wall) iterations, 5 actions per replan, frameskip 5 | CEM L2: horizon 6, 300 samples, 10 elites, 30 iterations, 6 actions stepped (one plan per episode), frameskip 5 |
| Goals | Push-T: a validation-set state 25 steps ahead; Wall: random states | same |
| Episodes | 20 per environment, seed 99 (config default; episode i = seed · i + 1) | 20 per task, `meta.seed` 1 (config default) |
| Video | the authors' 2 × 2 layout: top row the simulator rollout, bottom row the world model's decoded prediction, goal image in the right column | simulator frames of the executed plan |
| Python / torch | 3.9 / torch 2.3.0 (the authors' `environment.yaml` pins) | 3.10 / torch 2.7.0 (`pyproject.toml`; no upstream lock, resolved `--exclude-newer 2026-04-12`) |
| Published (paper) | Table 1, MPC CEM, 50 pairs: Push-T **0.90**, Wall **0.96** (PointMaze 0.98) | Table 2, CEM L2, mean (std) over 3 seeds × 96 ep: Push-T JEPA-WM **70.2 (2.8)** / DINO-WM 66.0 (4.7); Wall JEPA-WM **78.8 (3.9)** / DINO-WM 64.1 (4.6) |

- **Changes to the authors' runs.** `dino-wm`: 20 instead of 50 pairs, and MPC capped at `planner.max_iter=5` replans (125 simulator steps; Push-T goals are 25 steps away). The released configs set `max_iter: null`, which replans until every episode has succeeded (`planning/mpc.py`) and so never ends while one keeps failing. `jepa-wms`: 20 instead of 96 episodes; two config entries point at files that are not released and are changed — the state-readout head (`heads_cfg.pretrain_dec_path.state_head` = `${JEPAWM_LOGS}/...`, optional plots only) is dropped and the 3.6 GB VM2M image decoder (decodes the planner's imagined frames) is off, so `decode_each_iteration` and `optional_plots` are false. The planning loop and its success test are unchanged.
- **PointMaze is not run.** Its env needs `d4rl` + `mujoco-py` (MuJoCo 2.1 binaries and a Cython build). dino_wm's `env/__init__.py` imports it for every environment, so the runner registers two empty modules under those names when they are not installed; Push-T and Wall never touch them. Metaworld, RoboCasa and DROID (jepa-wms) need `metaworld` from git master, the authors' robosuite / robocasa forks with 20 GB of kitchen assets, or the 5.6 TB DROID set; the V-JEPA-2-AC baseline exists only for DROID / RoboCasa.
- **The DINO-WM baseline twice.** `dino-wm` runs the original authors' checkpoints with their own planner (MPC); `jepa-wms --task-set dino-wm-baseline` runs the jepa-wms authors' **retrained** DINO-WM with their CEM planner, which is the comparison their Table 2 makes. The two DINO-WM numbers are not the same measurement.
- **Third-party reproduction.** [jepa-wms issue #25](https://github.com/facebookresearch/jepa-wms/issues/25) (2026-03) ran the released checkpoints with the released configs: Push-T 62.5-65.6 %, Wall 75.0 %, below the paper's means.
- **Cost.** Both are fp32 and compute-bound. DINO-WM Push-T plans every MPC step for all 20 episodes at once: 30 CEM iterations × 300 samples × 5-step rollouts of a 588-token ViT predictor, about 60 TFLOP per episode-iteration, so `dino-wm` prefers the L40S of a `g6e.xlarge` (about 3× an L4 in fp32); `jepa-wms` plans once per episode and takes a `g6.xlarge`. Both need 200 GB of EBS for the unpacked Wall dataset. `--task-set probe` (2 episodes per environment) measures the time per episode before a full run.
- **Measured on AWS (ap-northeast-2, 2026-11-02).** `dino-wm` (L40S): Push-T 20/20, Wall 19/20 (run `2026-11-02-2225`; ~200 s and ~71 s per episode); `jepa-wms --task-set compare` (L4): JEPA-WM Push-T 6/10, Wall 8/10, the jepa-wms DINO-WM baseline Push-T 4/10, Wall 6/10 (run `2026-11-02-2103`; ~400 s per CEM episode). On the L40S, CUDA has to be initialised before `decord` and the env packages are imported: the first two `dino-wm` probes died with SIGSEGV inside `torch.cuda._lazy_init` when `plan.py` loaded the checkpoint (faulthandler stack), and the runner now creates the CUDA context first. See [results](results.md#latent-world-model-planners-dino-wm-jepa-wms).
- **Checked locally (CPU, aarch64, 2026-11-02).** Both checkpoint zips / files and both datasets match their sha256; the jepa-wms `jepa_wm_pusht` and `dino_wm_pusht` checkpoints load with "All keys matched successfully" (epoch 50) and the Push-T dataset loads (18,685 train / 21 validation trajectories); the dino_wm planning loop runs end to end on CPU through the runner for Push-T and Wall (2 episodes each with the planner cut to 4 samples, 1 CEM iteration, 1 MPC step: results.json, renamed videos, about 13-15 minutes each; the success rates of such a cut planner mean nothing). The jepa-wms episode loop is CUDA-only (`planning/gc_agent.py` builds a `cuda:0` generator), so it runs first on AWS.
- **Licenses.** `jepa-wms` is **non-commercial** (CC BY-NC 4.0 for the code, the checkpoints and the re-hosted data). `dino-wm` is MIT; DINOv2 code and weights are Apache-2.0. See [licenses](licenses.md).
- **VLA-JEPA** (`ginwind/VLA-JEPA`, a LIBERO / SimplerEnv VLA with a JEPA objective) is a VLA, not a planner: it is the separate target `vla-jepa` ([Note 22](#note-22-vla-jepa-lerobot-port-on-libero)).

### Note 22: vla-jepa (LeRobot port on LIBERO)

VLA-JEPA ([arXiv 2602.10098](https://arxiv.org/abs/2602.10098); code [`ginwind/VLA-JEPA`](https://github.com/ginwind/VLA-JEPA) @`ec907778`, Apache-2.0) is a language-conditioned VLA: a Qwen3-VL-2B backbone feeds a DiT-B flow-matching action head (chunk of 7, 4 Euler steps), and a V-JEPA 2 ViT-L encoder with an action-conditioned predictor adds a latent world-model loss **during training only** (LeRobot `docs/source/vla_jepa.mdx`: "At inference time only the Qwen backbone and action head are used"). LeRobot ships the policy class (`src/lerobot/policies/vla_jepa`, extra `vla_jepa`) at the commit the shared lerobot-libero template already pins (`ae2b57f1`, [Note 6](#note-6-shared-lerobot-libero-template-xvla-smolvla-lingbot-va)), so this target is a `models/vla-jepa.yaml` on that template, not a new runner.

- **Checkpoint.** `lerobot/VLA-JEPA-LIBERO` @`735d9f69` (Apache-2.0, not gated; 6.16 GB `model.safetensors` + LeRobot processor files), converted by LeRobot from `ginwind/VLA-JEPA` @`db32b4a4` `LIBERO/checkpoints/VLA-JEPA-LIBERO.pt`. One checkpoint for all four suites: the authors' `LIBERO/config.yaml` trains on `data_mix: libero_all` for 30k steps from their DROID / SSv2 pre-training checkpoint. Inputs `observation.images.image` / `image2` (agentview + wrist) and an 8-D state = the LeRobot LIBERO env's default keys, so no camera mapping is needed.
- **Backbones pinned.** The policy constructor calls `from_pretrained` on `Qwen/Qwen3-VL-2B-Instruct` and `facebook/vjepa2-vitl-fpc64-256` by name (the checkpoint then overwrites their weights; the Qwen processor / chat template is used as is). The template downloads both at pinned revisions (`89644892`, Apache-2.0, 4.27 GB; `b3c1679b`, MIT, 1.3 GB — `original/model.pth` is skipped) and passes the snapshot paths as `--policy.qwen_model_name` / `--policy.jepa_encoder_name` (new `model.extra_downloads` key of the template; `summary.txt` records both as `stack_<field>`).
- **Settings.** As the authors' `examples/LIBERO/eval_libero.py`: LIBERO rendered at 256 × 256 (`LIBERO_ENV_RESOLUTION`, `--env.observation_height/width=256`; the policy resizes to 224), 10 settle steps and the same per-suite step caps (LeRobot's `TASK_SUITE_MAX_STEPS` is the same table), the 7-step chunk executed open loop, gripper snapped and binarised (checkpoint config), seed 7. The authors run 50 episodes per task of every suite; here one LIBERO task per entry, episodes = init states 0..n-1.
- **Tasks.** Default: LIBERO-Goal tasks 1, 6, 8 × 10 episodes (same as `univla` / `ge-act`); `--task-set actions`: tasks 0, 5, 7 × 10; `probe`: task 1 × 2 (timing); `spatial` / `object`: tasks 0, 1 × 10; `long`: LIBERO-10 tasks 2, 3 × 10. Each entry names its LIBERO task (`env_name`), checked against LIBERO's task table at render time and against the installed LIBERO on the instance, and writes `env_name`, `action_type` and the published suite number into `summary.txt`.
- **Published.** Paper Table 1 (50 episodes per task): Spatial 96.2, Object 99.6, Goal 97.2, LIBERO-10 95.8, average 97.2. LeRobot's port, `docs/source/vla_jepa.mdx` "Expected results" (10 episodes per task with `lerobot-eval`): spatial 95.0, object 100.0, goal 98.0, libero_10 93.0 (that table's success counts 93 / 100 / 98 / 96 do not all match its rates; quoted as written).
- **Newer LeRobot commits.** `vla_jepa` changes on LeRobot main after the pin (#4813 starting-noise argument, #4835 / #5098 world-predictor RoPE) touch the starting-noise API and the training-only world predictor; inference is the same.
- **Instance.** `g6.xlarge` first (Qwen3-VL-2B + V-JEPA 2 ViT-L + DiT-B in bf16, ~6 GB of weights on a 24 GB L4; the 16 GB host loads the 4.3 GB backbone and the 6.2 GB checkpoint), `.2xlarge` as capacity fallback.
- **Checked locally (CPU, aarch64, 2026-11-03).** LeRobot at `ae2b57f1` (`uv sync --locked --extra vla_jepa`, Python 3.12), the three pinned snapshots, `HF_HUB_OFFLINE=1`: the config loads with the two backbone paths as overrides, the processors are the checkpoint's (normalizer; clip, pre-snap, unnormalizer, binarize), the model builds and loads (2.77B parameters; the only unexpected keys are four tensors of the training-only predictor and `embed_tokens`, which is tied to `lm_head` in Qwen3-VL-2B and equal to it in the checkpoint), and one `select_action` on a dummy 2-view 256 × 256 observation returns a 7-D action with the gripper at -1 and 6 more queued (100 s on CPU, 223 s end to end). The LIBERO episode loop needs a GPU + EGL and runs first on AWS.
- **Measured (ap-northeast-2, `g6.xlarge`, 2026-11-03).** Default (run `2026-11-03-0339`): bowl on stove, cream cheese in bowl, bowl on plate 30/30; `--task-set actions` (run `2026-11-03-0455`): open drawer 9/10, push plate 9/10, turn on stove 10/10; `probe` 2/2. About 32-44 minutes from `deploy` to the last result. Published LIBERO-Goal: 97.2% (paper), 98.0% (LeRobot). See [results](results.md#vla-jepa-vla-jepa).
- **SimplerEnv (`lerobot/VLA-JEPA-SimplerEnv` @`8b9ec33f`) is not added.** The SimplerEnv targets here (`spatialvla`, `magma`) run each model's own SimplerEnv plugin inside a SimplerEnv checkout; LeRobot has no SimplerEnv env type at the pin, and the LeRobot port has no SimplerEnv inference wrapper. Running it would need a new policy adapter for the SimplerEnv harness (image / state preprocessing of the port, the Bridge / Fractal un-normalisation, chunk execution) — new code, not a yaml. The authors' own SimplerEnv path (`examples/SimplerEnv`, websocket policy server + SimplerEnv client, `VLA-JEPA-SimplerEnv.pt` from `ginwind/VLA-JEPA`) is the candidate for that. Paper Table 2 (visual matching): Google Robot pick 88.3, move 64.1, drawer 59.3, place 49.1 (avg 65.2); WidowX spoon 75.0, carrot 70.8, block 12.5, eggplant 70.8 (avg 57.3).

### Note 23: vlaeval (vla-evaluation-harness, run unmodified)

[vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness) (Ai2, `vla-eval` on PyPI, Apache-2.0) splits an evaluation into a model server (WebSocket + msgpack, one script per model family with a PEP 723 header) and a benchmark client that runs inside one Docker image per benchmark. It has no cloud path of its own. This target runs it **as published** on one EC2 GPU instance: nothing of the harness is copied into this repository or patched on the instance.

- **Install.** `uv venv --python 3.11` (vla-eval needs >= 3.11; the DLAMI ships 3.10), `uv pip install vla-eval==0.9.0` run from a directory outside the harness checkout (the checkout's `pyproject.toml` sets uv `exclude-newer = 2026-11-02T00:20Z`, before the 0.9.0 upload on 2026-11-03, so installing inside it finds no 0.9.0), and `git clone --branch v0.9.0` for `configs/` (not in the wheel; the commit is checked: `8798464b`). The model server's own environment is resolved by `uv` from its PEP 723 header with the harness's exclude-newer date, as `vla-eval serve` does.
- **Pins.** Benchmark image `ghcr.io/allenai/vla-evaluation-harness/libero@sha256:bc9930a7…` (8.84 GB) pulled by digest; the rendered run config names the digest, so `vla-eval run` never pulls a moving tag. Model server config `configs/model_servers/lerobot/pi05_libero.yaml` (π0.5, chunk 10). Its checkpoint `lerobot/pi05_libero_finetuned` (the Hub redirects it to `lerobot/pi05_libero_finetuned_v044`) is downloaded at revision `7152f13e` and passed with the harness's own `--arg checkpoint=<snapshot path>`, so the server does not follow the moving `main` branch.
- **Config.** The run config is the one of the measured run below, rendered per `tasks:` entry: `server.timeout: 900` (a cold server warmed up for 216 s on the first episode; the default 30 s fails it with "no EPISODE_READY"), `docker.user: host`, `recording.record_video: true` (off by default), LIBERO `params: {suite, seed: 7, num_steps_wait: 10}`. All three are documented harness config keys.
- **HF token.** π0.5 loads its tokenizer from the gated `google/paligemma-3b-pt-224`. The token in SSM `/vla-simulator/hf-token` (the same one other gated targets use) must belong to an account that accepted that license; the UserData checks access before installing anything and stops with a message naming the repo (`vlasim doctor` checks the parameter exists).
- **Results.** `assets/vlaeval/vlaeval_results.py` (stdlib) reads the harness's `<Benchmark>_<suite>_aggregate.json` and writes `task-N/summary.txt` (suite, k / n from the per-episode `metrics.success`, errored episodes counted as failures and as `n_errors`, seed, pins: `stack_vla_eval_version`, `_commit`, `_server_config`, `_benchmark_image`, `_model_server`) and the protocol card; episode videos are uploaded as `<suite>_t<NN>--episode_<k>-<success|failure>.mp4` with a caption naming the LIBERO instruction. The harness output itself (aggregate JSON, recording SQLite, per-step `.jsonl`) is kept under `task-N/vla-eval/`.
- **Adding a pair.** One `tasks:` (or `task_sets`) entry names a `vlaeval.benchmarks` image and a `vlaeval.model_servers` config; a new benchmark or server is one more block there (image + digest, or config path + checkpoint revision + gated repos). Consecutive entries with the same server reuse the running server. Other benchmarks than LIBERO also need their simulator in `repeatability.py` and a `seeds.py` note; none is added yet.
- **Batch A task sets (smoke-measured on AWS 2026-11-09).** Three more benchmarks of the harness, each with a model server it pairs with, at smoke sizes. All three images are the `0.9.0` tag (= `latest`) of the harness's own `ghcr.io/allenai/vla-evaluation-harness/*` packages, anonymously pullable and pinned by digest (read 2026-11-09; no mirror needed).
  - `--task-set libero-mem`: LIBERO-Mem (`libero-mem@sha256:5f78cc5e…`, 5.57 GB compressed), 2 tasks × 3 episodes, seed 7, with the default π0.5 LIBERO server **zero-shot** (the harness has no LIBERO-Mem model; the score is a pipeline check, not comparable to anything). Same `suite` / `seed` params as LIBERO.
  - `--task-set calvin`: CALVIN ABC→D (`calvin@sha256:6d018bfe…`, 2.56 GB), 5 sequences of 5 chained subtasks, seed 0, with `configs/model_servers/xvla/calvin.yaml`; checkpoint `2toINF/X-VLA-Calvin-ABC_D` (ungated, Apache-2.0, 3.52 GB, self-contained with its tokenizer and remote code) downloaded at `d76710ee` (= `main`, 2025-11-15) and passed as `--arg model_path=<snapshot path>` (`xvla.py` loads it with `from_pretrained`). The server tells the benchmark its observation params (wrist image, state, absolute actions, 720-step subtasks) over the handshake, as in the harness's reproduction (4.30 vs 4.43 published). `summary.txt` adds `calvin_avg_len` (the CALVIN headline = the harness's `mean_completed_subtasks`), `calvin_chain_<i>_of_5` and `harness_mean_completed_subtasks`; on the harness's own 1000-sequence X-VLA aggregate (`docs/reproductions/data/xvla-calvin/merged.json`, a test fixture) the adapter gives 4.301 and 0.956 / 0.918 / 0.871 / 0.817 / 0.739, its published table.
  - `--task-set kinetix`: Kinetix (`kinetix@sha256:a8f6565d…`, 5.50 GB), 4 of the 12 RTC levels × 5 rollouts, symbolic observations, 256 steps, action noise 0.1, seed 0, with `configs/model_servers/rtc/rtc.yaml` (RTC soft masking, sync, d = 0, s = 1; the paper's 12-level average at d = 0 is 0.922, a 4-level subset is not comparable). No Hugging Face checkpoint: the server fetches the public `gs://rtc-assets/bc/31/policies/worlds_l_<level>.pkl` (12 × 12,290,914 bytes, md5-checked against the bucket listing; read 2026-11-09: `car_launch` `zNOPr17BVXGqhTp1n2tY3w==`, `cartpole_thrust` `WzmRpTOO93OQAwafeF953g==`, `catapult` `8Uqj/WhAdxIXlyGlx4cGHg==`, `catcher_v3` `D9l/UYq5yV5simp3718YKQ==`, `chain_lander` `su+p9E55rStmT1KwlAihDg==`, `grasp_easy` `VNLdwqohsp052EKbU9LLDg==`, `h17_unicycle` `k1luKyXXHxX5qJQGvU1EuQ==`, `hard_lunar_lander` `YiRqlmUrrUigk18/E9Uc9A==`, `mjc_half_cheetah` `IqcFW66s61T//jtgZ67WXg==`, `mjc_swimmer` `gmpn6oFfdzYKX819ifMmug==`, `mjc_walker` `OvP5USPNqZnujUM7ykpN8g==`, `trampoline` `SoHXZLt0ETGnsGKWlP/62A==`) and clones `Physical-Intelligence/real-time-chunking-kinetix` at `9296f31d` (pinned in the harness's `rtc.py`); the bucket objects are not version-pinned here. Episodes record no video (symbolic observations have no frame), so this benchmark sets `expect_videos: false`.
  - `--task-set batch-a`: the three entries above on one instance, in that order; each task starts its own server (the previous one is stopped and the next health check waits until nothing answers on :8000 any more). Estimate ~60 min launch to terminate, ~US$2.7 on `g6e.2xlarge` in `ap-northeast-2` (separately ~25-30 min, ~US$1.1-1.4 each).
  - **Per-benchmark keys** of `vlaeval.benchmarks` (all optional; defaults = the LIBERO behaviour, so the default run renders the same run config): `suite_param` / `seed_param` name the constructor arg that gets the entry's suite / seed (`null` = none is injected: the harness builds `Benchmark(**params)` without filtering, so CALVIN and Kinetix would fail on `suite`); `docker_env` (`NAME=value` list → `docker.env` of the run config: CALVIN `NVIDIA_DRIVER_CAPABILITIES=all` as the harness's own config; Kinetix `XLA_PYTHON_CLIENT_PREALLOCATE=false`, because the container's JAX and the RTC server's JAX share the GPU and each preallocates 75 % by default); `expect_videos: false`; `init_states` (the `summary.txt` text). Benchmark param values may now be one token of `[A-Za-z0-9_./:-]` up to 256 characters (paths, `module:Class` strings); they reach the instance only inside the tasks JSON and are written to the run config by `jq`, never put on a command line. Lists are still rejected.
  - **Measured** (`--task-set batch-a`, `g6e.2xlarge`, `ap-northeast-2`, driver 570.133.20, 2026-11-09). Run `2026-11-09-0712` (the shipped UserData): LIBERO-Mem **1/6**, CALVIN **4/5** sequences (avg len **4.00**; chain 1-of-5 … 5-of-5 over 5 sequences, not comparable to the paper's 4.43 over 1000), Kinetix **19/20** (4-level subset), all `eval_exit` 0, ~38 min launch to terminate, ~US$1.75. Earlier runs of the same sets: LIBERO-Mem 2/6 (`2026-11-09-0401`) and 0/6 (`2026-11-09-0617`, with a blanket torch constraint that also moved the LeRobot server off its published 2.11.0+cu128 to 2.10.0), CALVIN 3/5 avg len 3.60 and Kinetix 19/20 (`2026-11-09-0617`). LIBERO-Mem at 6 zero-shot episodes swings between 0 and 2: read it as a pipeline check.
  - **CUDA 12 drivers and open torch pins (measured).** `xvla.py` declares `torch>=2.2`; uv resolves torch 2.15, whose PyPI wheels are built for CUDA 13, and the X-VLA server died at model load on the DLAMI's 570 driver ("The NVIDIA driver on your system is too old", run `2026-11-09-0401`). When the driver is older than 580, the UserData now runs `uv lock --script` on each server script the run uses and adds `constraint-dependencies = ["torch<2.11"]` (torch 2.10.0+cu128) to the PEP 723 header **only** where the lock pulls `cu13` wheels; `lerobot.py` resolves 2.11.0+cu128 on its own and stays as published. Every `summary.txt` records it as `stack_torch_constraint` (`torch<2.11 (xvla.py)` in batch A, `none` on drivers ≥ 580). This is the one edit the stack makes to the harness checkout.
  - **A model server that does not start fails its own tasks only.** The run marks those tasks failed, uploads `server.log` under the task prefix and goes on with the next server (before, the whole run stopped and Kinetix never ran). The run still signals failure at the end.
  - **Risks (not measured).** `repeatability.py` and `seeds.py` still describe the LIBERO default for the whole target.
- **Measured by hand (before this target existed), 2026-11-08, `ap-northeast-2`, `g6e.2xlarge` (1 × L40S), "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)" 20261106 (driver 595), driven over SSM.** LIBERO-Spatial 10 tasks × 5 episodes, seed 7: **50/50** in 453 s of evaluation (8.9 s per episode); install 0.3 min, image pull 2.7 min, server load 2.0 min, first episode 3.6 min, **~36 min launch to terminate, ~US$1.65** (US$2.75652 per hour). The same run sharded over 4 processes (`run_sharded.sh -n 4`) gave 48/50 in 159 s (task 4, episodes 0 and 2; one run cannot tell chance from sharding). LeRobot's published LIBERO-Spatial number for this checkpoint: 97.0 (50 episodes per task). This target reproduces that recipe.
- **Measured with this target, 2026-11-09, `ap-northeast-2`, `g6e.2xlarge`** (the PyTorch DLAMI the other Docker targets use, driver 570.133.20; run `2026-11-09-0136`): **49/50** (task 3, episode 2 missed), 964 s of evaluation including the cold first episode, **28 min launch to terminate, ~US$1.30**, 50 videos in the gallery. A second run on the renamed stack `VLAEval-Demo` (run `2026-11-09-0239`): **48/50**, 27 min, ~US$1.22. LeRobot publishes 97.0% for this checkpoint (50 episodes per task); 97/100 over the two runs.
- **Licenses.** The harness is Apache-2.0. LIBERO is MIT; the LIBERO image is built and published by the harness. `lerobot/pi05_libero_finetuned` carries the Gemma terms on its card and needs the gated PaliGemma tokenizer. Each keeps its own license ([licenses](licenses.md)).

## Task sets and action types

Most benchmarks default to pick-and-place, but several pinned checkpoints were trained (and are evaluated upstream) on other motions too. `--task-set NAME` runs `task_sets.NAME` of `models/<target>.yaml` instead of its `tasks:` list, on the same stack, results bucket, template and checkpoint revision; `actions` is the set of those non-pick-and-place tasks:

```bash
python vlasim.py deploy --vla rldx-kitchen --task-set actions       # doors, drawers, stove knob, faucet, coffee button
python vlasim.py deploy --vla rldx-simpler --task-set actions       # Google robot: open / close drawer
python vlasim.py deploy --vla gr00t --task-set actions              # LIBERO-Goal: open drawer, turn on stove, push plate
python vlasim.py deploy --vla lingbot-vla2 --task-set actions       # RoboTwin: handover, lift, bell, switch, alarm, hammer, stamp
python vlasim.py deploy --vla smolvla-robotwin --task-set actions   # same 7 RoboTwin tasks
python vlasim.py deploy --vla lerobot-gym --task-set aloha          # ACT: ALOHA bimanual insertion, cube transfer
python vlasim.py deploy --vla spatialvla --task-set actions         # SimplerEnv Google robot: open / close the top drawer
python vlasim.py deploy --vla magma --task-set actions              # SimplerEnv Google robot: open the drawer, put the apple in
python vlasim.py deploy --vla univla --task-set actions             # LIBERO-Goal: open the drawer, push the plate, turn on the stove
python vlasim.py deploy --vla vla-jepa --task-set actions           # same LIBERO-Goal tasks (VLA-JEPA LeRobot port)
python vlasim.py deploy --vla ge-act --task-set actions             # same three LIBERO-Goal tasks
```

`lerobot-gym` has no pick-and-place task at all, so its sets split it by simulator (`aloha`, `pusht`) instead of having an `actions` set.

`--task-set` cannot be combined with `--libero-variant plus`, `--libero-objects real2sim`, `--robotwin-protocol` (each of those replaces the task list itself), `--collect`, `--bridge` or `--judge`. A set may override model keys for its run (`gr00t`'s set switches `hf_subfolder` to `libero_goal`, the same mechanism `--libero-variant plus` uses). Module: [`task_sets.py`](../task_sets.py); tests: `tests/test_task_sets.py`.

**Action type.** Every task carries one tag. The yaml can set `action_type` per task; otherwise it comes from the task name (`task_sets.classify`, an exact-name table for the known RoboTwin / RoboCasa tasks plus verb rules). A task that chains a pick-and-place with opening, closing or turning a fixture is tagged by the fixture motion. The `gr00t`, `rldx*` and RoboTwin templates write `action_type:` into `summary.txt`; `vlasim results` carries it into its per-task JSON rows and the CSV / JSON table (`action_types`), derives it from the task name for older runs, and takes `--action-type TAG` and `--per-task`. GR00T rollout videos are named `<task>--<uuid>_s<0|1>.mp4` (they were `<uuid>_s<0|1>.mp4`); the other templates already put the task in the file name.

| Tag | Meaning |
|---|---|
| `pick-place` | grasp an object and put it somewhere |
| `articulated` | open / close / turn a jointed fixture: door, drawer, knob, faucet |
| `press` | press or tap a button, bell or switch |
| `push` | move an object by pushing it, without grasping |
| `bimanual` | two arms on one object (handover, two-handed lift) |
| `tool-use` | grasp a tool and act on another object with it (hammer, stamp) |
| `locomotion` | the robot walks or drives as part of the task |
| `navigation` | move a point agent / mobile base to a goal position, no object contact (2D mazes, Wall) |

| Target | `actions` task | Tag | Why it is in the checkpoint's training / eval set |
|---|---|---|---|
| `rldx-kitchen` | `OpenSingleDoor`, `OpenDrawer`, `CloseDrawer` | articulated | RLWRLD `run_scripts/eval/robocasa_kitchen/eval_robocasa.sh` lists all 24 tasks for `RLDX-1-FT-ROBOCASA` (24-task average 70.6%, no per-task rate published) |
| | `TurnOnStove`, `TurnOnSinkFaucet` | articulated | same list |
| | `CoffeePressButton` | press | same list |
| `rldx-simpler` | `google_robot_open_drawer`, `google_robot_close_drawer` | articulated | RLWRLD `eval_simpler.sh google_vm` runs them with `RLDX-1-FT-SIMPLER-GOOGLE` (Google-VM average 81.5% over the 4 tasks); registered by `register_simpler_envs()` |
| `gr00t` | `open_the_middle_drawer_of_the_cabinet`, `turn_on_the_stove` | articulated | `libero_goal` checkpoint of `nvidia/GR00T-N1.7-LIBERO` (same revision); Isaac-GR00T `examples/LIBERO` reports Goal 97.5% |
| | `push_the_plate_to_the_front_of_the_stove` | push | same |
| `smolvla-robotwin`, `lingbot-vla2` | `handover_block`, `lift_pot` | bimanual | RoboTwin 2.0 tasks; the default subset already runs them (measured 2026-10-30, [results](results.md#expected-results)) |
| | `click_bell`, `turn_switch`, `click_alarmclock` | press | `click_bell` in the default subset; `turn_switch` / `click_alarmclock` are in the 50-task suite both checkpoints were trained on |
| | `beat_block_hammer`, `stamp_seal` | tool-use | default subset |
| `lerobot-gym` (default run) | `AlohaInsertion-v0`, `AlohaTransferCube-v0` | bimanual | single-task ACT checkpoints trained on these sims (human demos); [Note 17](#note-17-lerobot-gym-act-on-aloha-diffusion-policy-on-push-t) |
| | `PushT-v0` | push | single-task Diffusion Policy checkpoint trained on Push-T |
| `spatialvla` | `OpenTopDrawerCustomInScene-v0`, `CloseTopDrawerCustomInScene-v0` | articulated | `scripts/drawer_visual_matching.sh` of the authors' harness; README Google "Open/Close Drawer" 59.3% (visual matching, zero-shot) |
| `magma` | `PlaceIntoClosedTopDrawerCustomInScene-v0` | articulated | Magma's own `put_drawer_visual_matching.sh` (the drawer is opened before the apple goes in) |
| `univla`, `ge-act` | `open_the_middle_drawer_of_the_cabinet`, `turn_on_the_stove` | articulated | LIBERO-Goal tasks 0 and 7; both checkpoints are trained on all of LIBERO-Goal ([Note 20](#note-20-libero-with-the-authors-own-loop-univla-ge-act)) |
| | `push_the_plate_to_the_front_of_the_stove` | push | LIBERO-Goal task 5 |
| `vla-jepa` | `open_the_middle_drawer_of_the_cabinet`, `turn_on_the_stove` / `push_the_plate_to_the_front_of_the_stove` | articulated / push | same LIBERO-Goal tasks 0, 7 / 5; one checkpoint trained on all four suites ([Note 22](#note-22-vla-jepa-lerobot-port-on-libero)) |
| `dino-wm`, `jepa-wms` (default run) | `pusht` | push | goal-image planning, not language ([Note 21](#note-21-latent-world-model-planners-dino-wm-jepa-wms)); task sets `pusht`, `wall`, `probe` (+ `dino-wm-baseline`, `compare` for `jepa-wms`) |
| | `wall` | navigation | same |

The default `gr00t` LIBERO-10 tasks (stove + moka pot, bowl into drawer + close) and the `rldx` / `rldx-gr1` microwave and drawer tasks are tagged `articulated` too: the [showcase](showcase.md) groups its clips by these tags.

## Post-run judge (Flora-Rewarder)

`flora-judge` is **not a `--vla` target**: it scores the rollout videos of a run that already
finished. `python vlasim.py judge --vla <target> [--run RUN_ID]` (= `python deploy.py --judge --vla
<target> ...`) launches one GPU instance that reads `s3://<results bucket of the target>/<RUN_ID>/task-N/videos/`,
scores every episode with NVIDIA [Flora-Rewarder](https://huggingface.co/nvidia/Flora-Rewarder)
(a Qwen3.5-27B video reward model, bf16 ~54 GB, pinned `@031299e4`), writes `judge.json` + `judge.txt`
next to that run, e-mails a summary (SNS) and terminates itself. Without `--run` it takes the newest
run that has videos.

| Judge | Model | Output per episode | Stack | GPU instance |
|---|---|---|---|---|
| `flora-judge` | Flora-Rewarder (NVIDIA, **NC**) outcome reward | integer score 1 (complete failure) to 5 (complete success) + one-line rationale | `FloraJudge-<target stack>` (e.g. `FloraJudge-SmolVLA-Demo`) | `g7e.2xlarge` |

- **Scoring code is NVIDIA's own.** The instance downloads the model repository at the pinned
  revision and runs its `batch_utils.terminal_reward --with-rationale` (the official outcome prompt,
  up to 64 uniformly sampled frames, structured-output parser) with the pinned
  `batch_utils/requirements.txt` (`vllm==0.24.0`, `transformers==5.8.1`, ...). Nothing from that
  repository is vendored here. The rollout mp4 is passed as one video (it is already the target's
  camera mosaic).
- **Flora success = score >= 4** (`model.success_threshold` in `models/flora-judge.yaml`). 3 is the
  model's midpoint and counts as failure. `judge.json` also keeps the raw score, so any threshold
  can be applied afterwards.
- **What it needs per episode.** The task instruction, from (in order) the video's `.txt` sidecar
  (`Task:` line), the file name (OpenVLA-OFT style `--task=...`, openpi style `rollout_<task>_success.mp4`), or, for the LeRobot targets
  (`xvla`, `smolvla`, `lingbot-va`, `fastwam`), LIBERO's task map (pinned commit) indexed by the
  suite and task id in `eval_info.json`. Instructions cut to 40 characters in OpenVLA-OFT style
  file names are completed from the same task map when the prefix is unique. The simulator's success flag comes from `success=True|False`
  in the file name, the sidecar `result:` line or `eval_info.json`. An episode without an
  instruction is not scored; one without a sim flag is scored but left out of the agreement.
  LeRobot runs recorded before this version named videos `eval_episode_N.mp4` without the task
  prefix (and lost videos when a task entry had several task ids), so their episodes cannot be
  matched to `eval_info.json`; re-run the target to judge them.
- **Instance.** `g7e.2xlarge` (1 x RTX PRO 6000 Blackwell 96 GB) → `g7e.4xlarge` → `g6e.12xlarge`
  (4 x L40S 48 GB, tensor parallel 4) → `p5.4xlarge` (1 x H100 80 GB, the model card's test GPU).
  The judge uses the **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)**, whose driver
  supports G7e; the policy targets keep their PyTorch 2.4.1 DLAMI. All five supported regions,
  including `ap-northeast-2`, offer `g7e`.
- **Measured (2026-10-28, `ap-northeast-2`).** `g7e.2xlarge` and `g7e.4xlarge` had no capacity in
  either Seoul AZ (nor in any `us-west-2` AZ, together with `g6e.12xlarge` and `p5.4xlarge`), so both
  runs used the `g6e.12xlarge` fallback (tensor parallel 4). Instance wall time 13-15 min
  (weights 52 GB in ~7 min, vLLM install ~1 min, scoring 263 s for 10 episodes / 365 s for 47).
  | judged run | episodes | sim success | agreement | confusion (sim ok/flora ok, ok/fail, fail/ok, fail/fail) |
  |---|---|---|---|---|
  | `cosmos-policy` LIBERO-Spatial `2026-10-17-0254` | 10 | 100% | 90% (9/10) | 9 / 1 / 0 / 0 |
  | `pi` (π0.5) four LIBERO suites `2026-10-15-0948` | 47 | 85% | 74.5% (35/47) | 32 / 8 / 4 / 3 |

  Read the disagreements as episodes to look at, not as judge errors: LIBERO stops at the first
  success frame, so Flora often gives 4 ("bowl still held by the gripper") instead of 5, and its
  rationales flag sim successes that look partial on video (wrong compartment, one of two objects,
  swapped mugs) as well as sim failures that look complete on video.
- **Permissions.** The instance role can list and read only `<RUN_ID>/*` of the judged bucket and
  write only `<RUN_ID>/judge*` and the run's video captions (`task-*/videos/*.txt`, where the score
  is appended for the [video history](gallery.md)).
- **License.** Flora-Rewarder is under the **NVIDIA OneWay Noncommercial License (22 Mar 2022)** —
  research and development only, not for commercial or production use (model card). Base model
  Qwen3.5-27B is Apache-2.0. The weights are not gated on Hugging Face (no token needed).

```bash
python vlasim.py judge --vla smolvla                        # newest smolvla run with videos
python vlasim.py judge --vla openvla-oft --libero-suite spatial --run 2026-10-19-0558
aws logs tail /flora-judge/userdata --follow --region ap-northeast-2
python vlasim.py results --vla smolvla                     # prints the judge agreement below the table
python vlasim.py destroy --judge --vla smolvla              # delete the judge stack; judge.json stays
```

The judge stack has no bucket: `judge.json` lives with the run in the target's retained results
bucket, so `destroy.py --judge` only removes the VPC, SNS topic and Lambda. The judged target's stack
does not have to exist any more (its bucket is retained); the gallery score badge needs it.

## GPU instances per target

`instance.preferred` in `models/<target>.yaml` is tried in order; the AZ selector falls back through the list (and across AZs) when capacity is short. `ebs_gb` is the root gp3 volume. Prices per instance are in [cost.md](cost.md).

| `--vla` | `instance.preferred` (first choice first) | EBS (GB) |
|---|---|---|
| `gr00t` | `g6.12xlarge` → `g5.12xlarge` → `g6.xlarge` → `g5.xlarge` | 400 |
| `gr00t-bridge` | `g6.2xlarge` → `g5.2xlarge` → `g6.xlarge` → `g5.xlarge` | 200 |
| `gr00t-gr1` | `g6.12xlarge` → `g5.12xlarge` → `g6.xlarge` → `g5.xlarge` | 400 |
| `gr00t-g1` | `g6.12xlarge` → `g5.12xlarge` → `g6.xlarge` → `g5.xlarge` | 500 |
| `pi` | `g5.xlarge` → `g5.2xlarge` → `g6.xlarge` → `g6.2xlarge` | 200 |
| `openvla-oft` | `g6.xlarge` → `g5.xlarge` → `g6.2xlarge` → `g5.2xlarge` | 250 |
| `lap` | `g6.xlarge` → `g6.2xlarge` → `g5.xlarge` → `g5.2xlarge` | 200 |
| `molmoact2` | `g6e.2xlarge` → `g6e.4xlarge` | 250 |
| `xvla` | `g6.2xlarge` → `g5.2xlarge` → `g6.xlarge` → `g5.xlarge` | 150 |
| `smolvla` | `g6.xlarge` → `g5.xlarge` → `g6.2xlarge` → `g5.2xlarge` | 120 |
| `lingbot-va` | `g6e.2xlarge` → `g6e.4xlarge` | 250 |
| `vla-adapter` | `g6.xlarge` → `g5.xlarge` → `g6.2xlarge` → `g5.2xlarge` | 150 |
| `fastwam` | `g6e.2xlarge` → `g6e.4xlarge` | 250 |
| `cosmos-policy` | `g6e.xlarge` → `g6e.2xlarge` → `g6.2xlarge` → `g5.2xlarge` | 300 |
| `aspire` | `g6e.2xlarge` → `g6e.4xlarge` | 250 |
| `real2sim` | `g5.2xlarge` → `g6.2xlarge` → `g5.4xlarge` → `g6.4xlarge` | 150 |
| `rldx` | `g6.xlarge` → `g6.2xlarge` → `g5.xlarge` → `g5.2xlarge` | 150 |
| `rldx-simpler` | `g6.2xlarge` → `g5.2xlarge` → `g6.xlarge` → `g5.xlarge` | 200 |
| `rldx-bridge` | `g6.2xlarge` → `g5.2xlarge` → `g6.xlarge` → `g5.xlarge` | 200 |
| `rldx-gr1` | `g6.2xlarge` → `g5.2xlarge` → `g6.xlarge` → `g5.xlarge` | 200 |
| `rldx-kitchen` | `g6.2xlarge` → `g5.2xlarge` → `g6.xlarge` → `g5.xlarge` | 220 |
| `openarm-lift-act` | `g6.12xlarge` → `g5.12xlarge` → `g6.24xlarge` → `g5.24xlarge` → `g6e.12xlarge` → `g6e.24xlarge` | 300 |
| `openarm-isaac` | `g6.12xlarge` → `g5.12xlarge` → `g6.2xlarge` → `g5.2xlarge` | 500 |
| `robolab-cosmos3` | `edge` (default): `g6e.2xlarge` → `g6e.4xlarge`; `nano`: `g6e.12xlarge` → `g6e.24xlarge`; `edge-full`: `g6e.48xlarge` | 600 / 700 / 1000 |
| `robolab-pi05` | `subset` (default): `g6e.2xlarge` → `g6e.4xlarge`; `full`: `g6e.48xlarge` | 500 / 900 |
| `smolvla-robotwin` | `g6.2xlarge` → `g5.2xlarge` → `g6.4xlarge` → `g5.4xlarge` | 200 |
| `lingbot-vla2` | `g6e.2xlarge` → `g6e.4xlarge` | 250 |
| `lerobot-gym` | `g6.xlarge` → `g5.xlarge` → `g6.2xlarge` → `g5.2xlarge` | 100 |
| `multitask-dit` | `g6.xlarge` → `g5.xlarge` → `g6.2xlarge` → `g5.2xlarge` | 120 |
| `spatialvla` | `g6.xlarge` → `g5.xlarge` → `g6.2xlarge` → `g5.2xlarge` | 120 |
| `magma` | `g6.2xlarge` → `g5.2xlarge` → `g6e.xlarge` → `g6.4xlarge` | 150 |
| `univla` | `g6e.xlarge` → `g6e.2xlarge` (24 GB L4 ran out of memory, measured) | 150 |
| `ge-act` | `g6.2xlarge` → `g5.2xlarge` → `g6e.xlarge` → `g6.4xlarge` | 150 |
| `dino-wm` | `g6e.xlarge` → `g6.2xlarge` → `g5.2xlarge` → `g6e.2xlarge` (fp32 CEM / MPC; 32 GB host for 20 env subprocesses) | 200 |
| `jepa-wms` | `g6.xlarge` → `g5.xlarge` → `g6.2xlarge` → `g6e.xlarge` | 200 |
| `vla-jepa` | `g6.xlarge` → `g5.xlarge` → `g6.2xlarge` → `g5.2xlarge` | 150 |
| `vlaeval` | `g6e.2xlarge` → `g6e.4xlarge` → `g6e.xlarge` (π0.5 server + LIBERO container on one L40S, measured) | 250 |
| `rlinf-oft-libero` | `smoke` (default, 1 GPU): `g6e.8xlarge` → `g6e.4xlarge` → `g6e.16xlarge` → `p5.4xlarge`; `full` (8 GPU): `p5.48xlarge` → `p4de.24xlarge` → `p5en.48xlarge` | 500 / 1500 |
| `flora-judge` ([post-run judge](#post-run-judge-flora-rewarder)) | `g7e.2xlarge` → `g7e.4xlarge` → `g6e.12xlarge` → `p5.4xlarge` | 250 |

## Deploy commands

Each target is one `vlasim deploy --vla <target>` (which runs the `doctor` preflight,
then forwards every flag to `deploy.py` unchanged). The `--email` is only needed the
first time, or if you skipped `init`.

```bash
# GR00T N1.7 — LIBERO kitchen tasks (~120 min)
python vlasim.py deploy --vla gr00t --email you@example.com

# GR00T N1.7 — SimplerEnv Bridge (WidowX 250, 4 tasks x 10 episodes, ~90-150 min, local mode only)
python vlasim.py deploy --vla gr00t-bridge --email you@example.com

# GR00T N1.6 + GR1 humanoid — RoboCasa tabletop tasks (~90 min)
python vlasim.py deploy --vla gr00t-gr1 --email you@example.com

# GR00T N1.6 + Unitree G1 — whole-body loco-manipulation (GR00T-WholeBodyControl, ~20-30 min)
python vlasim.py deploy --vla gr00t-g1 --email you@example.com

# π0.5 — LIBERO spatial + object tasks (~4 hrs)
python vlasim.py deploy --vla pi --email you@example.com

# OpenVLA-OFT — LIBERO-10 long-horizon (~3 hrs, local mode only; default suite)
python vlasim.py deploy --vla openvla-oft --email you@example.com

# OpenVLA-OFT — LIBERO-Spatial short-horizon (~1.5 hrs)
python vlasim.py deploy --vla openvla-oft --libero-suite spatial --email you@example.com

# Other short-horizon suites: object, goal
python vlasim.py deploy --vla openvla-oft --libero-suite object --email you@example.com
python vlasim.py deploy --vla openvla-oft --libero-suite goal   --email you@example.com

# LAP-3B — LIBERO-Spatial (zero-shot cross-embodiment VLA, JAX policy server, ~1.5-2.5 hrs)
python vlasim.py deploy --vla lap --email you@example.com

# MolmoAct2 (Ai2) — LIBERO-Goal (Apache-2.0 5B Action Reasoning Model, fp32 in-process, g6e GPU ≥48GB, ~40-60 min)
# Use --region us-west-2 (g6e capacity); us-east-1 currently has no g6e capacity.
python vlasim.py deploy --vla molmoact2 --email you@example.com --region us-west-2

# LeRobot × LIBERO family (shared template; Apache-2.0)
python vlasim.py deploy --vla xvla --email you@example.com         # X-VLA-0.9B, LIBERO-Spatial (~15-30 min)
python vlasim.py deploy --vla smolvla --email you@example.com      # SmolVLA-450M, LIBERO-Spatial (~15-30 min)
python vlasim.py deploy --vla lingbot-va --email you@example.com   # LingBot-VA world action model, LIBERO-10, g6e (~2.5 h for 6 episodes)
python vlasim.py deploy --vla vla-adapter --email you@example.com  # VLA-Adapter-Pro (MIT), LIBERO-Spatial
python vlasim.py deploy --vla fastwam --email you@example.com      # FastWAM world action model (research-only weights), LIBERO-Spatial (~30-40 min)
python vlasim.py deploy --vla cosmos-policy --email you@example.com  # Cosmos Policy (NVIDIA WAM, research-only weights; HF token must accept the Cosmos-Predict2 license), LIBERO-Spatial (~60-90 min)
python vlasim.py deploy --vla multitask-dit --email you@example.com  # LBM-style Multi-task DiT (community LIBERO-10 checkpoint, not TRI weights), 2 tasks x 5 ep
python vlasim.py deploy --vla aspire --email you@example.com         # ASPIRE agentic skill discovery (Claude Code on Amazon Bedrock; HF token must have SAM3 access), LIBERO-Spatial 2 tasks (~45 min, ~US$3 Bedrock)

# LeRobot gym sims (Apache-2.0): ACT on ALOHA (bimanual insertion, cube transfer) + Diffusion Policy on Push-T (~30-60 min, estimate)
python vlasim.py deploy --vla lerobot-gym --email you@example.com

# SimplerEnv with the authors' own eval (MIT; times measured 2026-11-02, ap-northeast-2)
python vlasim.py deploy --vla spatialvla --email you@example.com   # SpatialVLA-4B zero-shot: Google + WidowX, 93 ep (~108 min)
python vlasim.py deploy --vla magma --email you@example.com        # Magma-8B zero-shot: WidowX + Google, 68 ep (~83 min)

# LIBERO with the authors' own loop (times measured 2026-11-02, ap-northeast-2; ge-act is non-commercial: CC BY-NC-SA 4.0)
python vlasim.py deploy --vla univla --email you@example.com       # UniVLA img sft: 3 LIBERO-Goal tasks x 10 ep (~12 min, g6e.xlarge)
python vlasim.py deploy --vla ge-act --email you@example.com       # GE-Act: same tasks (~22 min)

# VLA-JEPA, LeRobot port through upstream lerobot-eval (Apache-2.0; not measured yet)
python vlasim.py deploy --vla vla-jepa --task-set probe --email you@example.com   # 1 LIBERO-Goal task x 2 ep, measures time per episode
python vlasim.py deploy --vla vla-jepa --email you@example.com     # same 3 LIBERO-Goal tasks x 10 ep as univla / ge-act

# vla-eval (allenai/vla-evaluation-harness) unmodified; needs the HF token (PaliGemma tokenizer is gated)
python vlasim.py deploy --vla vlaeval --email you@example.com      # LIBERO-Spatial x LeRobot pi0.5, 10 tasks x 5 ep (~40 min)
python vlasim.py deploy --vla vlaeval --task-set probe --email you@example.com   # 1 task x 1 ep, cold-start timing
python vlasim.py deploy --vla vlaeval --task-set batch-a --email you@example.com # LIBERO-Mem + CALVIN + Kinetix smokes (~38 min, ~US$1.75 measured)

# Latent world-model planners: goal image, not language (measured 2026-11-02; jepa-wms is non-commercial: CC BY-NC 4.0)
python vlasim.py deploy --vla dino-wm --task-set probe --email you@example.com   # 2 ep per env, measures time per episode
python vlasim.py deploy --vla dino-wm --email you@example.com       # DINO-WM MPC + CEM: Push-T, Wall, 20 ep each
python vlasim.py deploy --vla jepa-wms --email you@example.com      # JEPA-WM CEM: Push-T, Wall, 10 ep each
python vlasim.py deploy --vla jepa-wms --task-set compare --email you@example.com  # + the authors' DINO-WM baseline

# Real2Sim — capture -> 3DGS -> collision meshes -> MuJoCo grasp replay (~12 min measured, not a policy)
# Default input HOPE-Video is CC BY-NC-SA 4.0 (non-commercial); your own capture: input.* in models/real2sim.yaml
python vlasim.py deploy --vla real2sim --email you@example.com --region ap-northeast-2

# RLDX-1 (RLWRLD) — LIBERO (open VLA, MSAT/Qwen3-VL-8B, ZeroMQ server + sim client, ~80-120 min)
# NON-COMMERCIAL weights (RLWRLD Model License v1.0) — research / simulation benchmarking only.
python vlasim.py deploy --vla rldx --email you@example.com

# RLDX-1 (RLWRLD) — SimplerEnv Google-VM (real-robot OXE_FRACTAL embodiment, ~35-45 min)
# NON-COMMERCIAL weights (RLWRLD Model License v1.0) — research / simulation benchmarking only.
python vlasim.py deploy --vla rldx-simpler --email you@example.com

# RLDX-1 (RLWRLD) — SimplerEnv Bridge (WidowX 250, OXE_BRIDGE_ORIG, 4 tasks x 10 episodes, ~100-160 min)
# NON-COMMERCIAL weights (RLWRLD Model License v1.0) — research / simulation benchmarking only.
python vlasim.py deploy --vla rldx-bridge --email you@example.com

# RLDX-1 (RLWRLD) — RoboCasa GR-1 Tabletop (bimanual humanoid + waist, MuJoCo, ~25-35 min)
# NON-COMMERCIAL weights (RLWRLD Model License v1.0) — research / simulation benchmarking only.
python vlasim.py deploy --vla rldx-gr1 --email you@example.com

# RLDX-1 (RLWRLD) — RoboCasa Kitchen 24-task (PandaOmron mobile manipulator, MuJoCo, ~25-35 min)
# NON-COMMERCIAL weights (RLWRLD Model License v1.0) — research / simulation benchmarking only.
python vlasim.py deploy --vla rldx-kitchen --email you@example.com
# Same stack, non-pick-and-place tasks of the same checkpoints (see "Task sets and action types" above):
#   --task-set actions works for rldx-kitchen, rldx-simpler, gr00t, smolvla-robotwin and lingbot-vla2
python vlasim.py deploy --vla rldx-kitchen --task-set actions --email you@example.com

# OpenArm Lift-Cube — scripted ACT demo collection in Isaac Lab (HDF5 only; needs a 4-GPU instance)
python vlasim.py deploy --vla openarm-lift-act --email you@example.com

# OpenArm bimanual π0.5 (folding_latest) in Isaac Lab — experimental pipe proof, success ~0 by design (note 11)
python vlasim.py deploy --vla openarm-isaac --email you@example.com

# NVIDIA RoboLab (Isaac Lab, DROID Franka; assets CC BY-NC-SA 4.0) — 12-task subset x 5 episodes (~4-6 h, note 14)
# model.variant in models/robolab-*.yaml: nano (g6e.12xlarge) or the full RoboLab-120 protocol (g6e.48xlarge, up to 12 h)
python vlasim.py deploy --vla robolab-cosmos3 --email you@example.com   # Cosmos3-Edge-Policy-DROID
python vlasim.py deploy --vla robolab-pi05 --email you@example.com      # pi0.5-DROID
# RoboTwin 2.0 bimanual (LeRobot lerobot-eval; CuRobo planner is NON-COMMERCIAL, note 15)
python vlasim.py deploy --vla smolvla-robotwin --email you@example.com   # 5 tasks x 5 ep, clean (measured ~55 min, 48%)
python vlasim.py deploy --vla lingbot-vla2 --email you@example.com       # LingBot-VLA 2.0, same subset, g6e (measured ~49 min, 80%)
# Official protocol (49 tasks x 100 ep) in 12-hour shards; deploy.py prints N if the plan does not fit
python vlasim.py deploy --vla smolvla-robotwin --robotwin-protocol clean --robotwin-shard 1/19 --email you@example.com
# RL post-training with RLinf: eval SFT -> GRPO -> eval RL checkpoint (note 16; smoke measured 56 min, ~6 min per step)
python vlasim.py deploy --vla rlinf-oft-libero --email you@example.com                     # smoke: 1 GPU, LoRA, 5 steps (56 min measured)
python vlasim.py deploy --vla rlinf-oft-libero --rl-profile full --email you@example.com   # 8 GPU, recipe defaults, 100 steps (stack completes after the baseline)
python vlasim.py deploy --vla rlinf-oft-libero --rl-eval-run RUN_ID --email you@example.com  # re-evaluate a saved RL checkpoint

# Post-run judge: score a finished run's videos with NVIDIA Flora-Rewarder (NON-COMMERCIAL license, ~45-90 min)
python vlasim.py judge --vla smolvla --email you@example.com
```

On first deploy you will receive an SNS subscription confirmation email — **click the link** to enable notifications.

> **Direct invocation.** `vlasim deploy` is a thin wrapper — every command above also
> runs as `python deploy.py --vla <target> ...` with the same flags, skipping the
> `doctor` preflight. Use that if you've already run `doctor` and want to bypass it
> (equivalently, `vlasim deploy --skip-doctor`).

---

## Monitor (optional)

Every target streams its boot and evaluation log to the CloudWatch Logs group `/<target>/userdata` in the deploy region:

```bash
aws logs tail /<target>/userdata --follow --region us-east-1
# for example
aws logs tail /gr00t-g1/userdata --follow --region us-east-1
aws logs tail /rldx-kitchen/userdata --follow --region us-east-1
aws logs tail /flora-judge/userdata --follow --region us-east-1   # post-run judge
```

## Cleanup

```bash
python vlasim.py destroy --vla gr00t
python vlasim.py destroy --vla gr00t-bridge
python vlasim.py destroy --vla gr00t-gr1
python vlasim.py destroy --vla gr00t-g1
python vlasim.py destroy --vla pi
python vlasim.py destroy --vla openvla-oft                         # default suite (10)
python vlasim.py destroy --vla openvla-oft --libero-suite spatial  # non-default suite
python vlasim.py destroy --vla openvla-oft --libero-suite object   # non-default suite
python vlasim.py destroy --vla openvla-oft --libero-suite goal     # non-default suite
python vlasim.py destroy --vla lap
python vlasim.py destroy --vla molmoact2
python vlasim.py destroy --vla xvla
python vlasim.py destroy --vla smolvla
python vlasim.py destroy --vla lingbot-va
python vlasim.py destroy --vla vla-adapter
python vlasim.py destroy --vla fastwam
python vlasim.py destroy --vla cosmos-policy
python vlasim.py destroy --vla aspire
python vlasim.py destroy --vla real2sim
python vlasim.py destroy --vla rldx
python vlasim.py destroy --vla rldx-simpler
python vlasim.py destroy --vla rldx-bridge
python vlasim.py destroy --vla rldx-gr1
python vlasim.py destroy --vla rldx-kitchen
python vlasim.py destroy --vla openarm-lift-act
python vlasim.py destroy --vla openarm-isaac
python vlasim.py destroy --vla robolab-cosmos3
python vlasim.py destroy --vla robolab-pi05
python vlasim.py destroy --vla smolvla-robotwin
python vlasim.py destroy --vla lingbot-vla2
python vlasim.py destroy --judge --vla smolvla   # post-run judge stack of smolvla (judge.json stays with the run)
```

> `vlasim destroy` forwards to `destroy.py` unchanged — `python destroy.py --vla <target>`
> works identically if you prefer to call it directly.

The S3 results bucket is **retained** after stack deletion to preserve simulation outputs.
