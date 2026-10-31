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
- **Cost (estimate).** 70 episodes per target unless noted; hours = install + assets + eval; price = first preferred instance, us-east-1 On-Demand, plus ~US$0.3 of NAT processing for the 6.4 GB asset pack. Measured eval minutes for `smolvla` and `openvla-oft` are in [results.md](results.md#latency-and-cost).

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
python vlasim.py deploy --vla aspire --email you@example.com         # ASPIRE agentic skill discovery (Claude Code on Amazon Bedrock; HF token must have SAM3 access), LIBERO-Spatial 2 tasks (~45 min, ~US$3 Bedrock)

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
