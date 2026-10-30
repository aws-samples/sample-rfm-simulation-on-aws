# Supported targets

Back to the [README](../README.md) · [Documentation index](README.md)

Each target is one `--vla <target>` value for `python vlasim.py deploy` / `destroy` (and `deploy.py` / `destroy.py`). The table lists the model, simulator, robot and CloudFormation stack name; numbered notes below the table carry the license, pinning and validation detail for each target. Sample clips are in [showcase.md](showcase.md) and measured success rates in [results.md](results.md).

## Target table

| `--vla` | `--libero-suite` | Model | Sim Environment | Robot | Stack |
|---------|------------------|-------|----------------|-------|-------|
| `gr00t` | — | GR00T N1.7-LIBERO | LIBERO-10 kitchen tasks | Franka Panda (7-DOF) | `GR00T-Demo` |
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
| `rldx-gr1` | — | RLDX-1 (RLWRLD MSAT / Qwen3-VL-8B, eager)[¹](#note-1-rldx-1-weights-license) [³](#note-3-rldx-gr1) | RoboCasa GR-1 Tabletop (24-task) | GR-1 humanoid (bimanual + waist) | `RLDX-GR1-Demo` |
| `openarm-isaac` | — | π0.5 (LeRobot pi05 `folding_latest`, 16-DOF bimanual), pipe-proof only[¹¹](#note-11-openarm-isaac-experimental) | Isaac Lab bimanual Reach (empty scene) | OpenArm (bimanual, 2 × 7-DOF + grippers) | `OpenArm-Isaac-Demo` |
| `robolab-cosmos3` | — | NVIDIA Cosmos3-Edge-Policy-DROID (4B world action model; Cosmos3-Nano-Policy-DROID 16B via `model.variant: nano`), cosmos-framework policy server[¹³](#note-13-robolab) | NVIDIA RoboLab (Isaac Lab 2.2 / Isaac Sim 5.0): 12-task subset of RoboLab-120 (default) or the full 120 tasks | DROID Franka Panda + Robotiq 2F-85 | `RoboLab-Cosmos3-Demo` |
| `robolab-pi05` | — | π0.5-DROID (`pi05_droid_jointpos`, openpi policy server)[¹³](#note-13-robolab) | NVIDIA RoboLab, same protocol as `robolab-cosmos3` | DROID Franka Panda + Robotiq 2F-85 | `RoboLab-Pi05-Demo` |

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

### Note 13: RoboLab

`robolab-cosmos3` and `robolab-pi05` run [NVIDIA RoboLab](https://github.com/NVlabs/RoboLab) (pinned `@9cce00ca`, 2026-10-27) the way its README describes it: a **policy server** and a **headless Isaac Sim client** talking the OpenPI msgpack WebSocket protocol, here both on one instance. One shared template, `templates/robolab-userdata.sh.j2`, does, in order: disable GPU ECC and reboot once (Isaac Sim on L40S, same as `openarm-isaac`); `git lfs pull` RoboLab's ~6.2 GB of assets; build RoboLab's own image with `docker/build_docker.sh` on `nvcr.io/nvidia/isaac-lab:2.2.0` pinned by digest (Isaac Lab 2.2.0 / Isaac Sim 5.0.0, the stack every leaderboard row except one was measured on); build the policy server image; download the checkpoint once; start one server per shard (port 8000+i) and run `policies/<backend>/run.py --headless --instruction-type default` per shard over a round-robin slice of the task list. Shards run in parallel, one per GPU (`gpus_per_shard: 2` gives the server its own GPU). `assets/robolab/robolab_report.py` merges the shards' `episode_results.jsonl` into `task-0/summary.txt` (suite `RoboLab/subset12` or `RoboLab/RoboLab-120` in `vlasim results`), `task-0/robolab_results.json` (per task, per difficulty, Wilson intervals, subtask score) and the gallery videos (`<task>_run<r>_env<e>_success=<bool>[_viewport].mp4`, at most `max_videos_per_task` per task and camera).

- **Policy servers.** `robolab-cosmos3`: [NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework) `@86e1ee52`, `python -m cosmos_framework.scripts.action_policy_server_robolab`, image from the framework's Dockerfile on the **CUDA 12.8** base (`cu128-train` + `policy-server` groups; the framework default CUDA 13.0 needs driver ≥ 580, newer than the project DLAMI), apex skipped. Edge is served with `--format-prompt-as-json True --guidance-interval 960 1001` (framework docs); Nano with the defaults. Checkpoints: `nvidia/Cosmos3-Edge-Policy-DROID@0110e64c` (9.2 GB) / `nvidia/Cosmos3-Nano-Policy-DROID@24aa5d14` (33 GB), public, not gated. `robolab-pi05`: the [xuningy/openpi](https://github.com/xuningy/openpi) fork RoboLab links to, `@aa642056` (the same commit RoboLab's image pins for `openpi-client`), `serve_policy.py policy:checkpoint --policy.config=pi05_droid_jointpos --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos` (public GCS, 12.4 GB) with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`.
- **Variants** (`model.variant` in the model yaml; the CDK stack reads the instance list, EBS and timeout of the selected variant): `robolab-cosmos3` = `edge` (default; 1 × L40S, 12 tasks × 5 episodes), `nano` (`g6e.12xlarge`, 2 shards of sim GPU + server GPU), `edge-full` (`g6e.48xlarge`, 8 shards, RoboLab-120 × 10 = the leaderboard protocol); `robolab-pi05` = `subset` (default) and `full` (`g6e.48xlarge`, 8 shards).
- **Subset.** 12 of the 120 tasks, 6 simple / 4 moderate / 2 complex, covering the visual, relational and procedural axes, × 5 episodes (`--num-envs 5`), default instructions: `BananaInBowlTask`, `RubiksCubeAndBananaTask` (the README's sanity pair), `PickUpGreenObjectTask`, `SmallerObjectButterInBinTask`, `MustardInLeftBinTask`, `BowlStackingLeftOnRightTask`, `BananasInBinThreeTotalTask`, `PickUpBluePitcherTask`, `RubiksCubeBehindBowlTask`, `Stack3RubiksCubeTask`, `BlockStackingSpecifiedOrderTask`, `ReorientAllMugsTask`. A subset rate is **not** a RoboLab-120 number; compare it only with another run of the same subset.
- **Licenses.** RoboLab code Apache-2.0, but most bundled scenes, objects and robot models are **CC BY-NC-SA 4.0 (non-commercial)** (RoboLab `THIRD_PARTY_NOTICES.md`); Isaac Sim runs under NVIDIA's license (the container EULA is accepted with `ACCEPT_EULA=Y`); Cosmos3 weights and cosmos-framework are OpenMDW-1.1 (commercial use allowed); openpi is Apache-2.0. See [licenses.md](licenses.md).
- **Status.** Rendered, linted and synthesized; **not run on AWS yet**. Leaderboard references are in [results.md](results.md#expected-results).

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
| `rldx-gr1` | `g6.2xlarge` → `g5.2xlarge` → `g6.xlarge` → `g5.xlarge` | 200 |
| `rldx-kitchen` | `g6.2xlarge` → `g5.2xlarge` → `g6.xlarge` → `g5.xlarge` | 220 |
| `openarm-lift-act` | `g6.12xlarge` → `g5.12xlarge` → `g6.24xlarge` → `g5.24xlarge` → `g6e.12xlarge` → `g6e.24xlarge` | 300 |
| `openarm-isaac` | `g6.12xlarge` → `g5.12xlarge` → `g6.2xlarge` → `g5.2xlarge` | 500 |
| `robolab-cosmos3` | `edge` (default): `g6e.2xlarge` → `g6e.4xlarge`; `nano`: `g6e.12xlarge` → `g6e.24xlarge`; `edge-full`: `g6e.48xlarge` | 600 / 700 / 1000 |
| `robolab-pi05` | `subset` (default): `g6e.2xlarge` → `g6e.4xlarge`; `full`: `g6e.48xlarge` | 500 / 900 |
| `flora-judge` ([post-run judge](#post-run-judge-flora-rewarder)) | `g7e.2xlarge` → `g7e.4xlarge` → `g6e.12xlarge` → `p5.4xlarge` | 250 |

## Deploy commands

Each target is one `vlasim deploy --vla <target>` (which runs the `doctor` preflight,
then forwards every flag to `deploy.py` unchanged). The `--email` is only needed the
first time, or if you skipped `init`.

```bash
# GR00T N1.7 — LIBERO kitchen tasks (~120 min)
python vlasim.py deploy --vla gr00t --email you@example.com

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

# NVIDIA RoboLab (Isaac Lab, DROID Franka; assets CC BY-NC-SA 4.0) — 12-task subset x 5 episodes (~4-6 h, note 13)
# model.variant in models/robolab-*.yaml: nano (g6e.12xlarge) or the full RoboLab-120 protocol (g6e.48xlarge, up to 12 h)
python vlasim.py deploy --vla robolab-cosmos3 --email you@example.com   # Cosmos3-Edge-Policy-DROID
python vlasim.py deploy --vla robolab-pi05 --email you@example.com      # pi0.5-DROID

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
python vlasim.py destroy --vla rldx-gr1
python vlasim.py destroy --vla rldx-kitchen
python vlasim.py destroy --vla openarm-lift-act
python vlasim.py destroy --vla openarm-isaac
python vlasim.py destroy --vla robolab-cosmos3
python vlasim.py destroy --vla robolab-pi05
python vlasim.py destroy --judge --vla smolvla   # post-run judge stack of smolvla (judge.json stays with the run)
```

> `vlasim destroy` forwards to `destroy.py` unchanged — `python destroy.py --vla <target>`
> works identically if you prefer to call it directly.

The S3 results bucket is **retained** after stack deletion to preserve simulation outputs.
