# Licenses and acknowledgments

Back to the [README](../README.md) · [Documentation index](README.md)

## Third-party components and licenses

This sample's own code (CDK, generators, templates) is **MIT-0** (see [`LICENSE`](../LICENSE)). It orchestrates several third-party models and simulators, each under its own license — pulled at **runtime** (not vendored in this repo):

| Component | Used by | Code license | Weights / assets license | Notes |
|-----------|---------|--------------|--------------------------|-------|
| [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | `gr00t`, `gr00t-gr1`, `gr00t-g1` | Apache-2.0 | `nvidia/GR00T-N1.7-LIBERO`: NVIDIA Open Model License; **`nvidia/GR00T-N1.6-3B`: NVIDIA One-Way Noncommercial License (NON-COMMERCIAL)**; `nvidia/Cosmos-Reason2-2B` (gated): NVIDIA Open Model License | `gr00t-g1` uses the community fine-tune `cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate` (card: Apache-2.0; derived from GR00T N1.6 — review the base-model terms) |
| [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) | `gr00t-g1` | Apache-2.0 | NVIDIA Open Model License | uses the [xieleo5/robosuite](https://github.com/xieleo5/robosuite) fork (MIT) |
| [Physical Intelligence openpi](https://github.com/Physical-Intelligence/openpi) (π0.5) | `pi` | Apache-2.0 | per the openpi project | |
| [OpenVLA-OFT](https://github.com/moojink/openvla-oft) | `openvla-oft` | MIT | MIT (`moojink/openvla-7b-oft-finetuned-libero-*`) | uses [transformers-openvla-oft](https://github.com/moojink/transformers-openvla-oft) (Apache-2.0) |
| [LAP-3B](https://github.com/lihzha/lap) | `lap` | Apache-2.0 | Apache-2.0 (`lihzha/LAP-3B-Libero`) | |
| [MolmoAct2 (Ai2)](https://github.com/allenai/molmoact2) | `molmoact2` | Apache-2.0 (served from the [allenai/lerobot](https://github.com/allenai/lerobot) fork, Apache-2.0) | open weights (`allenai/MolmoAct2-LIBERO`); the model card has no license field — check it | |
| [Hugging Face LeRobot](https://github.com/huggingface/lerobot) | `xvla`, `smolvla`, `lingbot-va`, `fastwam`, `openarm-isaac` | Apache-2.0 | see the rows below | shared `lerobot-libero` template |
| [X-VLA](https://github.com/2toINF/X-VLA) | `xvla` | Apache-2.0 | Apache-2.0 (`lerobot/xvla-libero`) | |
| SmolVLA | `smolvla` | Apache-2.0 (LeRobot) | Apache-2.0 (`lerobot/smolvla_libero`) | |
| [LingBot-VA](https://github.com/Robbyant/lingbot-va) | `lingbot-va` | Apache-2.0 | Apache-2.0 (`lerobot/lingbot_va_libero_long`) | |
| [FastWAM](https://github.com/yuantianyuan01/FastWAM) | `fastwam` | MIT | **unspecified** — `lerobot/fastwam_libero_uncond_2cam224` states no license; research/evaluation only | also fetches `google/umt5-xxl` and `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (both Apache-2.0) |
| π0.5 `folding_latest` (LeRobot) | `openarm-isaac` | Apache-2.0 (LeRobot) | Apache-2.0 (`lerobot/folding_latest`) | |
| [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter) | `vla-adapter` | MIT | MIT (`VLA-Adapter/LIBERO-Spatial-Pro`) | |
| [NVIDIA Cosmos Policy](https://github.com/NVlabs/cosmos-policy) | `cosmos-policy` | Apache-2.0 | **NVIDIA One-Way Noncommercial License, NSCLv1 (NON-COMMERCIAL)** (`nvidia/Cosmos-Policy-LIBERO-Predict2-2B`); base `nvidia/Cosmos-Predict2-2B-Video2World` (gated): NVIDIA Open Model License | |
| [NVIDIA ASPIRE](https://github.com/NVlabs/ASPIRE) | `aspire` | Apache-2.0 | tools: **SAM3 — SAM License (gated)**; **[Contact-GraspNet](https://github.com/NVlabs/contact_graspnet) — NVIDIA Source Code License (NON-COMMERCIAL)**; [PyRoki](https://github.com/chungmin99/pyroki) — MIT | the agent calls Claude models on Amazon Bedrock in your account |
| [HOPE-Video](https://github.com/swtyree/hope-dataset) (NVIDIA; Lin et al., IROS 2021) | `real2sim` (default input) | — | **CC BY-NC-SA 4.0 (NON-COMMERCIAL)** — images, poses, scene clouds and object meshes | downloaded at run time (Google Drive via gdown, MD5-checked); results derived from it inherit the license |
| [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) + [gsplat](https://github.com/nerfstudio-project/gsplat) | `real2sim` | Apache-2.0 / Apache-2.0 | — | splatfacto 3D Gaussian Splatting |
| [MuJoCo](https://github.com/google-deepmind/mujoco) + [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | `real2sim` | Apache-2.0 | UR5e: BSD-3-Clause; Robotiq 2F-85: BSD-2-Clause | Menagerie models fetched at a pinned commit |
| [Open3D](https://github.com/isl-org/Open3D), [trimesh](https://github.com/mikedh/trimesh), [CoACD](https://github.com/SarahWeiii/CoACD), [OpenUSD](https://github.com/PixarAnimationStudios/OpenUSD) (`usd-core`) | `real2sim` | MIT / MIT / MIT / Tomorrow Open Source Technology License (Apache-2.0 based) | — | segmentation, meshes, USD export |
| **[RLDX-1 (RLWRLD)](https://github.com/RLWRLD/RLDX-1)** | `rldx`, `rldx-simpler`, `rldx-gr1`, `rldx-kitchen` | Apache-2.0 | **RLWRLD Model License v1.0 (NON-COMMERCIAL)** (`RLWRLD/RLDX-1-FT-*`); backbone `Qwen/Qwen3-VL-8B-Instruct` Apache-2.0 | "simulation benchmarking" is an explicit intended use; commercial deployment is out of scope — research / simulation benchmarking only |
| [NVIDIA Flora-Rewarder](https://huggingface.co/nvidia/Flora-Rewarder) | `flora-judge` (post-run judge) | `batch_utils/` scoring scripts in the model repository (downloaded at run time; vLLM Apache-2.0, Transformers Apache-2.0) | **NVIDIA OneWay Noncommercial License, 22 Mar 2022 (NON-COMMERCIAL)** (`nvidia/Flora-Rewarder`); base `Qwen/Qwen3.5-27B` Apache-2.0 | research and development only, not for commercial or production use; outputs are annotations, not a safety monitor (model card). Not gated |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | LIBERO targets, `flora-judge` (task map for LeRobot instructions) | MIT | MIT | |
| [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) (Fei et al., arXiv 2510.13626) | `--libero-variant plus` (`smolvla`, `openvla-oft`) | **none declared** (no LICENSE file; GitHub reports none) — research / evaluation use | asset pack `Sylvest/LIBERO-plus` (Hugging Face dataset): MIT on the dataset card | fork of LIBERO; code cloned at a pinned commit and the 6.4 GB asset pack downloaded at run time, never vendored |
| [SimplerEnv](https://github.com/allenzren/SimplerEnv) + [ManiSkill2_real2sim](https://github.com/allenzren/ManiSkill2_real2sim) | `rldx-simpler` | MIT / Apache-2.0 | — | cloned at runtime by `setup_SimplerEnv.sh` |
| [RoboCasa GR-1 Tabletop tasks](https://github.com/robocasa/robocasa-gr1-tabletop-tasks) | `gr00t-gr1`, `rldx-gr1` | MIT | — | cloned at runtime by `setup_gr1.sh` |
| RoboCasa (Kitchen) + [robosuite](https://github.com/ARISE-Initiative/robosuite) | `rldx-kitchen` | MIT | RoboCasa asset terms | cloned at runtime by `setup_robocasa.sh` ([squarefk/robocasa](https://github.com/squarefk/robocasa), fork of the RoboCasa project); ~5 GB scene assets download during setup |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | LIBERO / RoboCasa / G1 targets | Apache-2.0 | — | |
| [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) + Isaac Sim | `openarm-isaac`, `openarm-lift-act` | BSD-3-Clause (Isaac Lab) | NGC image `nvcr.io/nvidia/isaac-lab` bundles Isaac Sim under NVIDIA's license terms | |
| [OpenArm Isaac Lab](https://github.com/enactic/openarm_isaac_lab) | `openarm-isaac`, `openarm-lift-act` | Apache-2.0 | — | |

Licenses were checked on 2026-10-22 against each upstream repository's license file and Hugging Face model card; upstream terms can change, so re-check before use. **This is sample code, not production-ready software.** It downloads third-party models, datasets and simulators at runtime into your AWS account; you are responsible for reviewing and complying with each component's license and terms of use (several weights above are non-commercial or have no stated license). See [`NOTICE`](../NOTICE).

The `rldx`/`rldx-simpler`/`rldx-gr1`/`rldx-kitchen` targets are built on **RLDX-1 by [RLWRLD](https://github.com/RLWRLD/RLDX-1)** — gratefully acknowledged. RLDX-1's eval harness vendors LIBERO, SimplerEnv, RoboCasa, and GR-1 task suites as submodules (all MIT); this sample clones the RLDX-1 repo at a pinned commit at runtime and applies small in-place runtime fixes to the clone only — the upstream repo is never modified and no model weights are committed here. Per target: `rldx` needs only the shared dependency-pin fixes; `rldx-simpler` adds a SimplerEnv dispatch + obs/action adapter patch; `rldx-gr1` needs no code patch (dispatch + obs/action routing already correct at the pin); `rldx-kitchen` needs no dispatch/obs/action patch but adds three RoboCasa-Kitchen-specific runtime fixes (making the cloned asset downloader non-interactive for headless setup; a `numpy`/`numba` pin reconciliation so the import-reachable `albumentations` resolves against the kitchen fork; and a cosmetic video-recorder narrowing so the saved clip shows each camera once rather than duplicating it at two resolutions).

## Related projects

- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) — NVIDIA GR00T foundation model
- [openpi](https://github.com/Physical-Intelligence/openpi) — Physical Intelligence π0.5
- [openvla-oft](https://github.com/moojink/openvla-oft) — OpenVLA-OFT (Optimized Fine-Tuning)
- [lap](https://github.com/lihzha/lap) — LAP: Language-Action Pre-Training (zero-shot cross-embodiment, JAX)
- [RLDX-1](https://github.com/RLWRLD/RLDX-1) — RLWRLD dexterity-first VLA (MSAT / Qwen3-VL-8B, Apache-2.0 code)
- [LIBERO](https://libero-project.github.io/) — Long-horizon robot benchmark
- [SimplerEnv](https://github.com/allenzren/SimplerEnv) + [ManiSkill2_real2sim](https://github.com/allenzren/ManiSkill2_real2sim) — real-to-sim eval for Google Fractal / WidowX Bridge (MIT)
- [robocasa-gr1-tabletop-tasks](https://github.com/robocasa/robocasa-gr1-tabletop-tasks) — RoboCasa GR-1 humanoid tabletop benchmark (24 tasks, MIT)
