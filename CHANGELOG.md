# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Video history gallery: the completion e-mail links to a password-protected Amazon CloudFront page with every video of every run under the notification e-mail, grouped by target, newest run first (shared `RFM-Gallery` stack: private S3 bucket + OAC, CloudFront Function + KeyValueStore Basic auth; per-target `GalleryPublisher` Lambda). `vlasim.py gallery [--reset-password]` prints the URL and password; `deploy.py --no-gallery` / `deployment.gallery: false` opt out. Videos already in a kept results bucket are added on the next deploy (backfill). The `aws s3 sync` download line stays in the e-mail. See [docs/gallery.md](docs/gallery.md).

### Changed

- `destroy.py` asks what to do with the data: `--keep-data` (default, today's behaviour: results bucket and gallery are kept) or `--delete-all` (also deletes the results bucket, the stack's Lambda log groups and this target's gallery videos, and the whole gallery when no other target stack is left in the region). A terminal gets a numbered prompt; without one, destroy keeps the data.
- `requirements.txt` installs `boto3[crt]` (the CloudFront KeyValueStore API needs SigV4A).

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
