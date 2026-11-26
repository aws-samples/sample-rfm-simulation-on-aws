# Security

If you discover a potential security issue in this project, notify AWS/Amazon Security via the [vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public GitHub issue.

See also the "Security issue notifications" section of [CONTRIBUTING.md](CONTRIBUTING.md#security-issue-notifications).

## Pinned evaluation stacks and Dependabot alerts

Some targets install a uv lock under `assets/` (`*-requirements.txt`, compiled from the `*-requirements.in` next to it). Each lock reproduces the Python version and the model stack the authors published (for example SpatialVLA pins `torch==2.5.1` and `transformers==4.47.0` on Python 3.10), because a newer stack changes the measured success rates. Several of those pinned versions have published advisories, and the patched releases cannot be installed into these stacks: `pillow>=13.0.0` and `filelock>=4.2.1` need Python 3.11, and the `torch`, `transformers`, `keras` and `protobuf` fixes are major versions the upstream code does not run on.

These packages only run on the short-lived GPU instance that one evaluation creates. The instance security group is outbound only (access is through SSM Session Manager), processes the simulator's own frames, loads model weights from the repositories named in `models/<target>.yaml`, and is deleted with the stack (`destroy.py`). Dependabot alerts on these lock files are therefore dismissed as *tolerable risk* with a link to this section. Alerts on the CDK app (`cdk/`), the GitHub workflows and the root `requirements.txt` are fixed, not dismissed. The one exception is a package that `aws-cdk-lib` bundles inside its own tarball (for example `minimatch`): npm `overrides` cannot replace a bundled dependency, so such an alert is dismissed as *tolerable risk* until an `aws-cdk-lib` release ships the patched version, and the next Dependabot bump of `aws-cdk-lib` closes it.

When a target moves to a newer upstream release, recompile its lock and re-measure it; alerts that the new lock resolves close on their own.

## Isaac Lab CVE-2026-65141

NVIDIA Isaac Lab 0.0 - 2.3.2 evaluates lambda-looking strings with `eval()` in `isaaclab.utils.string.string_to_callable` (CVE-2026-65141, NVIDIA security bulletin of 2026-10-13, CVSS 7.8). There is no 2.x fix, so the Isaac Lab targets run fixed releases pinned by digest: `openarm-isaac` and `openarm-lift-act` on `nvcr.io/nvidia/isaac-lab:3.0.0`, `robolab-cosmos3` and `robolab-pi05` on `3.0.0-beta2` (the stack RoboLab supports). `tests/test_isaaclab3_cve.py` keeps 2.x images out of the models, templates and `generate.py`. To check an Isaac Lab install yourself, run `tools/check_isaaclab_cve_2026_65141.py` with its Python (exit 0 = fixed, 1 = vulnerable). Details: [docs/targets.md](docs/targets.md#isaac-lab-30-cve-2026-65141).
