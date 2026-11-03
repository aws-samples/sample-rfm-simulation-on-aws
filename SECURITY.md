# Security

If you discover a potential security issue in this project, notify AWS/Amazon Security via the [vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public GitHub issue.

See also the "Security issue notifications" section of [CONTRIBUTING.md](CONTRIBUTING.md#security-issue-notifications).

## Pinned evaluation stacks and Dependabot alerts

Some targets install a uv lock under `assets/` (`*-requirements.txt`, compiled from the `*-requirements.in` next to it). Each lock reproduces the Python version and the model stack the authors published (for example SpatialVLA pins `torch==2.5.1` and `transformers==4.47.0` on Python 3.10), because a newer stack changes the measured success rates. Several of those pinned versions have published advisories, and the patched releases cannot be installed into these stacks: `pillow>=13.0.0` and `filelock>=4.2.1` need Python 3.11, and the `torch`, `transformers`, `keras` and `protobuf` fixes are major versions the upstream code does not run on.

These packages only run on the short-lived GPU instance that one evaluation creates. The instance security group is outbound only (access is through SSM Session Manager), processes the simulator's own frames, loads model weights from the repositories named in `models/<target>.yaml`, and is deleted with the stack (`destroy.py`). Dependabot alerts on these lock files are therefore dismissed as *tolerable risk* with a link to this section. Alerts on the CDK app (`cdk/`), the GitHub workflows and the root `requirements.txt` are fixed, not dismissed.

When a target moves to a newer upstream release, recompile its lock and re-measure it; alerts that the new lock resolves close on their own.
