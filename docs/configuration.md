# Setup and configuration

Back to the [README](../README.md) · [Documentation index](README.md)

## `vlasim.py` commands

A single CLI (`vlasim.py`) wraps the whole setup: `doctor`, `init`, `deploy`, `destroy`, `results` and `gallery` ([video history](gallery.md)). It runs on the standard library alone, so `doctor` works on a fresh host **before** any dependencies are installed. The command sequence is in the [README Quick Start](../README.md#quick-start).

`doctor` checks: Python/Node/CDK versions and installs, AWS credentials, region
support, CDK bootstrap, GPU On-Demand vCPU quota (per target), `notify_email`, and —
for targets that pull a gated Hugging Face model — the HF token in SSM.
It exits non-zero if any check **FAILs**, so the `deploy` preflight stops a missing
quota or un-bootstrapped account in seconds rather than mid-deploy (idle GPU billing).

```
$ python vlasim.py doctor --vla gr00t-g1
  [ OK ] Python 3.12.1 (>= 3.10)
  [ OK ] Node v22.22.1
  [ OK ] AWS credentials valid (account ..., region us-east-1)
  [ OK ] CDK bootstrap present in us-east-1
  [ OK ] On-Demand G/VR vCPU quota = 768 in us-east-1 (>= 4 for g6.xlarge ...)
  [FAIL] notify_email is still the placeholder
         -> run: python vlasim.py init --email YOU@EXAMPLE.COM
```

`deploy` and `destroy` forward every flag to `deploy.py` / `destroy.py` unchanged
(`--bridge`, `--collect`, `--libero-suite`, `--region`, …). Use `--skip-doctor` to
bypass the deploy preflight. Run `python vlasim.py deploy --help` for the full list.

Supported regions: `us-east-1`, `us-west-2`, `ap-northeast-1`, `ap-northeast-2`, `eu-central-1`

## Hugging Face token (gated models)

`gr00t` (backbone `nvidia/Cosmos-Reason2-2B`),
`gr00t-g1` (the N1.7 adapter in `server.hf_repo`) and `openarm-isaac` (tokenizer
`google/paligemma-3b-pt-224`) download gated repos. Accept the model
license on huggingface.co with your account, then store a read token in **each region you deploy
to** (SSM parameters are regional). `cosmos-policy` (gated base `nvidia/Cosmos-Predict2-2B-Video2World`) and `aspire` (SAM3) also need a token whose account has been granted access; see [note 9](targets.md#note-9-cosmos-policy) and [note 10](targets.md#note-10-aspire).

```bash
aws ssm put-parameter --region us-east-1 --name /vla-simulator/hf-token \
  --type SecureString --value <YOUR_HF_TOKEN>
```

## Manual setup

Equivalent to `init`, if you prefer to run the steps yourself:

```bash
pip install -r requirements.txt          # boto3, pyyaml, jinja2
cd cdk && npm install && cd ..           # CDK deps (aws-cdk is pinned in cdk/package.json)
aws configure                            # or: export AWS_PROFILE=...
# then edit deployment.region / notify_email in simulator-config.yaml
```

## Direct invocation

`vlasim deploy` is a thin wrapper — every deploy command in [targets.md](targets.md#deploy-commands) also runs as `python deploy.py --vla <target> ...` with the same flags, skipping the `doctor` preflight. Use that if you've already run `doctor` and want to bypass it (equivalently, `vlasim deploy --skip-doctor`). `vlasim destroy` forwards to `destroy.py` unchanged — `python destroy.py --vla <target>` works identically.

On first deploy you will receive an SNS subscription confirmation email — **click the link** to enable notifications.

## Configuration reference

### `simulator-config.yaml` (shared)

| Key | Default | Description |
|-----|---------|-------------|
| `deployment.region` | `us-east-1` | AWS region |
| `deployment.notify_email` | *(required)* | SNS notification email |
| `deployment.s3_results_prefix` | `vla-sim-results` | S3 bucket name prefix |
| `deployment.auto_terminate` | `true` | Auto-terminate EC2 after completion |
| `deployment.gallery` | `true` | Publish run videos to the [video history](gallery.md) page linked from the e-mail (`--no-gallery` for one deploy) |

### `models/gr00t.yaml`

| Key | Description |
|-----|-------------|
| `model.hf_repo` | HuggingFace repo for checkpoint download |
| `model.hf_subfolder` | Subfolder within repo (e.g. `libero_10`) |
| `model.hf_model_revision` | Pinned commit SHA for reproducibility |
| `model.isaac_groot_commit` | Pinned Isaac-GR00T commit SHA |
| `instance.preferred` | GPU instance type fallback list |
| `instance.ebs_gb` | Root EBS volume size (GB) |
| `tasks` | List of LIBERO tasks to evaluate |

### `models/pi.yaml`

| Key | Description |
|-----|-------------|
| `model.openpi_commit` | Pinned openpi commit SHA |
| `instance.preferred` | GPU instance type fallback list |
| `tasks[].suite` | LIBERO suite name (`libero_spatial`, `libero_object`, etc.) |
| `tasks[].num_episodes` | Episodes per task |

Every other target has its own `models/<target>.yaml` with the same `model` / `instance` / `tasks` layout (plus `bridge`, `server` or `collection` blocks where the target supports them); the comments in each file document the target-specific keys. The per-task evaluation knobs (`condition`, `extra_args`, `replan`, `seed`, `step_scale`, `random_init`, `replan_head`, `model.mujoco_pin`, `model.eval_args`) are described under [Expected Results](results.md#expected-results).
