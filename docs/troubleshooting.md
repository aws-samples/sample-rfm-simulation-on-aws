# Troubleshooting

Back to the [README](../README.md) · [Documentation index](README.md)

Start with `python vlasim.py doctor --vla <target>`: it checks the toolchain, credentials, region, CDK bootstrap, GPU quota, `notify_email` and the Hugging Face token in SSM, and prints the fix for every FAIL ([configuration.md](configuration.md#vlasimpy-commands)). Per-target known issues and pins are in the [target notes](targets.md#notes).

## "Embodiment tag 'LIBERO_PANDA' is not supported"

The **base** GR00T N1.7-3B model does not support LIBERO simulation. Use the fine-tuned checkpoint:

```yaml
# models/gr00t.yaml
model:
  hf_repo: nvidia/GR00T-N1.7-LIBERO
  hf_subfolder: libero_10
```

## "No visible GPU devices" (π0.5 task-0)

JAX inside the Docker container fails if GPU passthrough isn't ready before the task loop starts. The `[2.5/5]` step in the UserData script polls `docker run --gpus all nvidia-smi` until GPU is confirmed before proceeding.

## Stack rollback / InsufficientInstanceCapacity

The `AzSelectorConstruct` Lambda automatically retries across AZs and falls back through the instance type preference list (`g6.12xlarge` → `g5.12xlarge` → `g6.xlarge` → `g5.xlarge` for GR00T). No manual intervention needed — CDK rollback means no capacity was found; re-deploying will retry.

## CDK synth cdk.out conflict (parallel deploys)

Use model-specific output directories:

```bash
npx cdk deploy GR00T-Demo --output cdk.out-gr00t -c vla=gr00t -c region=us-east-1 ...
npx cdk deploy Pi-Demo    --output cdk.out-pi    -c vla=pi    -c region=us-east-1 ...
```

`deploy.py` handles this automatically.

## apt lock on DLAMI boot

`unattended-upgrades` holds the apt lock for ~2 minutes on fresh boot. The UserData waits up to 120 seconds then forcibly kills the process. No action needed. Every template also masks the `apt-daily` timers (they re-launch `unattended-upgrades` later in the boot), removes the `command-not-found` apt hook (it once hung `apt-get update` for 3 h) and bounds each `apt-get update` with `timeout 600`.

---
