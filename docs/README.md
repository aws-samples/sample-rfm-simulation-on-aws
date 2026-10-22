# Documentation

Back to the [README](../README.md).

| Page | What it covers |
|---|---|
| [targets.md](targets.md) | Every `--vla` target: model, simulator, robot, stack name, per-target notes (licenses, pins, validation), GPU instances, deploy / monitor / cleanup commands |
| [showcase.md](showcase.md) | Full rollout gallery: GIF previews and MP4 links for every target with captured video |
| [results.md](results.md) | Downloading results, `vlasim results`, Expected Results, same-stack baseline, per-target smoke results, evaluation knobs |
| [configuration.md](configuration.md) | `vlasim.py` commands, `doctor` checks, supported regions, Hugging Face token, manual setup, `simulator-config.yaml` and `models/*.yaml` keys |
| [architecture.md](architecture.md) | Deploy flow (CDK stack, AZ selector, UserData) and the project structure |
| [cost.md](cost.md) | On-Demand prices per instance type and an EC2 cost estimate per target |
| [bridge-mode.md](bridge-mode.md) | Advanced, experimental: forward inference to a gRPC endpoint you host |
| [troubleshooting.md](troubleshooting.md) | Known errors and what the templates already do about them |
| [licenses.md](licenses.md) | Third-party code and weight licenses per target, acknowledgments, related projects |

Media for the gallery lives in [`showcase/`](showcase/) (one folder per target, GIF preview plus full-quality MP4).
