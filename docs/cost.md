# Cost estimate

Back to the [README](../README.md) · [Documentation index](README.md)

Prices are **us-east-1 Linux On-Demand, checked 2026-10-22** with the AWS Price List API (`aws pricing get-products --region us-east-1 --service-code AmazonEC2`, shared tenancy, no pre-installed software). Other regions differ; check the [Amazon EC2 On-Demand pricing page](https://aws.amazon.com/ec2/pricing/on-demand/) for the region you deploy to. `deploy.py` launches On-Demand instances only; there is no Spot option.

## Hourly price per instance type

| Instance | GPU | vCPU | Memory | On-Demand USD/h |
|---|---|---|---|---:|
| `g6.xlarge` | 1 × L4 24 GB | 4 | 16 GiB | 0.8048 |
| `g6.2xlarge` | 1 × L4 24 GB | 8 | 32 GiB | 0.9776 |
| `g6.12xlarge` | 4 × L4 24 GB | 48 | 192 GiB | 4.6016 |
| `g5.xlarge` | 1 × A10G 24 GB | 4 | 16 GiB | 1.006 |
| `g5.2xlarge` | 1 × A10G 24 GB | 8 | 32 GiB | 1.212 |
| `g5.12xlarge` | 4 × A10G 24 GB | 48 | 192 GiB | 5.672 |
| `g6e.xlarge` | 1 × L40S 48 GB | 4 | 32 GiB | 1.861 |
| `g6e.2xlarge` | 1 × L40S 48 GB | 8 | 64 GiB | 2.24208 |
| `g6e.4xlarge` | 1 × L40S 48 GB | 16 | 128 GiB | 3.00424 |
| `g6e.12xlarge` | 4 × L40S 48 GB | 48 | 384 GiB | 10.49264 |
| `g7e.2xlarge` | 1 × RTX PRO 6000 Blackwell 96 GB | 8 | 64 GiB | 3.36312 |
| `g7e.4xlarge` | 1 × RTX PRO 6000 Blackwell 96 GB | 16 | 128 GiB | 3.99816 |
| `p5.4xlarge` | 1 × H100 80 GB | 16 | 256 GiB | 6.88 |

The `g6e.12xlarge`, `g7e` and `p5.4xlarge` rows were checked on 2026-10-28 (same API); they are used by the post-run judge only. In `ap-northeast-2`, `g7e.2xlarge` is US$4.13478 per hour.

## Per-target estimate

Instance = the first entry of `instance.preferred` (the AZ selector may fall back to a later, differently priced type; see [targets.md](targets.md#gpu-instances-per-target)). Runtime = the typical end-to-end time noted for the deploy command in [targets.md](targets.md#deploy-commands). Cost = hourly price × runtime, EC2 only, rounded.

| `--vla` | Instance | Typical runtime | Approx. EC2 cost (USD) |
|---|---|---|---:|
| `gr00t` | `g6.12xlarge` | ~120 min | ~9.20 |
| `gr00t-gr1` | `g6.12xlarge` | ~90 min | ~6.90 |
| `gr00t-g1` | `g6.12xlarge` | ~20-30 min | ~1.53-2.30 |
| `pi` | `g5.xlarge` | ~4 h | ~4.02 |
| `openvla-oft` (suite `10`, default) | `g6.xlarge` | ~3 h | ~2.41 |
| `openvla-oft --libero-suite spatial` | `g6.xlarge` | ~1.5 h | ~1.21 |
| `lap` | `g6.xlarge` | ~1.5-2.5 h | ~1.21-2.01 |
| `molmoact2` | `g6e.2xlarge` | ~40-60 min | ~1.49-2.24 (validated in us-west-2) |
| `xvla` | `g6.2xlarge` | ~15-30 min | ~0.24-0.49 |
| `smolvla` | `g6.xlarge` | ~15-30 min | ~0.20-0.40 |
| `lingbot-va` | `g6e.2xlarge` | ~2.5 h (6 episodes) | ~5.61 |
| `vla-adapter` | `g6.xlarge` | not recorded | 0.8048 per hour |
| `fastwam` | `g6e.2xlarge` | ~30-40 min | ~1.12-1.49 |
| `cosmos-policy` | `g6e.xlarge` | ~60-90 min | ~1.86-2.79 |
| `aspire` | `g6e.2xlarge` | ~45 min | ~1.68, plus ~3 Amazon Bedrock (agent, 2 tasks) |
| `real2sim` | `g5.2xlarge` | ~12 min measured (2026-10-29, `ap-northeast-2`, HOPE-Video scene 0: instance boot to done 11 min 53 s, of which packages, both venvs and the HOPE-Video download ~5 min, splatfacto training 272 s; stack create / delete not included) | ~0.24; ~0.30 in `ap-northeast-2` (US$1.4903 per hour there, Price List API 2026-10-29) |
| `rldx` | `g6.xlarge` | ~80-120 min | ~1.07-1.61 |
| `rldx-simpler` | `g6.2xlarge` | ~35-45 min | ~0.57-0.73 |
| `rldx-gr1` | `g6.2xlarge` | ~25-35 min | ~0.41-0.57 |
| `rldx-kitchen` | `g6.2xlarge` | ~25-35 min | ~0.41-0.57 |
| `openarm-lift-act` | `g6.12xlarge` | not recorded | 4.6016 per hour |
| `openarm-isaac` | `g6.12xlarge` | not recorded (creation timeout 240 min) | 4.6016 per hour |
| `smolvla --libero-variant plus` (70 episodes) | `g6.xlarge` | ~70-90 min (estimate, not yet measured) | ~0.94-1.21, plus ~0.29 NAT processing for the 6.4 GB asset pack |
| `openvla-oft --libero-suite spatial --libero-variant plus` (70 episodes) | `g6.xlarge` | ~2-2.5 h (estimate, not yet measured) | ~1.61-2.01, plus ~0.29 NAT processing for the 6.4 GB asset pack |
| `flora-judge` (post-run judge, `vlasim judge`) | `g7e.2xlarge` (measured on the `g6e.12xlarge` fallback, US$12.90/h in `ap-northeast-2`) | 13-15 min instance time measured for 10-47 episodes (+ ~10 min stack create/delete) | ~0.85-1.10 on `g7e.2xlarge`; ~2.80-3.25 measured on `g6e.12xlarge` |

## Not included

- **NAT gateway** of the stack VPC (local mode): US$0.045 per hour plus US$0.045 per GB processed in us-east-1. Model and dataset downloads go through it, so a multi-GB checkpoint adds data-processing charges. The NAT gateway is billed until you run `vlasim.py destroy`.
- **EBS** root volume (gp3, US$0.08 per GB-month in us-east-1), sized by `instance.ebs_gb` (120-500 GB per target).
- **S3** storage for videos and logs in the retained results bucket, CloudWatch Logs, SNS and the AZ selector Lambda (usually small).
- **Amazon Bedrock** usage for `aspire` (reported at US$0.91-2.30 per task in the [showcase](showcase.md)).

With `deployment.auto_terminate: true` (the default) the instance terminates after the run; the rest of the stack stays until you destroy it.
