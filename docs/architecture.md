# Architecture

Back to the [README](../README.md) · [Documentation index](README.md)

## Deploy flow

```
deploy.py
  │
  ├─ generate.py          → assets/userdata/{vla}.sh
  │
  └─ CDK deploy
       ├─ VPC + SG
       ├─ S3 ResultsBucket (RETAIN)
       ├─ SNS + EmailSubscription
       ├─ IAM Role
       ├─ AzSelector Lambda  →  EC2 (G5 / G6 / G6e, per target)
       └─ WaitCondition      ←  cfn_signal from UserData
```

1. `generate.py` renders `templates/<template>-userdata.sh.j2` with the values in `models/<target>.yaml` and `simulator-config.yaml` into `assets/userdata/<target>.sh` (gitignored).
2. `cdk deploy` creates one stack per target (stack names in [targets.md](targets.md#target-table)): a VPC with public and private subnets and one NAT gateway (in bridge mode the stack looks up the VPC you pass instead), a security group, the S3 results bucket (retained on delete), an SNS topic with your email subscription, and the instance IAM role.
3. The `AzSelector` Lambda (`cdk/lib/az-selector.ts`) launches the EC2 GPU instance. It walks the `instance.preferred` list of the target (see [targets.md](targets.md#gpu-instances-per-target)) and every AZ in the region until it finds capacity.
4. The UserData script installs the model and simulator at pinned commits, runs the evaluation, uploads videos and `summary.txt` to S3, sends the SNS email, signals the CloudFormation `WaitCondition`, and (with `deployment.auto_terminate: true`) terminates the instance.

**Two deployment modes:**

- **Local mode** — model runs directly on the EC2 instance (default)
- **Bridge mode** (experimental) — EC2 runs the simulation env; model inference is forwarded to a gRPC inference endpoint you provide (see [bridge-mode.md](bridge-mode.md))

## Project structure

```
sample-rfm-simulation-on-aws/
├── vlasim.py                 # CLI front end: doctor / init / deploy / destroy / results / gallery / judge
├── deploy.py                 # 1-click deploy entrypoint
├── destroy.py                # Stack teardown (--keep-data / --delete-all)
├── gallery.py                # Video history gallery helpers (id, password, SSM, KeyValueStore)
├── results.py                # Read-only result collector behind `vlasim results`
├── generate.py               # Renders assets/userdata/{vla}.sh from templates/ + models/
├── simulator-config.yaml     # Shared deployment settings
├── requirements.txt          # boto3[crt], pyyaml, jinja2
├── models/                   # One config per --vla target (23) + the post-run judge (flora-judge.yaml)
│   ├── gr00t.yaml            # GR00T N1.7 on LIBERO-10
│   ├── gr00t-bridge.yaml     # GR00T N1.7 on SimplerEnv Bridge (WidowX)
│   ├── gr00t-gr1.yaml        # GR00T N1.6 + GR1 humanoid (RoboCasa)
│   ├── gr00t-g1.yaml         # GR00T N1.6 + Unitree G1 whole-body loco-manip (optional N1.7 adapter)
│   ├── pi.yaml               # π0.5 (openpi)
│   ├── openvla-oft.yaml      # OpenVLA-OFT (per-suite checkpoints)
│   ├── lap.yaml              # LAP-3B
│   ├── molmoact2.yaml        # MolmoAct2 (Ai2)
│   ├── xvla.yaml             # X-VLA-0.9B (LeRobot)
│   ├── smolvla.yaml          # SmolVLA-450M (LeRobot)
│   ├── lingbot-va.yaml       # LingBot-VA world action model (LeRobot)
│   ├── fastwam.yaml          # FastWAM world action model (LeRobot)
│   ├── vla-adapter.yaml      # VLA-Adapter-Pro
│   ├── cosmos-policy.yaml    # NVIDIA Cosmos Policy
│   ├── aspire.yaml           # NVlabs ASPIRE (Claude on Amazon Bedrock)
│   ├── real2sim.yaml         # Real2Sim: capture -> 3DGS -> collision meshes -> MuJoCo grasp
│   ├── rldx.yaml             # RLDX-1 on LIBERO-10
│   ├── rldx-simpler.yaml     # RLDX-1 on SimplerEnv Google Fractal
│   ├── rldx-bridge.yaml      # RLDX-1 on SimplerEnv Bridge (WidowX)
│   ├── rldx-gr1.yaml         # RLDX-1 on RoboCasa GR-1 Tabletop
│   ├── rldx-kitchen.yaml     # RLDX-1 on RoboCasa Kitchen
│   ├── openarm-lift-act.yaml # OpenArm Lift-Cube scripted ACT collection (Isaac Lab)
│   ├── openarm-isaac.yaml    # OpenArm bimanual π0.5 pipe proof (Isaac Lab, experimental)
│   └── flora-judge.yaml      # post-run judge: NVIDIA Flora-Rewarder outcome score (not a --vla target)
├── templates/                # Jinja2 UserData templates (16)
│   ├── gr00t-userdata.sh.j2             # shared: gr00t (LIBERO), gr00t-bridge (SimplerEnv), by sim_id
│   ├── gr00t-gr1-userdata.sh.j2
│   ├── gr00t-g1-userdata.sh.j2          # GR00T-WholeBodyControl loco-manip, optional 2-venv N1.7 server
│   ├── pi-userdata.sh.j2                # Docker Compose
│   ├── openvla-oft-userdata.sh.j2       # single conda env
│   ├── vla-adapter-userdata.sh.j2       # copy of the openvla-oft template
│   ├── lap-userdata.sh.j2               # uv 2-venv: JAX policy + LIBERO sim
│   ├── molmoact2-userdata.sh.j2         # allenai/lerobot fork, in-process
│   ├── lerobot-libero-userdata.sh.j2    # shared: xvla, smolvla, lingbot-va, fastwam
│   ├── cosmos-policy-userdata.sh.j2     # project's own CUDA 12.8 Docker image
│   ├── aspire-userdata.sh.j2            # LIBERO-Pro fork + tool services + Claude Code on Bedrock
│   ├── real2sim-userdata.sh.j2          # 3DGS venv + geometry/MuJoCo venv, stage scripts embedded
│   ├── rldx-userdata.sh.j2              # shared: rldx, rldx-simpler, rldx-bridge, rldx-gr1, rldx-kitchen
│   ├── openarm-lift-act-userdata.sh.j2  # Isaac Lab container, scripted ACT collection
│   ├── openarm-isaac-userdata.sh.j2     # Isaac Lab container, bimanual π0.5 eval
│   ├── flora-judge-userdata.sh.j2       # post-run judge: vLLM + NVIDIA batch_utils terminal reward
│   └── _gallery-sns.sh.j2               # included: "Video history" block of the SNS message
├── assets/
│   ├── userdata/             # Generated scripts (gitignored)
│   ├── bridge/
│   │   ├── gr00t/            # ZMQ-gRPC bridge for GR00T (generated gr00t_pb2*.py stubs)
│   │   ├── pi/               # gRPC bridge for π0.5 (pi.proto)
│   │   └── lap/              # WebSocket↔gRPC bridge for LAP-3B (lap.proto, port 50055)
│   ├── stack-knobs/          # stackpatch.py: per-task eval-script edits for the evaluation knobs
│   ├── gr00t-g1/             # G1 trajectory collection + HDF5 → LeRobot conversion
│   ├── openarm-lift-act/     # Lift-Cube env overlay, scripted collector, HDF5 → LeRobot
│   ├── openarm-isaac/        # bimanual env overlay + LeRobot rollout driver
│   ├── real2sim/             # capture prep, splat -> metres, segmentation, grasp replay, report
│   └── flora-judge/          # flora_judge.py: videos -> episodes, scores -> judge.json (stdlib)
├── cdk/
│   ├── bin/app.ts            # CDK app: one stack per --vla target
│   ├── bin/gallery.ts        # CDK app: shared RFM-Gallery stack (video history)
│   ├── bin/judge.ts          # CDK app: post-run judge stack (FloraJudge-<target stack>)
│   ├── lib/
│   │   ├── vla-simulator-stack.ts   # Unified CDK stack
│   │   ├── az-selector.ts           # GPU capacity Lambda (AZ/type fallback)
│   │   ├── gallery-stack.ts         # S3 + CloudFront (OAC) + Function + KeyValueStore
│   │   ├── judge-stack.ts           # post-run judge: VPC + SNS + AzSelector, reads the judged bucket
│   │   └── nag.ts                   # cdk-nag acknowledgement helpers
│   ├── functions/gallery-auth.js    # CloudFront Function: Basic auth for /g/<id>/
│   ├── lambda/gallery_publisher/    # Lambda: copy videos to the gallery, rebuild index.html
│   └── test/*.test.ts        # Jest synth tests
├── docs/
│   ├── *.md                  # This documentation
│   └── showcase/             # Sample rollout MP4 + GIF preview per target
└── .github/
    ├── workflows/ci.yml      # tsc + jest + cdk synth + userdata render for every target
    ├── workflows/dependabot-automerge.yml
    └── dependabot.yml
```
