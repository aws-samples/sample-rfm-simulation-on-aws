# Bridge mode (advanced, experimental)

Back to the [README](../README.md) · [Documentation index](README.md)

> **Experimental. Bring your own endpoint.** Bridge mode needs a policy inference endpoint that you deploy and operate yourself. This repository does not deploy one, and the default `ssm:/vla-hub/*` values in `models/*.yaml` are only parameter names that you create (or replace with literal values). The `/vla-hub/*` names are kept unchanged on purpose because existing endpoints are configured against them. Local mode (the default) needs none of this.

Targets that accept `--bridge` (checked in `deploy.py`): `gr00t`, `gr00t-gr1` and `gr00t-g1` (a GR00T gRPC endpoint plus `bridge.vpc_id`), and `pi` and `lap` (`bridge.vpc_id` plus an NLB endpoint, with the openpi WebSocket-to-gRPC bridge on the instance). Every other target is local mode only, and `deploy.py` exits with an error if you pass `--bridge`.

Bridge mode (experimental) connects the simulation environment on EC2 to a policy inference endpoint that **you** host (for example a GPU service on Amazon ECS behind a Network Load Balancer in the same VPC), without downloading the model locally. This repository does not deploy that endpoint: you bring a gRPC server that implements the contract in `assets/bridge/<model>/` (`pi.proto`, `lap.proto`, or the GR00T stubs in `gr00t_pb2*.py`). Endpoints can be given as a literal `host:port` or as an `ssm:/path` that `deploy.py` resolves at deploy time; the default paths under `/vla-hub/*` are only parameter names — create them yourself or replace them with literal values.

### GR00T bridge

```bash
# Set endpoint in models/gr00t.yaml (literal host:port, or an SSM parameter you created):
#   bridge.remote_grpc_endpoint: ssm:/vla-hub/gr00t/n1-7/grpc-endpoint
#   bridge.vpc_id: ssm:/vla-hub/vpc-id

python vlasim.py deploy --vla gr00t --bridge
```

### π0.5 bridge

```bash
# Set in models/pi.yaml:
#   bridge.vpc_id: vpc-xxxxxxxxxxxxxxxxx
#   bridge.nlb_endpoint: internal-xxx.elb.us-east-1.amazonaws.com:50052

python vlasim.py deploy --vla pi --bridge
```

### LAP-3B bridge

LAP shares π0.5's openpi WebSocket bridge pattern. The instance runs only the LIBERO
sim + a WebSocket↔gRPC bridge (`assets/bridge/lap/`); the LAP-3B JAX model runs on the
remote LAP gRPC endpoint you provide (port 50055). No local checkpoint download.

```bash
# Set in models/lap.yaml (or use ssm: for nlb_endpoint):
#   bridge.vpc_id: vpc-xxxxxxxxxxxxxxxxx                       # the VPC of your endpoint
#   bridge.nlb_endpoint: internal-xxx.elb.us-west-2.amazonaws.com:50055
#                        or ssm:/vla-hub/lap/3b/grpc-endpoint

python vlasim.py deploy --vla lap --bridge
```

> Bridge differs from π0.5 only in the wire contract: LAP uses a nested observation
> (`observation.base_0_rgb` / `left_wrist_0_rgb` / `state`), a 10-dim state
> (eef_pos 3 + eef_rot6d 6 + gripper 1), a `frame_description` field, and a (10,7)
> action chunk — see `assets/bridge/lap/lap.proto`.

---
