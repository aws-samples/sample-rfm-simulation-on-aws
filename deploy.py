#!/usr/bin/env python3
"""
RFM Simulation on AWS 1-Click Deploy
Usage:
  python deploy.py --vla gr00t              # GR00T N1.7 + LIBERO local mode
  python deploy.py --vla gr00t-bridge       # GR00T N1.7 + SimplerEnv Bridge (WidowX) local mode
  python deploy.py --vla gr00t-gr1          # GR00T N1.6 + GR1 humanoid + RoboCasa local mode
  python deploy.py --vla pi                 # π0.5  local mode
  python deploy.py --vla openvla-oft        # OpenVLA-OFT + LIBERO-10 local mode
  python deploy.py --vla lap                # LAP-3B + LIBERO-Spatial local mode
  python deploy.py --vla openarm-isaac      # π0.5 (folding_latest) + Isaac Lab bimanual OpenArm local mode
  python deploy.py --vla openarm-lift-act   # OpenArm unimanual Lift-Cube × ACT — scripted demo COLLECTION (HDF5)
  python deploy.py --vla molmoact2          # MolmoAct2 (Ai2 5B) + LIBERO-Goal — fp32 in-process eval (allenai/lerobot fork)
  python deploy.py --vla real2sim           # Real2Sim: capture -> 3DGS -> collision meshes -> MuJoCo grasp (HOPE-Video, non-commercial)
  python deploy.py --vla gr00t --bridge     # GR00T bridge mode (bring-your-own gRPC endpoint, experimental)
  python deploy.py --vla gr00t-gr1 --bridge # GR00T-GR1 bridge mode (your endpoint must serve N1.6)
  python deploy.py --vla pi    --bridge     # π0.5  bridge mode (bring-your-own gRPC endpoint, experimental)
  python deploy.py --vla smolvla --no-gallery  # without the video history gallery
  python deploy.py --vla rlinf-oft-libero   # RLinf: baseline eval -> GRPO -> re-eval (smoke: 1 GPU, ~2-4 h)
  python deploy.py --vla rlinf-oft-libero --rl-profile full   # 8 GPU, recipe defaults
  python deploy.py --judge --vla smolvla    # score the latest smolvla run's videos with Flora-Rewarder
  python deploy.py --judge --vla smolvla --run 2026-10-20-0130   # ... a given run

Steps:
  1. Load simulator-config.yaml + models/{vla}.yaml
  2. If bridge mode: resolve SSM-backed endpoints
  3. python generate.py → assets/userdata/{vla}.sh
  4. Video history gallery (unless --no-gallery): deploy RFM-Gallery when missing, password + KVS entry
  5. cdk deploy {VLA}-Demo

Notes:
  - On first deploy, a SNS subscription confirmation email will arrive.
    Click "Confirm subscription" to receive completion notifications.
  - After simulation completes, run: python destroy.py --vla {gr00t|pi}
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import boto3
import yaml
from botocore.exceptions import ClientError

import gallery
import libero_plus
import libero_r2s
import repeatability
import robotwin_plan
import seeds
from generate import (
    OFT_DEFAULT_SUITE,
    OFT_LIBERO_SUITES,
    ROBOLAB_TARGETS,
    libero_plus_supported,
    libero_r2s_supported,
    normalise_libero_suite,
    oft_stack_name,
    r2s_default_suite,
    robolab_plan,
)
from targets import JUDGE_TARGETS, VLA_TARGETS

BASE_DIR = Path(__file__).parent



# Upstream-LeRobot × LIBERO targets (shared lerobot-libero template) → CDK stack names.
# Keep in sync with generate.LEROBOT_LIBERO_TARGETS and cdk/bin/app.ts (tests/test_targets_parity.py checks).
LEROBOT_LIBERO_STACKS = {
    "xvla": "XVLA-Demo",
    "smolvla": "SmolVLA-Demo",
    "lingbot-va": "LingBot-VA-Demo",
    "fastwam": "FastWAM-Demo",
}
LEROBOT_LIBERO_TARGETS = tuple(LEROBOT_LIBERO_STACKS)
# Local-only targets with their own template (same CLI handling as the LeRobot family).
OTHER_LOCAL_STACKS = {"vla-adapter": "VLA-Adapter-Demo", "cosmos-policy": "Cosmos-Policy-Demo", "aspire": "ASPIRE-Demo",
                      "real2sim": "Real2Sim-Demo", "robolab-cosmos3": "RoboLab-Cosmos3-Demo",
                      "robolab-pi05": "RoboLab-Pi05-Demo", "rlinf-oft-libero": "RLinf-OFT-LIBERO-Demo"}
# RLinf RL post-training targets (evaluate -> GRPO -> re-evaluate); --rl-profile picks models/<vla>.yaml profiles.
RLINF_TARGETS = ("rlinf-oft-libero",)
RL_PROFILES = ("smoke", "full")

# Upstream-LeRobot × RoboTwin 2.0 targets (shared lerobot-robotwin template) → CDK stack names.
# Keep in sync with generate.LEROBOT_ROBOTWIN_TARGETS and cdk/bin/app.ts (tests/test_targets_parity.py checks).
LEROBOT_ROBOTWIN_STACKS = {
    "smolvla-robotwin": "SmolVLA-RoboTwin-Demo",
    "lingbot-vla2": "LingBot-VLA2-Demo",
}
LEROBOT_ROBOTWIN_TARGETS = tuple(LEROBOT_ROBOTWIN_STACKS)
# One deploy must finish inside the 12 h CloudFormation WaitCondition (PT720M) including ~1.5 h of setup.
ROBOTWIN_MAX_LOOP_HOURS = 10.0
# Planning figure for the runtime estimate (s per simulated step incl. policy inference), not measured yet.
ROBOTWIN_EST_SEC_PER_STEP = 0.15

RUN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{4}$")  # RUN_ID=$(date '+%Y-%m-%d-%H%M') in every template
VIDEO_KEY_RE = re.compile(r"^[^/]+/task-\d+/videos/[^/]+\.mp4$")


def results_bucket_name(prefix: str, stack_name: str, region: str, account: str) -> str:
    """Results bucket of a target stack. Same rule as cdk/lib/vla-simulator-stack.ts: drop "-demo"
    only when the full name overflows 63 chars."""
    name = f"{prefix}-{stack_name.lower()}-{region}-{account}"
    if len(name) > 63:
        name = f"{prefix}-{stack_name.lower().removesuffix('-demo')}-{region}-{account}"
    return name


def pick_run(run_ids, video_counts: dict, wanted: str = ""):
    """The run to judge: ``wanted`` if given (must have videos), else the newest run with videos.

    run_ids: top-level prefixes of the results bucket (without "/"); video_counts: {run_id: n mp4}.
    Returns (run_id, error message); exactly one of them is set."""
    runs = sorted(r for r in run_ids if RUN_RE.match(r))
    if wanted:
        if wanted not in runs:
            return None, f"run {wanted} not found (runs: {', '.join(runs[-5:]) or 'none'})"
        if not video_counts.get(wanted):
            return None, f"run {wanted} has no task-N/videos/*.mp4"
        return wanted, None
    with_videos = [r for r in runs if video_counts.get(r)]
    if not with_videos:
        return None, "no run with task-N/videos/*.mp4 in this bucket"
    return with_videos[-1], None


def judge_stack_name(judge: str, target_stack: str) -> str:
    """Same as judgeStackName() in cdk/lib/judge-stack.ts."""
    return {"flora-judge": "FloraJudge"}[judge] + "-" + target_stack


def _deploy_judge(args, deployment: dict, region: str, email: str, safe_email: str, safe_region: str) -> None:
    """--judge: score a finished run of --vla with the post-run judge (cdk/bin/judge.ts)."""
    from destroy import STACK_NAMES  # noqa: PLC0415 - same stack-name map as destroy.py

    judge = args.judge
    libero_suite = normalise_libero_suite(args.libero_suite)
    target_stack = oft_stack_name(libero_suite) if args.vla == "openvla-oft" else STACK_NAMES[args.vla]
    account = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
    outputs = gallery.stack_outputs(region, target_stack) or {}
    bucket = outputs.get("S3BucketName") or results_bucket_name(
        deployment.get("s3_results_prefix", "vla-sim-results"), target_stack, region, account)
    s3 = boto3.client("s3", region_name=region)
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        print(f"[error] Results bucket {bucket} not found in {region}: run python deploy.py --vla {args.vla} first.",
              file=sys.stderr)
        sys.exit(1)
    run_ids, counts = [], {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Delimiter="/"):
        run_ids += [p["Prefix"].rstrip("/") for p in page.get("CommonPrefixes", [])]
    for run in sorted(r for r in run_ids if RUN_RE.match(r)):
        if args.run and run != args.run:
            continue
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=run + "/"):
            counts[run] = counts.get(run, 0) + sum(1 for o in page.get("Contents", []) if VIDEO_KEY_RE.match(o["Key"]))
    run, err = pick_run(run_ids, counts, args.run or "")
    if err:
        print(f"[error] {bucket}: {err}", file=sys.stderr)
        sys.exit(1)
    stack_name = judge_stack_name(judge, target_stack)
    try:
        s3.head_object(Bucket=bucket, Key=f"{run}/judge.json")
        print(f"[judge] s3://{bucket}/{run}/judge.json exists and will be replaced")
    except ClientError:
        pass

    print(f"[judge] Judge:   {judge}")
    print(f"[judge] Target:  {args.vla} ({target_stack})")
    print(f"[judge] Run:     {run} ({counts[run]} videos) in s3://{bucket}/")
    print(f"[judge] Region:  {region}")
    print(f"[judge] Stack:   {stack_name}")
    print(f"[judge] Logs:    aws logs tail /{judge}/userdata --follow --region {region}")
    print()
    subprocess.run(  # nosec B603 - sys.executable is the current Python interpreter, not user input
        [sys.executable, "generate.py", "--judge-target", judge, "--output-dir", "assets/userdata"],
        cwd=str(BASE_DIR), check=True,
    )
    print()
    print("[judge] Starting CDK deploy (one GPU instance; ~45-90 min: 54 GB weights + vLLM install + scoring)...")
    cdk_cmd = [
        "npx", "cdk", "deploy", stack_name,
        "--app", "npx ts-node --prefer-ts-exts bin/judge.ts",
        "-c", f"region={safe_region}",
        "-c", f"notify_email={safe_email}",
        "-c", f"judge={judge}",
        "-c", f"judge_target={args.vla}",
        "-c", f"judge_target_stack={target_stack}",
        "-c", f"judge_bucket={bucket}",
        "-c", f"judge_run={run}",
        "--require-approval", "never",
        "--output", f"cdk.out-{judge}-{target_stack.lower()}",
    ]
    subprocess.run(cdk_cmd, cwd=str(BASE_DIR / "cdk"), check=True)  # nosec B603,B607 - hardcoded command list
    print()
    print(f"[judge] Done. judge.json: s3://{bucket}/{run}/judge.json  (judge.txt next to it)")
    print(f"  Summary: python vlasim.py results --vla {shlex.quote(args.vla)} --region {region}")
    suite = f" --libero-suite {shlex.quote(libero_suite)}" if args.vla == "openvla-oft" else ""
    print(f"  To clean up: python destroy.py --judge --vla {shlex.quote(args.vla)}{suite}")


def _resolve_ssm(value: str, region: str) -> str:
    """If value has the form 'ssm:/path', look it up in SSM Parameter Store."""
    if not value.startswith("ssm:"):
        return value
    param_name = value[4:]
    ssm_client = boto3.client("ssm", region_name=region)
    try:
        resp = ssm_client.get_parameter(Name=param_name)
        resolved = resp["Parameter"]["Value"]
        print(f"[ssm] {param_name} → {resolved}")
        return resolved
    except ssm_client.exceptions.ParameterNotFound:
        print(f"[error] SSM parameter not found: {param_name}", file=sys.stderr)
        print("  Bridge mode needs your own inference endpoint. Create this parameter (or put a literal "
              "host:port / vpc-id in the model yaml bridge block), or deploy without --bridge.",
              file=sys.stderr)
        sys.exit(1)


def _validate_email(email: str) -> str:
    if not re.match(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$", email):
        print(f"[error] Invalid email format: {email}", file=sys.stderr)
        sys.exit(1)
    return shlex.quote(email)


def _validate_region(region: str) -> str:
    if not re.match(r"^[a-z]{2,3}(-[a-z]+)+-\d+$", region):
        print(f"[error] Invalid AWS region format: {region}", file=sys.stderr)
        sys.exit(1)
    return shlex.quote(region)


def _validate_vpc_id(vpc_id: str) -> str:
    if not re.match(r"^vpc-[0-9a-f]{8,17}$", vpc_id):
        print(f"[error] Invalid VPC ID format: {vpc_id} (expected: vpc-XXXXXXXX)", file=sys.stderr)
        sys.exit(1)
    return shlex.quote(vpc_id)


def _load_config(vla: str) -> dict:
    """Merge simulator-config.yaml + models/{vla}.yaml. Model settings take precedence."""
    sim_path = BASE_DIR / "simulator-config.yaml"
    model_path = BASE_DIR / "models" / f"{vla}.yaml"
    if not sim_path.exists():
        print(f"[error] Config not found: {sim_path}", file=sys.stderr)
        sys.exit(1)
    if not model_path.exists():
        print(f"[error] Model config not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    sim_cfg = yaml.safe_load(sim_path.read_text())
    model_cfg = yaml.safe_load(model_path.read_text())
    merged_deployment = {**sim_cfg.get("deployment", {}), **model_cfg.get("deployment", {})}
    return {**sim_cfg, **model_cfg, "deployment": merged_deployment}


def _maybe_import_orphan_bucket(
    cdk_dir: str,
    stack_name: str,
    region: str,
    safe_email: str,
    safe_vpc_id: str,
    s3_results_prefix: str,
    extra_ctx: list[str],
    cdk_out_dir: str = "cdk.out",
    vla: str = "",
    libero_suite: str = "",
) -> None:
    """If the S3 bucket is orphaned (RETAINed after its stack was deleted), re-adopt it with cdk import."""
    sts = boto3.client("sts", region_name=region)
    account_id = sts.get_caller_identity()["Account"]
    bucket_name = results_bucket_name(s3_results_prefix, stack_name, region, account_id)

    s3 = boto3.client("s3", region_name=region)
    try:
        s3.head_bucket(Bucket=bucket_name)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchBucket"):
            return
        # 403 = bucket exists but restricted; fall through

    cfn = boto3.client("cloudformation", region_name=region)
    try:
        status = cfn.describe_stacks(StackName=stack_name)["Stacks"][0]["StackStatus"]
    except ClientError:
        status = None
    if status in ("ROLLBACK_COMPLETE", "REVIEW_IN_PROGRESS"):
        # A failed first create leaves the stack in ROLLBACK_COMPLETE (or, after a rejected
        # change set, REVIEW_IN_PROGRESS) with no resources and the RETAINed bucket outside it. `cdk deploy` would delete that stack and create a new one, which then fails
        # with "ResultsBucket ... already exists". Delete it here so the bucket is re-adopted below.
        print(f"[pre-check] {stack_name} is {status} (failed first create) — deleting it ...")
        cfn.delete_stack(StackName=stack_name)
        cfn.get_waiter("stack_delete_complete").wait(StackName=stack_name)
    elif status in ("CREATE_COMPLETE", "UPDATE_COMPLETE", "UPDATE_ROLLBACK_COMPLETE"):
        # A finished run leaves its stack behind (the instance terminates itself). Re-running the target
        # would update that stack, and CloudFormation cannot update the run's WaitCondition
        # ("Update to resource type AWS::CloudFormation::WaitCondition is not supported"). Delete the
        # compute stack — the results bucket is RETAINed — and re-adopt the bucket below, which is what
        # `destroy.py --keep-data` + deploy does. Refuse while the run's instance is still alive.
        outputs = cfn.describe_stacks(StackName=stack_name)["Stacks"][0].get("Outputs", [])
        instance_id = next((o["OutputValue"] for o in outputs if o["OutputKey"] == "InstanceId"), None)
        if instance_id:
            ec2 = boto3.client("ec2", region_name=region)
            try:
                reservations = ec2.describe_instances(InstanceIds=[instance_id])["Reservations"]
                state = reservations[0]["Instances"][0]["State"]["Name"] if reservations else "terminated"
            except ClientError:
                state = "terminated"
            if state not in ("terminated", "shutting-down"):
                print(f"[pre-check] {stack_name}: the previous run's instance {instance_id} is still {state}.")
                print("            Wait for it to finish (or stop it), then run deploy again.")
                sys.exit(1)
        print(f"[pre-check] {stack_name} is {status} from a previous run — deleting the compute stack "
              "(results bucket and gallery are kept) ...")
        cfn.delete_stack(StackName=stack_name)
        cfn.get_waiter("stack_delete_complete").wait(StackName=stack_name)
    elif status is not None:
        return  # Stack still exists (in progress or failed) — let cdk report it

    print(f"[pre-check] Orphan S3 bucket detected: {bucket_name}")
    print("[pre-check] Re-adopting via `cdk import` before deploy ...")

    mapping = {"ResultsBucketA95A2103": {"BucketName": bucket_name}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        json.dump(mapping, fh)
        mapping_file = fh.name

    vla_ctx = ["-c", f"vla={vla}"] if vla else []
    suite_ctx = ["-c", f"libero_suite={shlex.quote(libero_suite)}"] if libero_suite else []
    cdk_import_cmd = [
        "npx", "cdk", "import", stack_name,
        "-c", f"region={shlex.quote(region)}",
        "-c", f"notify_email={safe_email}",
        *vla_ctx,
        *suite_ctx,
        "--resource-mapping", mapping_file,
        "--force",
        "--output", cdk_out_dir,
        *extra_ctx,
    ]
    subprocess.run(cdk_import_cmd, cwd=cdk_dir, check=True)  # nosec B603,B607 - hardcoded command list
    print()


def _robotwin_units(args, config: dict) -> list:
    """RoboTwin run plan for --vla <RoboTwin target> (exits on a bad flag); [] for every other target."""
    if args.vla not in LEROBOT_ROBOTWIN_STACKS:
        if args.robotwin_protocol != "subset" or args.robotwin_shard:
            print(f"[error] --robotwin-protocol / --robotwin-shard apply to {', '.join(LEROBOT_ROBOTWIN_TARGETS)} only.",
                  file=sys.stderr)
            sys.exit(1)
        return []
    if args.bridge or args.collect:
        return []  # rejected below with the target-specific message
    try:
        units = robotwin_plan.plan(config.get("tasks") or [], protocol=args.robotwin_protocol,
                                   shard=args.robotwin_shard)
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)
    hours = robotwin_plan.worst_case_hours(units, ROBOTWIN_EST_SEC_PER_STEP)
    if hours > ROBOTWIN_MAX_LOOP_HOURS:
        need = 1  # smallest N whose round-robin shards all fit
        while any(robotwin_plan.worst_case_hours(
                robotwin_plan.plan(config.get("tasks") or [], protocol=args.robotwin_protocol, shard=f"{k}/{need}"),
                ROBOTWIN_EST_SEC_PER_STEP) > ROBOTWIN_MAX_LOOP_HOURS for k in range(1, need + 1)):
            need += 1
        print(f"[error] RoboTwin plan = {robotwin_plan.episodes(units)} episodes, up to ~{hours:.0f} h of episode loop "
              f"at {ROBOTWIN_EST_SEC_PER_STEP} s/step: one deploy is limited to 12 h (CloudFormation WaitCondition). "
              f"Split it: --robotwin-shard K/{int(need)} for K = 1..{int(need)}.", file=sys.stderr)
        sys.exit(1)
    return units


def main():
    parser = argparse.ArgumentParser(description="RFM Simulation on AWS 1-Click Deploy")
    parser.add_argument("--vla", required=True,
                        choices=VLA_TARGETS,
                        help="Target (model + simulator) to deploy")
    parser.add_argument("--bridge", action="store_true",
                        help="Bridge mode (experimental): forward inference to your own gRPC endpoint instead of a local model")
    parser.add_argument("--email", "-e", metavar="ADDRESS",
                        help="Notification email (overrides simulator-config.yaml)")
    parser.add_argument("--region", "-r", metavar="REGION",
                        help="AWS region (overrides simulator-config.yaml)")
    parser.add_argument(
        "--libero-suite", default=OFT_DEFAULT_SUITE, choices=list(OFT_LIBERO_SUITES),
        help="openvla-oft only: LIBERO suite (default: 10 = LIBERO-Long; `long` aliases `10`)",
    )
    parser.add_argument("--libero-variant", default="libero", choices=list(libero_plus.VARIANTS),
                        help="LIBERO targets: `plus` = LIBERO-Plus robustness sweep (7 perturbation dimensions x N "
                             "variants, 1 episode each) instead of the target's LIBERO tasks; same stack and results "
                             "bucket, so `vlasim results --robustness` compares it with the LIBERO baseline. "
                             "Settings: libero_plus in models/<vla>.yaml (default: libero)")
    parser.add_argument("--perturb", metavar="LIST",
                        help=f"--libero-variant plus: dimensions to sweep, comma-separated "
                             f"({','.join(libero_plus.CATEGORIES)}) or all (default: libero_plus.perturb)")
    parser.add_argument("--perturb-n", type=int, metavar="N",
                        help="--libero-variant plus: variants per dimension (default: libero_plus.variants_per_category; "
                             "0 = every variant, ~350 per dimension)")
    parser.add_argument("--libero-objects", default="libero", choices=list(libero_r2s.OBJECT_SOURCES),
                        help="openvla-oft / pi: `real2sim` replaces LIBERO's HOPE objects by the objects a real2sim "
                             "stage-1 run reconstructed (--real2sim-run) and evaluates public vs swapped scenes on the "
                             "same task ids, initial states and seed (LIBERO-R2S/<condition>); same stack and results "
                             "bucket. Settings: libero_r2s in models/<vla>.yaml (default: libero)")
    parser.add_argument("--real2sim-run", metavar="S3_URI",
                        help="--libero-objects real2sim: s3://<real2sim results bucket>/<RUN_ID> of a finished stage-1 "
                             "run (the stack may read exactly <RUN_ID>/real2sim/segment/ and splat_world.ply)")
    parser.add_argument("--r2s-conditions", metavar="LIST",
                        help=f"--libero-objects real2sim: conditions ({','.join(libero_r2s.CONDITIONS)}; public is "
                             "always included; default: libero_r2s.conditions)")
    parser.add_argument("--r2s-trials", type=int, metavar="N",
                        help="--libero-objects real2sim: trials per selected task and condition (default: "
                             "libero_r2s.trials_per_task)")
    parser.add_argument("--r2s-suite", choices=list(libero_r2s.SUITES),
                        help="pi + --libero-objects real2sim: LIBERO suite (default: libero_r2s.suite = libero_object; "
                             "openvla-oft follows --libero-suite)")
    parser.add_argument("--robotwin-protocol", default="subset", choices=list(robotwin_plan.PROTOCOLS),
                        help="RoboTwin targets (smolvla-robotwin, lingbot-vla2): subset = the tasks in "
                             "models/<vla>.yaml (default, 5 tasks x 5 episodes, clean); clean / randomized = the "
                             "official 49 tasks x 100 episodes of that setting; full = both settings. Anything "
                             "but subset needs --robotwin-shard (one deploy has a 12 h limit)")
    parser.add_argument("--robotwin-shard", default="", metavar="K/N",
                        help="RoboTwin targets: run every N-th (setting, task) unit starting at K (1-based); "
                             "deploy K = 1..N one after the other (same stack, one RUN_ID each)")
    parser.add_argument("--seed", default="model", metavar="model|auto|N",
                        help="Seed for every task of this run: model = models/<vla>.yaml seeds (default); auto = draw "
                             "one now and print it; N = reuse it. Give every target of a comparison the same N "
                             "(docs/results.md \"Shared seed\"); protocol.yaml records it.")
    parser.add_argument("--repeats", default="auto", metavar="auto|N",
                        help="Runs of the whole evaluation in this deploy, each under its own RUN_ID (same stack, "
                             "seed and start states). auto (default) = 2 where an identical re-run does not repeat "
                             "(GPU physics: RoboLab; repeatability.py), 1 elsewhere; --repeats 1 turns it off. "
                             "results --compare needs two runs per side there before it calls a winner.")
    parser.add_argument("--rl-profile", default="smoke", choices=list(RL_PROFILES),
                        help="rlinf-oft-libero: smoke = 1 GPU, LoRA, a few GRPO steps (~2-4 h); full = 8 GPU, recipe "
                             "defaults, the stack completes after the baseline eval and training continues up to "
                             "profiles.full.max_wall_hours (models/<vla>.yaml)")
    parser.add_argument("--rl-steps", type=int, metavar="N",
                        help="rlinf-oft-libero: GRPO steps (default: the profile's steps)")
    parser.add_argument("--rl-eval-run", metavar="RUN_ID",
                        help="rlinf-oft-libero: no training; evaluate the RL checkpoint that run RUN_ID of this stack "
                             "saved to the results bucket (baseline + RL eval, same protocol)")
    parser.add_argument("--no-gallery", action="store_true",
                        help="Do not publish videos to the video history gallery (CloudFront page linked from the "
                             "completion e-mail). Default: on (deployment.gallery in simulator-config.yaml).")
    parser.add_argument("--update-gallery", action="store_true",
                        help="Re-deploy the shared RFM-Gallery stack even if it already exists (after an upgrade).")
    parser.add_argument("--collect", action="store_true",
                        help="gr00t-g1 only: trajectory COLLECTION mode (success-only obs/action HDF5 + "
                             "LeRobotDataset for N1.7 G1 adapter FT) instead of rollout/video.")
    parser.add_argument("--judge", nargs="?", const=JUDGE_TARGETS[0], choices=JUDGE_TARGETS, metavar="JUDGE",
                        help="Post-run judge instead of a simulation: score the rollout videos of a finished --vla "
                             f"run with {JUDGE_TARGETS[0]} (NVIDIA Flora-Rewarder, NON-COMMERCIAL license) and write "
                             "judge.json + judge.txt next to that run.")
    parser.add_argument("--run", metavar="RUN_ID",
                        help="--judge only: run id to score (default: the newest run with videos)")
    args = parser.parse_args()

    if args.run and not args.judge:
        print("[error] --run is only used with --judge.", file=sys.stderr)
        sys.exit(1)
    if args.run and not RUN_RE.match(args.run):
        print(f"[error] --run must look like YYYY-MM-DD-HHMM, got {args.run!r}", file=sys.stderr)
        sys.exit(1)
    if args.judge and (args.bridge or args.collect):
        print("[error] --judge cannot be combined with --bridge or --collect.", file=sys.stderr)
        sys.exit(1)

    if args.collect and args.vla != "gr00t-g1":
        print("[error] --collect is only supported for --vla gr00t-g1.", file=sys.stderr)
        sys.exit(1)
    if args.collect and args.bridge:
        print("[error] --collect is local mode only (bridge delegates inference; no local obs/action "
              "dict to dump).", file=sys.stderr)
        sys.exit(1)

    plus = args.libero_variant == "plus"
    if not plus and (args.perturb is not None or args.perturb_n is not None):
        print("[error] --perturb / --perturb-n need --libero-variant plus.", file=sys.stderr)
        sys.exit(1)
    if plus and (args.judge or args.collect or args.bridge):
        print("[error] --libero-variant plus cannot be combined with --judge, --collect or --bridge.", file=sys.stderr)
        sys.exit(1)

    rl_flags = args.rl_profile != "smoke" or args.rl_steps is not None or args.rl_eval_run
    if rl_flags and args.vla not in RLINF_TARGETS:
        print(f"[error] --rl-profile / --rl-steps / --rl-eval-run apply to {', '.join(RLINF_TARGETS)} only.",
              file=sys.stderr)
        sys.exit(1)
    if args.rl_steps is not None and args.rl_steps < 1:
        print("[error] --rl-steps must be >= 1.", file=sys.stderr)
        sys.exit(1)
    if args.rl_eval_run and not RUN_RE.match(args.rl_eval_run):
        print(f"[error] --rl-eval-run must look like YYYY-MM-DD-HHMM, got {args.rl_eval_run!r}", file=sys.stderr)
        sys.exit(1)
    if args.rl_eval_run and args.rl_steps is not None:
        print("[error] --rl-eval-run evaluates an existing checkpoint; --rl-steps does not apply.", file=sys.stderr)
        sys.exit(1)
    r2s = args.libero_objects == "real2sim"
    if not r2s and any(x is not None for x in (args.real2sim_run, args.r2s_conditions, args.r2s_trials, args.r2s_suite)):
        print("[error] --real2sim-run / --r2s-* need --libero-objects real2sim.", file=sys.stderr)
        sys.exit(1)
    if r2s and (args.judge or args.collect or args.bridge or plus):
        print("[error] --libero-objects real2sim cannot be combined with --judge, --collect, --bridge or "
              "--libero-variant plus.", file=sys.stderr)
        sys.exit(1)

    config = _load_config(args.vla)
    deployment = config.get("deployment", {})
    bridge_cfg = config.get("bridge", {})
    plus_plan = None
    r2s_plan = None
    if r2s:
        if not libero_r2s_supported(args.vla, config):
            print(f"[error] --libero-objects real2sim is not available for --vla {args.vla} (openvla-oft and pi; "
                  "needs a libero_r2s block in models/<vla>.yaml; see docs/targets.md).", file=sys.stderr)
            sys.exit(1)
        if not args.real2sim_run:
            print("[error] --libero-objects real2sim needs --real2sim-run s3://<bucket>/<RUN_ID>.", file=sys.stderr)
            sys.exit(1)
        if args.r2s_suite and args.vla != "pi":
            print("[error] --r2s-suite applies to pi (openvla-oft follows --libero-suite).", file=sys.stderr)
            sys.exit(1)
        r2s_suite = args.r2s_suite or r2s_default_suite(config, args.vla, args.libero_suite)
        try:
            r2s_plan, _ = libero_r2s.plan(config["libero_r2s"], args.real2sim_run, r2s_suite,
                                                  conditions=args.r2s_conditions, trials=args.r2s_trials)
        except ValueError as e:
            print(f"[error] {e}", file=sys.stderr)
            sys.exit(1)
    if plus:
        if not libero_plus_supported(args.vla, config):
            print(f"[error] --libero-variant plus is not available for --vla {args.vla} (needs a libero_plus "
                  "block in models/<vla>.yaml; see docs/targets.md#libero-plus-robustness-sweep).", file=sys.stderr)
            sys.exit(1)
        try:
            suite_override = None
            if args.vla == "openvla-oft":
                suite_override = config["model"]["suites"][normalise_libero_suite(args.libero_suite)]["task_suite_name"]
            plus_plan, _ = libero_plus.plan(config["libero_plus"], perturb=args.perturb, n=args.perturb_n,
                                            suite=suite_override)
        except ValueError as e:
            print(f"[error] {e}", file=sys.stderr)
            sys.exit(1)
    robotwin_units = _robotwin_units(args, config)

    region = args.region or deployment["region"]
    email = args.email or deployment["notify_email"]

    if email == "YOUR_EMAIL@example.com":
        print("[error] Email not set. Use --email YOUR@EMAIL.COM or edit notify_email in "
              "simulator-config.yaml", file=sys.stderr)
        sys.exit(1)

    safe_email = _validate_email(email)
    safe_region = _validate_region(region)

    if args.judge:
        _deploy_judge(args, deployment, region, email, safe_email, safe_region)
        return

    # ── Bridge mode resolution ────────────────────────────────────
    vpc_id = ""
    extra_cdk_ctx: list[str] = []
    generate_extra: list[str] = []
    seed_args: list[str] = []
    try:
        seed_plan = seeds.resolve(args.seed)
    except ValueError as e:
        parser.error(str(e))
    if seed_plan["value"] is not None:
        # Resolve once here so the drawn value is printed and the same one reaches generate.py.
        seed_args = ["--seed", str(seed_plan["value"])]
        if seed_plan["drawn_utc"]:
            seed_args += ["--seed-drawn-utc", seed_plan["drawn_utc"]]
        ctl = seeds.control(args.vla)
        if seed_plan["mode"] == "drawn":
            print(f"[seed] drew {seed_plan['value']} at {seed_plan['drawn_utc']} — deploy every other target of "
                  f"this comparison with --seed {seed_plan['value']}")
        if ctl.get("where") is None:
            print(f"[seed] {args.vla} cannot be seeded by the tool: {ctl.get('reason')} "
                  "(protocol.yaml records it as not-controllable)")

    try:
        n_repeats, repeats_why = repeatability.resolve_repeats(args.vla, args.repeats)
    except ValueError as e:
        parser.error(str(e))
    repeat_args = ["--repeats", str(n_repeats)]
    rplan = robolab_plan(config, args.vla) if args.vla in ROBOLAB_TARGETS else {}
    if n_repeats > 1:
        print(f"[runs] {n_repeats} runs in this deploy: {repeats_why}")
        fit_error = repeatability.check_fit(args.vla, n_repeats, rplan)
        if fit_error:
            print(f"[error] {fit_error}", file=sys.stderr)
            sys.exit(1)
    elif repeatability.for_target(args.vla)["repeatable"] == "no":
        print(f"[warn] one run in this deploy: {repeats_why}")

    if args.bridge:
        if args.vla == "openvla-oft":
            print("[error] Bridge mode not supported for openvla-oft (local only).", file=sys.stderr)
            sys.exit(1)
        if args.vla == "gr00t-bridge":
            print("[error] Bridge mode not supported for gr00t-bridge (local only — the remote gRPC bridge is "
                  "wired for the LIBERO rollout venv). The name refers to the SimplerEnv Bridge benchmark.",
                  file=sys.stderr)
            sys.exit(1)
        if args.vla == "molmoact2":
            print("[error] Bridge mode not supported for molmoact2 (local only — MolmoAct2Policy runs in-process via lerobot-eval on the sim GPU).",
                  file=sys.stderr)
            sys.exit(1)
        if args.vla in LEROBOT_LIBERO_TARGETS or args.vla in OTHER_LOCAL_STACKS or args.vla in LEROBOT_ROBOTWIN_STACKS:
            print(f"[error] Bridge mode not supported for {args.vla} (local only — the policy runs in-process on the sim GPU).",
                  file=sys.stderr)
            sys.exit(1)
        if args.vla == "openarm-isaac":
            print("[error] Bridge mode not supported for openarm-isaac (local only — LeRobot pi05 runs in-process on the sim GPU).",
                  file=sys.stderr)
            sys.exit(1)
        if args.vla == "openarm-lift-act":
            print("[error] Bridge mode not supported for openarm-lift-act (local only — scripted demo collection, no policy server).",
                  file=sys.stderr)
            sys.exit(1)
        if args.vla.startswith("rldx"):
            print(f"[error] Bridge mode not supported for {args.vla} (local only — fused-kernel/ZeroMQ serving "
                  "is not supported by the bridge gRPC adapters).", file=sys.stderr)
            sys.exit(1)
        if args.vla in ("gr00t", "gr00t-gr1", "gr00t-g1"):
            raw_grpc = str(bridge_cfg.get("remote_grpc_endpoint", "")).strip()
            raw_vpc = str(bridge_cfg.get("vpc_id", "")).strip()
            if not raw_grpc or not raw_vpc:
                print(f"[error] Bridge mode requires bridge.remote_grpc_endpoint and bridge.vpc_id "
                      f"in models/{args.vla}.yaml", file=sys.stderr)
                sys.exit(1)
            resolved_grpc = _resolve_ssm(raw_grpc, region)
            resolved_vpc = _resolve_ssm(raw_vpc, region)
            vpc_id = resolved_vpc
            safe_vpc = _validate_vpc_id(vpc_id)
            extra_cdk_ctx = ["-c", f"vpc_id={safe_vpc}"]
            generate_extra = ["--resolved-grpc", resolved_grpc, "--resolved-vpc", resolved_vpc]
        else:  # pi, lap — both use NLB endpoint + openpi WebSocket bridge
            raw_vpc = str(bridge_cfg.get("vpc_id", "")).strip()
            raw_nlb = str(bridge_cfg.get("nlb_endpoint", "")).strip()
            if not raw_vpc or not raw_nlb:
                print(f"[error] Bridge mode requires bridge.vpc_id and bridge.nlb_endpoint "
                      f"in models/{args.vla}.yaml", file=sys.stderr)
                sys.exit(1)
            resolved_vpc = _resolve_ssm(raw_vpc, region)
            resolved_nlb = _resolve_ssm(raw_nlb, region) if raw_nlb.startswith("ssm:") else raw_nlb
            vpc_id = resolved_vpc
            safe_vpc = _validate_vpc_id(vpc_id)
            extra_cdk_ctx = ["-c", f"vpc_id={safe_vpc}", "-c", f"nlb_endpoint={shlex.quote(resolved_nlb)}"]
            generate_extra = ["--resolved-vpc", resolved_vpc, "--resolved-nlb", resolved_nlb]

    libero_suite = normalise_libero_suite(args.libero_suite)
    if args.vla == "gr00t":
        stack_name = "GR00T-Demo"
    elif args.vla == "gr00t-bridge":
        stack_name = "GR00T-Bridge-Demo"
    elif args.vla == "gr00t-gr1":
        stack_name = "GR00T-GR1-Demo"
    elif args.vla == "gr00t-g1":
        stack_name = "GR00T-G1-Demo"
    elif args.vla == "openvla-oft":
        stack_name = oft_stack_name(libero_suite)
    elif args.vla == "lap":
        stack_name = "LAP-Demo"
    elif args.vla == "rldx":
        stack_name = "RLDX-Demo"
    elif args.vla == "rldx-simpler":
        stack_name = "RLDX-Simpler-Demo"
    elif args.vla == "rldx-bridge":
        stack_name = "RLDX-Bridge-Demo"
    elif args.vla == "rldx-gr1":
        stack_name = "RLDX-GR1-Demo"
    elif args.vla == "rldx-kitchen":
        stack_name = "RLDX-Kitchen-Demo"
    elif args.vla == "openarm-isaac":
        stack_name = "OpenArm-Isaac-Demo"
    elif args.vla == "openarm-lift-act":
        stack_name = "OpenArm-Lift-ACT-Demo"
    elif args.vla == "molmoact2":
        stack_name = "MolmoAct2-Demo"
    elif args.vla in LEROBOT_LIBERO_STACKS:
        stack_name = LEROBOT_LIBERO_STACKS[args.vla]
    elif args.vla in OTHER_LOCAL_STACKS:
        stack_name = OTHER_LOCAL_STACKS[args.vla]
        if args.vla in RLINF_TARGETS:
            generate_extra += ["--rl-profile", args.rl_profile]
            if args.rl_steps is not None:
                generate_extra += ["--rl-steps", str(int(args.rl_steps))]
            if args.rl_eval_run:
                generate_extra += ["--rl-eval-run", args.rl_eval_run]
            # Selects profiles.<profile>.instance (instance list, EBS, WaitCondition timeout); same stack + bucket.
            extra_cdk_ctx += ["-c", f"rl_profile={args.rl_profile}"]
    elif args.vla in LEROBOT_ROBOTWIN_STACKS:
        stack_name = LEROBOT_ROBOTWIN_STACKS[args.vla]
        generate_extra += ["--robotwin-protocol", args.robotwin_protocol]
        if args.robotwin_shard:
            generate_extra += ["--robotwin-shard", args.robotwin_shard]
        # Selects the protocol's entry of instance.creationpolicy_timeout (same stack + results bucket).
        extra_cdk_ctx += ["-c", f"robotwin_protocol={args.robotwin_protocol}"]
    else:
        stack_name = "Pi-Demo"
    mode = "collect" if args.collect else ("bridge" if args.bridge else "local")
    s3_results_prefix = deployment.get("s3_results_prefix", "vla-sim-results")
    cdk_dir = str(BASE_DIR / "cdk")

    print(f"[deploy] VLA:    {args.vla}")
    print(f"[deploy] Region: {region}")
    print(f"[deploy] Email:  {email}")
    print(f"[deploy] Mode:   {mode}")
    print(f"[deploy] Stack:  {stack_name}")
    if args.vla == "openvla-oft":
        print(f"[deploy] Suite:  libero_{libero_suite}")
    if plus_plan:
        n_ep = libero_plus.episodes(plus_plan)
        print(f"[deploy] LIBERO-Plus: {plus_plan['suite']}, {','.join(plus_plan['perturb'])} x "
              f"{plus_plan['variants_per_category'] or 'all'} variants = {n_ep or 'all'} episodes")
    if r2s_plan:
        print(f"[deploy] LIBERO-R2S: {r2s_plan['suite']}, {','.join(r2s_plan['conditions'])} x "
              f"{r2s_plan['trials_per_task']} trials per selected task, objects from {r2s_plan['run_uri']}/")
    if robotwin_units:
        shard = f" shard {args.robotwin_shard}" if args.robotwin_shard else ""
        print(f"[deploy] RoboTwin: {args.robotwin_protocol}{shard}, {len(robotwin_units)} task run(s), "
              f"{robotwin_plan.episodes(robotwin_units)} episodes")
    if vpc_id:
        print(f"[deploy] VPC:    {vpc_id}")
    runs_line = repeatability.plan_line(args.vla, n_repeats, rplan.get("episodes"))
    if runs_line:
        print(f"[deploy] Runs:   {runs_line}")
    print()

    # 1. Generate UserData
    print("[1/2] Generating UserData script...")
    oft_suite_args: list[str] = []
    if args.vla == "openvla-oft":
        oft_suite_args = ["--libero-suite", shlex.quote(libero_suite)]
    collect_args: list[str] = ["--collect"] if args.collect else []
    plus_args: list[str] = []
    plus_cdk_ctx: list[str] = []
    if plus:
        plus_args = ["--libero-variant", "plus"]
        if args.perturb is not None:
            plus_args += ["--perturb", ",".join(plus_plan["perturb"])]
        if args.perturb_n is not None:
            plus_args += ["--perturb-n", str(plus_plan["variants_per_category"])]
        # Same stack + results bucket as the LIBERO run; the context only raises the timeout / EBS size.
        plus_cdk_ctx = ["-c", "libero_variant=plus"]
    if r2s_plan:
        plus_args = ["--libero-objects", "real2sim", "--real2sim-run", args.real2sim_run]
        if args.r2s_conditions is not None:
            plus_args += ["--r2s-conditions", ",".join(r2s_plan["conditions"])]
        if args.r2s_trials is not None:
            plus_args += ["--r2s-trials", str(r2s_plan["trials_per_task"])]
        if args.r2s_suite:
            plus_args += ["--r2s-suite", args.r2s_suite]
        # Same stack + results bucket as the LIBERO run; the context grants s3:GetObject on exactly the stage-1
        # outputs the UserData downloads and raises the timeout / EBS size (libero_r2s in models/<vla>.yaml).
        plus_cdk_ctx = ["-c", "libero_objects=real2sim", "-c", f"real2sim_run={r2s_plan['run_uri']}"]
    subprocess.run(  # nosec B603 - sys.executable is the current Python interpreter, not user input
        [
            sys.executable, "generate.py",
            "--vla", args.vla,
            "--output-dir", "assets/userdata",
            *oft_suite_args,
            *collect_args,
            *plus_args,
            *generate_extra,
            *seed_args,
            *repeat_args,
        ],
        cwd=str(BASE_DIR),
        check=True,
    )
    print()

    # Use a VLA-specific output dir so parallel deploys don't clash on cdk.out.
    # For OpenVLA-OFT, include the suite so e.g. spatial + 10 can synth independently.
    if args.vla == "openvla-oft" and libero_suite != OFT_DEFAULT_SUITE:
        cdk_out_dir = f"cdk.out-{args.vla}-{libero_suite}"
    else:
        cdk_out_dir = f"cdk.out-{args.vla}"

    # 1.2 Video history gallery (shared RFM-Gallery stack + this e-mail's password)
    gallery_ctx: list[str] = []
    gallery_info = None
    if deployment.get("gallery", True) and not args.no_gallery:
        gallery_info = gallery.ensure_gallery(region, email, update_stack=args.update_gallery)
        gallery_ctx = [
            "-c", f"gallery_bucket={gallery_info['bucket']}",
            "-c", f"gallery_hash={gallery_info['id']}",
            "-c", f"gallery_url={gallery_info['url']}",
        ]
        print(f"[gallery] Video history: {gallery_info['url']}")
        print()

    # 1.5 Pre-deploy: re-adopt orphan S3 bucket
    _maybe_import_orphan_bucket(
        cdk_dir, stack_name, region, safe_email,
        shlex.quote(vpc_id) if vpc_id else "",
        s3_results_prefix, [*extra_cdk_ctx, *gallery_ctx, *plus_cdk_ctx],
        cdk_out_dir=cdk_out_dir,
        vla=args.vla,
        libero_suite=libero_suite if args.vla == "openvla-oft" else "",
    )

    # 2. CDK deploy
    print("[2/2] Starting CDK deploy...")
    if mode == "bridge":
        if args.vla in ("gr00t", "gr00t-gr1", "gr00t-g1"):
            print("  Estimated time: ~60 min (install ~20-30min + bridge eval ~30min)")
        else:
            print("  Estimated time: ~30-40 min (install ~10min + bridge eval ~20-30min)")
    elif r2s_plan:
        n_ep = libero_r2s.episodes_hint(r2s_plan)
        print(f"  Estimated time: install as for LIBERO + stage-1 download, object build and physics check ~5 min "
              f"(pi: + LIBERO client image ~10 min) + about {n_ep} episodes (5 libero_object tasks of the HOPE-Video "
              f"scene; fewer if the physics check keeps an object stock) at ~45 s (OpenVLA-OFT) / ~40 s (pi0.5) on L4")
    elif plus_plan:
        n_ep = libero_plus.episodes(plus_plan)
        print(f"  Estimated time: install as for LIBERO + LIBERO-Plus fork and {plus_plan['assets_bytes'] / 1e9:.1f} GB "
              f"assets ~10 min + {n_ep or 'all'} episodes (~15-30 s each for SmolVLA, ~45 s for OpenVLA-OFT on L4)")
    else:
        if args.vla == "gr00t":
            print("  Estimated time: ~120 min (install ~30min + model download ~10min + sim ~80min)")
        elif args.vla == "gr00t-bridge":
            print("  Estimated time: ~90-150 min (uv sync + simpler_uv venv + ManiSkill2_real2sim ~30-40min + 6.9GB ckpt + backbone DL ~10min + 4 tasks x (server load ~5-8min + 10 SAPIEN episodes))")
        elif args.vla == "gr00t-gr1":
            print("  Estimated time: ~90 min (install ~20min + model download ~5min + sim ~60min)")
        elif args.vla == "gr00t-g1":
            print("  Estimated time: ~120-150 min (WBC install ~25min [submodule+Git-LFS+3 editable] + ckpt DL + loco-manip sim ~60-90min)")
        elif args.vla == "openvla-oft":
            if libero_suite == "10":
                print("  Estimated time: ~180 min (conda/pip ~30min + HF download ~10min + LIBERO-10 eval ~120min)")
            else:
                print(f"  Estimated time: ~90 min (conda/pip ~30min + HF download ~10min + LIBERO-{libero_suite} eval ~45min)")
        elif args.vla == "lap":
            print("  Estimated time: ~90-150 min (uv venvs ~25min + HF download ~10min + LIBERO-Spatial eval ~30-50min + buffer)")
        elif args.vla == "rldx":
            print("  Estimated time: ~80-120 min (uv sync + libero_uv venv ~30-40min + ckpt+backbone DL ~15min + server warmup ~3min + eager rollout)")
        elif args.vla == "rldx-simpler":
            print("  Estimated time: ~90-140 min (uv sync + simpler_uv venv + ManiSkill2_real2sim clone ~35-45min + ckpt+backbone DL ~15min + server warmup ~3min + SAPIEN eager rollout)")
        elif args.vla == "rldx-bridge":
            print("  Estimated time: ~100-160 min (uv sync + simpler_uv venv + ManiSkill2_real2sim clone ~35-45min + ckpt+backbone DL ~15min + server warmup up to ~18min + 4 tasks x 10 SAPIEN eager episodes)")
        elif args.vla == "rldx-gr1":
            print("  Estimated time: ~25-35 min (Gate-3 measured 2026-06-25, g5.2xlarge: uv syncs + robocasa_uv venv + tabletop assets ~12min + ckpt+backbone DL ~4min + fast server warmup ~210s + MuJoCo eager rollout ~10min)")
        elif args.vla == "rldx-kitchen":
            print("  Estimated time: ~25-35 min (Gate-3 measured 2026-06-26, g6.2xlarge: uv syncs + robocasa_uv venv + kitchen asset DL ~3-4min + ckpt+backbone DL ~4min + server warmup ~300s + MuJoCo eager rollout ~10min)")
        elif args.vla == "openarm-isaac":
            print("  Estimated time: ~120-180 min (docker pull Isaac Lab image ~20-30min + openarm/lerobot install + ckpt DL + sim boot ~10min + single rollout)")
        elif args.vla == "openarm-lift-act":
            print("  Estimated time: ~60-90 min (docker pull Isaac Lab image ~20-30min + openarm install + sim boot ~10min + scripted collection; first run small num_demos)")
        elif args.vla == "molmoact2":
            print("  Estimated time: ~60-90 min for the pipe-proof cap (uv sync fork source ~20-30min + MolmoAct2-LIBERO ckpt ~22GB DL ~10min + fp32 LIBERO eval 2 tasks x 5 ep). Full suite (all 10 tasks x 50 ep) is MANY hours in fp32 — scale via models/molmoact2.yaml tasks.")
        elif args.vla == "real2sim":
            print("  Estimated time: ~12 min measured on g5.2xlarge (packages + two uv venvs + HOPE-Video download ~5min + splatfacto ~5min + segment + MuJoCo grasp replay ~2min), plus stack create/delete")
        elif args.vla in ("robolab-cosmos3", "robolab-pi05"):
            print("  Estimated time: ~4-6 h for the default 12-task subset (ECC-off reboot + RoboLab LFS assets ~6 GB + "
                  "Isaac Lab / policy-server image builds ~45-75 min + checkpoint DL + 12 tasks x 5 episodes ~2-3 h); "
                  "model.variant in models/<vla>.yaml selects nano / the full RoboLab-120 protocol (see docs/targets.md)"
                  + (f"; the episode part runs {n_repeats} times (--repeats)" if n_repeats > 1 else ""))
        elif args.vla in RLINF_TARGETS:
            prof = (config.get("profiles") or {}).get(args.rl_profile) or {}
            steps = args.rl_steps or prof.get("steps")
            if args.rl_eval_run:
                print(f"  Estimated time: ~2-3 h (image pull ~15 min + verify + SFT ckpt ~10 min + two evals of "
                      f"{prof.get('eval_num_envs')} episodes); RL checkpoint of run {args.rl_eval_run}")
            elif prof.get("early_signal"):
                print(f"  Estimated time: the stack completes after setup + baseline eval (~1.5-3 h); {steps} GRPO steps, "
                      f"the RL eval and the report follow by e-mail (cap {prof.get('max_wall_hours')} h, then the "
                      "instance terminates itself). Time per step is not published; first runs measure it.")
            else:
                print(f"  Estimated time: ~2-4 h (image pull ~15 min + verify + SFT ckpt ~10 min + baseline eval + "
                      f"{steps} GRPO steps + RL eval); not measured yet")
        elif args.vla == "cosmos-policy":
            print("  Estimated time: ~60-120 min (docker build ~10-15min + uv sync ~10-20min + ckpt DL + LIBERO eval, 1 ep x 10 tasks)")
        elif args.vla in LEROBOT_ROBOTWIN_TARGETS:
            hours = robotwin_plan.worst_case_hours(robotwin_units, ROBOTWIN_EST_SEC_PER_STEP)
            print("  Estimated time: ~60-90 min setup (uv sync + RoboTwin stack + CuRobo build + ~4 GB assets"
                  f"{' + ~11 GB textures' if any(u['setting'] == 'randomized' for u in robotwin_units) else ''}) + "
                  f"episode loop up to ~{hours:.1f} h (every episode at its step limit, "
                  f"{ROBOTWIN_EST_SEC_PER_STEP} s/step; not measured yet)")
        elif args.vla in LEROBOT_LIBERO_TARGETS:
            print("  Estimated time: ~40-80 min for the pipe-proof cap (uv sync LeRobot source ~15-30min + ckpt DL + LIBERO eval; scale via models/<vla>.yaml tasks)")
        else:
            print("  Estimated time: ~90-120 min per suite (install ~30min + eval ~60-90min)")
    print("  If this is your first deploy, confirm the SNS subscription email to receive notifications.")
    print()

    oft_cdk_ctx: list[str] = []
    if args.vla == "openvla-oft":
        oft_cdk_ctx = ["-c", f"libero_suite={shlex.quote(libero_suite)}"]
    cdk_cmd = [
        "npx", "cdk", "deploy", stack_name,
        "-c", f"region={safe_region}",
        "-c", f"notify_email={safe_email}",
        "-c", f"vla={args.vla}",
        *oft_cdk_ctx,
        *plus_cdk_ctx,
        "--require-approval", "never",
        "--output", cdk_out_dir,
        *extra_cdk_ctx,
        *gallery_ctx,
    ]
    subprocess.run(cdk_cmd, cwd=cdk_dir, check=True)  # nosec B603,B607 - hardcoded command list

    # 3. Gallery backfill: runs from before the gallery existed (results bucket kept by destroy)
    if gallery_info:
        outputs = gallery.stack_outputs(region, stack_name) or {}
        if outputs.get("S3BucketName") and outputs.get("GalleryPrefix"):
            try:
                n = gallery.backfill(region, outputs["S3BucketName"], gallery_info["bucket"], outputs["GalleryPrefix"])
                if n:
                    print(f"[gallery] Added {n} files from earlier runs to the video history")
            except Exception as e:  # noqa: BLE001 - backfill is best effort; the run itself succeeded
                print(f"[gallery] [warn] Backfill of earlier runs failed: {e}")

    print()
    print("[deploy] Done! Simulation is running in the background.")
    print(f"  A completion notification will be sent to: {email}")
    if gallery_info:
        print(f"  Video history (all your runs, by model): {gallery_info['url']}")
        print("  Password: python vlasim.py gallery")
    vla_safe = shlex.quote(args.vla)
    if plus_plan:
        print(f"  Robustness table (after the run): python vlasim.py results --robustness --vla {stack_name.lower()}")
    if args.vla == "openvla-oft" and libero_suite != OFT_DEFAULT_SUITE:
        print(f"  To clean up: python destroy.py --vla {vla_safe} --libero-suite {shlex.quote(libero_suite)}")
    else:
        print(f"  To clean up: python destroy.py --vla {vla_safe}")


if __name__ == "__main__":
    main()
