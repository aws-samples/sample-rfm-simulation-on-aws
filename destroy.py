#!/usr/bin/env python3
"""
RFM Simulation on AWS Stack Destroy
Usage: python destroy.py --vla gr00t        [--region REGION] [--keep-data | --delete-all [--yes]]
       python destroy.py --vla gr00t-gr1    [--region REGION]
       python destroy.py --vla pi           [--region REGION]
       python destroy.py --vla openvla-oft  [--region REGION] [--libero-suite SUITE]
       python destroy.py --vla lap          [--region REGION]
       python destroy.py --vla openarm-isaac [--region REGION]
       python destroy.py --vla openarm-lift-act [--region REGION]
       python destroy.py --judge --vla smolvla [--region REGION]   # the Flora judge stack of smolvla

Data choice:
  --keep-data   Delete the compute stack only. The S3 results bucket (RemovalPolicy.RETAIN) and the
                video history gallery (S3 + CloudFront) stay. Default without a TTY.
  --delete-all  Also empty + delete the results bucket and remove this target's videos from the
                gallery. When no other target stack is left in the region, the shared RFM-Gallery
                stack, its bucket, the CloudFront KeyValueStore and the gallery SSM parameters go too.
                Asks to type the stack name unless --yes.
  Neither flag on a TTY: numbered prompt (default 1 = keep data).
"""

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

import gallery
from generate import (
    OFT_DEFAULT_SUITE,
    OFT_LIBERO_SUITES,
    normalise_libero_suite,
    oft_stack_name,
)
from targets import JUDGE_TARGETS, VLA_TARGETS

BASE_DIR = Path(__file__).parent

STACK_NAMES = {
    "gr00t": "GR00T-Demo",
    "gr00t-bridge": "GR00T-Bridge-Demo",
    "gr00t-gr1": "GR00T-GR1-Demo",
    "gr00t-g1": "GR00T-G1-Demo",
    "pi": "Pi-Demo",
    "openvla-oft": "OpenVLA-OFT-Demo",   # default suite (10); non-default via oft_stack_name()
    "lap": "LAP-Demo",
    "rldx": "RLDX-Demo",
    "rldx-simpler": "RLDX-Simpler-Demo",
    "rldx-bridge": "RLDX-Bridge-Demo",
    "rldx-gr1": "RLDX-GR1-Demo",
    "rldx-kitchen": "RLDX-Kitchen-Demo",
    "openarm-isaac": "OpenArm-Isaac-Demo",
    "openarm-lift-act": "OpenArm-Lift-ACT-Demo",
    "molmoact2": "MolmoAct2-Demo",
    "xvla": "XVLA-Demo",
    "smolvla": "SmolVLA-Demo",
    "lingbot-va": "LingBot-VA-Demo",
    "vla-adapter": "VLA-Adapter-Demo",
    "fastwam": "FastWAM-Demo",
    "cosmos-policy": "Cosmos-Policy-Demo",
    "aspire": "ASPIRE-Demo",
    "real2sim": "Real2Sim-Demo",
    "robolab-cosmos3": "RoboLab-Cosmos3-Demo",
    "robolab-pi05": "RoboLab-Pi05-Demo",
}


def _validate_region(region: str) -> str:
    if not re.match(r"^[a-z]{2,3}(-[a-z]+)+-\d+$", region):
        print(f"[error] Invalid AWS region format: {region}", file=sys.stderr)
        sys.exit(1)
    return shlex.quote(region)


def main():
    parser = argparse.ArgumentParser(description="RFM Simulation on AWS Stack Destroy")
    parser.add_argument("--vla", required=True,
                        choices=VLA_TARGETS,
                        help="Target stack to destroy")
    parser.add_argument("--region", "-r", metavar="REGION",
                        help="AWS region (overrides simulator-config.yaml)")
    parser.add_argument(
        "--libero-suite", default=OFT_DEFAULT_SUITE, choices=list(OFT_LIBERO_SUITES),
        help="openvla-oft only: LIBERO suite of the stack to destroy (default: 10)",
    )
    data = parser.add_mutually_exclusive_group()
    data.add_argument("--keep-data", action="store_true",
                      help="Delete the compute stack only; keep the results bucket and the video history gallery")
    data.add_argument("--delete-all", action="store_true",
                      help="Also delete the results bucket and this target's gallery videos "
                           "(and the whole gallery when no other target stack is left in the region)")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="--delete-all: do not ask to type the stack name")
    parser.add_argument("--judge", nargs="?", const=JUDGE_TARGETS[0], choices=JUDGE_TARGETS, metavar="JUDGE",
                        help="Delete the post-run judge stack of --vla (deploy.py --judge) instead of the target. "
                             "judge.json/judge.txt stay next to the run in the target's results bucket.")
    args = parser.parse_args()

    if args.judge:
        if args.delete_all:
            print("[error] --judge has no data of its own (judge.json lives in the target's results bucket); "
                  "use destroy.py --vla <target> --delete-all for that.", file=sys.stderr)
            sys.exit(1)
        _destroy_judge(args)
        return

    sim_path = BASE_DIR / "simulator-config.yaml"
    config = yaml.safe_load(sim_path.read_text())
    region = args.region or config["deployment"]["region"]
    safe_region = _validate_region(region)

    libero_suite = normalise_libero_suite(args.libero_suite)
    if args.vla == "openvla-oft":
        stack_name = oft_stack_name(libero_suite)
    else:
        stack_name = STACK_NAMES[args.vla]

    print(f"[destroy] VLA:    {args.vla}")
    if args.vla == "openvla-oft":
        print(f"[destroy] Suite:  libero_{libero_suite}")
    print(f"[destroy] Stack:  {stack_name}")
    print(f"[destroy] Region: {region}")

    choice = gallery.resolve_destroy_choice(args.keep_data, args.delete_all, sys.stdin.isatty())
    if choice == gallery.DELETE_ALL and not args.yes:
        if not sys.stdin.isatty():
            print("[error] --delete-all without a TTY needs --yes.", file=sys.stderr)
            sys.exit(1)
        typed = input(f"This permanently deletes the results and videos of {stack_name}. "
                      f"Type the stack name to confirm: ").strip()
        if typed != stack_name:
            print("[destroy] Stack name did not match; nothing was deleted.")
            sys.exit(1)

    # Read what to clean up before the stack (and its outputs) are gone.
    outputs = {}
    if choice == gallery.DELETE_ALL:
        outputs = _stack_outputs(region, stack_name)
    if choice == gallery.KEEP:
        print("[destroy] Deleting CDK stack... (S3 bucket and video history gallery will be retained)")
    else:
        print("[destroy] Deleting CDK stack, then its results bucket and gallery videos...")
    print()

    oft_suite_ctx: list[str] = []
    if args.vla == "openvla-oft":
        oft_suite_ctx = ["-c", f"libero_suite={shlex.quote(libero_suite)}"]

    # `cdk destroy` synthesizes the app first, and the stack refuses to synth without the
    # rendered UserData (assets/userdata/<vla>.sh, gitignored). Render it when it is missing so
    # destroy also works from a fresh checkout or a different machine than the one that deployed.
    if not (BASE_DIR / "assets" / "userdata" / f"{args.vla}.sh").exists():
        subprocess.run(  # nosec B603 - sys.executable is the current Python interpreter, not user input
            [
                sys.executable, "generate.py",
                "--vla", args.vla,
                "--output-dir", "assets/userdata",
                *(["--libero-suite", shlex.quote(libero_suite)] if args.vla == "openvla-oft" else []),
            ],
            cwd=str(BASE_DIR),
            check=True,
        )

    # Same VLA-specific output dir as deploy.py, so parallel destroys don't clash on cdk.out.
    if args.vla == "openvla-oft" and libero_suite != OFT_DEFAULT_SUITE:
        cdk_out_dir = f"cdk.out-{args.vla}-{libero_suite}"
    else:
        cdk_out_dir = f"cdk.out-{args.vla}"

    subprocess.run(  # nosec B603,B607 - hardcoded command list, no user input in executable path
        [
            "npx", "cdk", "destroy", stack_name,
            "-c", f"region={safe_region}",
            "-c", f"vla={args.vla}",
            *oft_suite_ctx,
            "--force",
            "--output", cdk_out_dir,
        ],
        cwd=str(BASE_DIR / "cdk"),
        check=True,
    )

    print()
    print("[destroy] Stack deleted.")
    if choice == gallery.KEEP:
        print("  The S3 results bucket and the video history gallery still exist.")
        print(f"  To delete them as well: python destroy.py --vla {shlex.quote(args.vla)}"
              + (f" --libero-suite {shlex.quote(libero_suite)}" if args.vla == "openvla-oft" else "")
              + " --delete-all")
        return
    _delete_data(region, stack_name, outputs)


def _destroy_judge(args) -> None:
    """Delete the judge stack (VPC, SNS, AzSelector; no bucket). CloudFormation directly: a cdk destroy
    would have to re-synthesize the judge app with the run it scored."""
    import boto3  # noqa: PLC0415

    from deploy import judge_stack_name  # noqa: PLC0415

    config = yaml.safe_load((BASE_DIR / "simulator-config.yaml").read_text())
    region = args.region or config["deployment"]["region"]
    _validate_region(region)
    suite = normalise_libero_suite(args.libero_suite)
    target_stack = oft_stack_name(suite) if args.vla == "openvla-oft" else STACK_NAMES[args.vla]
    stack_name = judge_stack_name(args.judge, target_stack)
    print(f"[destroy] Judge stack: {stack_name} ({region})")
    if not _stack_exists(region, stack_name):
        print("[destroy] Not found (already deleted?).")
        return
    cfn = boto3.client("cloudformation", region_name=region)
    cfn.delete_stack(StackName=stack_name)
    cfn.get_waiter("stack_delete_complete").wait(StackName=stack_name, WaiterConfig={"Delay": 15, "MaxAttempts": 160})
    print("[destroy] Judge stack deleted. judge.json / judge.txt stay next to the judged run.")


def _stack_outputs(region: str, stack_name: str) -> dict:
    import boto3  # noqa: PLC0415
    from botocore.exceptions import ClientError  # noqa: PLC0415

    try:
        stack = boto3.client("cloudformation", region_name=region).describe_stacks(StackName=stack_name)["Stacks"][0]
    except ClientError:
        return {}
    return {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}


def _results_bucket_name(stack_name: str, region: str, account: str) -> str:
    """Same rule as the CDK stack / deploy.py: drop "-demo" only when the name overflows 63 chars."""
    config = yaml.safe_load((BASE_DIR / "simulator-config.yaml").read_text())
    prefix = config.get("deployment", {}).get("s3_results_prefix", "vla-sim-results")
    name = f"{prefix}-{stack_name.lower()}-{region}-{account}"
    if len(name) > 63:
        name = f"{prefix}-{stack_name.lower().removesuffix('-demo')}-{region}-{account}"
    return name


def _remaining_target_stacks(region: str, exclude: str) -> list:
    """Target stacks (any --vla, any OpenVLA-OFT suite) that still exist in the region."""
    import boto3  # noqa: PLC0415

    names = set(STACK_NAMES.values()) | {oft_stack_name(s) for s in OFT_LIBERO_SUITES}
    names.discard(exclude)
    cfn = boto3.client("cloudformation", region_name=region)
    left = []
    for page in cfn.get_paginator("list_stacks").paginate():
        for st in page["StackSummaries"]:
            if st["StackName"] in names and st["StackStatus"] != "DELETE_COMPLETE":
                left.append(st["StackName"])
    return sorted(set(left))


def _delete_data(region: str, stack_name: str, outputs: dict) -> None:
    import boto3  # noqa: PLC0415

    account = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
    bucket = outputs.get("S3BucketName") or _results_bucket_name(stack_name, region, account)
    n = gallery.delete_bucket(bucket, region)
    print(f"[destroy] Results bucket {bucket}: " + ("not found (already deleted?)" if n is None else f"deleted ({n} objects)"))

    # Lambda log groups outlive the stack (AzSelector, bucket notification handler, GalleryPublisher, ...).
    logs = boto3.client("logs", region_name=region)
    groups = [g["logGroupName"]
              for page in logs.get_paginator("describe_log_groups").paginate(logGroupNamePrefix=f"/aws/lambda/{stack_name}-")
              for g in page.get("logGroups", [])]
    for name in groups:
        logs.delete_log_group(logGroupName=name)
    print(f"[destroy] Lambda log groups of {stack_name}: deleted ({len(groups)})")

    remaining = _remaining_target_stacks(region, stack_name)
    gallery_outputs = gallery.stack_outputs(region, gallery.GALLERY_STACK_NAME)
    if remaining:
        prefix = outputs.get("GalleryPrefix", "")
        g_bucket = outputs.get("GalleryBucket") or (gallery_outputs or {}).get("GalleryBucketName", "")
        if prefix and g_bucket:
            n = gallery.delete_prefix(g_bucket, prefix, region)
            # g/<id>/<target>/ -> rebuild g/<id>/index.html without this target
            gallery.rebuild_index(g_bucket, "/".join(prefix.split("/")[:2]) + "/", region)
            print(f"[destroy] Gallery: removed {n} objects under {prefix} and rebuilt the index")
        else:
            print("[destroy] Gallery: this stack did not publish to the gallery; nothing to remove")
        print(f"[destroy] Gallery kept: other target stacks remain in {region}: {', '.join(remaining)}")
        return

    print(f"[destroy] No other target stack left in {region}: removing the shared video history gallery")
    g_bucket = (gallery_outputs or {}).get("GalleryBucketName") or outputs.get("GalleryBucket") \
        or f"rfm-gallery-{account}-{region}"
    if gallery_outputs is not None or _stack_exists(region, gallery.GALLERY_STACK_NAME):
        gallery.destroy_gallery_stack(region)  # CloudFront distribution, function and KeyValueStore
    n = gallery.delete_bucket(g_bucket, region)
    print(f"[destroy] Gallery bucket {g_bucket}: " + ("not found" if n is None else f"deleted ({n} objects)"))
    n = gallery.delete_ssm_params(region)
    print(f"[destroy] Gallery SSM parameters under {gallery.SSM_PREFIX}: deleted ({n})")


def _stack_exists(region: str, stack_name: str) -> bool:
    import boto3  # noqa: PLC0415
    from botocore.exceptions import ClientError  # noqa: PLC0415

    try:
        st = boto3.client("cloudformation", region_name=region).describe_stacks(StackName=stack_name)["Stacks"][0]
    except ClientError:
        return False
    return st["StackStatus"] != "DELETE_COMPLETE"


if __name__ == "__main__":
    main()
