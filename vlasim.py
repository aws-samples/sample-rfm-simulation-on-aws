#!/usr/bin/env python3
"""
vlasim — RFM Simulation on AWS deployment CLI

A thin, dependency-light front end that turns the multi-step manual setup into a
three-command flow:

    python vlasim.py doctor              # check prerequisites (safe to run anytime)
    python vlasim.py init  --email YOU@EXAMPLE.COM
    python vlasim.py deploy --vla gr00t-g1

Subcommands
-----------
  doctor   Verify the host can deploy: Python/Node/CDK versions and installs,
           AWS credentials, region support, CDK bootstrap, GPU instance quota,
           and notify_email. Read-only. Prints PASS/WARN/FAIL/SKIP per check and
           exits non-zero if any check FAILs.

  init     One-time setup: `pip install -r requirements.txt`, `npm install` in
           cdk/, and fill simulator-config.yaml (notify_email, region). Optional
           `--bootstrap` runs `cdk bootstrap` for the target account/region.

  deploy   Run a deploy. Runs `doctor` for the target first (preflight) so a
           missing GPU quota or un-bootstrapped account is caught in seconds
           rather than mid-deploy. Forwards all flags to deploy.py unchanged.
           Use --skip-doctor to bypass the preflight.

  destroy  Tear down a stack. Forwards all flags to destroy.py unchanged.

  results  Read every vla-sim-results-* bucket in the account and print one row
           per (stack, run, suite, condition): k/n successes, rate, 95% Wilson
           interval, and the stack fingerprint from summary.txt. Read-only.
           Forwards all flags to results.py (--vla, --region, --since,
           --all-runs, --csv, --json).

  judge    Score the rollout videos of a finished run with a post-run judge
           (NVIDIA Flora-Rewarder, outcome score 1..5 per episode, NON-COMMERCIAL
           license) and write judge.json + judge.txt next to that run. Runs the
           doctor for the judge's GPU instance first, then deploy.py --judge.

  gallery  Print the video history gallery URL and password for the notification
           e-mail (all runs under that e-mail, grouped by target). Read-only;
           --reset-password issues a new password.

Design note: `doctor` deliberately imports boto3/pyyaml/jinja2 lazily. Its whole
purpose is to run on a fresh host where those are not installed yet, report what
is missing, and tell you to run `init`. AWS-dependent checks degrade to SKIP when
boto3 is absent instead of crashing.
"""

import argparse
import re
import shutil
import subprocess  # nosec B404 - used only with hardcoded command lists, no shell
import sys
from pathlib import Path

from targets import JUDGE_TARGETS, VLA_TARGETS

BASE_DIR = Path(__file__).resolve().parent
CDK_DIR = BASE_DIR / "cdk"
CONFIG_PATH = BASE_DIR / "simulator-config.yaml"
REQUIREMENTS = BASE_DIR / "requirements.txt"
DEPLOY_PY = BASE_DIR / "deploy.py"
DESTROY_PY = BASE_DIR / "destroy.py"

# Single source: targets.py (also used by deploy.py / destroy.py / generate.py).
VLA_CHOICES = list(VLA_TARGETS)

# Regions the userdata templates + DLAMI lookups are validated against (README).
SUPPORTED_REGIONS = {
    "us-east-1", "us-west-2", "ap-northeast-1", "ap-northeast-2", "eu-central-1",
}

PLACEHOLDER_EMAIL = "YOUR_EMAIL@example.com"
EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
REGION_RE = re.compile(r"^[a-z]{2,3}(-[a-z]+)+-\d+$")

MIN_PYTHON = (3, 10)
MIN_NODE_MAJOR = 18         # aws-cdk 2.x requires Node >= 18
RECOMMENDED_NODE_MAJOR = 20

# On-demand G/VR vCPU-based Service Quota (EC2). A fresh account often defaults to
# 0 here, which is the classic cause of a mid-deploy InsufficientInstanceCapacity /
# VcpuLimitExceeded failure — exactly what `doctor` exists to catch up front.
GVR_ONDEMAND_QUOTA_CODE = "L-DB2E81BA"

# vCPU by instance size suffix (covers every size used in models/*.yaml preferred lists).
_SIZE_VCPU = {
    "xlarge": 4, "2xlarge": 8, "4xlarge": 16, "8xlarge": 32,
    "12xlarge": 48, "16xlarge": 64, "24xlarge": 96, "48xlarge": 192,
}


# ───────────────────────────── small helpers ──────────────────────────────

class Doctor:
    """Accumulates check results and renders them as aligned status lines."""

    def __init__(self):
        self.failures = 0
        self.warnings = 0

    def ok(self, msg):
        print(f"  [ OK ] {msg}")

    def warn(self, msg, fix=None):
        self.warnings += 1
        print(f"  [WARN] {msg}")
        if fix:
            print(f"         -> {fix}")

    def fail(self, msg, fix=None):
        self.failures += 1
        print(f"  [FAIL] {msg}")
        if fix:
            print(f"         -> {fix}")

    def skip(self, msg, why):
        print(f"  [SKIP] {msg} ({why})")


def _instance_vcpu(instance_type):
    """g6.12xlarge -> 48. Returns None for an unrecognised size."""
    return _SIZE_VCPU.get(instance_type.split(".")[-1])


def _read_config():
    """Return (region, email) from simulator-config.yaml, best-effort.

    Uses PyYAML when available; otherwise falls back to a comment-stripping regex
    so `doctor` still works before `init` has installed dependencies.
    """
    if not CONFIG_PATH.exists():
        return (None, None)
    text = CONFIG_PATH.read_text()
    try:
        import yaml  # noqa: PLC0415  (lazy by design)
        cfg = yaml.safe_load(text) or {}
        dep = cfg.get("deployment", {}) or {}
        return (dep.get("region"), dep.get("notify_email"))
    except Exception:  # nosec B110 - PyYAML missing or unparseable; regex fallback below
        pass

    def grab(key):
        m = re.search(rf"^\s*{key}\s*:\s*([^\s#]+)", text, re.M)
        return m.group(1) if m else None

    return (grab("region"), grab("notify_email"))


def _model_preferred(vla):
    """Return the `instance.preferred` list from models/{vla}.yaml, best-effort."""
    model_path = BASE_DIR / "models" / f"{vla}.yaml"
    if not model_path.exists():
        return None
    text = model_path.read_text()
    try:
        import yaml  # noqa: PLC0415
        cfg = yaml.safe_load(text) or {}
        inst = dict(cfg.get("instance", {}) or {})
        # models with a variants: block (robolab-*): model.variant's instance keys win, as in the CDK stack
        variant = (cfg.get("variants") or {}).get(str((cfg.get("model") or {}).get("variant", "")))
        if isinstance(variant, dict):
            inst.update(variant.get("instance") or {})
        pref = inst.get("preferred")
        if pref:
            return list(pref)
    except Exception:  # nosec B110 - regex fallback below
        pass
    m = re.search(r"^\s*preferred:\s*\[([^\]]+)\]", text, re.M)
    if not m:
        return None
    return [t.strip() for t in m.group(1).split(",") if t.strip()]


def _run(cmd, **kwargs):
    """subprocess.run wrapper that echoes the command for transparency."""
    printable = " ".join(cmd)
    print(f"  $ {printable}")
    return subprocess.run(cmd, **kwargs)  # nosec B603 - hardcoded/validated command lists only


# ───────────────────────────── doctor ──────────────────────────────────────

def run_doctor(vla=None, region=None, email=None, quiet_header=False, bridge=False):
    """Run all prerequisite checks. Returns the number of FAILs (0 == ready)."""
    d = Doctor()
    if not quiet_header:
        print("vlasim doctor — checking prerequisites\n")

    cfg_region, cfg_email = _read_config()
    region = region or cfg_region
    email = email or cfg_email

    # 1. Python version ------------------------------------------------------
    pyver = sys.version_info
    if pyver[:2] >= MIN_PYTHON:
        d.ok(f"Python {pyver.major}.{pyver.minor}.{pyver.micro} "
             f"(>= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})")
    else:
        d.fail(f"Python {pyver.major}.{pyver.minor} is too old",
               f"install Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")

    # 2. Python dependencies -------------------------------------------------
    missing_py = []
    for mod, pkg in (("boto3", "boto3"), ("yaml", "pyyaml"), ("jinja2", "jinja2")):
        try:
            __import__(mod)
        except ImportError:
            missing_py.append(pkg)
    have_boto3 = "boto3" not in missing_py
    if not missing_py:
        d.ok("Python deps installed (boto3, pyyaml, jinja2)")
    else:
        d.fail(f"Missing Python deps: {', '.join(missing_py)}",
               "run: python vlasim.py init   (or: pip install -r requirements.txt)")
    try:
        __import__("awscrt")
        d.ok("AWS CRT installed (video history gallery: CloudFront KeyValueStore)")
    except ImportError:
        d.warn("AWS CRT (awscrt) not installed: deploy cannot register the video history gallery password",
               "run: pip install -r requirements.txt   (or deploy with --no-gallery)")

    # 3. Node.js -------------------------------------------------------------
    node = shutil.which("node")
    if not node:
        d.fail("Node.js not found", "install Node.js >= 18 (https://nodejs.org)")
    else:
        try:
            out = subprocess.run([node, "--version"], capture_output=True,  # nosec B603
                                 text=True, check=True).stdout.strip()
            major = int(re.sub(r"[^\d.]", "", out).split(".")[0])
            if major < MIN_NODE_MAJOR:
                d.fail(f"Node {out} is too old for aws-cdk 2.x",
                       f"install Node >= {RECOMMENDED_NODE_MAJOR}")
            elif major < RECOMMENDED_NODE_MAJOR:
                d.warn(f"Node {out} works but Node >= {RECOMMENDED_NODE_MAJOR} is recommended")
            else:
                d.ok(f"Node {out}")
        except (subprocess.CalledProcessError, ValueError):
            d.warn("Node present but version could not be parsed")

    # 4. npm -----------------------------------------------------------------
    if shutil.which("npm"):
        d.ok("npm found")
    else:
        d.fail("npm not found", "install Node.js (ships with npm)")

    # 5. AWS CLI (optional — boto3 covers credential checks) -----------------
    if shutil.which("aws"):
        d.ok("AWS CLI found")
    else:
        d.warn("AWS CLI not found (optional — used for log tailing / result sync)")

    # 6. simulator-config.yaml present --------------------------------------
    if CONFIG_PATH.exists():
        d.ok("simulator-config.yaml present")
    else:
        d.fail("simulator-config.yaml not found",
               f"expected at {CONFIG_PATH}")

    # 7. notify_email set ----------------------------------------------------
    if email is None:
        d.warn("notify_email not resolved from config",
               "run: python vlasim.py init --email YOU@EXAMPLE.COM")
    elif email == PLACEHOLDER_EMAIL:
        d.fail("notify_email is still the placeholder",
               "run: python vlasim.py init --email YOU@EXAMPLE.COM")
    elif not EMAIL_RE.match(email):
        d.fail(f"notify_email is not a valid address: {email}",
               "fix deployment.notify_email in simulator-config.yaml")
    else:
        d.ok(f"notify_email = {email}")

    # 8. region supported ----------------------------------------------------
    if region is None:
        d.warn("region not resolved from config")
    elif not REGION_RE.match(region):
        d.fail(f"region has an invalid format: {region}")
    elif region not in SUPPORTED_REGIONS:
        d.warn(f"region {region} is outside the validated set "
               f"({', '.join(sorted(SUPPORTED_REGIONS))})",
               "DLAMI / capacity not verified there; deploy may still work")
    else:
        d.ok(f"region = {region}")

    # 9. CDK dependencies installed -----------------------------------------
    if (CDK_DIR / "node_modules").is_dir():
        d.ok("cdk/node_modules present")
    else:
        d.fail("cdk/node_modules missing",
               "run: python vlasim.py init   (or: cd cdk && npm install)")

    # 10. CDK runnable -------------------------------------------------------
    if shutil.which("npx") and (CDK_DIR / "node_modules").is_dir():
        try:
            out = subprocess.run(["npx", "cdk", "--version"], cwd=str(CDK_DIR),  # nosec B603,B607
                                 capture_output=True, text=True, check=True).stdout.strip()
            d.ok(f"cdk CLI {out}")
        except subprocess.CalledProcessError:
            d.warn("`npx cdk --version` failed", "try: cd cdk && npm install")
    else:
        d.skip("cdk CLI", "node_modules not installed yet")

    # ── AWS account checks (need boto3 + credentials) ──────────────────────
    if not have_boto3:
        d.skip("AWS credentials / bootstrap / quota", "boto3 not installed — run init")
        return _summary(d, vla)

    if not REGION_RE.match(region or ""):
        d.skip("AWS credentials / bootstrap / quota", "no valid region resolved")
        return _summary(d, vla)

    import boto3  # noqa: PLC0415
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError  # noqa: PLC0415

    account = None
    try:
        ident = boto3.client("sts", region_name=region).get_caller_identity()
        account = ident["Account"]
        d.ok(f"AWS credentials valid (account {account}, region {region})")
    except (NoCredentialsError, ClientError, BotoCoreError) as e:
        d.fail(f"AWS credentials check failed: {type(e).__name__}",
               "run `aws configure` or set AWS_PROFILE / AWS credentials env vars")

    # 12. CDK bootstrap ------------------------------------------------------
    if account:
        try:
            boto3.client("cloudformation", region_name=region).describe_stacks(
                StackName="CDKToolkit")
            d.ok(f"CDK bootstrap present in {region}")
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if "does not exist" in str(e) or code in ("ValidationError",):
                d.fail(f"CDK not bootstrapped in {region}",
                       "run: python vlasim.py init --bootstrap")
            else:
                d.warn(f"could not verify CDK bootstrap: {code or type(e).__name__}")
        except BotoCoreError as e:
            d.warn(f"could not verify CDK bootstrap: {type(e).__name__}")

    # 13. GPU instance quota (best-effort — WARN only) ----------------------
    if account:
        _check_gpu_quota(d, boto3, ClientError, BotoCoreError, region, vla)

    # 14. HF token for gated models (per region) ----------------------------
    if account and vla in HF_TOKEN_TARGETS and not bridge:
        _check_hf_token(d, boto3, ClientError, BotoCoreError, region, vla)

    return _summary(d, vla)


# Targets whose local-mode UserData downloads a gated Hugging Face repo with the token stored at
# SSM HF_TOKEN_PARAM. Without it the instance installs for 15+ min and then fails with a 401.
HF_TOKEN_PARAM = "/vla-simulator/hf-token"
HF_TOKEN_TARGETS = {
    "gr00t": "nvidia/Cosmos-Reason2-2B",
    "gr00t-bridge": "nvidia/Cosmos-Reason2-2B",
    "gr00t-g1": "the N1.7 server.hf_repo in models/gr00t-g1.yaml",
    "openarm-isaac": "google/paligemma-3b-pt-224",
    "cosmos-policy": "nvidia/Cosmos-Predict2-2B-Video2World",
}


def _check_hf_token(d, boto3, ClientError, BotoCoreError, region, vla):
    """FAIL when the gated-model token is missing in the deploy region (SSM is regional)."""
    repo = HF_TOKEN_TARGETS[vla]
    try:
        boto3.client("ssm", region_name=region).get_parameter(Name=HF_TOKEN_PARAM)
        d.ok(f"HF token present at SSM {HF_TOKEN_PARAM} in {region} (gated: {repo})")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code == "ParameterNotFound":
            d.fail(f"HF token missing at SSM {HF_TOKEN_PARAM} in {region} ({vla} pulls gated {repo})",
                   f"accept the model license on huggingface.co, then: aws ssm put-parameter "
                   f"--region {region} --name {HF_TOKEN_PARAM} --type SecureString --value <HF_TOKEN>")
        else:
            d.warn(f"could not verify HF token in SSM: {code or type(e).__name__}")
    except BotoCoreError as e:
        d.warn(f"could not verify HF token in SSM: {type(e).__name__}")


def _check_gpu_quota(d, boto3, ClientError, BotoCoreError, region, vla):
    """Compare the On-Demand G/VR vCPU quota against the smallest viable instance.

    Always WARN-only: Service Quotas can be access-denied, and the AzSelector
    Lambda falls back across instance types/AZs at deploy time anyway. The goal is
    to surface an obvious 0-quota fresh account, not to gate the deploy.
    """
    # Required vCPU = the smallest instance in the target's preferred list (the
    # AzSelector can fall back to it), or a generic single-GPU 4 vCPU floor.
    need = 4
    detail = "a single-GPU instance"
    if vla:
        pref = _model_preferred(vla)
        if pref:
            vcpus = [v for v in (_instance_vcpu(i) for i in pref) if v]
            if vcpus:
                need = min(vcpus)
                smallest = pref[vcpus.index(need)] if need in vcpus else pref[-1]
                detail = f"{smallest} ({need} vCPU, smallest in {vla} preferred list)"
    try:
        sq = boto3.client("service-quotas", region_name=region)
        resp = sq.get_service_quota(ServiceCode="ec2", QuotaCode=GVR_ONDEMAND_QUOTA_CODE)
        limit = int(resp["Quota"]["Value"])
        if limit <= 0:
            d.warn(f"On-Demand G/VR vCPU quota is {limit} in {region} — cannot launch {detail}",
                   "request a quota increase: Service Quotas > EC2 > "
                   "'Running On-Demand G and VR instances'")
        elif limit < need:
            d.warn(f"On-Demand G/VR vCPU quota is {limit} in {region}, "
                   f"but {detail} needs {need}",
                   "request a quota increase (EC2 'Running On-Demand G and VR instances')")
        else:
            d.ok(f"On-Demand G/VR vCPU quota = {limit} in {region} (>= {need} for {detail})")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", type(e).__name__)
        d.warn(f"could not read G/VR vCPU quota ({code})",
               "verify manually in the Service Quotas console (EC2)")
    except BotoCoreError as e:
        d.warn(f"could not read G/VR vCPU quota ({type(e).__name__})")


def _summary(d, vla):
    print()
    target = f" for --vla {vla}" if vla else ""
    if d.failures == 0 and d.warnings == 0:
        print(f"doctor: all checks passed{target}. Ready to deploy.")
    elif d.failures == 0:
        print(f"doctor: ready{target}, with {d.warnings} warning(s) above (non-blocking).")
    else:
        print(f"doctor: {d.failures} failure(s), {d.warnings} warning(s){target}. "
              f"Resolve the FAILs above before deploying.")
    return d.failures


# ───────────────────────────── init ────────────────────────────────────────

def _set_yaml_scalar(text, key, value):
    """Replace the scalar value of `key:` in a YAML string, preserving indentation
    and any inline `# comment`. Returns (new_text, changed_bool)."""
    pattern = re.compile(
        rf"^(?P<lead>\s*{re.escape(key)}\s*:\s*)(?P<val>[^\s#]+)(?P<gap>[ \t]*)(?P<cmt>#.*)?$",
        re.M,
    )

    changed = {"hit": False}

    def repl(m):
        if m.group("val") == value:
            changed["hit"] = True  # already set; treat as a (no-op) success
            return m.group(0)
        changed["hit"] = True
        cmt = m.group("cmt") or ""
        gap = "  " if cmt else ""
        return f"{m.group('lead')}{value}{gap}{cmt}".rstrip()

    new_text = pattern.sub(repl, text, count=1)
    return new_text, changed["hit"]


def _fill_config(email=None, region=None):
    if not CONFIG_PATH.exists():
        print(f"[init] simulator-config.yaml not found at {CONFIG_PATH}", file=sys.stderr)
        return
    text = CONFIG_PATH.read_text()
    cur_region, cur_email = _read_config()

    # Email: prompt interactively only if still the placeholder and not provided.
    if email is None and cur_email in (None, PLACEHOLDER_EMAIL) and sys.stdin.isatty():
        entered = input("[init] Notification email (SNS): ").strip()
        email = entered or None

    if email is not None:
        if not EMAIL_RE.match(email):
            print(f"[init] '{email}' is not a valid email — skipping email update.",
                  file=sys.stderr)
        else:
            text, hit = _set_yaml_scalar(text, "notify_email", email)
            print(f"[init] notify_email -> {email}" if hit
                  else "[init] notify_email key not found (left unchanged)")
    elif cur_email in (None, PLACEHOLDER_EMAIL):
        print("[init] notify_email left as placeholder — set it before deploy "
              "(--email or edit simulator-config.yaml)")

    if region is not None:
        if not REGION_RE.match(region):
            print(f"[init] '{region}' is not a valid region — skipping region update.",
                  file=sys.stderr)
        else:
            if region not in SUPPORTED_REGIONS:
                print(f"[init] note: {region} is outside the validated region set.")
            text, hit = _set_yaml_scalar(text, "region", region)
            print(f"[init] region -> {region}" if hit
                  else "[init] region key not found (left unchanged)")

    CONFIG_PATH.write_text(text)


def _cdk_bootstrap(region):
    npx = shutil.which("npx")
    if not npx:
        print("[init] npx not found — cannot run cdk bootstrap.", file=sys.stderr)
        return 1
    try:
        import boto3  # noqa: PLC0415
        account = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
    except Exception as e:  # noqa: BLE001
        print(f"[init] could not resolve AWS account for bootstrap ({type(e).__name__}). "
              f"Ensure credentials are configured.", file=sys.stderr)
        return 1
    # The CDK app requires a valid `vla` context and a generated userdata asset even
    # for bootstrap. Seed those with the cheapest local-only path so bootstrap works
    # before any real deploy target is chosen.
    seed_vla = "gr00t"
    print(f"[init] generating bootstrap seed userdata for {seed_vla}")
    res = _run([sys.executable, str(BASE_DIR / "generate.py"), "--vla", seed_vla], cwd=str(BASE_DIR))
    if res.returncode != 0:
        print("[init] generate.py failed; cannot prepare CDK bootstrap context.", file=sys.stderr)
        return res.returncode
    target = f"aws://{account}/{region}"
    print(f"[init] cdk bootstrap {target}")
    res = _run([npx, "cdk", "bootstrap", target, "-c", f"region={region}", "-c", f"vla={seed_vla}"],
               cwd=str(CDK_DIR))
    return res.returncode


def run_init(rest):
    p = argparse.ArgumentParser(prog="vlasim init", add_help=True,
                                description="One-time setup: install deps + fill config.")
    p.add_argument("--email", "-e", metavar="ADDRESS",
                   help="Notification email to write into simulator-config.yaml")
    p.add_argument("--region", "-r", metavar="REGION",
                   help="AWS region to write into simulator-config.yaml")
    p.add_argument("--bootstrap", action="store_true",
                   help="Also run `cdk bootstrap` for the target account/region")
    p.add_argument("--skip-install", action="store_true",
                   help="Skip pip/npm install (only fill config / bootstrap)")
    args = p.parse_args(rest)

    print("vlasim init — installing prerequisites\n")

    if not args.skip_install:
        # 1. Python deps
        print("[1/3] pip install -r requirements.txt")
        res = _run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)])
        if res.returncode != 0:
            print("[init] pip install failed.", file=sys.stderr)
            return res.returncode

        # 2. CDK deps
        npm = shutil.which("npm")
        if not npm:
            print("[init] npm not found — install Node.js >= 18 and re-run.", file=sys.stderr)
            return 1
        print("\n[2/3] npm install (cdk/)")
        res = _run([npm, "install"], cwd=str(CDK_DIR))
        if res.returncode != 0:
            print("[init] npm install failed.", file=sys.stderr)
            return res.returncode
    else:
        print("[init] --skip-install: skipping pip/npm install")

    # 3. Config
    print("\n[3/3] Configuring simulator-config.yaml")
    _fill_config(email=args.email, region=args.region)

    # Optional bootstrap
    if args.bootstrap:
        region, _ = _read_config()
        region = args.region or region
        if not region:
            print("[init] no region resolved — cannot bootstrap.", file=sys.stderr)
            return 1
        rc = _cdk_bootstrap(region)
        if rc != 0:
            return rc

    print("\n[init] Done. Next: python vlasim.py doctor")
    return 0


# ───────────────────────────── deploy / destroy ────────────────────────────

def _peek_core_args(rest):
    """Best-effort extraction of --vla / --region / --email from forwarded tokens,
    used only to run the preflight doctor. Unknown args are ignored."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--vla")
    p.add_argument("--region", "-r")
    p.add_argument("--email", "-e")
    known, _ = p.parse_known_args(rest)
    return known.vla, known.region, known.email


def run_deploy(rest):
    # If the user just wants deploy.py's help, hand it straight through.
    if any(tok in ("-h", "--help") for tok in rest):
        return _run([sys.executable, str(DEPLOY_PY), "--help"], cwd=str(BASE_DIR)).returncode

    skip_doctor = "--skip-doctor" in rest
    forward = [t for t in rest if t != "--skip-doctor"]

    vla, region, email = _peek_core_args(forward)

    if not skip_doctor:
        print("=== Preflight (vlasim doctor) ===")
        failures = run_doctor(vla=vla, region=region, email=email, quiet_header=True,
                              bridge="--bridge" in forward)
        print("=================================\n")
        if failures:
            print("[deploy] Preflight FAILED. Fix the [FAIL] items above, or re-run with "
                  "--skip-doctor to bypass.", file=sys.stderr)
            return 1

    cmd = [sys.executable, str(DEPLOY_PY)] + forward
    return _run(cmd, cwd=str(BASE_DIR)).returncode


def run_judge(rest):
    """vlasim judge --vla TARGET [--run RUN_ID] [...] -> doctor (judge instance quota) + deploy.py --judge."""
    if any(tok in ("-h", "--help") for tok in rest):
        print("usage: python vlasim.py judge --vla TARGET [--run RUN_ID] [--region R] [--email A] "
              "[--libero-suite S] [--skip-doctor]\n\nForwards to deploy.py --judge (see deploy.py --help: "
              "--judge, --run).")
        return 0
    skip_doctor = "--skip-doctor" in rest
    forward = [t for t in rest if t != "--skip-doctor"]
    if "--judge" not in forward:
        forward = ["--judge"] + forward
    _, region, email = _peek_core_args(forward)
    if not skip_doctor:
        print("=== Preflight (vlasim doctor, judge instance) ===")
        failures = run_doctor(vla=JUDGE_TARGETS[0], region=region, email=email, quiet_header=True)
        print("=================================\n")
        if failures:
            print("[judge] Preflight FAILED. Fix the [FAIL] items above, or re-run with --skip-doctor.",
                  file=sys.stderr)
            return 1
    return _run([sys.executable, str(DEPLOY_PY)] + forward, cwd=str(BASE_DIR)).returncode


def run_destroy(rest):
    cmd = [sys.executable, str(DESTROY_PY)] + rest
    return _run(cmd, cwd=str(BASE_DIR)).returncode


def run_results(rest):
    cmd = [sys.executable, str(BASE_DIR / "results.py")] + rest
    return _run(cmd, cwd=str(BASE_DIR)).returncode


def run_gallery(rest):
    p = argparse.ArgumentParser(prog="vlasim gallery", add_help=True,
                                description="Show the video history gallery URL + password for your e-mail.")
    p.add_argument("--email", "-e", help="Notification e-mail (else from config)")
    p.add_argument("--region", "-r", help="Region of the gallery (else from config)")
    p.add_argument("--reset-password", action="store_true",
                   help="Issue a new password (the old one stops working within a few seconds)")
    args = p.parse_args(rest)
    cfg_region, cfg_email = _read_config()
    region = args.region or cfg_region
    email = args.email or cfg_email
    if not email or email == PLACEHOLDER_EMAIL or not EMAIL_RE.match(email):
        print("[gallery] No valid notification e-mail; pass --email or run: python vlasim.py init --email ...",
              file=sys.stderr)
        return 1
    if not region or not REGION_RE.match(region):
        print("[gallery] No valid region; pass --region.", file=sys.stderr)
        return 1
    import gallery  # noqa: PLC0415 - needs boto3 (doctor must run without it)

    try:
        info = gallery.reset_password(region, email) if args.reset_password else gallery.lookup(region, email)
    except LookupError as e:
        print(f"[gallery] {e}. Deploy a target first (python vlasim.py deploy --vla ...).", file=sys.stderr)
        return 1
    print(f"Video history for {email} ({region}):")
    print(f"  URL:      {info['url']}")
    if info["password"]:
        print(f"  Password: {info['password']}   (any user name)")
    else:
        print("  Password: not set yet; the next deploy creates one (or run: python vlasim.py gallery --reset-password)")
    if args.reset_password:
        print("  New password issued; CloudFront picks it up within a few seconds.")
    return 0


# ───────────────────────────── doctor entrypoint ───────────────────────────

def run_doctor_cli(rest):
    p = argparse.ArgumentParser(prog="vlasim doctor", add_help=True,
                                description="Check deployment prerequisites (read-only).")
    p.add_argument("--vla", choices=VLA_CHOICES,
                   help="Check quota/instance fit for a specific target")
    p.add_argument("--region", "-r", help="Override region (else from config)")
    p.add_argument("--email", "-e", help="Override email (else from config)")
    args = p.parse_args(rest)
    return 1 if run_doctor(vla=args.vla, region=args.region, email=args.email) else 0


# ───────────────────────────── dispatch ────────────────────────────────────

USAGE = """vlasim — RFM Simulation on AWS deployment CLI

Usage:
  python vlasim.py doctor  [--vla TARGET] [--region R] [--email A]
  python vlasim.py init    [--email A] [--region R] [--bootstrap] [--skip-install]
  python vlasim.py deploy  --vla TARGET [--email A] [--region R] [--skip-doctor] [deploy.py flags...]
  python vlasim.py destroy --vla TARGET [destroy.py flags...]
  python vlasim.py results [--vla PREFIX] [--region R] [--since RUN_ID] [--all-runs] [--csv F] [--json F]
  python vlasim.py gallery [--email A] [--region R] [--reset-password]
  python vlasim.py judge   --vla TARGET [--run RUN_ID] [--region R] [--skip-doctor]

Three-step quick start:
  python vlasim.py doctor                         # see what's missing
  python vlasim.py init --email you@example.com   # install deps + fill config
  python vlasim.py deploy --vla gr00t-g1          # preflight + deploy

`deploy` and `destroy` forward all flags to deploy.py / destroy.py unchanged
(e.g. --bridge, --collect, --libero-suite, --no-gallery, --keep-data, --delete-all).
Run `deploy --help` for the full list.
"""


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0

    cmd, rest = argv[0], argv[1:]
    dispatch = {
        "doctor": run_doctor_cli,
        "init": run_init,
        "deploy": run_deploy,
        "destroy": run_destroy,
        "results": run_results,
        "gallery": run_gallery,
        "judge": run_judge,
    }
    handler = dispatch.get(cmd)
    if handler is None:
        print(f"vlasim: unknown command '{cmd}'\n", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    try:
        return handler(rest) or 0
    except KeyboardInterrupt:
        print("\n[vlasim] interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
