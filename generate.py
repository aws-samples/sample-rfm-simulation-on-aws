"""
generate.py — read a model yaml + simulator-config.yaml and write assets/userdata/{vla}.sh

Steps:
  1. Load simulator-config.yaml (shared) + models/{vla}.yaml (per model)
  2. Take resolved bridge values (ssm:/path → value; deploy.py resolves and passes them in)
  3. Render the templates/{vla}-userdata.sh.j2 Jinja2 template
  4. Write assets/userdata/{vla}.sh

Usage:
    python generate.py --vla gr00t       [--config simulator-config.yaml] [--dry-run]
    python generate.py --vla gr00t-gr1   [--config simulator-config.yaml] [--dry-run]
    python generate.py --vla pi          [--resolved-vpc vpc-xxx --resolved-nlb host:port]
    python generate.py --vla openvla-oft [--config simulator-config.yaml] [--dry-run]
    python generate.py --vla lap         [--config simulator-config.yaml] [--dry-run]
    python generate.py --vla rldx        [--config simulator-config.yaml] [--dry-run]
    python generate.py --vla rldx-simpler [--config simulator-config.yaml] [--dry-run]
    python generate.py --vla rldx-bridge [--config simulator-config.yaml] [--dry-run]
    python generate.py --vla gr00t-bridge [--config simulator-config.yaml] [--dry-run]
    python generate.py --vla rldx-gr1    [--config simulator-config.yaml] [--dry-run]
    python generate.py --vla rldx-kitchen [--config simulator-config.yaml] [--dry-run]
    python generate.py --vla openarm-isaac [--config simulator-config.yaml] [--dry-run]
    python generate.py --vla openarm-lift-act [--config simulator-config.yaml] [--dry-run]
    python generate.py --vla molmoact2   [--config simulator-config.yaml] [--dry-run]
    python generate.py --vla real2sim    [--config simulator-config.yaml] [--dry-run]
    python generate.py --vla robolab-cosmos3 [--dry-run]   # model.variant in the yaml: edge | nano | edge-full
    python generate.py --vla smolvla-robotwin [--robotwin-protocol subset|clean|randomized|full] [--robotwin-shard K/N]
    python generate.py --vla rlinf-oft-libero [--rl-profile smoke|full] [--rl-steps N] [--rl-eval-run RUN_ID]
    python generate.py --judge-target flora-judge [--dry-run]   # post-run judge (deploy.py --judge)
"""

import argparse
import base64
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

import libero_plus
import libero_r2s
import robotwin_plan
from targets import JUDGE_TARGETS, VLA_TARGETS

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"

# OpenVLA-OFT supported LIBERO suites (`long` is an alias for `10`).
OFT_LIBERO_SUITES = ("spatial", "object", "goal", "10", "long")
OFT_DEFAULT_SUITE = "10"


def _tasks_json_for_bash(tasks: list) -> str:
    """Serialize tasks to JSON, then bash-escape for embedding in a single-quoted
    string (templates use `TASKS_JSON='{{ tasks_json }}'`).

    Inside bash single quotes the ONLY metacharacter is `'` itself, so replacing
    each `'` with the canonical `'\\''` (close-quote, literal quote, reopen-quote)
    makes ANY free-text task description safe. This is a no-op (byte-identical
    output) for descriptions without apostrophes. Without it, an apostrophe in a
    task description (e.g. "gr00t's") prematurely terminates the bash string →
    UserData syntax error → script aborts before the rollout loop → no cfn_signal
    → stack hangs on WaitCondition until timeout.
    """
    return json.dumps(tasks, ensure_ascii=False).replace("'", "'\\''")


def normalise_libero_suite(suite: str) -> str:
    """`long` → `10` (OFT paper: LIBERO-Long = LIBERO-10)."""
    return "10" if suite == "long" else suite


def oft_stack_name(suite: str) -> str:
    """Stack id for a given suite. Default (`10`) preserves legacy name for
    backwards compatibility with existing deployments."""
    s = normalise_libero_suite(suite)
    if s == OFT_DEFAULT_SUITE:
        return "OpenVLA-OFT-Demo"
    return f"OpenVLA-OFT-{s.capitalize()}-Demo"


# --libero-variant plus: targets whose template carries the LIBERO-Plus switch (each also needs a
# `libero_plus:` block in models/<id>.yaml). See libero_plus.py and docs/targets.md.
LIBERO_PLUS_TEMPLATE_TARGETS = ("openvla-oft", "xvla", "smolvla", "lingbot-va", "fastwam", "pi", "gr00t",
                                "vla-adapter", "molmoact2", "rldx", "lap")


def libero_plus_supported(vla: str, config: dict) -> bool:
    return vla in LIBERO_PLUS_TEMPLATE_TARGETS and bool(config.get("libero_plus"))


def _libero_plus_ctx(config: dict, vla: str, perturb, n, suite=None) -> tuple:
    """(template context, task entries) for a --libero-variant plus render. Exits on a bad setting."""
    if not libero_plus_supported(vla, config):
        ok = [v for v in LIBERO_PLUS_TEMPLATE_TARGETS if (BASE_DIR / "models" / f"{v}.yaml").exists()
              and (yaml.safe_load((BASE_DIR / "models" / f"{v}.yaml").read_text()) or {}).get("libero_plus")]
        print(f"[error] --libero-variant plus is not available for --vla {vla} "
              f"(supported: {', '.join(ok)}; needs a libero_plus block in models/<vla>.yaml "
              f"and a template with the switch).", file=sys.stderr)
        sys.exit(1)
    try:
        cfg, tasks = libero_plus.plan(config["libero_plus"], perturb=perturb, n=n, suite=suite)
    except ValueError as e:
        print(f"[error] models/{vla}.yaml libero_plus: {e}", file=sys.stderr)
        sys.exit(1)
    pins = str(cfg.get("pip_pins", "")).split()
    bad = [x for x in pins if not _PIN_RE.match(x)]
    if bad or not pins:
        print(f"[error] models/{vla}.yaml: libero_plus.pip_pins must be name==version entries: {bad or pins}",
              file=sys.stderr)
        sys.exit(1)
    for key in ("repo_commit", "assets_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(cfg[key])):
            print(f"[error] libero_plus.{key} must be a 40-hex commit", file=sys.stderr)
            sys.exit(1)
    if not re.fullmatch(r"[0-9a-f]{64}", str(cfg["assets_sha256"])):
        print("[error] libero_plus.assets_sha256 must be 64 hex chars", file=sys.stderr)
        sys.exit(1)
    asset_dir = BASE_DIR / "assets" / "libero-plus"
    ctx = {
        **{k: _shell_safe(cfg[k], f"libero_plus.{k}") for k in (
            "repo_url", "repo_commit", "assets_repo", "assets_revision", "assets_file", "assets_sha256", "suite")},
        "assets_gb": f"{int(cfg['assets_bytes']) / 1e9:.1f}",
        "pip_pins": " ".join(pins),
        "perturb": cfg["perturb"],
        "variants_per_category": cfg["variants_per_category"],
        "episodes_per_variant": cfg["episodes_per_variant"],
        "seed": cfg["seed"],
        "hf_repo": _shell_safe(str(cfg.get("hf_repo", "") or ""), "libero_plus.hf_repo"),
        "hf_model_revision": _shell_safe(str(cfg.get("hf_model_revision", "") or ""), "libero_plus.hf_model_revision"),
        "select_py": (asset_dir / "select_variants.py").read_text(),
        "oft_patch_py": (asset_dir / "oft_task_ids_patch.py").read_text(),
        "openpi_patch_py": (asset_dir / "openpi_task_ids_patch.py").read_text(),
        "openpi_verify_py": (asset_dir / "openpi_verify.py").read_text(),
    }
    return ctx, tasks


# --libero-objects real2sim: targets whose template carries the LIBERO-R2S switch (each also needs a
# `libero_r2s:` block in models/<id>.yaml). See libero_r2s.py and docs/targets.md.
LIBERO_R2S_TEMPLATE_TARGETS = ("openvla-oft", "pi")


def libero_r2s_supported(vla: str, config: dict) -> bool:
    return vla in LIBERO_R2S_TEMPLATE_TARGETS and bool(config.get("libero_r2s"))


def _libero_r2s_ctx(config: dict, vla: str, r2s: tuple, suite: str) -> tuple:
    """(template context, task entries) for a --libero-objects real2sim render. Exits on a bad setting.
    r2s = (run uri, conditions or None, trials per task or None)."""
    if not libero_r2s_supported(vla, config):
        print(f"[error] --libero-objects real2sim is not available for --vla {vla} (supported: "
              f"{', '.join(LIBERO_R2S_TEMPLATE_TARGETS)}; needs a libero_r2s block in models/<vla>.yaml).",
              file=sys.stderr)
        sys.exit(1)
    run, conds, trials = r2s
    try:
        cfg, tasks = libero_r2s.plan(config["libero_r2s"], run, suite, conditions=conds, trials=trials)
    except ValueError as e:
        print(f"[error] models/{vla}.yaml libero_r2s: {e}", file=sys.stderr)
        sys.exit(1)
    asset_dir = BASE_DIR / "assets" / "libero-r2s"
    ctx = {
        "run_uri": _shell_safe(cfg["run_uri"], "--real2sim-run"),
        "run_id": cfg["run_prefix"].split("/", 1)[0],
        "suite": suite,
        "conditions": cfg["conditions"],
        "modes": [c for c in cfg["conditions"] if c != "public"],
        "trials_per_task": cfg["trials_per_task"],
        "scale": cfg["scale"],
        "mass": cfg["mass"],
        "select": cfg["select"],
        "objects": str(cfg["objects"] or ""),
        "seed": cfg["seed"],
        "build_py": (asset_dir / "build_r2s_objects.py").read_text(),
        "select_py": (asset_dir / "r2s_select.py").read_text(),
        "check_py": (asset_dir / "check_scene.py").read_text(),
        "report_py": (asset_dir / "r2s_libero_report.py").read_text(),
    }
    return ctx, tasks


def r2s_default_suite(config: dict, vla: str, libero_suite: str = OFT_DEFAULT_SUITE) -> str:
    """LIBERO suite of a --libero-objects real2sim run: openvla-oft follows --libero-suite (one checkpoint per
    suite); pi takes libero_r2s.suite (default libero_object; pi05_libero covers every suite)."""
    if vla == "openvla-oft":
        suites = (config.get("model") or {}).get("suites") or {}
        return (suites.get(normalise_libero_suite(libero_suite)) or {}).get("task_suite_name", "")
    return str((config.get("libero_r2s") or {}).get("suite", "libero_object"))


def _load_merged_config(simulator_config_path: Path, vla: str) -> dict:
    """Merge simulator-config.yaml + models/{vla}.yaml. Model settings take precedence."""
    sim_cfg = yaml.safe_load(simulator_config_path.read_text())
    model_path = BASE_DIR / "models" / f"{vla}.yaml"
    if not model_path.exists():
        print(f"[error] Model config not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    model_cfg = yaml.safe_load(model_path.read_text())
    # deployment: simulator-config is the base; a deployment block in the model yaml overrides it
    merged_deployment = {**sim_cfg.get("deployment", {}), **model_cfg.get("deployment", {})}
    return {**sim_cfg, **model_cfg, "deployment": merged_deployment}


def _build_gr00t_ctx(config: dict, resolved_grpc: str, model_id: str) -> dict:
    """Build the shared context for gr00t / gr00t-gr1."""
    tasks = config.get("tasks", [])
    if not tasks:
        print(f"[error] models/{model_id}.yaml has no tasks entry.", file=sys.stderr)
        sys.exit(1)

    tasks_json = _tasks_json_for_bash(tasks)
    deployment = config.get("deployment", {})
    model = config.get("model", {})
    default_hf_repo = "nvidia/GR00T-N1.7-3B" if model_id == "gr00t" else "nvidia/GR00T-N1.6-3B"

    ctx = {
        "tasks_json": tasks_json,
        "deployment": deployment,
        "isaac_groot_commit": model.get("isaac_groot_commit", ""),
        "uv_version": model.get("uv_version", ""),
        "hf_repo": model.get("hf_repo", default_hf_repo),
        "hf_subfolder": model.get("hf_subfolder", ""),
        "hf_model_revision": model.get("hf_model_revision", ""),
        "robosuite_commit": model.get("robosuite_commit", ""),  # gr00t-g1 only (WBC robosuite fork pin)
        # FIX 14 — renderer pins the LIBERO venv must hold after setup_libero.sh resolves
        # LIBERO's requirements.txt (mujoco is unconstrained there). Only models/gr00t.yaml
        # sets it; gr00t-gr1/gr00t-g1 leave it "" and their templates never reference it, so
        # those two render byte-identically.
        "sim_protect_pins": model.get("sim_protect_pins", ""),
        "remote_grpc_endpoint": resolved_grpc,
    }

    if resolved_grpc:
        bridge_dir = BASE_DIR / "assets" / "bridge" / "gr00t"
        missing = [f for f in ("zmq_grpc_bridge.py", "gr00t_pb2.py", "gr00t_pb2_grpc.py")
                   if not (bridge_dir / f).exists()]
        if missing:
            print(f"[error] Bridge files missing: {missing} (check assets/bridge/gr00t/)", file=sys.stderr)
            sys.exit(1)
        ctx["zmq_grpc_bridge_b64"] = base64.encodebytes(
            (bridge_dir / "zmq_grpc_bridge.py").read_bytes()
        ).decode()
        ctx["gr00t_pb2_b64"] = base64.encodebytes(
            (bridge_dir / "gr00t_pb2.py").read_bytes()
        ).decode()
        ctx["gr00t_pb2_grpc_b64"] = base64.encodebytes(
            (bridge_dir / "gr00t_pb2_grpc.py").read_bytes()
        ).decode()

    return ctx


_HF_PATTERN_RE = re.compile(r"^[A-Za-z0-9_.*/\-]+$")


def _gr00t_sim_ctx(config: dict, vla: str) -> dict:
    """Simulator parameterisation of gr00t-userdata.sh.j2 (rldx pattern: one template, branched by sim_id).

    `gr00t` (LIBERO) sets none of these keys and renders byte-identically to the pre-parameterisation
    template. `gr00t-bridge` sets sim_id: simpler — the SimplerEnv WidowX/Bridge rollout venv, the Vulkan
    loader SAPIEN needs, and a checkpoint download that skips the optimizer/RNG states.
    log_prefix = vla so the CloudWatch log group matches the stack's /${vla}/* IAM grant.
    """
    model = config.get("model", {})
    tasks = config.get("tasks", [])
    sim_id = model.get("sim_id", "libero")
    if sim_id not in ("libero", "simpler"):
        print(f"[error] models/{vla}.yaml: model.sim_id must be libero or simpler, got {sim_id!r}", file=sys.stderr)
        sys.exit(1)
    patterns = list(model.get("hf_ignore_patterns") or [])
    bad = [x for x in patterns if not _HF_PATTERN_RE.match(str(x))]
    if bad:
        print(f"[error] models/{vla}.yaml: model.hf_ignore_patterns has unsafe entries: {bad}", file=sys.stderr)
        sys.exit(1)
    pins = (str(model.get("sim_extra_deps", "")) + " " + str(model.get("sim_protect_pins", ""))).split()
    bad = [x for x in pins if not re.fullmatch(r"[A-Za-z0-9_.\-]+(\[[A-Za-z0-9_,\-]+\])?(==|<|>=)[A-Za-z0-9_.+\-]+", x)]
    if bad:
        print(f"[error] models/{vla}.yaml: sim_extra_deps / sim_protect_pins entries must be name==version: {bad}",
              file=sys.stderr)
        sys.exit(1)
    first_env = str(tasks[0].get("env_name", "")) if tasks else ""
    if sim_id == "simpler" and not re.fullmatch(r"simpler_env_(google|widowx)/[a-z0-9_]+", first_env):
        print(f"[error] models/{vla}.yaml: SimplerEnv tasks need env_name simpler_env_(google|widowx)/<task>, "
              f"got {first_env!r}", file=sys.stderr)
        sys.exit(1)
    return {
        "log_prefix": vla,
        "sim_id": sim_id,
        "needs_vulkan": bool(model.get("needs_vulkan", False)),
        "sim_extra_deps": str(model.get("sim_extra_deps", "")).strip(),
        "first_env_name": first_env,
        "hf_ignore_patterns": "[" + ", ".join(repr(str(x)) for x in patterns) + "]" if patterns else "",
        "robot_label": _shell_safe(model.get("robot_label", ""), "robot_label"),
        "sim_label": _shell_safe(model.get("sim_label", ""), "sim_label"),
    }


def generate_gr00t(config: dict, resolved_grpc: str, resolved_vpc: str, dry_run: bool, vla: str = "gr00t",
                   plus: tuple = None) -> str:
    """GR00T N1.7 x {LIBERO (gr00t) | SimplerEnv WidowX/Bridge (gr00t-bridge)} — shared gr00t-userdata.sh.j2."""
    lp_ctx = None
    if plus is not None:
        # --libero-variant plus: one task entry per perturbation dimension; the variant ids are picked on the
        # instance and each variant runs as its own gym id. The checkpoint subfolder follows the sweep suite
        # (nvidia/GR00T-N1.7-LIBERO ships libero_spatial / _object / _goal / _10).
        lp_ctx, lp_tasks = _libero_plus_ctx(config, vla, *plus)
        if lp_ctx["hf_repo"] or resolved_grpc:
            print(f"[error] models/{vla}.yaml: libero_plus.hf_repo / bridge mode are not supported for gr00t.",
                  file=sys.stderr)
            sys.exit(1)
        t0 = config["tasks"][0]
        keep = {k: t0[k] for k in ("embodiment", "n_envs", "video_fps", "video_crf", "steps_per_render") if k in t0}
        steps = int(config["libero_plus"].get("max_episode_steps", t0.get("max_episode_steps", 720)))
        config = {**config, "tasks": [dict(t, env_name=f"libero_plus/{t['perturbation']}", **keep,
                                           n_episodes=lp_ctx["episodes_per_variant"], max_episode_steps=steps)
                                      for t in lp_tasks],
                  "model": {**config["model"], "hf_subfolder": lp_ctx["suite"]}}
    ctx = _build_gr00t_ctx(config, resolved_grpc, vla)
    ctx.update(_gr00t_sim_ctx(config, vla))
    ctx["libero_plus"] = lp_ctx
    if ctx["sim_id"] != "libero" and resolved_grpc:
        print(f"[error] {vla}: bridge mode (remote gRPC inference) is LIBERO-only.", file=sys.stderr)
        sys.exit(1)
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template, autoescape not applicable
    return env.get_template("gr00t-userdata.sh.j2").render(**ctx)


def generate_gr00t_gr1(config: dict, resolved_grpc: str, resolved_vpc: str, dry_run: bool) -> str:
    ctx = _build_gr00t_ctx(config, resolved_grpc, "gr00t-gr1")
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template, autoescape not applicable
    return env.get_template("gr00t-gr1-userdata.sh.j2").render(**ctx)


def generate_gr00t_g1(config: dict, resolved_grpc: str, resolved_vpc: str, dry_run: bool,
                      collect: bool = False) -> str:
    """GR00T N1.6 G1 whole-body loco-manip. Two modes (same WBC/ZMQ/checkpoint stack):

      • rollout (default): run rollout_policy.py → MP4 + success rate to S3 (the verified 3/10 demo).
      • collect (--collect): run collect_trajectories.py → success-only (obs,action) HDF5 to S3, for
        imitation fine-tuning of an N1.7 G1 adapter. Forces n_action_steps=1 (per-timestep transitions)
        and embeds the 2 collect assets as base64. Everything else (setup, ZMQ server, ckpt) is shared.
    """
    ctx = _build_gr00t_ctx(config, resolved_grpc, "gr00t-g1")
    ctx["collect"] = collect

    # N1.7 rollout server (Step 2): if models/gr00t-g1.yaml has a `server:` block with a checkpoint
    # source (either `hf_repo` for the public, reproducible path OR `s3_ckpt_uri` for a private bucket),
    # the inference server loads the fine-tuned N1.7 G1 adapter ckpt in a SEPARATE venv (Isaac-GR00T
    # 65cc4a), while the WBC sim keeps the N1.6 stack (77866395). Mutually exclusive with --collect
    # (collect runs the N1.6 teacher) and bridge mode (remote inference). Absent → the N1.6 demo path.
    server = config.get("server", {}) or {}
    server_hf_repo = server.get("hf_repo", "").strip()
    server_s3 = server.get("s3_ckpt_uri", "").strip()
    server_n17 = bool(server_hf_repo or server_s3)
    if server_hf_repo and server_s3:
        print("[error] server.hf_repo and server.s3_ckpt_uri are mutually exclusive — pick one "
              "checkpoint source for the N1.7 rollout server.", file=sys.stderr)
        sys.exit(1)
    if server_n17 and collect:
        print("[error] server.* (N1.7 rollout) and --collect are mutually exclusive: "
              "collect runs the N1.6 cloudwalk teacher, not the N1.7 adapter.", file=sys.stderr)
        sys.exit(1)
    if server_n17 and resolved_grpc:
        print("[error] server.* (N1.7 local server) and bridge mode (remote_grpc_endpoint) "
              "are mutually exclusive.", file=sys.stderr)
        sys.exit(1)
    ctx["server_n17"] = server_n17
    ctx["server_hf_repo"] = server_hf_repo
    ctx["server_hf_revision"] = server.get("hf_revision", "").strip()
    ctx["server_s3_ckpt_uri"] = server_s3
    ctx["server_isaac_groot_commit"] = server.get("isaac_groot_commit", "")
    ctx["server_hf_token_ssm"] = server.get("hf_token_ssm", "/vla-simulator/hf-token")
    ctx["server_embodiment_tag"] = server.get("embodiment_tag", "UNITREE_G1")

    if collect:
        if resolved_grpc:
            print("[error] --collect is local mode only (bridge delegates inference, no obs/action "
                  "dict locally to dump).", file=sys.stderr)
            sys.exit(1)
        asset_dir = BASE_DIR / "assets" / "gr00t-g1"
        asset_files = {
            "collect_traj_b64": "collect_trajectories.py",
            "hdf5_to_lerobot_b64": "gr00t_hdf5_to_lerobot.py",
        }
        missing = [fn for fn in asset_files.values() if not (asset_dir / fn).exists()]
        if missing:
            print(f"[error] gr00t-g1 collect asset files missing: {missing} (check assets/gr00t-g1/)",
                  file=sys.stderr)
            sys.exit(1)
        for ctx_key, fname in asset_files.items():
            ctx[ctx_key] = base64.encodebytes((asset_dir / fname).read_bytes()).decode()
        collection = config.get("collection", {})
        ctx["collection"] = {
            "lerobot_task": collection.get("lerobot_task", "pick up the apple and place it on the plate"),
            "repo_id": collection.get("repo_id", "local/gr00t-g1-applepnp"),
            "max_total_episodes": collection.get("max_total_episodes", 0),
            # num_demos = SUCCESSFUL demos to export (collect target). Decoupled from tasks[].n_episodes
            # (which is the rollout/eval episode count) so a small validation collect run doesn't alter eval.
            "num_demos": collection.get("num_demos", 0),  # 0 → fall back to tasks[].n_episodes
        }
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template, autoescape not applicable
    return env.get_template("gr00t-g1-userdata.sh.j2").render(**ctx)


def generate_openvla_oft(config: dict, libero_suite: str, dry_run: bool, plus: tuple = None, r2s: tuple = None) -> str:
    """plus = (perturb, n) for --libero-variant plus (the sweep runs on the selected suite), else None.
    r2s = (run uri, conditions, trials) for --libero-objects real2sim (the selected suite), else None."""
    suite = normalise_libero_suite(libero_suite)
    model = config.get("model", {})
    suites = model.get("suites", {})
    suite_cfg = suites.get(suite)
    if not suite_cfg:
        print(f"[error] models/openvla-oft.yaml model.suites has no '{suite}' entry. "
              f"Available: {sorted(suites.keys())}", file=sys.stderr)
        sys.exit(1)

    tasks_by_suite = config.get("tasks", {})
    if isinstance(tasks_by_suite, list):
        # Backwards-compat: flat list treated as the default suite.
        tasks = tasks_by_suite
    else:
        tasks = tasks_by_suite.get(suite, [])
    if not tasks:
        print(f"[error] models/openvla-oft.yaml has no tasks for suite '{suite}'.", file=sys.stderr)
        sys.exit(1)

    lp_ctx = None
    if plus is not None:
        lp_ctx, lp_tasks = _libero_plus_ctx(config, "openvla-oft", *plus, suite=suite_cfg["task_suite_name"])
        if lp_ctx["hf_repo"]:
            print("[error] models/openvla-oft.yaml: libero_plus.hf_repo is not supported (the checkpoint "
                  "follows --libero-suite)", file=sys.stderr)
            sys.exit(1)
        tasks = [dict(t, num_trials_per_task=lp_ctx["episodes_per_variant"]) for t in lp_tasks]

    r2s_ctx = None
    if r2s is not None:
        r2s_ctx, r2s_tasks = _libero_r2s_ctx(config, "openvla-oft", r2s, suite_cfg["task_suite_name"])
        tasks = [dict(t, num_trials_per_task=r2s_ctx["trials_per_task"]) for t in r2s_tasks]

    tasks_json = _tasks_json_for_bash(tasks)
    deployment = config.get("deployment", {})

    ctx = {
        "tasks_json": tasks_json,
        "deployment": deployment,
        "libero_suite": suite,
        "hf_repo": suite_cfg["hf_repo"],
        "hf_model_revision": suite_cfg.get("hf_model_revision", ""),
        "task_suite_name": suite_cfg["task_suite_name"],
        "oft_commit": model.get("oft_commit", ""),
        "transformers_fork_commit": model.get("transformers_fork_commit", ""),
        "libero_commit": model.get("libero_commit", "master"),
        "center_crop": model.get("center_crop", True),
        # FIX 14 — renderer pins the eval env must hold after libero_requirements.txt resolves.
        "sim_protect_pins": model.get("sim_protect_pins", ""),
        "stackpatch_py": (BASE_DIR / "assets" / "stack-knobs" / "stackpatch.py").read_text(),
        # FIX 15 — tensorflow-metadata pin (its declared protobuf floor understates what its
        # generated code needs, so the resolver can pick a version that cannot import here).
        "tfmd_version": model.get("tfmd_version", ""),
        "libero_plus": lp_ctx,
        "libero_r2s": r2s_ctx,
    }

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template
    return env.get_template("openvla-oft-userdata.sh.j2").render(**ctx)


def generate_rldx(config: dict, dry_run: bool, vla: str = "rldx", plus: tuple = None) -> str:
    """RLDX-1 (RLWRLD): MSAT/Qwen3-VL-8B × {LIBERO | SimplerEnv | ...}. Local mode only.

    ONE shared template (rldx-userdata.sh.j2) parameterised by `sim_id` (openvla-oft
    pattern), NOT a per-sim template fork. The server+client two-venv plumbing
    (cfn_signal / die / S3 upload / SNS / ZeroMQ warmup) is sim-agnostic single-source;
    only the [5/8] sim-venv setup+fix block, the server args, the rollout values, and
    the result labels diverge by sim. LIBERO (sim_id default) renders byte-identically
    to the pre-parameterisation template (checked by diffing dry-run renders).

    `vla` (= the model id / CDK context value) drives the CloudWatch log_prefix so the
    `/rldx/userdata` log group matches the stack's `/${vla}/*` IAM grant. A mismatch
    deploys fine but silently drops logs (no `aws logs tail`).

    No bridge assets (fused-kernel/ZeroMQ serving does not fit the bridge gRPC adapters).
    """
    tasks = config.get("tasks", [])
    if not tasks:
        print(f"[error] models/{vla}.yaml has no tasks entry.", file=sys.stderr)
        sys.exit(1)

    tasks_json = _tasks_json_for_bash(tasks)
    deployment = config.get("deployment", {})
    model = config.get("model", {})
    if model.get("simpler_robot", "google") not in ("google", "widowx"):
        print(f"[error] models/{vla}.yaml: model.simpler_robot must be google or widowx", file=sys.stderr)
        sys.exit(1)
    lp_ctx = None
    if plus is not None:
        # --libero-variant plus: RLDX-1's own LIBERO-Plus path at rldx_commit (rldx/eval/sim/LIBERO_PLUS,
        # libero_plus_sim/<variant> gym ids, setup_libero_plus.sh). One rollout per variant, 1 episode each.
        lp_ctx, lp_tasks = _libero_plus_ctx(config, vla, *plus)
        if lp_ctx["hf_repo"] or model.get("sim_id", "libero") != "libero":
            print(f"[error] models/{vla}.yaml: libero_plus needs the LIBERO sim and the model's own hf_repo",
                  file=sys.stderr)
            sys.exit(1)
        tasks = [dict(t, num_episodes=lp_ctx["episodes_per_variant"]) for t in lp_tasks]
        tasks_json = _tasks_json_for_bash(tasks)

    ctx = {
        "tasks_json": tasks_json,
        "tasks": tasks,
        "deployment": deployment,
        "rldx_commit": model.get("rldx_commit", "").strip(),
        "hf_repo": model.get("hf_repo", "RLWRLD/RLDX-1-FT-LIBERO"),
        "hf_model_revision": model.get("hf_model_revision", ""),
        "backbone_hf_repo": model.get("backbone_hf_repo", ""),
        "embodiment_tag": model.get("embodiment_tag", "GENERAL_EMBODIMENT"),
        "compile": str(model.get("compile", "") or "").strip(),
        "max_episode_steps": model.get("max_episode_steps", 720),
        "n_action_steps": model.get("n_action_steps", 8),
        "n_envs": model.get("n_envs", 1),
        # ── sim parameterisation (empty/default → LIBERO, byte-identical) ──
        # log_prefix = vla so the CW log group matches the stack's /${vla}/* IAM grant.
        "log_prefix": vla,
        "sim_id": model.get("sim_id", "libero"),
        "sim_venv_subpath": model.get("sim_venv_subpath", ""),
        "sim_setup_script": model.get("sim_setup_script", ""),
        "sim_register_module": model.get("sim_register_module", ""),
        # Empty sim_register_fn → the sim registers at IMPORT time (a module-level loop, e.g.
        # GR-1's gymnasium_groot.py:171), so the verify block just imports the module. A
        # non-empty value → the sim exposes a callable register_*() (e.g. SIMPLER's
        # register_simpler_envs) the verify block must call. Defaults to "" for forward sims.
        "sim_register_fn": model.get("sim_register_fn", ""),
        # FIX 4 (dispatch sed) + FIX 6/7 (obs/action policy patch) are SIMPLER-ONLY: they repair
        # bugs that exist only on the SIMPLER eval path at this pin. Default them to whether the
        # sim IS SIMPLER so rldx-simpler.yaml (which predates these flags) renders byte-identical
        # without having to set them; every other sim (LIBERO, GR-1, ...) leaves them False.
        # GR-1 source-verified clean: dispatch routes gr1_unified via startswith (rollout_policy
        # .py:237) and is_libero is correctly False (rldx_policy.py:523) → neither fix applies.
        "sim_needs_dispatch_fix": bool(
            model.get("sim_needs_dispatch_fix", model.get("sim_id", "libero") == "simpler")
        ),
        "sim_needs_policy_fix": bool(
            model.get("sim_needs_policy_fix", model.get("sim_id", "libero") == "simpler")
        ),
        # SimplerEnv robot for the FIX 6/7 adapter: "google" (Fractal, quaternion state — the default, so
        # rldx-simpler renders byte-identically) or "widowx" (Bridge, euler state + pad, rldx-bridge).
        "simpler_robot": model.get("simpler_robot", "google"),
        # FIX 5 — non-LIBERO sims whose external_dependencies/<sim> gitlink is absent at the pin
        # (SimplerEnv) must be pre-cloned --recursive so the setup script's already-populated
        # fallback runs. Empty for LIBERO (vendored dir present) → template skips the pre-clone.
        "sim_clone_url": model.get("sim_clone_url", ""),
        "sim_clone_subpath": model.get("sim_clone_subpath", ""),
        # FIX 1 (broadened) — pins the full rldx-dep install must not bump (sim renderer pins).
        "sim_protect_pins": model.get("sim_protect_pins", ""),
        # FIX 8 — SAPIEN/ManiSkill sims (SimplerEnv) render via Vulkan, which the DLAMI does
        # NOT ship (no libvulkan.so.1 loader, no ICD JSON) → `import sapien.core` dies with
        # `ImportError: libvulkan.so.1`. Install the Vulkan loader + point the NVIDIA ICD at
        # the pre-installed driver (libGLX_nvidia.so.0). Empty for LIBERO/MuJoCo sims (no
        # Vulkan) → template skips the block, LIBERO render stays byte-identical.
        # Source of truth: simpler-env/ManiSkill2_real2sim/docker/{Dockerfile,nvidia_icd.json}.
        "needs_vulkan": model.get("needs_vulkan", ""),
        # FIX 9 — RoboCasa-Kitchen's setup runs download_kitchen_assets.py, which has NO argparse
        # (the `-y` it's passed is ignored) and calls input() unconditionally → EOFError on a
        # headless host. True for kitchen → the template shadows input() in the cloned script.
        # Empty/False for every other sim (GR-1's downloader has a real argparse -y) → byte-ident.
        "sim_needs_asset_yes_fix": bool(model.get("sim_needs_asset_yes_fix", False)),
        # FIX 10 — RoboCasa-Kitchen's GrootRoboCasaEnv emits both a res256 and a res512 copy of
        # each of the 3 PandaOmron cameras; the policy only uses res256, but the video recorder
        # hstacks every "video.*" obs key → a 6-panel strip (each camera shown twice). True for
        # kitchen → the template narrows the recorder to res256, yielding a clean 3-panel video.
        # Other sims have no res512 keys, so empty/False elsewhere → byte-identical.
        "sim_drop_res512_video": bool(model.get("sim_drop_res512_video", False)),
        # server CLI args appended after --use-sim-policy-wrapper. LIBERO ships --no-strict;
        # SIMPLER (eval_simpler.sh) runs strict (no flag). NOTE: eval_simpler.sh ALSO passes
        # --num-inference-timesteps 10, but that is NOT a ServerConfig field at pin ecbfaf80
        # (tyro rejects unknown args → server crash); inference steps come from the ckpt
        # config. Omitted by default; re-add via server_extra_args after a Gate-1 check.
        "server_extra_args": model.get("server_extra_args", ["--no-strict"]),
        "robot_label": model.get("robot_label", ""),
        "sim_label": model.get("sim_label", ""),
        # Server warmup watchdog (seconds). Default 900 (15min) = LIBERO byte-identical.
        # SIMPLER (Gate-3 run 2): MSAT/Qwen3-VL-8B shard load is interleaved with an fp32
        # up-cast of all trainable params → ~250-316s/shard × 3 on g6.2xlarge (8 vCPU/32GB),
        # ~17-18min cold. Raised for SIMPLER so the watchdog doesn't kill a healthy load
        # mid-shard. Stays well under creationpolicy_timeout (PT180M).
        "server_warmup_timeout": int(model.get("server_warmup_timeout", 900)),
        "libero_plus": lp_ctx,
    }

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template
    return env.get_template("rldx-userdata.sh.j2").render(**ctx)


def generate_lap(config: dict, resolved_nlb: str, dry_run: bool, plus: tuple = None) -> str:
    tasks = config.get("tasks", [])
    if not tasks:
        print("[error] models/lap.yaml has no tasks entry.", file=sys.stderr)
        sys.exit(1)
    lp_ctx = None
    if plus is not None:
        # --libero-variant plus: same Python 3.8 LIBERO client stack as pi (openpi's third_party/libero), so
        # the pi recipe applies: fork in place of that submodule, filtered requirements, openpi task-loop patch.
        lp_ctx, lp_tasks = _libero_plus_ctx(config, "lap", *plus)
        if lp_ctx["hf_repo"]:
            print("[error] models/lap.yaml: libero_plus.hf_repo is not supported", file=sys.stderr)
            sys.exit(1)
        tasks = [dict(t, num_trials_per_task=lp_ctx["episodes_per_variant"]) for t in lp_tasks]

    tasks_json = _tasks_json_for_bash(tasks)
    deployment = config.get("deployment", {})
    model = config.get("model", {})

    # Bridge assets (embedded into UserData as heredoc — same pattern as generate_pi).
    # Empty strings render harmless heredocs in local mode (BRIDGE_MODE=false skips them).
    bridge_dir = BASE_DIR / "assets" / "bridge" / "lap"
    lap_proto_path = bridge_dir / "lap.proto"
    lap_grpc_bridge_path = bridge_dir / "lap_grpc_bridge.py"

    ctx = {
        "tasks_json": tasks_json,
        "deployment": deployment,
        "lap_commit": model.get("lap_commit", "").strip(),
        "hf_repo": model.get("hf_repo", "lihzha/LAP-3B-Libero"),
        "hf_model_revision": model.get("hf_model_revision", ""),
        "policy_config": model.get("policy_config", "lap_libero"),
        "policy_type": model.get("policy_type", "flow"),
        "lap_proto": lap_proto_path.read_text() if lap_proto_path.exists() else "",
        "lap_grpc_bridge_py": lap_grpc_bridge_path.read_text() if lap_grpc_bridge_path.exists() else "",
        "libero_plus": lp_ctx,
    }

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template
    return env.get_template("lap-userdata.sh.j2").render(**ctx)


def generate_openarm_isaac(config: dict, dry_run: bool) -> str:
    """OpenArm-Isaac: π0.5 (LeRobot pi05 folding_latest) × Isaac Lab bimanual Reach.

    Embeds the 3 sibling python assets (env.py / run_eval.py / overlay env_cfg) as base64 so the
    userdata can materialise them on the instance and mount them into the Isaac Lab container.
    """
    tasks = config.get("tasks", [])
    if not tasks:
        print("[error] models/openarm-isaac.yaml has no tasks entry.", file=sys.stderr)
        sys.exit(1)

    tasks_json = _tasks_json_for_bash(tasks)
    deployment = config.get("deployment", {})
    model = config.get("model", {})

    asset_dir = BASE_DIR / "assets" / "openarm-isaac"
    asset_files = {
        "env_py_b64": "env.py",
        "run_eval_py_b64": "run_eval.py",
        "overlay_cfg_b64": "openarm_bi_vla_env_cfg.py",
        # Rewrites LeRobot @ pin for Python 3.11 (NGC isaac-lab Kit runtime): relaxes the
        # requires-python gate + converts PEP-695 type syntax. See the asset's docstring.
        "patch_lerobot_b64": "patch_lerobot_py311.py",
    }
    missing = [fn for fn in asset_files.values() if not (asset_dir / fn).exists()]
    if missing:
        print(f"[error] openarm-isaac asset files missing: {missing} (check assets/openarm-isaac/)", file=sys.stderr)
        sys.exit(1)

    ctx = {
        "tasks_json": tasks_json,
        "deployment": deployment,
        "hf_repo": model.get("hf_repo", "lerobot/folding_latest"),
        "hf_model_revision": model.get("hf_model_revision", ""),
        "openarm_isaac_commit": model.get("openarm_isaac_commit", ""),
        "lerobot_commit": model.get("lerobot_commit", ""),
        "isaac_lab_image": model.get("isaac_lab_image", "nvcr.io/nvidia/isaac-lab:2.3.0"),
    }
    for ctx_key, fname in asset_files.items():
        ctx[ctx_key] = base64.encodebytes((asset_dir / fname).read_bytes()).decode()

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template
    return env.get_template("openarm-isaac-userdata.sh.j2").render(**ctx)


def generate_openarm_lift_act(config: dict, dry_run: bool) -> str:
    """OpenArm-Lift-ACT: teleop-free scripted demo COLLECTION for the unimanual Lift-Cube × ACT env.

    Embeds 2 sibling python assets (overlay env_cfg + collection driver) as base64 so the userdata
    can materialise them on the instance and mount them into the Isaac Lab container. No LeRobot,
    no checkpoint, no rollout video — the deliverable is a success-only demo HDF5 uploaded to S3.
    """
    model = config.get("model", {})
    collection = config.get("collection", {})
    if not collection:
        print("[error] models/openarm-lift-act.yaml has no collection entry.", file=sys.stderr)
        sys.exit(1)

    asset_dir = BASE_DIR / "assets" / "openarm-lift-act"
    asset_files = {
        "overlay_cfg_b64": "openarm_uni_lift_act_env_cfg.py",
        "collect_sm_b64": "collect_demos_sm.py",
    }
    missing = [fn for fn in asset_files.values() if not (asset_dir / fn).exists()]
    if missing:
        print(f"[error] openarm-lift-act asset files missing: {missing} (check assets/openarm-lift-act/)", file=sys.stderr)
        sys.exit(1)

    ctx = {
        "openarm_isaac_commit": model.get("openarm_isaac_commit", ""),
        "isaac_lab_image": model.get("isaac_lab_image", "nvcr.io/nvidia/isaac-lab:2.3.0"),
        "task_id": model.get("task_id", "Isaac-Lift-Cube-OpenArm-ACT-v0"),
        "register_module": model.get(
            "register_module",
            "openarm.tasks.manager_based.openarm_manipulation.unimanual.lift.config.openarm_uni_lift_act_env_cfg",
        ),
        "collection": {
            "num_demos": collection.get("num_demos", 5),
            "num_envs": collection.get("num_envs", 16),
            "position_threshold": collection.get("position_threshold", 0.01),
            "max_steps": collection.get("max_steps", 200000),
            "cam_res": collection.get("cam_res", 224),
            "task": collection.get("task", "lift the cube"),
        },
    }
    for ctx_key, fname in asset_files.items():
        ctx[ctx_key] = base64.encodebytes((asset_dir / fname).read_bytes()).decode()

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template
    return env.get_template("openarm-lift-act-userdata.sh.j2").render(**ctx)


def generate_molmoact2(config: dict, dry_run: bool, plus: tuple = None) -> str:
    """MolmoAct2 (Ai2) × LIBERO eval. Local in-process LeRobot inference (no policy server).

    MolmoAct2 is NOT in upstream LeRobot — installed from the allenai/lerobot FORK at a pinned
    commit via `uv sync --locked --extra molmoact2 --extra libero`. The template runs the fork's
    built-in `lerobot-eval --env.type=libero` directly (one process). No embedded python assets.
    Eval dtype is fp32 (official LIBERO recipe) → requires a single GPU >=40GB (g6e/L40S); the
    userdata enforces this with a VRAM-floor preflight.
    """
    tasks = config.get("tasks", [])
    if not tasks:
        print("[error] models/molmoact2.yaml has no tasks entry.", file=sys.stderr)
        sys.exit(1)
    lp_ctx = None
    if plus is not None:
        # --libero-variant plus: same recipe as the lerobot-libero targets (the allenai fork carries
        # LiberoPlusEnv and docker/Dockerfile.benchmark.libero_plus). The eval seed of the first LIBERO
        # task entry is kept so only the scene changes.
        lp_ctx, lp_tasks = _libero_plus_ctx(config, "molmoact2", *plus)
        if lp_ctx["hf_repo"]:
            print("[error] models/molmoact2.yaml: libero_plus.hf_repo is not supported", file=sys.stderr)
            sys.exit(1)
        seed = tasks[0].get("seed", 1000)
        tasks = [dict(t, n_episodes=lp_ctx["episodes_per_variant"], seed=seed) for t in lp_tasks]

    tasks_json = _tasks_json_for_bash(tasks)
    deployment = config.get("deployment", {})
    model = config.get("model", {})

    ctx = {
        "tasks_json": tasks_json,
        "deployment": deployment,
        "hf_repo": model.get("hf_repo", "allenai/MolmoAct2-LIBERO"),
        "hf_model_revision": model.get("hf_model_revision", ""),
        "lerobot_repo": model.get("lerobot_repo", "https://github.com/allenai/lerobot.git"),
        "lerobot_branch": model.get("lerobot_branch", "molmoact2-policy"),
        "lerobot_commit": model.get("lerobot_commit", ""),
        "python_version": model.get("python_version", "3.12"),
        "model_dtype": model.get("model_dtype", "float32"),
        "use_amp": bool(model.get("use_amp", False)),
        "norm_tag": model.get("norm_tag", "libero"),
        "inference_action_mode": model.get("inference_action_mode", "continuous"),
        "enable_inference_cuda_graph": bool(model.get("enable_inference_cuda_graph", True)),
        "camera_name_mapping": model.get(
            "camera_name_mapping",
            '{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}',
        ),
        "libero_plus": lp_ctx,
    }

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template
    return env.get_template("molmoact2-userdata.sh.j2").render(**ctx)


# Targets that run an upstream-LeRobot policy through `lerobot-eval --env.type=libero` with the shared
# lerobot-libero template. Adding one = a models/<id>.yaml + the CDK/CLI name maps; no new template.
LEROBOT_LIBERO_TARGETS = ("xvla", "smolvla", "lingbot-va", "fastwam")


def generate_lerobot_libero(config: dict, dry_run: bool, vla: str, plus: tuple = None) -> str:
    """Upstream LeRobot policy × LIBERO eval, in-process (shared lerobot-libero-userdata.sh.j2).

    LeRobot is installed from source at a pinned commit (`uv sync --locked --extra <policy> --extra
    libero`); `--policy.path=<hf_repo>` lets lerobot-eval resolve policy.type from the checkpoint.
    Per-model extras come from models/<vla>.yaml: uv_extras, policy_module (import check),
    policy_args / env_args (extra lerobot-eval flags) and the instance VRAM / host-RAM floors.
    """
    tasks = config.get("tasks", [])
    if not tasks:
        print(f"[error] models/{vla}.yaml has no tasks entry.", file=sys.stderr)
        sys.exit(1)
    model = config.get("model", {})
    instance = config.get("instance", {})
    for key in ("hf_repo", "lerobot_commit", "uv_extras", "policy_module"):
        if not model.get(key):
            print(f"[error] models/{vla}.yaml: model.{key} is required", file=sys.stderr)
            sys.exit(1)
    lp_ctx = None
    if plus is not None:
        # --libero-variant plus: one task entry per perturbation dimension; the variant ids are picked on
        # the instance. The eval seed of the first LIBERO task entry is kept so only the scene changes.
        lp_ctx, lp_tasks = _libero_plus_ctx(config, vla, *plus)
        seed = tasks[0].get("seed", 1000)
        tasks = [dict(t, n_episodes=lp_ctx["episodes_per_variant"], seed=seed) for t in lp_tasks]
        if lp_ctx["hf_repo"]:
            model = {**model, "hf_repo": lp_ctx["hf_repo"], "hf_model_revision": lp_ctx["hf_model_revision"]}

    ctx = {
        "vla": vla,
        "tasks_json": _tasks_json_for_bash(tasks),
        "deployment": config.get("deployment", {}),
        "display_name": model.get("display_name", vla),
        "policy_desc": model.get("policy_desc", vla),
        "license": model.get("license", "see model card"),
        "hf_repo": model["hf_repo"],
        "hf_model_revision": model.get("hf_model_revision", ""),
        "lerobot_repo": model.get("lerobot_repo", "https://github.com/huggingface/lerobot.git"),
        "lerobot_commit": model["lerobot_commit"],
        "python_version": model.get("python_version", "3.12"),
        "uv_extras": list(model["uv_extras"]),
        "policy_module": model["policy_module"],
        "policy_args": list(model.get("policy_args") or []),
        "env_args": list(model.get("env_args") or []),
        "min_gpu_mib": int(instance.get("min_gpu_mib", 20000)),
        "min_host_mib": int(instance.get("min_host_mib", 15000)),
        "libero_plus": lp_ctx,
    }
    for a in ctx["policy_args"] + ctx["env_args"]:
        if "'" in a:
            print(f"[error] models/{vla}.yaml: single quote not allowed in eval args: {a}", file=sys.stderr)
            sys.exit(1)

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template
    return env.get_template("lerobot-libero-userdata.sh.j2").render(**ctx)


# Targets that run an upstream-LeRobot policy through `lerobot-eval --env.type=robotwin` (RoboTwin 2.0,
# SAPIEN + CuRobo) with the shared lerobot-robotwin template. Adding one = models/<id>.yaml + the name maps.
LEROBOT_ROBOTWIN_TARGETS = ("smolvla-robotwin", "lingbot-vla2")


def generate_lerobot_robotwin(config: dict, dry_run: bool, vla: str, protocol: str = "subset",
                              shard: str = "") -> str:
    """Upstream LeRobot policy x RoboTwin 2.0 eval, in-process (shared lerobot-robotwin-userdata.sh.j2).

    The run plan (robotwin_plan.plan) is flattened to one entry per (setting, task): the userdata runs one
    lerobot-eval call per entry with the official per-task step limit and writes task-N/ with
    `suite: RoboTwin2/<setting>`. `protocol` / `shard` replace the yaml `tasks:` subset with the official
    49-task x 100-episode protocol (or a slice of it).
    """
    model = config.get("model", {})
    instance = config.get("instance", {})
    for key in ("hf_repo", "hf_model_revision", "lerobot_commit", "uv_extras", "policy_module", "rename_map"):
        if not model.get(key):
            print(f"[error] models/{vla}.yaml: model.{key} is required", file=sys.stderr)
            sys.exit(1)
    pins = {**robotwin_plan.PINS, **(config.get("robotwin") or {})}
    try:
        units = robotwin_plan.plan(config.get("tasks") or [], protocol=protocol, shard=shard)
    except ValueError as e:
        print(f"[error] models/{vla}.yaml: {e}", file=sys.stderr)
        sys.exit(1)
    tasks = [dict(u, task_config=robotwin_plan.SETTINGS[u["setting"]]) for u in units]
    plan_tasks = sorted({u["task"] for u in units})
    zips = ["embodiments.zip", "objects.zip"]
    if any(u["setting"] == "randomized" for u in units):
        zips.append("background_texture.zip")

    extra_repos = []
    for r in model.get("extra_hf_repos") or []:
        if not (r.get("repo") and re.fullmatch(r"[0-9a-f]{40}", str(r.get("revision", "")))):
            print(f"[error] models/{vla}.yaml: extra_hf_repos entries need repo + a 40-hex revision", file=sys.stderr)
            sys.exit(1)
        extra_repos.append(dict(repo=r["repo"], revision=r["revision"], why=r.get("why", ""),
                                patterns_json=json.dumps(r.get("patterns") or None)))

    ctx = {
        "vla": vla,
        "tasks_json": _tasks_json_for_bash(tasks),
        "plan_tasks_json": json.dumps(plan_tasks),
        "first_seed": int(units[0]["seed"]),
        "n_units": len(units),
        "n_episodes": robotwin_plan.episodes(units),
        "protocol": protocol,
        "shard": shard,
        "display_name": model.get("display_name", vla),
        "policy_desc": model.get("policy_desc", vla),
        "license": model.get("license", "see model card"),
        "hf_repo": model["hf_repo"],
        "hf_model_revision": model["hf_model_revision"],
        "lerobot_repo": model.get("lerobot_repo", "https://github.com/huggingface/lerobot.git"),
        "lerobot_commit": model["lerobot_commit"],
        "python_version": model.get("python_version", "3.12"),
        "uv_extras": list(model["uv_extras"]),
        "policy_module": model["policy_module"],
        "policy_args": list(model.get("policy_args") or []),
        "env_args": list(model.get("env_args") or []),
        "rename_map_json": json.dumps(model["rename_map"], separators=(", ", ": ")),
        "extra_hf_repos": extra_repos,
        "min_gpu_mib": int(instance.get("min_gpu_mib", 20000)),
        "min_host_mib": int(instance.get("min_host_mib", 15000)),
        "robotwin_repo": pins["repo"],
        "robotwin_commit": pins["commit"],
        "assets_repo": pins["assets_repo"],
        "assets_revision": pins["assets_revision"],
        "asset_zips": zips,
        "curobo_repo": pins["curobo_repo"],
        "curobo_ref": pins["curobo_ref"],
        "curobo_commit": pins["curobo_commit"],
        "sapien_version": pins["sapien_version"],
        "mplib_version": pins["mplib_version"],
        "robotwin_pip": list(pins["pip"]),
        "sec_per_step": float(pins["sec_per_step"]),
        "hf_offline": bool(pins["hf_offline"]),
        "episode_length": int(pins.get("episode_length") or 0),
    }
    for a in ctx["policy_args"] + ctx["env_args"] + [ctx["rename_map_json"]] + [r["patterns_json"] for r in extra_repos]:
        if "'" in a:
            print(f"[error] models/{vla}.yaml: single quote not allowed in eval args: {a}", file=sys.stderr)
            sys.exit(1)
    for a in ctx["env_args"]:
        if a.split("=", 1)[0] in ("--env.task", "--env.episode_length", "--env.type"):
            print(f"[error] models/{vla}.yaml: {a.split('=', 1)[0]} is set by the template", file=sys.stderr)
            sys.exit(1)

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template
    return env.get_template("lerobot-robotwin-userdata.sh.j2").render(**ctx)


def generate_vla_adapter(config: dict, dry_run: bool, vla: str = "vla-adapter", plus: tuple = None) -> str:
    """VLA-Adapter (OpenHelix) × LIBERO. An OpenVLA-OFT fork: vla-adapter-userdata.sh.j2 is a copy of the
    openvla-oft template (not shared, so the validated openvla-oft target cannot regress) with the repo,
    checkpoint and extra run_libero_eval.py flags taken from models/<vla>.yaml."""
    tasks = config.get("tasks", [])
    if not tasks:
        print(f"[error] models/{vla}.yaml has no tasks entry.", file=sys.stderr)
        sys.exit(1)
    model = config.get("model", {})
    for key in ("hf_repo", "repo_url", "repo_commit", "task_suite_name"):
        if not model.get(key):
            print(f"[error] models/{vla}.yaml: model.{key} is required", file=sys.stderr)
            sys.exit(1)
    lp_ctx = None
    if plus is not None:
        # --libero-variant plus: one task entry per perturbation dimension (same OFT task-loop patch as
        # openvla-oft). The checkpoint is single-suite, so the sweep must run on that suite.
        lp_ctx, lp_tasks = _libero_plus_ctx(config, vla, *plus)
        if lp_ctx["hf_repo"] or lp_ctx["suite"] != model["task_suite_name"]:
            print(f"[error] models/{vla}.yaml: libero_plus must run on the checkpoint's suite "
                  f"({model['task_suite_name']}) with the model's own hf_repo", file=sys.stderr)
            sys.exit(1)
        tasks = [dict(t, num_trials_per_task=lp_ctx["episodes_per_variant"]) for t in lp_tasks]
    ctx = {
        "vla": vla,
        "tasks_json": _tasks_json_for_bash(tasks),
        "deployment": config.get("deployment", {}),
        "display_name": model.get("display_name", vla),
        "policy_desc": model.get("policy_desc", vla),
        "license": model.get("license", "see model card"),
        "hf_repo": model["hf_repo"],
        "hf_model_revision": model.get("hf_model_revision", ""),
        "repo_url": model["repo_url"],
        "repo_commit": model["repo_commit"],
        "task_suite_name": model["task_suite_name"],
        "libero_commit": model.get("libero_commit", "master"),
        "eval_args": list(model.get("eval_args") or []),
        "extra_pip": list(model.get("extra_pip") or []),
        "sim_protect_pins": model.get("sim_protect_pins", ""),
        "tfmd_version": model.get("tfmd_version", ""),
        "libero_plus": lp_ctx,
    }
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template
    return env.get_template("vla-adapter-userdata.sh.j2").render(**ctx)


def generate_cosmos_policy(config: dict, dry_run: bool, vla: str = "cosmos-policy") -> str:
    """Cosmos Policy (NVIDIA) × LIBERO, run inside the project's own Docker image.

    The repo is cloned at a pinned commit, `docker build -t cosmos-policy docker` builds its CUDA 12.8
    image, `uv sync --extra cu128 --group libero` installs into the bind-mounted repo, and
    run_libero_eval.py runs on GPU 0 with the LIBERO.md flags from models/<vla>.yaml (eval_args).
    """
    tasks = config.get("tasks", [])
    if not tasks:
        print(f"[error] models/{vla}.yaml has no tasks entry.", file=sys.stderr)
        sys.exit(1)
    model = config.get("model", {})
    instance = config.get("instance", {})
    for key in ("hf_repo", "repo_url", "repo_commit", "eval_args"):
        if not model.get(key):
            print(f"[error] models/{vla}.yaml: model.{key} is required", file=sys.stderr)
            sys.exit(1)
    for a in model["eval_args"]:
        if '"' in a or "'" in a or "$" in a:
            print(f"[error] models/{vla}.yaml: quotes and $ are not allowed in eval_args: {a}", file=sys.stderr)
            sys.exit(1)
    ctx = {
        "vla": vla,
        "tasks_json": _tasks_json_for_bash(tasks),
        "deployment": config.get("deployment", {}),
        "display_name": model.get("display_name", vla),
        "policy_desc": model.get("policy_desc", vla),
        "license": model.get("license", "see model card"),
        "hf_repo": model["hf_repo"],
        "hf_model_revision": model.get("hf_model_revision", ""),
        "repo_url": model["repo_url"],
        "repo_commit": model["repo_commit"],
        "python_version": model.get("python_version", "3.10"),
        "eval_args": list(model["eval_args"]),
        "min_gpu_mib": int(instance.get("min_gpu_mib", 20000)),
        "min_host_mib": int(instance.get("min_host_mib", 15000)),
    }
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template
    return env.get_template("cosmos-policy-userdata.sh.j2").render(**ctx)


def generate_aspire(config: dict, dry_run: bool, vla: str = "aspire") -> str:
    """ASPIRE (NVlabs agentic skill discovery) × LIBERO-Pro fork. Not a trained policy: Claude Code on
    Amazon Bedrock writes a per-task cap-harness program on development seeds, which is then frozen and
    scored on the LIBERO saved init states the LeRobot targets use. Own template (aspire-userdata.sh.j2)."""
    tasks = config.get("tasks", [])
    if not tasks:
        print(f"[error] models/{vla}.yaml has no tasks entry.", file=sys.stderr)
        sys.exit(1)
    model = config.get("model", {})
    for key in ("repo_url", "repo_commit", "profile", "providers", "agent", "sanity_program"):
        if not model.get(key):
            print(f"[error] models/{vla}.yaml: model.{key} is required", file=sys.stderr)
            sys.exit(1)
    ports = {"sam3": 8114, "contact_graspnet": 8115, "pyroki": 8116, "curobo": 8118}
    unknown = [p for p in model["providers"] if p not in ports]
    if unknown:
        print(f"[error] models/{vla}.yaml: unknown providers {unknown}", file=sys.stderr)
        sys.exit(1)
    instance = config.get("instance", {})
    preflight = bool(model.get("preflight", False))
    service_providers = [p for p in model["providers"] if not (preflight and p == "sam3")]
    ctx = {
        "vla": vla,
        "tasks_json": _tasks_json_for_bash(tasks),
        "deployment": config.get("deployment", {}),
        "display_name": model.get("display_name", vla),
        "policy_desc": model.get("policy_desc", vla),
        "license": model.get("license", "see model card"),
        "repo_url": model["repo_url"],
        "repo_commit": model["repo_commit"],
        "profile": model["profile"],
        "providers": list(model["providers"]),
        # preflight: bootstrap everything, but start no SAM3 service (gated weights) and run no episodes
        "preflight": preflight,
        "service_providers": service_providers,
        "service_ports": [ports[p] for p in service_providers],
        "agent": model["agent"],
        "eval_seeds": list(model.get("eval_seeds") or [1, 2, 3, 4, 5]),
        "sanity_program": model["sanity_program"],
        "min_gpu_mib": instance.get("min_gpu_mib", 0),
        "min_host_mib": instance.get("min_host_mib", 0),
    }
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template
    return env.get_template("aspire-userdata.sh.j2").render(**ctx)


# Stage scripts embedded into the real2sim UserData (written to /opt/r2s/assets on the instance).
REAL2SIM_ASSETS = ("hope_fetch.py", "prep_capture.py", "ns_to_world.py", "r2s_segment.py", "r2s_grasp_mj.py",
                   "r2s_grasp_ur5e.py", "r2s_report.py")
_PIN_RE = re.compile(r"^[A-Za-z0-9_.\-]+(\[[A-Za-z0-9_,\-]+\])?==[A-Za-z0-9_.+\-]+$")
_ARG_RE = re.compile(r"^[A-Za-z0-9_.=\-]+$")
_S3_ZIP_RE = re.compile(r"^s3://[a-z0-9][a-z0-9.\-]{1,61}[a-z0-9]/[A-Za-z0-9_./=+\-]+\.zip$")


def generate_real2sim(config: dict, dry_run: bool, vla: str = "real2sim") -> str:
    """Real2Sim (stage 1): capture -> nerfstudio splatfacto 3DGS -> table + per-object collision meshes (USD)
    -> scripted MuJoCo grasp in the reconstructed / scene-hull / ground-truth worlds. Own template
    (real2sim-userdata.sh.j2); the stage scripts in assets/real2sim/ are embedded as base64. The input is
    a HOPE-Video scene (downloaded at run time, CC BY-NC-SA 4.0) or your own capture zip on S3."""
    model = config.get("model", {})
    inp = config.get("input", {}) or {}
    instance = config.get("instance", {})

    def fail(msg):
        print(f"[error] models/{vla}.yaml: {msg}", file=sys.stderr)
        sys.exit(1)

    for key in ("splat_python", "torch_version", "torchvision_version", "torch_index", "gsplat_version",
                "gsplat_index", "nerfstudio_version", "sim_python", "sim_pins", "menagerie_repo", "menagerie_commit"):
        if not model.get(key):
            fail(f"model.{key} is required")
    if not re.fullmatch(r"[0-9a-f]{40}", str(model["menagerie_commit"])):
        fail("model.menagerie_commit must be a 40-hex commit")
    pins = str(model["sim_pins"]).split()
    bad = [p for p in pins if not _PIN_RE.match(p)]
    if bad:
        fail(f"model.sim_pins entries must be name==version: {bad}")
    seg_args = str(model.get("segment_args", "")).split()
    bad = [a for a in seg_args if not _ARG_RE.match(a)]
    if bad:
        fail(f"model.segment_args has unsafe tokens: {bad}")
    source = inp.get("source", "hope")
    if source not in ("hope", "capture"):
        fail("input.source must be hope or capture")
    hope_scene = int(inp.get("hope_scene", 0))
    if not 0 <= hope_scene <= 9:
        fail("input.hope_scene must be 0..9 (HOPE-Video has 10 scenes)")
    capture = str(inp.get("capture_s3_uri", "") or "").strip()
    if source == "capture" and not _S3_ZIP_RE.match(capture):
        fail(f"input.capture_s3_uri must be s3://<bucket>/<key>.zip for input.source: capture, got {capture!r}")
    asset_dir = BASE_DIR / "assets" / "real2sim"
    missing = [f for f in REAL2SIM_ASSETS if not (asset_dir / f).is_file()]
    if missing:
        print(f"[error] real2sim asset files missing: {missing} (check assets/real2sim/)", file=sys.stderr)
        sys.exit(1)
    ctx = {
        "vla": vla,
        "display_name": _shell_safe(model.get("display_name", vla), "display_name"),
        "policy_desc": _shell_safe(model.get("policy_desc", vla), "policy_desc"),
        "license": _shell_safe(model.get("license", "see models/real2sim.yaml"), "license"),
        **{k: _shell_safe(model[k], k) for k in ("splat_python", "torch_version", "torchvision_version", "torch_index",
                                                 "gsplat_version", "gsplat_index", "nerfstudio_version",
                                                 "sim_python", "menagerie_repo", "menagerie_commit")},
        "sim_pins": " ".join(pins),
        "segment_args": " ".join(seg_args),
        "max_iterations": int(model.get("max_iterations", 15000)),
        "train_timeout_s": int(model.get("train_timeout_s", 4200)),
        "splat_min_opacity": float(model.get("splat_min_opacity", 0.5)),
        "grasp_ur5e": bool(model.get("grasp_ur5e", True)),
        "max_video_targets": int(model.get("max_video_targets", 4)),
        "input_source": source,
        "hope_scene": hope_scene,
        "capture_s3_uri": capture,
        "min_gpu_mib": int(instance.get("min_gpu_mib", 20000)),
        "min_host_mib": int(instance.get("min_host_mib", 28000)),
        "assets": [(f, base64.encodebytes((asset_dir / f).read_bytes()).decode()) for f in REAL2SIM_ASSETS],
    }
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template
    return env.get_template("real2sim-userdata.sh.j2").render(**ctx)


ROBOLAB_TARGETS = ("robolab-cosmos3", "robolab-pi05")
_TASK_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.=/:@+-]+$")
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")


def resolve_variant(config: dict, vla: str) -> tuple:
    """(variant name, merged variant dict) for models with a ``variants:`` block selected by model.variant.
    The same rule as resolveInstanceCfg() in cdk/lib/vla-simulator-stack.ts (the instance list comes from it)."""
    variants = config.get("variants") or {}
    name = str((config.get("model") or {}).get("variant", "")).strip()
    if name not in variants:
        print(f"[error] models/{vla}.yaml: model.variant {name!r} is not one of {sorted(variants)}", file=sys.stderr)
        sys.exit(1)
    return name, dict(variants[name] or {})


def generate_robolab(config: dict, dry_run: bool, vla: str) -> str:
    """NVIDIA RoboLab (Isaac Lab, DROID Franka) x a DROID policy server, both on one instance.

    One shared template (robolab-userdata.sh.j2) for every policy backend (rldx pattern): `model.backend`
    picks the server image and launch line (cosmos3 = NVIDIA/cosmos-framework action_policy_server_robolab,
    pi05 = openpi serve_policy.py), `model.variant` picks checkpoint, protocol (task list x episodes),
    shard layout and instance list. Embeds assets/robolab/robolab_report.py (stdlib) as base64."""
    model = config.get("model", {})
    ev = config.get("eval", {}) or {}

    def fail(msg):
        print(f"[error] models/{vla}.yaml: {msg}", file=sys.stderr)
        sys.exit(1)

    backend = model.get("backend")
    if backend not in ("cosmos3", "pi05"):
        fail("model.backend must be cosmos3 or pi05")
    variant, var = resolve_variant(config, vla)
    for key in ("robolab_repo", "robolab_commit", "isaac_lab_tag", "server_repo", "server_commit"):
        if not model.get(key):
            fail(f"model.{key} is required")
    for key in ("robolab_commit", "server_commit"):
        if not _HEX40_RE.match(str(model[key])):
            fail(f"model.{key} must be a 40-hex commit")
    protocols = config.get("protocols") or {}
    proto = protocols.get(var.get("protocol"))
    if not proto:
        fail(f"variants.{variant}.protocol must name an entry of protocols: {sorted(protocols)}")
    tasks = list(proto.get("tasks") or [])
    bad = [t for t in tasks if not _TASK_RE.match(str(t))]
    if bad:
        fail(f"protocol task names must be RoboLab task class names: {bad}")
    shards, per_shard = int(var.get("shards", 1)), int(var.get("gpus_per_shard", 1))
    num_envs, num_runs = int(proto.get("num_envs", 1)), int(proto.get("num_runs", 1))
    if not (1 <= shards <= 8 and per_shard in (1, 2) and num_envs >= 1 and num_runs >= 1):
        fail("need 1 <= shards <= 8, gpus_per_shard 1 or 2, num_envs >= 1, num_runs >= 1")
    instruction_type = str(ev.get("instruction_type", "default"))
    video_mode = str(ev.get("video_mode", "all"))
    if instruction_type not in ("default", "vague", "specific") or video_mode not in ("all", "viewport", "sensor", "none"):
        fail("eval.instruction_type must be default|vague|specific and eval.video_mode all|viewport|sensor|none")

    if backend == "cosmos3":
        for key in ("hf_repo", "hf_model_revision"):
            if not var.get(key):
                fail(f"variants.{variant}.{key} is required")
        if not _HEX40_RE.match(str(var["hf_model_revision"])):
            fail(f"variants.{variant}.hf_model_revision must be a 40-hex commit")
        for key in ("server_base_image", "server_cuda_version", "server_uv_groups"):
            if not model.get(key):
                fail(f"model.{key} is required for backend cosmos3")
        checkpoint_ref = f"{var['hf_repo']}@{str(var['hf_model_revision'])[:8]}"
        policy_label = var["hf_repo"].split("/")[-1]
        client_script = "policies/cosmos3/run.py"
    else:
        for key in ("policy_config", "checkpoint_dir"):
            if not var.get(key):
                fail(f"variants.{variant}.{key} is required")
        checkpoint_ref = var["checkpoint_dir"]
        policy_label = f"pi0.5-DROID ({var['policy_config']})"
        client_script = "policies/pi0_family/run.py"
    tokens = [str(t) for t in (list(var.get("server_args") or []) + list(var.get("client_args") or [])
                               + list(model.get("server_uv_groups") or []))]
    tokens += [str(model[k]) for k in ("robolab_repo", "isaac_lab_tag", "server_repo", "server_base_image",
                                       "server_cuda_version") if model.get(k)]
    tokens += [str(var[k]) for k in ("hf_repo", "policy_config", "checkpoint_dir") if var.get(k)]
    bad = [t for t in tokens if not _TOKEN_RE.match(t)]
    if bad:
        fail(f"unsafe characters in args / repos / images: {bad}")

    asset = BASE_DIR / "assets" / "robolab" / "robolab_report.py"
    ctx = {
        "vla": vla,
        "backend": backend,
        "variant": variant,
        "display_name": _shell_safe(model.get("display_name", vla), "display_name"),
        "license": _shell_safe(model.get("license", "see models/" + vla + ".yaml"), "license"),
        "policy_label": _shell_safe(policy_label, "policy_label"),
        "checkpoint_ref": checkpoint_ref,
        "robolab_repo": model["robolab_repo"],
        "robolab_commit": model["robolab_commit"],
        "isaac_lab_tag": model["isaac_lab_tag"],
        "server_repo": model["server_repo"],
        "server_commit": model["server_commit"],
        "server_base_image": model.get("server_base_image", ""),
        "server_cuda_version": model.get("server_cuda_version", ""),
        "server_uv_groups": list(model.get("server_uv_groups") or []),
        "min_driver_major": int(model.get("min_driver_major", 535)),
        "hf_repo": var.get("hf_repo", ""),
        "hf_model_revision": str(var.get("hf_model_revision", "")),
        "server_args": [str(a) for a in var.get("server_args") or []],
        "policy_config": var.get("policy_config", ""),
        "checkpoint_dir": var.get("checkpoint_dir", ""),
        "client_script": client_script,
        "client_args": [str(a) for a in var.get("client_args") or []],
        "label": _shell_safe(proto.get("label", "RoboLab/" + str(var.get("protocol"))), "protocol label"),
        "description": _shell_safe(proto.get("description", ""), "protocol description"),
        "leaderboard": _shell_safe(var.get("leaderboard", "see docs/results.md"), "leaderboard"),
        "tasks": tasks,
        "num_envs": num_envs,
        "num_runs": num_runs,
        "instruction_type": instruction_type,
        "video_mode": video_mode,
        "seed": int(ev.get("seed", 0)),
        "max_videos_per_task": int(var.get("max_videos_per_task", ev.get("max_videos_per_task", 10))),
        "shards": shards,
        "gpus_per_shard": per_shard,
        "eval_timeout_min": int(var.get("eval_timeout_min", 300)),
        "server_warmup_timeout": int(var.get("server_warmup_timeout", 2700)),
        "min_gpu_mib": int(var.get("min_gpu_mib", 44000)),
        "min_host_mib": int(var.get("min_host_mib", 60000)),
        "report_py_b64": base64.encodebytes(asset.read_bytes()).decode(),
    }
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template
    return env.get_template("robolab-userdata.sh.j2").render(**ctx)


def _shell_safe(value: str, what: str) -> str:
    """Values pasted into double-quoted bash strings: no quotes, $, backticks or backslashes."""
    if any(c in str(value) for c in "\"'$`\\\n"):
        print(f"[error] {what}: quotes, $, backticks, backslashes and newlines are not allowed: {value!r}",
              file=sys.stderr)
        sys.exit(1)
    return str(value)


def generate_flora_judge(config: dict, dry_run: bool, judge: str = "flora-judge") -> str:
    """Flora-Rewarder post-run judge. The run to score (bucket, run id) is not baked in: deploy.py
    passes it as CDK context and the bootstrap exports it (cdk/lib/judge-stack.ts), so one rendered
    script serves every target. Embeds assets/flora-judge/flora_judge.py (stdlib-only manifest +
    judge.json builder); the model, its batch_utils scoring code and requirements are downloaded on
    the instance at the pinned revision."""
    model = config.get("model", {})
    instance = config.get("instance", {})
    for key in ("hf_repo", "hf_model_revision", "license", "libero_task_map_url"):
        if not model.get(key):
            print(f"[error] models/{judge}.yaml: model.{key} is required", file=sys.stderr)
            sys.exit(1)
    if not re.fullmatch(r"[0-9a-f]{40}", str(model["hf_model_revision"])):
        print(f"[error] models/{judge}.yaml: model.hf_model_revision must be a 40-hex commit", file=sys.stderr)
        sys.exit(1)
    threshold = int(model.get("success_threshold", 4))
    if not 2 <= threshold <= 5:
        print(f"[error] models/{judge}.yaml: model.success_threshold must be 2..5", file=sys.stderr)
        sys.exit(1)
    asset = BASE_DIR / "assets" / "flora-judge" / "flora_judge.py"
    ctx = {
        "judge_id": judge,
        "display_name": _shell_safe(model.get("display_name", judge), "display_name"),
        "hf_repo": _shell_safe(model["hf_repo"], "hf_repo"),
        "hf_model_revision": model["hf_model_revision"],
        "license": _shell_safe(model["license"], "license"),
        "python_version": model.get("python_version", "3.12"),
        "success_threshold": threshold,
        "max_num_seqs": int(model.get("max_num_seqs", 8)),
        "gpu_memory_utilization": float(model.get("gpu_memory_utilization", 0.88)),
        "annotate_captions": bool(model.get("annotate_captions", True)),
        "libero_task_map_url": _shell_safe(model["libero_task_map_url"], "libero_task_map_url"),
        "single_gpu_mib": int(instance.get("single_gpu_mib", 79000)),
        "min_total_gpu_mib": int(instance.get("min_total_gpu_mib", 150000)),
        "min_host_mib": int(instance.get("min_host_mib", 60000)),
        "download_timeout_s": int(instance.get("download_timeout_s", 5400)),
        "score_timeout_s": int(instance.get("score_timeout_s", 5400)),
        "judge_py_b64": base64.encodebytes(asset.read_bytes()).decode(),
    }
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template
    return env.get_template(f"{judge}-userdata.sh.j2").render(**ctx)


# RLinf evaluate -> RL -> re-evaluate targets (templates/rlinf-userdata.sh.j2). Profiles: models/<vla>.yaml profiles.
RLINF_TARGETS = ("rlinf-oft-libero",)
RL_PROFILES = ("smoke", "full")
_HYDRA_OVERRIDE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*=[A-Za-z0-9_.,:/\[\]\-]+$")


def rl_profile_cfg(config: dict, vla: str, profile: str) -> dict:
    """The profile block of models/<vla>.yaml with its instance keys merged over the top-level instance."""
    profiles = config.get("profiles") or {}
    if profile not in profiles:
        print(f"[error] models/{vla}.yaml: no profile '{profile}' (available: {', '.join(profiles)})", file=sys.stderr)
        sys.exit(1)
    prof = dict(profiles[profile])
    prof["instance"] = {**(config.get("instance") or {}), **(prof.get("instance") or {})}
    return prof


def generate_rlinf(config: dict, dry_run: bool, vla: str, profile: str = "smoke", steps=None, eval_run: str = "") -> str:
    """RLinf (Apache-2.0) RL post-training: baseline eval of the SFT checkpoint, N GRPO steps, eval of the RL
    checkpoint with the same eval config and env count, results.json / summary.txt. Runs in RLinf's own Docker
    image (pinned by digest) with the RLinf checkout (pinned commit) mounted; see models/<vla>.yaml."""
    model = config.get("model", {})

    def fail(msg):
        print(f"[error] models/{vla}.yaml: {msg}", file=sys.stderr)
        sys.exit(1)

    for key in ("rlinf_repo", "rlinf_commit", "docker_image", "docker_digest", "switch_env", "suite", "suites"):
        if not model.get(key):
            fail(f"model.{key} is required")
    if not re.fullmatch(r"[0-9a-f]{40}", str(model["rlinf_commit"])):
        fail("model.rlinf_commit must be a 40-hex commit")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(model["docker_digest"])):
        fail("model.docker_digest must be sha256:<64 hex>")
    suite = str(model["suite"])
    scfg = (model.get("suites") or {}).get(suite)
    if not scfg:
        fail(f"model.suites has no '{suite}' entry")
    for key in ("task_suite_name", "train_config", "eval_config", "hf_repo", "hf_model_revision"):
        if not scfg.get(key):
            fail(f"model.suites.{suite}.{key} is required")
    if not re.fullmatch(r"[0-9a-f]{40}", str(scfg["hf_model_revision"])):
        fail(f"model.suites.{suite}.hf_model_revision must be a 40-hex commit")
    pins = str(model.get("sim_protect_pins", "")).split()
    if not pins or any(not _PIN_RE.match(x) for x in pins):
        fail(f"model.sim_protect_pins must be name==version entries: {pins}")
    mujoco_version = next((x.split("==", 1)[1] for x in pins if x.lower().startswith("mujoco==")), "")
    prof = rl_profile_cfg(config, vla, profile)
    rl_steps = int(steps if steps is not None else prof.get("steps", 0))
    if rl_steps < 1:
        fail(f"profiles.{profile}.steps (or --rl-steps) must be >= 1")
    overrides = [str(o) for o in (prof.get("overrides") or [])]
    eval_overrides = [str(o) for o in (prof.get("eval_overrides") or [])]
    bad = [o for o in overrides + eval_overrides if not _HYDRA_OVERRIDE_RE.match(o)]
    if bad:
        fail(f"profiles.{profile}: overrides must be key=value hydra overrides without quotes or spaces: {bad}")
    if eval_run and not re.fullmatch(r"\d{4}-\d{2}-\d{2}-\d{4}", eval_run):
        print(f"[error] --rl-eval-run must look like YYYY-MM-DD-HHMM, got {eval_run!r}", file=sys.stderr)
        sys.exit(1)
    inst = prof["instance"]
    # RLinf's eval runs one env process per visible GPU and asserts env.eval.total_num_envs % processes == 0;
    # each process builds its envs from a round-robin share of the eval reset states, so a count that does not
    # split evenly fails before the first step (2026-11-01: 500 and 504 envs on 8 x H100). The eval is placed
    # on the first eval_gpus GPUs (default: all of instance.min_gpus).
    eval_num_envs = int(prof.get("eval_num_envs", 500))
    eval_gpus = int(prof.get("eval_gpus") or inst.get("min_gpus", 1))
    if not 1 <= eval_gpus <= int(inst.get("min_gpus", 1)):
        fail(f"profiles.{profile}: eval_gpus {eval_gpus} must be between 1 and instance.min_gpus")
    if eval_num_envs % eval_gpus:
        fail(
            f"profiles.{profile}: eval_num_envs {eval_num_envs} must be a multiple of eval_gpus {eval_gpus} "
            f"(RLinf runs one eval env process per GPU)"
        )
    ctx = {
        "vla": vla,
        "deployment": config.get("deployment", {}),
        "display_name": _shell_safe(model.get("display_name", vla), "display_name"),
        "rlinf_repo": _shell_safe(model["rlinf_repo"], "rlinf_repo"),
        "rlinf_commit": model["rlinf_commit"],
        "docker_image": _shell_safe(model["docker_image"], "docker_image"),
        "docker_digest": model["docker_digest"],
        "switch_env": _shell_safe(model["switch_env"], "switch_env"),
        "sim_protect_pins": " ".join(pins),
        "mujoco_version": mujoco_version,
        "task_suite_name": _shell_safe(scfg["task_suite_name"], "task_suite_name"),
        "train_config": _shell_safe(scfg["train_config"], "train_config"),
        "eval_config": _shell_safe(scfg["eval_config"], "eval_config"),
        "hf_repo": _shell_safe(scfg["hf_repo"], "hf_repo"),
        "hf_model_revision": scfg["hf_model_revision"],
        "published_sft": float(scfg.get("published_sft", 0) or 0),
        "published_rl": float(scfg.get("published_rl", 0) or 0),
        "profile": profile,
        "rl_steps": rl_steps,
        "steps_overridden": steps is not None,
        "save_interval": int(prof.get("save_interval") or min(25, rl_steps)),
        "eval_num_envs": eval_num_envs,
        "eval_gpus": eval_gpus,
        "overrides": overrides,
        "eval_overrides": eval_overrides,
        "early_signal": bool(prof.get("early_signal", False)),
        "max_wall_hours": int(prof.get("max_wall_hours", 12)),
        "eval_run": eval_run,
        "min_gpus": int(inst.get("min_gpus", 1)),
        "min_gpu_mib": int(inst.get("min_gpu_mib", 0)),
        "min_host_mib": int(inst.get("min_host_mib", 0)),
        "max_videos_per_phase": int(model.get("max_videos_per_phase", 20)),
        "report_py_b64": base64.encodebytes((BASE_DIR / "assets" / "rlinf" / "rlinf_report.py").read_bytes()).decode(),
    }
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template
    return env.get_template("rlinf-userdata.sh.j2").render(**ctx)


def generate_pi(config: dict, resolved_vpc: str, resolved_nlb: str, dry_run: bool, plus: tuple = None,
                r2s: tuple = None, r2s_suite: str = None) -> str:
    """r2s = (run uri, conditions, trials) for --libero-objects real2sim, else None."""
    tasks = config.get("tasks", [])
    if not tasks:
        print("[error] models/pi.yaml has no tasks entry.", file=sys.stderr)
        sys.exit(1)
    lp_ctx = None
    if plus is not None:
        # --libero-variant plus: one task entry per perturbation dimension (variant ids picked on the
        # instance); num_episodes = trials per variant. Same pi05_libero checkpoint as the LIBERO run.
        lp_ctx, lp_tasks = _libero_plus_ctx(config, "pi", *plus)
        if lp_ctx["hf_repo"]:
            print("[error] models/pi.yaml: libero_plus.hf_repo is not supported (openpi serves the "
                  "pi05_libero checkpoint itself).", file=sys.stderr)
            sys.exit(1)
        tasks = [dict(t, num_episodes=lp_ctx["episodes_per_variant"]) for t in lp_tasks]
    r2s_ctx = None
    if r2s is not None:
        r2s_ctx, r2s_tasks = _libero_r2s_ctx(config, "pi", r2s, r2s_suite or r2s_default_suite(config, "pi"))
        tasks = [dict(t, num_episodes=r2s_ctx["trials_per_task"]) for t in r2s_tasks]

    tasks_json = _tasks_json_for_bash(tasks)
    deployment = config.get("deployment", {})
    model = config.get("model", {})

    bridge_dir = BASE_DIR / "assets" / "bridge" / "pi"
    pi_proto_path = bridge_dir / "pi.proto"
    pi_grpc_bridge_path = bridge_dir / "pi_grpc_bridge.py"

    ctx = {
        "tasks_json": tasks_json,
        "deployment": deployment,
        "openpi_commit": model.get("openpi_commit", "").strip(),
        "pi_proto": pi_proto_path.read_text() if pi_proto_path.exists() else "",
        "pi_grpc_bridge_py": pi_grpc_bridge_path.read_text() if pi_grpc_bridge_path.exists() else "",
        "stackpatch_py": (BASE_DIR / "assets" / "stack-knobs" / "stackpatch.py").read_text(),
        "mujoco_pin": str(model.get("mujoco_pin", "") or "").strip(),
        "libero_plus": lp_ctx,
        "libero_r2s": r2s_ctx,
    }

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template, autoescape not applicable
    return env.get_template("pi-userdata.sh.j2").render(**ctx)


def main():
    parser = argparse.ArgumentParser(description="Generate the EC2 UserData script for an RFM Simulation target")
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--vla", choices=VLA_TARGETS, help="Target (model + simulator) to render")
    what.add_argument("--judge-target", choices=JUDGE_TARGETS,
                      help="Post-run judge to render (scores the videos of a finished run; see deploy.py --judge)")
    parser.add_argument(
        "--config", default=str(BASE_DIR / "simulator-config.yaml"),
        help="Shared config file path (default: simulator-config.yaml)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the rendered script instead of writing it")
    parser.add_argument("--output-dir", default=str(BASE_DIR / "assets" / "userdata"),
                        help="Output directory")
    # Resolved bridge values (deploy.py looks them up in SSM and passes them in)
    parser.add_argument("--resolved-grpc", default="", help="GR00T: resolved gRPC endpoint")
    parser.add_argument("--resolved-vpc", default="", help="resolved VPC ID")
    parser.add_argument("--resolved-nlb", default="", help="Pi: resolved NLB endpoint")
    parser.add_argument(
        "--libero-suite", default=OFT_DEFAULT_SUITE, choices=list(OFT_LIBERO_SUITES),
        help="openvla-oft: LIBERO suite (default: 10 = LIBERO-Long; `long` is alias for `10`)",
    )
    parser.add_argument("--libero-variant", default="libero", choices=list(libero_plus.VARIANTS),
                        help="LIBERO targets: `plus` swaps LIBERO for the LIBERO-Plus robustness fork and runs one "
                             "eval per perturbation dimension (models/<vla>.yaml libero_plus; default: libero)")
    parser.add_argument("--perturb", default=None, metavar="LIST",
                        help=f"--libero-variant plus: comma-separated dimensions ({','.join(libero_plus.CATEGORIES)}) "
                             "or all (default: libero_plus.perturb in the model yaml)")
    parser.add_argument("--perturb-n", type=int, default=None, metavar="N",
                        help="--libero-variant plus: variants per dimension, 1 episode each; 0 = every variant "
                             "(default: libero_plus.variants_per_category)")
    parser.add_argument("--libero-objects", default="libero", choices=list(libero_r2s.OBJECT_SOURCES),
                        help="LIBERO targets: `real2sim` replaces LIBERO's HOPE objects by the objects reconstructed in "
                             "a real2sim stage-1 run and evaluates public vs swapped scenes (models/<vla>.yaml "
                             "libero_r2s; default: libero)")
    parser.add_argument("--real2sim-run", default=None, metavar="S3_URI",
                        help="--libero-objects real2sim: s3://<results bucket>/<RUN_ID> of a finished real2sim run")
    parser.add_argument("--r2s-conditions", default=None, metavar="LIST",
                        help=f"--libero-objects real2sim: conditions ({','.join(libero_r2s.CONDITIONS)}; public is "
                             "always included; default: libero_r2s.conditions)")
    parser.add_argument("--r2s-trials", type=int, default=None, metavar="N",
                        help="--libero-objects real2sim: trials per selected task and condition "
                             "(default: libero_r2s.trials_per_task)")
    parser.add_argument("--r2s-suite", default=None, choices=list(libero_r2s.SUITES),
                        help="pi + --libero-objects real2sim: LIBERO suite (default: libero_r2s.suite; "
                             "openvla-oft uses --libero-suite)")
    parser.add_argument("--robotwin-protocol", default="subset", choices=list(robotwin_plan.PROTOCOLS),
                        help="RoboTwin targets: subset = the tasks of models/<vla>.yaml (default); clean / "
                             "randomized / full = the official 49 tasks x 100 episodes (full = both settings)")
    parser.add_argument("--robotwin-shard", default="", metavar="K/N",
                        help="RoboTwin targets: keep every N-th (setting, task) unit starting at K (1-based)")
    parser.add_argument("--rl-profile", default="smoke", choices=list(RL_PROFILES),
                        help="rlinf-oft-libero: smoke = 1 GPU, LoRA, a few GRPO steps; full = 8 GPU, recipe defaults "
                             "(models/<vla>.yaml profiles)")
    parser.add_argument("--rl-steps", type=int, default=None, metavar="N",
                        help="rlinf-oft-libero: GRPO steps (default: the profile's steps)")
    parser.add_argument("--rl-eval-run", default="", metavar="RUN_ID",
                        help="rlinf-oft-libero: skip training and evaluate the RL checkpoint saved by run RUN_ID")
    parser.add_argument("--collect", action="store_true",
                        help="gr00t-g1: trajectory COLLECTION mode (success-only obs/action HDF5 for "
                             "N1.7 G1 adapter FT) instead of the default rollout/video mode.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"[error] Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    if (args.robotwin_protocol != "subset" or args.robotwin_shard) and args.vla not in LEROBOT_ROBOTWIN_TARGETS:
        parser.error(f"--robotwin-protocol / --robotwin-shard apply to {', '.join(LEROBOT_ROBOTWIN_TARGETS)} only")

    if (args.rl_profile != "smoke" or args.rl_steps is not None or args.rl_eval_run) and args.vla not in RLINF_TARGETS:
        parser.error(f"--rl-profile / --rl-steps / --rl-eval-run apply to {', '.join(RLINF_TARGETS)} only")

    name = args.judge_target or args.vla
    config = _load_merged_config(config_path, name)
    plus = None
    if args.libero_variant == "plus":
        if args.judge_target:
            print("[error] --libero-variant applies to --vla targets, not judges.", file=sys.stderr)
            sys.exit(1)
        plus = (args.perturb, args.perturb_n)
        _libero_plus_ctx(config, args.vla, *plus)  # fail fast (unsupported target / bad settings)
    elif args.perturb is not None or args.perturb_n is not None:
        print("[error] --perturb / --perturb-n need --libero-variant plus.", file=sys.stderr)
        sys.exit(1)
    r2s = None
    r2s_flags = (args.real2sim_run, args.r2s_conditions, args.r2s_trials, args.r2s_suite)
    if args.libero_objects == "real2sim":
        if args.judge_target or plus is not None:
            print("[error] --libero-objects real2sim applies to --vla targets and cannot be combined with "
                  "--libero-variant plus.", file=sys.stderr)
            sys.exit(1)
        if not args.real2sim_run:
            print("[error] --libero-objects real2sim needs --real2sim-run s3://<bucket>/<RUN_ID>.", file=sys.stderr)
            sys.exit(1)
        if args.r2s_suite and args.vla != "pi":
            print("[error] --r2s-suite applies to pi (openvla-oft follows --libero-suite).", file=sys.stderr)
            sys.exit(1)
        r2s = (args.real2sim_run, args.r2s_conditions, args.r2s_trials)
        _libero_r2s_ctx(config, args.vla, r2s, args.r2s_suite or r2s_default_suite(config, args.vla, args.libero_suite))
    elif any(x is not None for x in r2s_flags):
        print("[error] --real2sim-run / --r2s-* need --libero-objects real2sim.", file=sys.stderr)
        sys.exit(1)
    print(f"VLA: {name}")
    print(f"Config: {config_path}")
    print(f"Model:  {BASE_DIR / 'models' / (name + '.yaml')}")

    if args.judge_target:
        rendered = generate_flora_judge(config, args.dry_run, judge=args.judge_target)
    elif args.vla in ("gr00t", "gr00t-bridge"):
        rendered = generate_gr00t(config, args.resolved_grpc, args.resolved_vpc, args.dry_run, vla=args.vla, plus=plus)
    elif args.vla == "gr00t-gr1":
        rendered = generate_gr00t_gr1(config, args.resolved_grpc, args.resolved_vpc, args.dry_run)
    elif args.vla == "gr00t-g1":
        rendered = generate_gr00t_g1(config, args.resolved_grpc, args.resolved_vpc, args.dry_run,
                                     collect=args.collect)
    elif args.vla == "openvla-oft":
        rendered = generate_openvla_oft(config, args.libero_suite, args.dry_run, plus=plus, r2s=r2s)
    elif args.vla == "lap":
        rendered = generate_lap(config, args.resolved_nlb, args.dry_run, plus=plus)
    elif args.vla in ("rldx", "rldx-simpler", "rldx-bridge", "rldx-gr1", "rldx-kitchen"):
        rendered = generate_rldx(config, args.dry_run, vla=args.vla, plus=plus)
    elif args.vla == "openarm-isaac":
        rendered = generate_openarm_isaac(config, args.dry_run)
    elif args.vla == "openarm-lift-act":
        rendered = generate_openarm_lift_act(config, args.dry_run)
    elif args.vla == "molmoact2":
        rendered = generate_molmoact2(config, args.dry_run, plus=plus)
    elif args.vla in LEROBOT_LIBERO_TARGETS:
        rendered = generate_lerobot_libero(config, args.dry_run, vla=args.vla, plus=plus)
    elif args.vla in LEROBOT_ROBOTWIN_TARGETS:
        rendered = generate_lerobot_robotwin(config, args.dry_run, vla=args.vla,
                                             protocol=args.robotwin_protocol, shard=args.robotwin_shard)
    elif args.vla == "cosmos-policy":
        rendered = generate_cosmos_policy(config, args.dry_run)
    elif args.vla == "vla-adapter":
        rendered = generate_vla_adapter(config, args.dry_run, plus=plus)
    elif args.vla == "aspire":
        rendered = generate_aspire(config, args.dry_run)
    elif args.vla == "real2sim":
        rendered = generate_real2sim(config, args.dry_run)
    elif args.vla in ROBOLAB_TARGETS:
        rendered = generate_robolab(config, args.dry_run, vla=args.vla)
    elif args.vla in RLINF_TARGETS:
        rendered = generate_rlinf(config, args.dry_run, vla=args.vla, profile=args.rl_profile, steps=args.rl_steps,
                                  eval_run=args.rl_eval_run)
    else:
        rendered = generate_pi(config, args.resolved_vpc, args.resolved_nlb, args.dry_run, plus=plus,
                               r2s=r2s, r2s_suite=args.r2s_suite)

    if args.dry_run:
        print("=" * 60)
        print(f"[DRY-RUN] {name}.sh preview:")
        print("=" * 60)
        print(rendered)
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{name}.sh"
    dest.write_text(rendered)

    # Syntax-validate the rendered UserData before it can reach a deploy. A bash
    # syntax error (e.g. an apostrophe in a task description breaking a single-quoted
    # string) otherwise only surfaces on the EC2 instance: the script aborts before
    # sending cfn_signal and the stack hangs on its WaitCondition until timeout
    # (idle GPU billing). `bash -n` catches it here, loudly, at generation time.
    bash_check = subprocess.run(  # nosec B603,B607 - hardcoded command, dest is our own output
        ["bash", "-n", str(dest)],
        capture_output=True,
        text=True,
    )
    if bash_check.returncode != 0:
        print(f"[error] Generated {dest} has a bash syntax error:", file=sys.stderr)
        print(bash_check.stderr.strip(), file=sys.stderr)
        print("  Most common cause: an apostrophe or unescaped quote in a task "
              "description in models/{vla}.yaml.", file=sys.stderr)
        sys.exit(1)

    print(f"Generated: {dest}")


if __name__ == "__main__":
    main()
