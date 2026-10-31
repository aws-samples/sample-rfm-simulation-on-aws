"""LIBERO real2sim object swap: shared constants and the evaluation plan (stdlib only).

Stage 1 (`vlasim deploy --vla real2sim`) turns a capture into per-object collision hulls. Stage 2 asks
whether a policy that is strong on the public LIBERO scene still grasps when the objects are replaced by
those reconstructions. A LIBERO target opts in with

    python vlasim.py deploy --vla openvla-oft --libero-suite object --libero-objects real2sim \\
        --real2sim-run s3://<stage-1 results bucket>/<RUN_ID>

generate.py replaces the target's tasks with one entry per condition (`condition: LIBERO-R2S/<name>`):
`public` (stock LIBERO objects), `collision` (reconstructed collision geometry, stock visuals) and `full`
(reconstructed geometry rendered with the capture's colour). Every condition runs the same task ids,
the same fixed initial states and the same seed, so the only difference is the objects. The UserData
downloads the stage-1 outputs (<run>/real2sim/segment/ + splat_world.ply), builds the object assets
(assets/libero-r2s/build_r2s_objects.py), checks the scenes' physics and points LIBERO's HOPE object
loader at one asset directory per condition (assets/libero-r2s/r2s_select.py). Off by default: without
the flag nothing in the rendered UserData changes.
"""
import re

CONDITIONS = ("public", "collision", "full")
PREFIX = "LIBERO-R2S/"
OBJECT_SOURCES = ("libero", "real2sim")
SUITES = ("libero_object", "libero_10", "libero_goal", "libero_spatial", "libero_90")

DEFAULTS = {
    "conditions": list(CONDITIONS),
    "trials_per_task": 10,
    "scale": "fit",      # fit = bounding-box volume of the stock object; metric = measured size
    "mass": "match",     # match = stock mass; density = stock density on the reconstructed volume
    "select": "interest",  # interest = tasks whose object of interest is replaced; any = object anywhere
    "seed": 7,
    "objects": "",       # optional override "libero_name=cluster,..." (non-HOPE captures)
}

# s3://bucket/RUN_ID[/real2sim][/]  ->  (bucket, "RUN_ID/real2sim")
_RUN_RE = re.compile(r"^s3://([a-z0-9][a-z0-9.-]{1,61}[a-z0-9])/((?:[A-Za-z0-9_.=+-]+/)*?[A-Za-z0-9_.=+-]+?)"
                     r"(?:/real2sim)?/?$")
_MAP_RE = re.compile(r"^([a-z_]+=\d+)(,[a-z_]+=\d+)*$")


def parse_run(uri):
    """Validate a stage-1 run URI. Returns (bucket, prefix) where prefix ends in /real2sim."""
    m = _RUN_RE.match(str(uri or "").strip())
    if not m or ".." in m.group(2):
        raise ValueError(f"--real2sim-run must be s3://<bucket>/<RUN_ID> of a real2sim stage-1 run, got {uri!r}")
    return m.group(1), m.group(2) + "/real2sim"


def parse_conditions(value):
    if value is None:
        return None
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    items = [str(x).strip().lower() for x in items if str(x).strip()]
    bad = [x for x in items if x not in CONDITIONS]
    if bad or not items:
        raise ValueError(f"unknown condition(s) {bad or items}; choose from {', '.join(CONDITIONS)}")
    return [c for c in CONDITIONS if c in items]


def plan(block, run_uri, suite, conditions=None, trials=None):
    """Merge DEFAULTS + models/<id>.yaml ``libero_r2s`` block + CLI overrides.

    Returns (cfg, tasks): one task entry per condition in the ``tasks`` schema the LIBERO templates
    loop over. Raises ValueError on a bad setting.
    """
    cfg = {**DEFAULTS, **(block or {})}
    if suite not in SUITES:
        raise ValueError(f"libero_r2s: suite must be one of {', '.join(SUITES)}, got {suite!r}")
    bucket, prefix = parse_run(run_uri)
    conds = parse_conditions(conditions if conditions is not None else cfg["conditions"])
    if "public" not in conds:
        conds = ["public", *conds]  # the reference is always evaluated in the same run
    n = int(cfg["trials_per_task"] if trials is None else trials)
    if n < 1:
        raise ValueError("trials per task must be >= 1")
    for key, ok in (("scale", ("fit", "metric")), ("mass", ("match", "density")), ("select", ("interest", "any"))):
        if cfg[key] not in ok:
            raise ValueError(f"libero_r2s.{key} must be one of {', '.join(ok)}, got {cfg[key]!r}")
    if cfg["objects"] and not _MAP_RE.match(str(cfg["objects"])):
        raise ValueError("libero_r2s.objects must be libero_name=cluster_id,...")
    cfg.update(conditions=conds, trials_per_task=n, seed=int(cfg["seed"]), suite=suite,
               run_bucket=bucket, run_prefix=prefix, run_uri=f"s3://{bucket}/{prefix}")
    tasks = [{
        "suite": suite,
        "r2s_condition": c,
        "r2s_dir": "stable_hope_objects" if c == "public" else f"stable_hope_objects_r2s_{c}",
        "condition": PREFIX + c,
        "seed": cfg["seed"],
        "description": f"LIBERO-R2S {suite} / {c} ({n} trials per selected task)",
    } for c in conds]
    return cfg, tasks


def episodes_hint(cfg, n_tasks=5):
    """Episode count for the default HOPE-Video scene (5 libero_object tasks), for time estimates."""
    return n_tasks * cfg["trials_per_task"] * len(cfg["conditions"])
