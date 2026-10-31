#!/usr/bin/env python3
"""
leaderboard.py — cross-model matrix with cost and latency, built on results.py.

    python vlasim.py results --matrix                      # every region, baseline conditions
    python vlasim.py results --matrix --all-conditions     # one extra row per stack condition
    python vlasim.py results --matrix --csv m.csv --json m.json
    python vlasim.py results --write-docs                  # refresh the block in docs/results.md

Rows are model targets, columns are benchmarks (normalised labels, see BENCH_COND_RE and
SUITE_LABELS). A cell is the latest run of that target on that benchmark under a baseline
condition, with k/n and a 95% Wilson interval. Shards of one evaluation that ran as separate
stacks at about the same time (same condition and stack fingerprint, different bucket, started
within --pool-hours of the latest one, same checkpoint, disjoint recorded `task_ids`) are pooled
into the cell; reruns of the same tasks are not.

Cost and time are estimates: wall time = RUN_ID (instance boot, UTC) to the last object the
run uploaded (or the optional `run_wall_s` summary field), times the pinned On-Demand price of
the instance type in that region (PRICES below, AWS Price List API). Stack create/delete,
NAT gateway, EBS and data transfer are not included.

This module only reads S3 (list + get); it never writes to AWS.
"""
import collections
import concurrent.futures
import csv
import datetime as dt
import json
import re
import sys
from pathlib import Path

import results

# --- benchmark labels -----------------------------------------------------------------------------

# `suite` (or the part of `env_name` before "/") -> column label.
SUITE_LABELS = {
    "libero_spatial": "LIBERO-Spatial",
    "libero_object": "LIBERO-Object",
    "libero_goal": "LIBERO-Goal",
    "libero_10": "LIBERO-10",
    "libero_long": "LIBERO-10",
    "libero_90": "LIBERO-90",
    "libero_sim": "LIBERO scenes (GR00T)",
    "simpler_bridge": "SimplerEnv-Bridge",
    "simpler_env_widowx": "SimplerEnv-Bridge",
    "simpler_google": "SimplerEnv-Google",
    "simpler_env_google": "SimplerEnv-Google",
    "robocasa_kitchen": "RoboCasa-Kitchen",
    "robocasa_panda_omron": "RoboCasa-Kitchen",
    "gr1_unified": "GR1-Tabletop",
    "gr00tlocomanip_g1_sim": "G1-LocoManip",
}
# A `condition` of the form <Benchmark>/<setting> names a benchmark variant rather than a stack
# condition: LIBERO-Plus/camera, RoboLab/subset12, RoboTwin2/demo_clean. It becomes the column.
BENCH_COND_RE = re.compile(r"^(?P<bench>[A-Z][A-Za-z0-9.+_-]*)/(?P<setting>\S+)$")
# Stack conditions that mean "the upstream / official evaluation stack".
BASELINE_CONDITIONS = {"", "-", "s0", "none", "default", "baseline", "official"}
# RoboTwin targets write the benchmark into `suite` (RoboTwin2/clean) and the protocol into `condition`.
# These protocol names are baseline runs when the suite itself is <Benchmark>/<setting>.
PROTOCOL_CONDITIONS = {"subset", "clean", "randomized", "full"}
# Column order: known benchmarks first, everything else alphabetically after them.
COLUMN_ORDER = ["LIBERO-Spatial", "LIBERO-Object", "LIBERO-Goal", "LIBERO-10", "LIBERO-90", "LIBERO-Plus", "LIBERO-R2S",
                "LIBERO scenes", "SimplerEnv", "RoboCasa", "GR1", "G1", "RoboLab", "RoboTwin"]

# Keys that change what a success rate means. Cells in one column whose values differ are marked.
COMPARE_KEYS_RE = re.compile(r"^(stack_.+|step_scale|random_init|replan_head|max_episode_steps)$")
# Stack keys that name the policy side (its eval command line, its policy-server pin). They always differ
# between two models on the same simulator, so they are not a comparability signal.
POLICY_STACK_KEYS = {"stack_eval_args", "stack_server"}


def suite_label(suite):
    s = (suite or "-").strip()
    return SUITE_LABELS.get(s.lower()) or SUITE_LABELS.get(s.split("/", 1)[0].lower()) or s.split("/", 1)[0]


def benchmark_label(suite, condition):
    """Column label for a (suite, condition) group."""
    lbl = suite_label(suite)
    m = BENCH_COND_RE.match((condition or "").strip())
    if not m:
        s = (suite or "").strip()
        if BENCH_COND_RE.match(s) and (condition or "").strip().lower() in PROTOCOL_CONDITIONS:
            return f"{s} (subset)" if condition.strip().lower() == "subset" else s  # RoboTwin2/clean (subset)
        return lbl
    bench = f"{m.group('bench')}/{m.group('setting')}"
    if m.group("bench").upper().startswith("LIBERO") and lbl.startswith("LIBERO-"):
        return f"{bench} ({lbl[len('LIBERO-'):]})"  # LIBERO-Plus/camera (Spatial)
    return bench


def is_baseline(condition, suite=None):
    c = (condition or "").strip()
    if c.lower() in PROTOCOL_CONDITIONS and BENCH_COND_RE.match((suite or "").strip()):
        return True
    return c.lower() in BASELINE_CONDITIONS or bool(BENCH_COND_RE.match(c))


def column_sort_key(label):
    for i, prefix in enumerate(COLUMN_ORDER):
        if label.startswith(prefix):
            return (i, label)
    return (len(COLUMN_ORDER), label)


# --- target (row) of a bucket ---------------------------------------------------------------------

# Stack-name prefixes that are not a target id but clearly belong to one.
STACK_ALIASES = {"oft": "openvla-oft", "lingbot": "lingbot-va", "cosmos": "cosmos-policy"}


def known_targets():
    try:
        from targets import VLA_TARGETS
        return list(VLA_TARGETS)
    except Exception:  # pragma: no cover - targets.py is in the repo
        return []


def target_of(stack, model_field=None, targets=None):
    """Model target for a bucket stack name: summary `model:` if it is a target id, else the
    longest target id the stack name starts with, else an alias, else the stack name minus -demo."""
    targets = known_targets() if targets is None else targets
    model = (model_field or "").split()[0] if (model_field or "").split() else ""
    if model in targets:
        return model
    hits = [t for t in targets if stack == t or stack.startswith(t + "-")]
    if hits:
        return max(hits, key=len)
    for prefix, t in STACK_ALIASES.items():
        if stack == prefix or stack.startswith(prefix + "-"):
            return t
    return re.sub(r"-demo$", "", stack)


# --- cost -----------------------------------------------------------------------------------------

# On-Demand Linux USD per hour, shared tenancy, no pre-installed software. Source: AWS Price List
# API (`aws pricing get-products --region us-east-1 --service-code AmazonEC2`, filters instanceType,
# regionCode, operatingSystem=Linux, tenancy=Shared, preInstalledSw=NA, capacitystatus=Used).
PRICES_CHECKED = "2026-10-30"
PRICES = {
    "us-east-1": {
        "g5.xlarge": 1.006, "g5.2xlarge": 1.212, "g5.4xlarge": 1.624, "g5.12xlarge": 5.672, "g5.24xlarge": 8.144,
        "g6.xlarge": 0.8048, "g6.2xlarge": 0.9776, "g6.4xlarge": 1.3232, "g6.12xlarge": 4.6016,
        "g6.24xlarge": 6.6752, "g6e.xlarge": 1.861, "g6e.2xlarge": 2.24208, "g6e.4xlarge": 3.00424,
        "g6e.12xlarge": 10.49264, "g6e.24xlarge": 15.06559, "g7e.2xlarge": 3.36312, "g7e.4xlarge": 3.99816,
        "p5.4xlarge": 6.88,
    },
    "us-west-2": {
        "g5.xlarge": 1.006, "g5.2xlarge": 1.212, "g5.4xlarge": 1.624, "g5.12xlarge": 5.672, "g5.24xlarge": 8.144,
        "g6.xlarge": 0.8048, "g6.2xlarge": 0.9776, "g6.4xlarge": 1.3232, "g6.12xlarge": 4.6016,
        "g6.24xlarge": 6.6752, "g6e.xlarge": 1.861, "g6e.2xlarge": 2.24208, "g6e.4xlarge": 3.00424,
        "g6e.12xlarge": 10.49264, "g6e.24xlarge": 15.06559, "g7e.2xlarge": 3.36312, "g7e.4xlarge": 3.99816,
        "p5.4xlarge": 6.88,
    },
    "ap-northeast-1": {
        "g5.xlarge": 1.459, "g5.2xlarge": 1.75776, "g5.4xlarge": 2.35528, "g5.12xlarge": 8.22609,
        "g5.24xlarge": 11.81123, "g6.xlarge": 1.1672, "g6.2xlarge": 1.41781, "g6.4xlarge": 1.91903,
        "g6.12xlarge": 6.67369, "g6.24xlarge": 9.68103, "g6e.xlarge": 2.699, "g6e.2xlarge": 3.25168,
        "g6e.4xlarge": 4.35703, "g6e.12xlarge": 15.21742, "g6e.24xlarge": 21.84956, "g7e.2xlarge": 4.87752,
        "g7e.4xlarge": 5.79852, "p5.4xlarge": 8.6,
    },
    "ap-northeast-2": {
        "g5.xlarge": 1.237, "g5.2xlarge": 1.4903, "g5.4xlarge": 1.99691, "g5.12xlarge": 6.97442,
        "g5.24xlarge": 10.01404, "g6.xlarge": 0.9896, "g6.2xlarge": 1.20208, "g6.4xlarge": 1.62704,
        "g6.12xlarge": 5.65823, "g6.24xlarge": 8.20797, "g6e.xlarge": 2.288, "g6e.2xlarge": 2.75652,
        "g6e.4xlarge": 3.69355, "g6e.12xlarge": 12.90014, "g6e.24xlarge": 18.52234, "g7e.2xlarge": 4.13478,
        "g7e.4xlarge": 4.91553,
    },
    "eu-central-1": {
        "g5.xlarge": 1.258, "g5.2xlarge": 1.5156, "g5.4xlarge": 2.03081, "g5.12xlarge": 7.09282,
        "g5.24xlarge": 10.18405, "g6.xlarge": 1.0064, "g6.2xlarge": 1.22249, "g6.4xlarge": 1.65466,
        "g6.12xlarge": 5.75429, "g6.24xlarge": 8.34732, "g6e.xlarge": 2.327, "g6e.2xlarge": 2.8035,
        "g6e.4xlarge": 3.75651, "g6e.12xlarge": 13.12002, "g6e.24xlarge": 18.83806, "g7e.2xlarge": 5.71915,
        "g7e.4xlarge": 6.79907,
    },
}

INSTANCE_TYPE_RE = re.compile(r"\b([a-z][a-z0-9-]*\d[a-z0-9-]*\.(?:\d*x?large|metal(?:-\d+xl)?|medium|small))\b")
EPISODES_TOOK_RE = re.compile(r"Collecting (\d+) episodes took ([\d.]+) seconds")
JUDGE_KEY_RE = re.compile(r"(^|/)judge[^/]*(/|$)")  # judge.json, judge.txt, judge-*.jsonl, judge-diag-fail/
CAPTION_KEY_RE = re.compile(r"/videos/[^/]+\.txt$")  # captions are re-uploaded when a judge score is added
MAX_WALL_S = 48 * 3600


def instance_type_in(text):
    m = INSTANCE_TYPE_RE.search(text or "")
    return m.group(1) if m else None


def price_per_hour(instance_type, region):
    return PRICES.get(region, {}).get(instance_type or "")


def run_started(run):
    """RUN_ID is `date '+%Y-%m-%d-%H%M'` at the top of the instance user data (UTC on the AMIs)."""
    try:
        return dt.datetime.strptime(run, "%Y-%m-%d-%H%M").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def upload_span_s(run, objects):
    """Seconds from RUN_ID to the last object of the run, ignoring post-run judge outputs and
    video captions (both can be written days later)."""
    start = run_started(run)
    ends = [o["LastModified"] for o in objects
            if not JUDGE_KEY_RE.search(o["Key"][len(run) + 1:]) and not CAPTION_KEY_RE.search(o["Key"])]
    if not start or not ends:
        return None
    span = (max(ends) - start).total_seconds()
    return span if 0 < span <= MAX_WALL_S else None


# --- S3 access over one listing per run -------------------------------------------------------------

class RunView:
    """results.S3-compatible view of one run: prefixes/keys answer from a single listing, text()
    does a cached GET. Lets results.task_row run without extra list calls."""

    def __init__(self, s3, region, bucket, run, objects):
        self.s3, self.region, self.bucket, self.run = s3, region, bucket, run
        self.objects = objects
        self.cache = {}

    def prefixes(self, region, bucket, prefix):
        out = set()
        for o in self.objects:
            k = o["Key"]
            if k.startswith(prefix) and "/" in k[len(prefix):]:
                out.add(prefix + k[len(prefix):].split("/", 1)[0] + "/")
        return sorted(out)

    def keys(self, region, bucket, prefix):
        return [o["Key"] for o in self.objects if o["Key"].startswith(prefix)]

    def text(self, region, bucket, key, head=None):
        if key not in self.cache:
            self.cache[key] = self.s3.text(region, bucket, key, head) if head else self.s3.text(region, bucket, key)
        return self.cache[key]


class S3(results.S3):
    def objects(self, region, bucket, prefix):
        out = []
        for page in self.c(region).get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            out += [dict(Key=o["Key"], LastModified=o["LastModified"]) for o in page.get("Contents", [])]
        return out

    def text(self, region, bucket, key, head=None):
        try:
            kw = {"Range": f"bytes=0-{head - 1}"} if head else {}
            body = self.c(region).get_object(Bucket=bucket, Key=key, **kw)["Body"].read()
        except Exception:
            return None
        return body.decode("utf-8", "replace")


# --- collection -------------------------------------------------------------------------------------

def list_buckets(args, boto3):
    acct = boto3.client("sts").get_caller_identity()["Account"]
    out = []
    for b in boto3.client("s3").list_buckets()["Buckets"]:
        m = results.BUCKET_RE.match(b["Name"])
        if not m or m.group("acct") != acct:
            continue
        if getattr(args, "vla", None) and not m.group("stack").startswith(args.vla):
            continue
        if getattr(args, "region", None) and m.group("region") != args.region:
            continue
        out.append((b["Name"], m.group("stack"), m.group("region")))
    return sorted(out)


def read_run(s3, bucket, stack, region, run):
    """Every task of one run: the results.task_row fields plus the raw summary key/values."""
    objects = s3.objects(region, bucket, run + "/")
    view = RunView(s3, region, bucket, run, objects)
    tasks = []
    for t in view.prefixes(region, bucket, run + "/"):
        if not re.search(r"/task-\d+/$", t):
            continue
        r = results.task_row(view, region, bucket, run, t)
        summ = view.text(region, bucket, t + "summary.txt")
        r["kv"] = dict(results.KV_RE.findall(summ)) if summ else {}
        r["prefix"] = t
        tasks.append(r)
    return dict(bucket=bucket, stack=stack, region=region, run=run, objects=objects, view=view, tasks=tasks)


def collect_runs(args, boto3, s3=None, workers=16):
    s3 = s3 or S3(boto3, getattr(args, "region", None))
    jobs = []
    for bucket, stack, region in list_buckets(args, boto3):
        try:
            runs = sorted(p.rstrip("/") for p in s3.prefixes(region, bucket, "") if results.RUN_RE.match(p.rstrip("/")))
        except Exception as e:
            print(f"# skip {stack}@{region}: {e.__class__.__name__}", file=sys.stderr)
            continue
        if getattr(args, "since", None):
            runs = [r for r in runs if r >= args.since]
        jobs += [(bucket, stack, region, r) for r in runs]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(lambda j: read_run(s3, *j), jobs))
    return [r for r in out if r["tasks"]]


# --- groups and cells -------------------------------------------------------------------------------

def fingerprint(kvs):
    fp = {}
    for kv in kvs:
        fp.update({k: v.strip() for k, v in kv.items() if COMPARE_KEYS_RE.match(k) and k not in POLICY_STACK_KEYS})
    return fp


def run_groups(run, targets=None):
    """One group per (suite, condition) of a run, with pooled k/n and its row/column."""
    groups = collections.OrderedDict()
    for t in run["tasks"]:
        if t["status"] != "done":
            continue
        col = benchmark_label(t["suite"], t["condition"])
        g = groups.setdefault((col, t["condition"]), dict(k=0, n=0, rates=[], tasks=[], kvs=[], suites=[]))
        g["suites"].append(t["suite"])
        g["tasks"].append(t)
        g["kvs"].append(t["kv"])
        if t.get("n"):
            g["k"] += t["k"] or 0
            g["n"] += t["n"]
        else:
            g["rates"].append(t["rate"])
    out = []
    for (col, cond), g in groups.items():
        model = next((kv.get("model") for kv in g["kvs"] if kv.get("model")), None)
        suites = sorted(set(g["suites"]))
        suite = suites[0] if len(suites) == 1 else f"{suites[0].split('/', 1)[0]}/* ({len(suites)})"
        out.append(dict(g, suite=suite, condition=cond, run=run, target=target_of(run["stack"], model, targets),
                        column=col, baseline=is_baseline(cond, suites[0]), fp=fingerprint(g["kvs"])))
    return out


def shard_ids(group):
    """Task ids a group evaluated (`task_ids: [0,1]` / `task_id: 3`); empty when not recorded."""
    out = set()
    for kv in group["kvs"]:
        raw = (kv.get("task_ids") or kv.get("task_id") or "").strip()
        out |= {x.strip() for x in raw.strip("[]").split(",") if x.strip().isdigit()}
    return frozenset(out)


def checkpoints(group):
    return {kv.get("checkpoint", "").strip() for kv in group["kvs"]}


def select_cells(runs, all_conditions=False, condition=None, pool_hours=6.0, targets=None):
    """{(row, column): [group, ...]} — the latest group per cell plus its pooled shards."""
    cand = collections.defaultdict(list)
    for run in runs:
        for g in run_groups(run, targets):
            if condition is not None:
                if g["condition"] != condition:
                    continue
                row = g["target"]
            elif g["baseline"]:
                row = g["target"]
            elif all_conditions:
                row = f"{g['target']} [{g['condition']}]"
            else:
                continue
            cand[(row, g["column"])].append(g)
    cells = {}
    for key, gs in cand.items():
        gs.sort(key=lambda g: (g["run"]["run"], g["run"]["stack"], g["run"]["region"]), reverse=True)
        latest = gs[0]
        t0 = run_started(latest["run"]["run"])
        pooled, seen, ids = [latest], {latest["run"]["bucket"]}, set(shard_ids(latest))
        for g in gs[1:]:
            t = run_started(g["run"]["run"])
            gids = shard_ids(g)
            if (g["run"]["bucket"] in seen or g["condition"] != latest["condition"] or g["fp"] != latest["fp"]
                    or checkpoints(g) != checkpoints(latest) or bool(g["n"]) != bool(latest["n"])
                    or not ids or not gids or ids & gids
                    or not t or not t0 or abs((t0 - t).total_seconds()) > pool_hours * 3600):
                continue
            pooled.append(g)
            seen.add(g["run"]["bucket"])
            ids |= gids
        cells[key] = pooled
    return cells


def cell_stats(groups):
    k = sum(g["k"] for g in groups)
    n = sum(g["n"] for g in groups)
    if n:
        lo, hi = results.wilson(k, n)
        return dict(k=k, n=n, rate=k / n, lo=lo, hi=hi)
    rates = [r for g in groups for r in g["rates"] if r is not None]
    return dict(k=None, n=None, rate=sum(rates) / len(rates) if rates else None, lo=None, hi=None)


def stack_differences(cells):
    """{column: {key: {row: value}}} for compare keys recorded in >= 2 cells of a column with
    different values."""
    by_col = collections.defaultdict(dict)
    for (row, col), groups in cells.items():
        by_col[col][row] = groups[0]["fp"]
    out = {}
    for col, fps in by_col.items():
        keys = {k for fp in fps.values() for k in fp}
        diff = {}
        for k in sorted(keys):
            vals = {row: fp[k] for row, fp in fps.items() if k in fp}
            if len(vals) >= 2 and len(set(vals.values())) > 1:
                diff[k] = vals
        if diff:
            out[col] = diff
    return out


# --- cost and latency -------------------------------------------------------------------------------

def run_instance_type(run):
    view = run["view"]
    for t in run["tasks"]:
        kv = t["kv"]
        it = (kv.get("instance_type") or "").strip() or instance_type_in(kv.get("instance"))
        if it:
            return it, "summary"
    for o in run["objects"]:
        if "/videos/" in o["Key"] and o["Key"].endswith(".txt"):
            it = instance_type_in(view.text(run["region"], run["bucket"], o["Key"]))
            if it:
                return it, "video caption"
            break
    for o in run["objects"]:
        if o["Key"].endswith("userdata.log") and o["Key"].count("/") <= 2:
            text = view.text(run["region"], run["bucket"], o["Key"], head=65536)
            m = re.search(r"Instance: \S+ \(([^)]+)\)", text or "")
            it = instance_type_in(m.group(1)) if m else None
            if it:
                return it, "userdata.log"
    return None, None


def run_wall(run):
    explicit = [results._float(t["kv"].get("run_wall_s")) for t in run["tasks"]]
    explicit = [x for x in explicit if x]
    if explicit:
        return max(explicit), "run_wall_s"
    span = upload_span_s(run["run"], run["objects"])
    return (span, "upload span") if span else (None, None)


def task_episodes(task):
    """Episodes of a task: n from results.task_row, else the episode-count fields some templates
    write without n_success (num_episodes; per-task count x number of task ids / n_tasks)."""
    if task.get("n"):
        return task["n"]
    kv = task["kv"]
    n = results._int(kv.get("num_episodes"))
    if n:
        return n
    per = next((results._int(kv.get(k)) for k in ("n_episodes_per_task", "num_episodes_per_task",
                                                   "num_trials_per_task") if results._int(kv.get(k))), None)
    ids = re.fullmatch(r"\[([\d,\s]+)\]", (kv.get("task_ids") or "").strip())
    tasks = len(ids.group(1).split(",")) if ids else results._int(kv.get("n_tasks"))
    return per * tasks if per and tasks else None


def run_cost(run):
    """Cached per run: wall time, instance type, $/h, $ and total episodes of the run."""
    if "cost" not in run:
        wall, wall_src = run_wall(run)
        itype, it_src = run_instance_type(run)
        price = price_per_hour(itype, run["region"])
        usd = wall / 3600 * price if wall and price else None
        n_eps = sum(task_episodes(t) or 0 for t in run["tasks"] if t["status"] == "done")
        run["cost"] = dict(wall_s=wall, wall_src=wall_src, instance_type=itype, instance_src=it_src,
                           usd_per_h=price, usd=usd, n_episodes=n_eps)
    return run["cost"]


def task_timing(run, task):
    """(episode_wall_s, policy_ms_per_step, source) for one task, from the best signal present."""
    kv, view, keys = task["kv"], run["view"], {o["Key"] for o in run["objects"]}
    ep = results._float(kv.get("episode_wall_s"))
    ms = results._float(kv.get("policy_ms_per_step"))
    src = "summary" if ep or ms else None
    if ep is None and task["prefix"] + "eval_info.json" in keys:
        try:
            info = json.loads(view.text(run["region"], run["bucket"], task["prefix"] + "eval_info.json") or "{}")
            ep = results._float(info.get("overall", {}).get("eval_ep_s"))
            src = src or ("eval_info.json" if ep else None)
        except ValueError:
            pass
    if ep is None and task["prefix"] + "rollout.log" in keys:
        hits = EPISODES_TOOK_RE.findall(view.text(run["region"], run["bucket"], task["prefix"] + "rollout.log") or "")
        n = sum(int(a) for a, _ in hits)
        if n:
            ep = sum(float(b) for _, b in hits) / n
            src = src or "rollout.log"
    return ep, ms, src


def cell_cost(groups):
    usd, eps, walls, types, srcs = 0.0, 0, [], [], set()
    complete = True
    for g in groups:
        c = run_cost(g["run"])
        types.append(c["instance_type"] or "?")
        if c["wall_s"]:
            walls.append(c["wall_s"])
            srcs.add(c["wall_src"])
        if c["usd"] is None or not c["n_episodes"]:
            complete = False
            continue
        usd += c["usd"]
        eps += c["n_episodes"]
    ep_w, ms_w, n_ep, n_ms, tsrc = 0.0, 0.0, 0, 0, set()
    for g in groups:
        for t in g["tasks"]:
            ep, ms, src = task_timing(g["run"], t)
            w = t.get("n") or 1
            if ep:
                ep_w, n_ep = ep_w + ep * w, n_ep + w
            if ms:
                ms_w, n_ms = ms_w + ms * w, n_ms + w
            if src:
                tsrc.add(src)
    return dict(instance_type="/".join(sorted(set(types))), runs=len(groups),
                wall_s=sum(walls) if walls else None, wall_src="+".join(sorted(srcs)) or None,
                usd_per_run=(usd / len(groups)) if complete and groups else None,
                usd_per_episode=(usd / eps) if complete and eps else None,
                episode_wall_s=(ep_w / n_ep) if n_ep else None, policy_ms_per_step=(ms_w / n_ms) if n_ms else None,
                timing_src="+".join(sorted(tsrc)) or None)


# --- rendering --------------------------------------------------------------------------------------

def fmt_cell(st, marks=""):
    if st["rate"] is None:
        return "-"
    if st["n"]:
        return f"{st['rate']:.2f} ({st['k']}/{st['n']}) [{st['lo']:.2f}-{st['hi']:.2f}]{marks}"
    return f"{st['rate']:.2f} (rate only){marks}"


def _f(v, fmt):
    return "-" if v is None else format(v, fmt)


def build(cells, with_cost=True):
    """Plain data for every cell: what the markdown, CSV and JSON outputs are made of."""
    diffs = stack_differences(cells)
    rows = sorted({r for r, _ in cells})
    cols = sorted({c for _, c in cells}, key=column_sort_key)
    out = []
    for (row, col), groups in sorted(cells.items(), key=lambda kv: (kv[0][0], column_sort_key(kv[0][1]))):
        st = cell_stats(groups)
        stacks_differ = sorted(k for k, vals in diffs.get(col, {}).items() if row in vals)
        rec = dict(target=row, benchmark=col, condition=groups[0]["condition"], suite=groups[0]["suite"],
                   k=st["k"], n=st["n"], rate=st["rate"], ci95_lo=st["lo"], ci95_hi=st["hi"],
                   stacks_differ=stacks_differ,
                   no_fingerprint=not groups[0]["fp"] or groups[0]["condition"].strip() in ("", "-"),
                   sources=[f"{g['run']['stack']}@{g['run']['region']} {g['run']['run']}" for g in groups],
                   fingerprint=groups[0]["fp"])
        if with_cost:
            rec.update(cell_cost(groups))
        out.append(rec)
    return dict(rows=rows, columns=cols, cells=out, stack_differences=diffs)


def render_markdown(data, with_cost=True, generated=None):
    rows, cols = data["rows"], data["columns"]
    by = {(c["target"], c["benchmark"]): c for c in data["cells"]}
    lines = []
    if generated:
        lines.append(f"_Generated {generated} by `python vlasim.py results --write-docs` from the results buckets "
                     f"in this account; estimates are labelled as such._")
        lines.append("")
    lines.append("| Target | " + " | ".join(cols) + " |")
    lines.append("|---|" + "---|" * len(cols))
    for r in rows:
        cells = []
        for c in cols:
            rec = by.get((r, c))
            if not rec:
                cells.append("")
                continue
            marks = ("†" if rec["stacks_differ"] else "") + ("‡" if rec["no_fingerprint"] else "")
            cells.append(fmt_cell(dict(rate=rec["rate"], k=rec["k"], n=rec["n"], lo=rec["ci95_lo"],
                                       hi=rec["ci95_hi"]), marks))
        lines.append(f"| `{r}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Cell = latest run under a baseline condition: rate (k/n) [95% Wilson interval]. "
                 "† = a stack fingerprint key differs from another cell in the same column (see below); "
                 "‡ = no stack fingerprint or no condition recorded (older summary; comparability unknown). Compare cells only when episode count, init states, "
                 "step cap and chunk execution match.")
    if data["stack_differences"]:
        lines.append("")
        lines.append("Stack differences per column:")
        lines.append("")
        for col, diff in data["stack_differences"].items():
            for k, vals in diff.items():
                v = ", ".join(f"`{row}`={val}" for row, val in sorted(vals.items()))
                lines.append(f"- {col}: `{k}`: {v}")
    if with_cost:
        lines.append("")
        lines.append(f"Cost and time per cell (**estimates**: EC2 On-Demand only, prices pinned {PRICES_CHECKED} "
                     "from the AWS Price List API):")
        lines.append("")
        lines.append("| Target | Benchmark | Runs | Instance | Instance-min (est.) | USD/run (est.) | USD/episode (est.) "
                     "| s/episode | ms/step |")
        lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|")
        for rec in data["cells"]:
            wall = None if rec["wall_s"] is None else rec["wall_s"] / 60
            lines.append(f"| `{rec['target']}` | {rec['benchmark']} | {rec['runs']} | {rec['instance_type']} "
                         f"| {_f(wall, '.0f')} | {_f(rec['usd_per_run'], '.2f')} | {_f(rec['usd_per_episode'], '.3f')} "
                         f"| {_f(rec['episode_wall_s'], '.1f')} | {_f(rec['policy_ms_per_step'], '.0f')} |")
        lines.append("")
        lines.append("Instance-min = sum over the pooled runs of RUN_ID (instance boot) to the last object the run "
                     "uploaded, or `run_wall_s` from `summary.txt`; USD/episode divides by every episode of those runs "
                     "(all suites of a run share its instance); stack create/delete, NAT gateway, EBS and data transfer are not included. "
                     "USD/run is per pooled run. s/episode comes from `episode_wall_s`, LeRobot `eval_info.json` "
                     "(`eval_ep_s`) or the GR00T rollout log; ms/step only from `policy_ms_per_step`.")
    lines.append("")
    lines.append("Sources: " + "; ".join(f"`{c['target']}` {c['benchmark']}: {', '.join(c['sources'])}"
                                          for c in data["cells"]) + ".")
    return "\n".join(lines) + "\n"


CSV_COLS = ["target", "benchmark", "condition", "suite", "k", "n", "rate", "ci95_lo", "ci95_hi", "stacks_differ",
            "no_fingerprint", "runs", "instance_type", "wall_s", "wall_src", "usd_per_run", "usd_per_episode",
            "episode_wall_s", "policy_ms_per_step", "timing_src", "sources"]


def write_csv(path, data):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for rec in data["cells"]:
            row = dict(rec)
            row["stacks_differ"] = " ".join(rec["stacks_differ"])
            row["sources"] = "; ".join(rec["sources"])
            w.writerow(row)


DOCS_BEGIN = "<!-- leaderboard:begin -->"
DOCS_END = "<!-- leaderboard:end -->"


def replace_block(text, block):
    """Replace what is between the leaderboard markers; raise ValueError when they are missing."""
    i, j = text.find(DOCS_BEGIN), text.find(DOCS_END)
    if i < 0 or j < i:
        raise ValueError(f"markers {DOCS_BEGIN} / {DOCS_END} not found")
    return text[:i + len(DOCS_BEGIN)] + "\n" + block.rstrip("\n") + "\n" + text[j:]


def run(args, boto3):
    runs = collect_runs(args, boto3)
    cells = select_cells(runs, all_conditions=args.all_conditions, condition=args.condition,
                         pool_hours=args.pool_hours)
    if not cells:
        print("No comparable results found (no done task under a baseline condition).")
        return 0
    with_cost = not args.no_cost
    data = build(cells, with_cost=with_cost)
    md = render_markdown(data, with_cost=with_cost)
    print(md, end="")
    if args.csv:
        write_csv(args.csv, data)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(data, prices_checked=PRICES_CHECKED), f, indent=1, default=str)
    if args.write_docs:
        path = Path(args.write_docs)
        block = render_markdown(data, with_cost=with_cost, generated=dt.date.today().isoformat())
        try:
            path.write_text(replace_block(path.read_text(), block))
        except (OSError, ValueError) as e:
            print(f"vlasim results --write-docs: {path}: {e}", file=sys.stderr)
            return 2
        print(f"\nwrote the leaderboard block in {path}", file=sys.stderr)
    return 0
