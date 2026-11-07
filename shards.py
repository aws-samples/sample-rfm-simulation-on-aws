"""Shard plan for ``vlasim deploy <target> --shards N``: split, names, merge and cost (docs/shards.md).

One benchmark is split into N shards; every shard is its own stack (its own instance and results bucket),
so N shards of the same target can run at the same time in one region. ``fanout.py`` schedules them.

Split rule (the RoboTwin ``--robotwin-shard K/N`` rule, generalised): the target's task units are put in a
fixed order and shard K of N (1-based) takes every N-th unit starting at unit K, ``units[K-1::N]``.
The shards are disjoint, their union is the full list, and the same (list, N) always gives the same shards.

* RoboLab (robolab-*): the unit is a task class. The order is the protocol's ``tasks:`` list, or for the
  full protocol (``tasks: []``) the 120 names of ``assets/robolab/robolab-120-tasks.txt`` (code-point sorted,
  pinned to ``model.robolab_commit``). Every shard runs all episodes of its tasks.
* Targets whose ``tasks:`` is a list of entries (LIBERO families, SimplerEnv, ...): the unit is one task id of
  an entry (``task_ids: [0, 1]`` -> two units) or the entry itself when it has no ``task_ids``. Shard K gets a
  tasks file (the ``--tasks-file`` mechanism of ``vlasim bench``) with its units regrouped per entry.
* RoboTwin targets: the unit is a (setting, task) pair; shard K/N is ``--robotwin-shard K/N``.

Every other target has no enumerable task list and ``split_plan`` raises ``ValueError``.

Stdlib-only and side-effect free on import (the tests and ``--dry-run`` import it without AWS deps).
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROBOLAB_120_FILE = BASE_DIR / "assets" / "robolab" / "robolab-120-tasks.txt"
ROBOLAB_TARGETS = ("robolab-cosmos3", "robolab-pi05")
ROBOTWIN_TARGETS = ("smolvla-robotwin", "lingbot-vla2")

TOKEN_RE = re.compile(r"^(?:r(?P<run>\d{1,2})-)?s(?P<k>\d{2,3})of(?P<n>\d{2,3})$")
MAX_SHARDS = 256
MAX_RUNS = 20
BUCKET_MAX = 63
ACCOUNT_LEN = 12  # AWS account ids are 12 digits; the CDK side assumes the same when the account is a token


# --- shard ids, stack and bucket names ----------------------------------------------------------------

def parse_shard(value: str) -> tuple[int, int]:
    """'K/N' (1-based) -> (K, N). Raises ValueError."""
    try:
        k, n = (int(x) for x in str(value).split("/"))
    except ValueError:
        raise ValueError(f"--shard must look like K/N, got {value!r}") from None
    if not 1 <= n <= MAX_SHARDS or not 1 <= k <= n:
        raise ValueError(f"--shard {value}: need 1 <= K <= N <= {MAX_SHARDS}")
    return k, n


def shard_token(k: int, n: int, run: int | None = None) -> str:
    """Stack-name suffix of shard k of n (and run r of a --runs expansion): s03of16, r2-s03of16."""
    if not 1 <= n <= MAX_SHARDS or not 1 <= k <= n:
        raise ValueError(f"shard {k}/{n}: need 1 <= K <= N <= {MAX_SHARDS}")
    w = max(2, len(str(n)))
    tok = f"s{k:0{w}d}of{n:0{w}d}"
    if run is not None:
        if not 1 <= run <= MAX_RUNS:
            raise ValueError(f"run {run}: need 1 <= R <= {MAX_RUNS}")
        tok = f"r{run}-{tok}"
    return tok


def parse_token(token: str) -> tuple[int | None, int, int]:
    """'r2-s03of16' -> (2, 3, 16); 's03of16' -> (None, 3, 16). Raises ValueError."""
    m = TOKEN_RE.match(str(token or ""))
    if not m:
        raise ValueError(f"not a shard token: {token!r} (expected s03of16 or r2-s03of16)")
    run = int(m.group("run")) if m.group("run") else None
    return run, int(m.group("k")), int(m.group("n"))


def stack_name(base: str, token: str) -> str:
    """Stack of one shard: <base stack>-<token>, e.g. RoboLab-Pi05-Demo-s03of16. Same rule as cdk/bin/app.ts."""
    return f"{base}-{token}" if token else base


def bucket_name(prefix: str, stack: str, region: str, account: str, token: str = "") -> str:
    """Results bucket of a stack. Same rule as resultsBucketName() in cdk/lib/vla-simulator-stack.ts.

    No token: <prefix>-<stack>-<region>-<account>, dropping "-demo" only when that overflows 63 chars
    (the pre-shard rule; existing buckets keep their names). Shard stack: <prefix>-<base>-<token>-<region>-
    <account> with base = the stack name without "-<token>" and "-demo", cut so the name fits 63 chars."""
    stack_l = stack.lower()
    if not token:
        name = f"{prefix}-{stack_l}-{region}-{account}"
        if len(name) > BUCKET_MAX:
            name = f"{prefix}-{stack_l.removesuffix('-demo')}-{region}-{account}"
        return name
    base = stack_l.removesuffix(f"-{token}").removesuffix("-demo")
    room = BUCKET_MAX - len(prefix) - len(token) - len(region) - ACCOUNT_LEN - 4
    if room < 3:
        raise ValueError(f"results bucket of {stack} in {region} cannot fit 63 chars; use a shorter "
                         "deployment.s3_results_prefix")
    base = base[:room].rstrip("-")
    return f"{prefix}-{base}-{token}-{region}-{account}"


# --- split -------------------------------------------------------------------------------------------

def split(items: list, n: int) -> list[list]:
    """Round-robin split: shard k (1-based) = items[k-1::n]. Disjoint, union = items, stable."""
    if n < 1:
        raise ValueError("need N >= 1")
    if len(items) < n:
        raise ValueError(f"{len(items)} task unit(s) cannot be split into {n} shards (at most {len(items)})")
    return [list(items[k::n]) for k in range(n)]


def robolab_120() -> tuple[str, list[str]]:
    """(commit, 120 task names) from assets/robolab/robolab-120-tasks.txt."""
    commit, names = "", []
    for line in ROBOLAB_120_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("commit:"):
            commit = line.split(":", 1)[1].strip()
            continue
        names.append(line)
    return commit, names


def robolab_tasks(config: dict, vla: str = "") -> tuple[list[str], int]:
    """(ordered task list, episodes per task) of the selected RoboLab variant/protocol. Raises ValueError."""
    model = config.get("model") or {}
    variants = config.get("variants") or {}
    vname = str(model.get("variant", "")).strip()
    if vname not in variants:
        raise ValueError(f"models/{vla}.yaml: model.variant {vname!r} is not one of {sorted(variants)}")
    var = variants[vname] or {}
    proto = (config.get("protocols") or {}).get(var.get("protocol")) or {}
    if not proto:
        raise ValueError(f"models/{vla}.yaml: variants.{vname}.protocol names no protocol")
    tasks = [str(t) for t in proto.get("tasks") or []]
    if not tasks:
        commit, tasks = robolab_120()
        if commit != str(model.get("robolab_commit", "")):
            raise ValueError(f"{ROBOLAB_120_FILE.name} lists the tasks of RoboLab @{commit[:8]} but models/{vla}.yaml "
                             f"pins robolab_commit {str(model.get('robolab_commit', ''))[:8]}; regenerate the file "
                             "(command in its header) before sharding the full protocol")
    per_task = int(proto.get("num_envs", 1)) * int(proto.get("num_runs", 1))
    return tasks, per_task


def robolab_shard(config: dict, vla: str, k: int, n: int) -> dict:
    """{tasks, task_ids, episodes_per_task}: shard k of n of the RoboLab protocol. task_ids are positions in the
    full ordered list (leaderboard.py pools shards with disjoint task_ids)."""
    tasks, per_task = robolab_tasks(config, vla)
    idx = split(list(range(len(tasks))), n)[k - 1]
    return dict(tasks=[tasks[i] for i in idx], task_ids=idx, episodes_per_task=per_task, n_tasks_total=len(tasks))


def _list_units(tasks: list) -> list[tuple[int, int | None]]:
    units = []
    for i, entry in enumerate(tasks):
        ids = entry.get("task_ids") if isinstance(entry, dict) else None
        if isinstance(ids, list) and ids:
            units += [(i, int(t)) for t in ids]
        else:
            units.append((i, None))
    return units


def tasks_file_shard(tasks: list, k: int, n: int) -> list[dict]:
    """Shard k of n of a `tasks:` list of entries, regrouped per entry (task_ids kept in their order)."""
    picked = split(_list_units(tasks), n)[k - 1]
    out, by_entry = [], {}
    for i, tid in picked:
        if tid is None:
            out.append(copy.deepcopy(tasks[i]))
            continue
        if i not in by_entry:
            by_entry[i] = dict(copy.deepcopy(tasks[i]), task_ids=[])
            out.append(by_entry[i])
        by_entry[i]["task_ids"].append(tid)
    return out


def kind(vla: str, config: dict) -> str:
    """How a target is split: robolab | robotwin | tasks-file. Raises ValueError when it cannot be split."""
    if vla in ROBOLAB_TARGETS:
        return "robolab"
    if vla in ROBOTWIN_TARGETS:
        return "robotwin"
    tasks = config.get("tasks")
    if isinstance(tasks, list) and tasks and all(isinstance(t, dict) for t in tasks):
        return "tasks-file"
    raise ValueError(f"{vla} cannot be sharded: its models/{vla}.yaml tasks: is not a list of task entries "
                     "(nothing to enumerate). Run it as one deploy.")


def split_plan(vla: str, config: dict, n: int, robotwin_protocol: str = "subset") -> list[dict]:
    """[{k, n, units, n_units, label, tasks|tasks_file|robotwin_shard}] for shards 1..n. Raises ValueError."""
    how = kind(vla, config)
    out = []
    if how == "robolab":
        tasks, per_task = robolab_tasks(config, vla)
        parts = split(tasks, n)
        for k, part in enumerate(parts, 1):
            out.append(dict(k=k, n=n, kind=how, units=part, n_units=len(part), episodes=len(part) * per_task))
    elif how == "robotwin":
        import robotwin_plan  # noqa: PLC0415 - stdlib-only module

        full = robotwin_plan.plan(config.get("tasks") or [], protocol=robotwin_protocol)
        split([f"{u['setting']}/{u['task']}" for u in full], n)  # raises when N > units
        for k in range(1, n + 1):
            units = robotwin_plan.plan(config.get("tasks") or [], protocol=robotwin_protocol, shard=f"{k}/{n}")
            out.append(dict(k=k, n=n, kind=how, units=[f"{u['setting']}/{u['task']}" for u in units],
                            n_units=len(units), episodes=robotwin_plan.episodes(units), robotwin_shard=f"{k}/{n}"))
    else:
        tasks = config["tasks"]
        for k in range(1, n + 1):
            part = tasks_file_shard(tasks, k, n)
            units = [f"{_entry_name(e)}#{t}" for e in part for t in (e.get("task_ids") or [None])]
            eps = sum(_entry_episodes(e) for e in part)
            out.append(dict(k=k, n=n, kind=how, units=units, n_units=len(units), episodes=eps or None,
                            tasks=part))
    return out


def _entry_name(e: dict) -> str:
    for key in ("suite", "env_name", "task", "task_name", "instruction"):
        if e.get(key):
            return str(e[key])
    return "task"


def _entry_episodes(e: dict) -> int:
    per = next((int(e[k]) for k in ("n_episodes", "num_episodes", "num_trials_per_task", "episodes") if e.get(k)), 0)
    return per * max(1, len(e.get("task_ids") or []))


# --- failure classification ---------------------------------------------------------------------------

CAPACITY_RE = re.compile(r"InsufficientInstanceCapacity|No capacity available for any instance type|"
                         r"MaxSpotInstanceCountExceeded|SpotMaxPriceTooLow|InsufficientCapacity|"
                         r"Unsupported: |not available in any VPC subnet AZ")
QUOTA_RE = re.compile(r"VcpuLimitExceeded|InstanceLimitExceeded|You have requested more vCPU capacity|"
                      r"PendingVerification|LimitExceeded")


def classify_failure(log_text: str) -> str:
    """capacity | quota | other, from the deploy log (cdk prints the CloudFormation reason of the failed
    AzSelector resource: 'No capacity available for any instance type' or the RunInstances error)."""
    text = log_text or ""
    if QUOTA_RE.search(text):
        return "quota"
    if CAPACITY_RE.search(text):
        return "capacity"
    return "other"


# --- cost ----------------------------------------------------------------------------------------------

STATE_REASON_TIME_RE = re.compile(r"\((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) GMT\)")


def parse_time(value) -> dt.datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    s = str(value).replace("Z", "+00:00")
    try:
        t = dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def termination_time(state_transition_reason: str) -> dt.datetime | None:
    """EC2 StateTransitionReason 'User initiated (2026-11-07 16:43:50 GMT)' -> datetime."""
    m = STATE_REASON_TIME_RE.search(state_transition_reason or "")
    if not m:
        return None
    return dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)


def instance_hours(launch, end) -> float | None:
    a, b = parse_time(launch), parse_time(end)
    if not a or not b or b < a:
        return None
    return (b - a).total_seconds() / 3600.0


def spot_cost(points: list, launch, end) -> tuple[float, float] | None:
    """(usd, effective usd/h) of one instance from EC2 Spot price history points [(time, usd/h)]: the price in
    effect at launch (the last change at or before it), then every change, each interval clipped to
    [launch, end] and weighted by its duration. None without a point at or before the end."""
    t0, t1 = parse_time(launch), parse_time(end)
    if not t0 or not t1 or t1 <= t0:
        return None
    pts = sorted((parse_time(t), float(p)) for t, p in points if parse_time(t))
    before = [p for p in pts if p[0] <= t0]
    inside = [p for p in pts if t0 < p[0] < t1]
    if before:
        seq = [(t0, before[-1][1])] + inside
    elif inside:
        seq = [(t0, inside[0][1])] + inside  # no change before launch in the window: first known price
    else:
        return None
    usd = 0.0
    for (ta, pa), nxt in zip(seq, seq[1:] + [(t1, None)]):
        usd += pa * (nxt[0] - ta).total_seconds() / 3600.0
    hours = (t1 - t0).total_seconds() / 3600.0
    return usd, usd / hours


def validate_results(kind: str, path) -> bool:
    """The downloaded results of one shard are usable for the merge: RoboLab = summary.txt + at least one raw
    episode_results.jsonl; other targets = at least one task-N/summary.txt."""
    d = Path(path)
    if not d.is_dir():
        return False
    if kind == "robolab":
        return (d / "summary.txt").is_file() and any(p.stat().st_size for p in d.glob("raw/*/episode_results.jsonl"))
    return any(d.glob("task-*/summary.txt"))


def unit_cost(instances: list[dict], price_of) -> dict:
    """Actual cost of one shard attempt set: sum over the instances it launched of (end - launch) x $/h.

    instances: [{instance_type, region, launch, end, market}] (end = termination time, or when the run was
    seen finished). price_of(instance_type, region, market) -> (usd_per_h, source) or (None, reason)."""
    total_h, usd, rows, unknown = 0.0, 0.0, [], []
    for inst in instances:
        h = instance_hours(inst.get("launch"), inst.get("end"))
        price, src = price_of(inst.get("instance_type"), inst.get("region"), inst.get("market", "on-demand"),
                              inst.get("az", ""), str(inst.get("launch") or ""), str(inst.get("end") or ""))
        row = dict(inst, instance_hours=round(h, 4) if h is not None else None, usd_per_h=price, price_source=src)
        row["launch"], row["end"] = str(inst.get("launch") or ""), str(inst.get("end") or "")
        if h is None or price is None:
            unknown.append(inst.get("instance_id") or "?")
        else:
            total_h += h
            usd += h * price
            row["usd"] = round(h * price, 4)
        rows.append(row)
    return dict(instance_hours=round(total_h, 4), usd=round(usd, 4), instances=rows, unpriced=unknown)


# --- merge ---------------------------------------------------------------------------------------------

KV_RE = re.compile(r"^([A-Za-z_][\w]*): ?(.*)$", re.M)
# summary.txt keys that describe one shard's instance, not the merged run
SHARD_LOCAL_KEYS = ("shards", "instance", "fanout_shard", "task_ids", "description", "n_episodes", "n_success", "success_rate",
                    "n_expected", "tasks_complete", "suite")


def read_kv(path: Path) -> dict:
    return {k: v.strip() for k, v in KV_RE.findall(Path(path).read_text())} if Path(path).is_file() else {}


def _load_report_module():
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("robolab_report", BASE_DIR / "assets" / "robolab" / "robolab_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def merge_robolab(shard_dirs: list[Path], out_dir: Path, tasks: list[str], episodes_per_task: int,
                  extra_meta: dict | None = None) -> dict:
    """Merge RoboLab shard results into the summary.txt + robolab_results.json one single run writes.

    shard_dirs: one per shard, each the downloaded <RUN_ID>/task-0/ (summary.txt + raw/shard-*/
    episode_results.jsonl). Runs assets/robolab/robolab_report.py, the script the instance runs, over all
    raw episode files, so the merged numbers come from the same code as a one-instance run."""
    report = _load_report_module()
    run_dirs, meta = [], {}
    for d in shard_dirs:
        d = Path(d)
        for raw in sorted((d / "raw").glob("*/episode_results.jsonl")):
            run_dirs.append(str(raw.parent))
        for k, v in read_kv(d / "summary.txt").items():
            if k not in SHARD_LOCAL_KEYS:
                meta.setdefault(k, v)
    meta.update({k: str(v) for k, v in (extra_meta or {}).items()})
    label = read_kv(Path(shard_dirs[0]) / "summary.txt").get("suite", "RoboLab") if shard_dirs else "RoboLab"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = [x for d in run_dirs for x in ("--run-dir", d)]
    argv += ["--upload-dir", str(out_dir), "--label", label, "--tasks", " ".join(tasks),
             "--episodes-per-task", str(episodes_per_task), "--max-videos-per-task", "0"]
    argv += [x for k, v in meta.items() for x in ("--meta", f"{k}={v}")]
    import contextlib  # noqa: PLC0415
    import io  # noqa: PLC0415

    with contextlib.redirect_stdout(io.StringIO()):
        report.main(argv)
    return json.loads((out_dir / "robolab_results.json").read_text())


def merge_tasks(shard_dirs: list[Path], out_dir: Path, extra_meta: dict | None = None) -> dict:
    """Merge the task-N/summary.txt of tasks-file / RoboTwin shards: every shard task becomes task-<i> of the
    merged run (copied verbatim) and summary.json pools k/n per suite. Same numbers results.py reads."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, i = [], 0
    for d in shard_dirs:
        for s in sorted(Path(d).glob("task-*/summary.txt"), key=lambda p: int(p.parent.name.split("-")[1])):
            kv = read_kv(s)
            dest = out_dir / f"task-{i}"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "summary.txt").write_text(s.read_text())
            n = _int(kv.get("n_episodes"))
            k = _int(kv.get("n_success"))
            rows.append(dict(task=i, shard=Path(d).name, suite=kv.get("suite") or kv.get("env_name") or "",
                             condition=kv.get("condition", ""), n=n, k=k))
            i += 1
    pooled = {}
    for r in rows:
        g = pooled.setdefault(r["suite"], dict(suite=r["suite"], n=0, k=0, tasks=0))
        g["tasks"] += 1
        if r["n"] is not None and r["k"] is not None:
            g["n"] += r["n"]
            g["k"] += r["k"]
    for g in pooled.values():
        g["success_rate"] = round(g["k"] / g["n"], 4) if g["n"] else None
    n_all = sum(g["n"] for g in pooled.values())
    k_all = sum(g["k"] for g in pooled.values())
    summary = dict(n_episodes=n_all, n_success=k_all, success_rate=round(k_all / n_all, 4) if n_all else None,
                   per_suite=sorted(pooled.values(), key=lambda g: g["suite"]), per_task=rows,
                   meta={k: str(v) for k, v in (extra_meta or {}).items()})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None
