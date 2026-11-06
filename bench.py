#!/usr/bin/env python3
"""
bench.py — self-run benchmark advisor: a few questions -> 2-4 RFMs -> the same actions on each -> one HTML report.

    python vlasim.py bench plan                          # 5 questions, pick actions, write bench-plan.yaml
    python vlasim.py bench plan --answers answers.yaml --yes
    python vlasim.py bench run --plan bench-plan.yaml --dry-run   # print the exact deploy commands, deploy nothing
    python vlasim.py bench run --plan bench-plan.yaml             # deploy every target in parallel
    python vlasim.py bench report --plan bench-plan.yaml          # one self-contained HTML (gallery if available)
    python vlasim.py bench destroy --plan bench-plan.yaml         # destroy.py --keep-data for every target

The catalog (bench/catalog.yaml) lists the targets that can run a shared benchmark, with license, GPU tier and
cost/time figures from the repo docs. `plan` filters it by the answers, keeps the 2-4 cheapest candidates that
cover one benchmark suite and lists the actions (LIBERO or RoboTwin tasks) valid for all of them. `run` writes one
--tasks-file per target (the target's own tasks: format: same task ids, episodes, seed and the condition label
bench-<id>) and calls deploy.py, so each target uses its existing CDK stack and results bucket. `report` reads back
only the task results with that condition (results.task_row, Wilson interval, leaderboard cost/wall time).

This is a self-run benchmark in the customer's account: the numbers depend on the stack and settings, and AWS
does not certify model performance. Stdlib-first; boto3 / PyYAML are imported where they are needed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import secrets
import shlex
import subprocess  # nosec B404 - hardcoded command lists, no shell
import sys
from pathlib import Path

import results
import task_sets

BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR / "bench" / "catalog.yaml"
PLAN_SCHEMA = "rfm-simulation/bench-plan/1"
DEFAULT_EPISODES = 10
MAX_CANDIDATES = 4
DISCLAIMER = ("Self-run benchmark in your account; results depend on the stack and settings; "
              "AWS does not certify model performance.")

# The questions of `bench plan`: key, prompt, [(answer value, label)]. --answers uses the same keys and values.
QUESTIONS = [
    ("robot", "What does your robot look like?", [
        ("single-arm", "One arm on a table (Franka Panda, LIBERO)"),
        ("bimanual", "Two arms working together (Aloha-AgileX, RoboTwin 2.0)"),
        ("humanoid", "Humanoid (GR1 / Unitree G1)")]),
    ("action", "What kind of action matters most?", [
        ("pick-place", "Pick an object and place it somewhere"),
        ("articulated", "Open / close / turn something: drawer, door, stove knob"),
        ("long-horizon", "Several steps in a row (long-horizon)"),
        ("spatial", "Pick the right object by its position (spatial reasoning)"),
        ("press", "Press a button or switch"),
        ("tool-use", "Use a tool on something (hammer, stamp)"),
        ("bimanual", "Hand an object over / lift it with two hands"),
        ("any", "No preference, show me everything")]),
    ("commercial", "Do you need a license that allows commercial use?", [
        ("yes", "Yes, only permissive (commercial-use) code and weights"),
        ("no", "No, research / evaluation is enough")]),
    ("gpu", "Which GPU size can the model use?", [
        ("24gb", "Small: one 24 GB GPU (L4 / A10G)"),
        ("48gb", "Medium: one 48 GB GPU (L40S)"),
        ("any", "Any")]),
    ("edge", "Should the model be small enough for an edge device?", [
        ("yes", "Yes (about 1B parameters or less)"),
        ("no", "No")]),
]
ANSWER_VALUES = {k: [v for v, _ in opts] for k, _, opts in QUESTIONS}

# Action groups (the `action` answer) -> how a task joins the group. LIBERO: by suite or by action_type tag.
GROUP_LABELS = {
    "pick-place": "Pick and place",
    "articulated": "Drawer / door / knob (articulated)",
    "long-horizon": "Long-horizon (several steps)",
    "spatial": "Spatial (which object, by layout)",
    "bimanual": "Two-handed (bimanual)",
    "tool-use": "Tool use",
    "press": "Press a button / switch",
}
LIBERO_SUITE_LABELS = {"libero_spatial": "LIBERO-Spatial", "libero_object": "LIBERO-Object",
                       "libero_goal": "LIBERO-Goal", "libero_10": "LIBERO-10 (Long)"}
GPU_RANK = {"24gb": 0, "48gb": 1}


# ───────────────────────────── catalog ─────────────────────────────

def load_catalog(path: Path = CATALOG_PATH) -> dict:
    import yaml  # noqa: PLC0415
    cat = yaml.safe_load(Path(path).read_text())
    if not isinstance(cat, dict) or not isinstance(cat.get("targets"), dict):
        raise ValueError(f"{path}: needs a targets: mapping")
    return cat


def libero_tasks() -> dict:
    """LIBERO task names per suite (generate.LIBERO_TASKS, task order = task id)."""
    from generate import LIBERO_TASKS  # noqa: PLC0415 - generate needs jinja2 / PyYAML
    return LIBERO_TASKS


def instruction(name: str) -> str:
    """LIBERO language instruction = the task name without the scene prefix, underscores as spaces."""
    bare = re.sub(r"^[A-Z0-9_]*SCENE\d+_", "", name)
    return bare.replace("_", " ")


def all_actions(benchmark: str) -> list:
    """Every action of a benchmark: {id, suite, task_id | task, env_name, instruction, groups}."""
    out = []
    if benchmark == "libero":
        for suite, names in libero_tasks().items():
            for i, name in enumerate(names):
                tag = task_sets.classify(name)
                groups = []
                if suite == "libero_spatial":
                    groups.append("spatial")
                if suite == "libero_10":
                    groups.append("long-horizon")
                if tag == "articulated":
                    groups.append("articulated")
                if tag == "pick-place" and suite in ("libero_object", "libero_goal"):
                    groups.append("pick-place")
                out.append(dict(id=f"{suite}:{i}", suite=suite, task_id=i, env_name=name,
                                instruction=instruction(name), action_type=tag, groups=groups))
    elif benchmark == "robotwin":
        import robotwin_plan  # noqa: PLC0415
        for name in robotwin_plan.ALL_TASKS:
            tag = task_sets.classify(name)
            groups = [{"pick-place": "pick-place", "articulated": "articulated", "bimanual": "bimanual",
                       "tool-use": "tool-use", "press": "press"}.get(tag, "other")]
            out.append(dict(id=f"clean:{name}", suite="clean", task=name, env_name=name,
                            instruction=name.replace("_", " ") + " (RoboTwin task name)", action_type=tag,
                            groups=groups))
    return out


# ───────────────────────────── estimate ─────────────────────────────

def _timeout_minutes(value) -> int | None:
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?", str(value or ""))
    if not m or not any(m.groups()):
        return None
    return int(m.group(1) or 0) * 60 + int(m.group(2) or 0)


def estimate(entry: dict, n_episodes: int, region: str) -> dict:
    """Minutes and USD for one target running n_episodes, from the catalog figures (an estimate, EC2 only).

    With sec_per_episode: the default run's upper time plus (n - default episodes) x s/episode, not below the
    default run's lower time (setup does not shrink). Without: the default run's upper time scaled by
    n / default episodes (an upper bound: setup is scaled too)."""
    import leaderboard  # noqa: PLC0415 - pinned On-Demand price table per region

    d = entry.get("default_run") or {}
    lo, hi = (d.get("minutes") or [None, None])[:2]
    d_ep = d.get("episodes")
    spe = (entry.get("sec_per_episode") or {}).get("value")
    minutes, basis = None, "unknown"
    if hi and d_ep:
        if spe:
            minutes = max(lo or 0, hi + (n_episodes - d_ep) * spe / 60.0)
            basis = f"default run {d_ep} ep {lo}-{hi} min + {spe} s/episode"
        else:
            minutes = hi * max(1.0, n_episodes / d_ep)
            basis = f"default run {d_ep} ep {hi} min, scaled by episodes (upper bound)"
    price = leaderboard.price_per_hour(entry.get("instance"), region)
    usd = round(minutes / 60.0 * price, 2) if minutes is not None and price else None
    return dict(instance=entry.get("instance"), usd_per_hour=price, minutes=None if minutes is None else round(minutes),
                usd=usd, basis=basis, prices_checked=leaderboard.PRICES_CHECKED)


# ───────────────────────────── selection ─────────────────────────────

def validate_answers(answers: dict) -> dict:
    out = {}
    for key, values in ANSWER_VALUES.items():
        raw = (answers or {}).get(key, "")
        v = {True: "yes", False: "no"}[raw] if isinstance(raw, bool) else str(raw).strip().lower()  # YAML yes/no
        if v not in values:
            raise ValueError(f"answer {key!r} must be one of {', '.join(values)}, got {answers.get(key)!r}")
        out[key] = v
    return out


def _license_reason(e: dict) -> str | None:
    if e.get("commercial_ok") is True:
        return None
    if e.get("commercial_ok") is False:
        return f"license not for commercial use ({e.get('license')})"
    return f"commercial use not confirmed ({e.get('license')}); review the license"


def select(catalog: dict, answers: dict, region: str = "us-east-1", episodes: int = DEFAULT_EPISODES,
           only: list | None = None) -> dict:
    """Candidates for the answers: {candidates: [{target, reason}], excluded: [{target, reason}], benchmark,
    suite}. Hard filters: embodiment, license, GPU tier, edge size; then the suite most candidates cover for the
    action type; then the cheapest MAX_CANDIDATES (estimate for `episodes` on one action). ``only`` = target names
    the customer wants to compare: the others are excluded ("not in your list") and a listed target that the
    filters exclude raises ValueError with the reason."""
    a = validate_answers(answers)
    excluded, kept = [], []
    for name, reason in sorted((catalog.get("not_benchable") or {}).items()):
        if reason.get("embodiment") == a["robot"]:
            excluded.append(dict(target=name, reason=f"not on a shared bench yet: {reason.get('reason')}"))
    for name, e in sorted(catalog["targets"].items()):
        if e.get("embodiment") != a["robot"]:
            continue  # other robot forms are not listed as exclusions (not relevant to the question)
        why = None
        if a["commercial"] == "yes":
            why = _license_reason(e)
        if not why and a["gpu"] != "any" and GPU_RANK.get(e.get("gpu_tier"), 9) > GPU_RANK[a["gpu"]]:
            why = f"needs a {e.get('gpu_tier')} GPU ({e.get('instance')}), budget is {a['gpu']}"
        if not why and a["edge"] == "yes" and e.get("edge_friendly") is not True:
            why = ("too large or slow for an edge device" if e.get("edge_friendly") is False
                   else "model size not recorded, cannot confirm it fits an edge device")
        if why:
            excluded.append(dict(target=name, reason=why))
        else:
            kept.append(name)
    if not kept:
        return dict(candidates=[], excluded=excluded, benchmark=None, suite=None, answers=a)
    benchmark = catalog["targets"][kept[0]]["benchmark"]
    acts = [x for x in all_actions(benchmark) if a["action"] == "any" or a["action"] in x["groups"]]
    suites = list(dict.fromkeys(x["suite"] for x in acts))
    if not suites:
        for name in kept:
            excluded.append(dict(target=name, reason=f"{benchmark} has no {a['action']} task"))
        return dict(candidates=[], excluded=excluded, benchmark=benchmark, suite=None, answers=a)
    cover = {s: [n for n in kept if s in catalog["targets"][n].get("suites", [])] for s in suites}
    best = max(suites, key=lambda s: len(cover[s]))  # first suite wins a tie (catalog order)
    chosen = []
    for name in kept:
        if name in cover[best]:
            chosen.append(name)
        else:
            excluded.append(dict(target=name, reason=f"checkpoint does not cover {best} (covers "
                                                     f"{', '.join(catalog['targets'][name].get('suites', []))})"))

    def cost_key(n):
        est = estimate(catalog["targets"][n], episodes, region)
        return (est["usd"] if est["usd"] is not None else 1e9, n)

    if only:
        why = {x["target"]: x["reason"] for x in excluded}
        for name in only:
            if name not in chosen:
                raise ValueError(f"{name} does not fit these answers: "
                                 f"{why.get(name, 'not a ' + a['robot'] + ' target in bench/catalog.yaml')}")
        for name in [n for n in chosen if n not in only]:
            excluded.append(dict(target=name, reason="not in your --targets list"))
        chosen = [n for n in chosen if n in only]
    chosen.sort(key=cost_key)
    for name in chosen[MAX_CANDIDATES:]:
        excluded.append(dict(target=name, reason=f"more expensive than the {MAX_CANDIDATES} cheapest candidates"))
    chosen = chosen[:MAX_CANDIDATES]
    cands = []
    for name in chosen:
        e = catalog["targets"][name]
        lic = "commercial use OK" if e.get("commercial_ok") is True else (
            "non-commercial" if e.get("commercial_ok") is False else "license to review")
        cands.append(dict(target=name, reason=f"{e.get('strength')}; {e.get('gpu_tier')} GPU ({e.get('instance')}); "
                                              f"{lic}; covers {best}"))
    return dict(candidates=cands, excluded=excluded, benchmark=benchmark, suite=best, answers=a)


def actions_for(catalog: dict, selection: dict) -> list:
    """Actions valid for every candidate (their suites), matching the action answer, grouped and ordered."""
    if not selection["candidates"]:
        return []
    names = [c["target"] for c in selection["candidates"]]
    common = set.intersection(*(set(catalog["targets"][n].get("suites", [])) for n in names))
    want = selection["answers"]["action"]
    out = []
    for x in all_actions(selection["benchmark"]):
        if x["suite"] not in common:
            continue
        if want != "any" and want not in x["groups"]:
            continue
        out.append(dict(x, group=want if want != "any" else (x["groups"] or ["other"])[0]))
    order = list(GROUP_LABELS) + ["other"]
    out.sort(key=lambda x: (order.index(x["group"]) if x["group"] in order else 99, x["id"]))
    return out


# ───────────────────────────── plan ─────────────────────────────

def make_plan(catalog: dict, selection: dict, actions: list, episodes: int, seed: int, region: str,
              plan_id: str | None = None, now: dt.datetime | None = None) -> dict:
    if not actions:
        raise ValueError("pick at least one action")
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    plan_id = plan_id or secrets.token_hex(3)
    now = now or dt.datetime.now(dt.timezone.utc)
    n_total = episodes * len(actions)
    per = []
    for c in selection["candidates"]:
        e = catalog["targets"][c["target"]]
        est = estimate(e, n_total, region)
        per.append(dict(target=c["target"], **est))
    warnings = []
    for p in per:
        limit = _timeout_minutes(_model_instance(p["target"]).get("creationpolicy_timeout"))
        if limit and p["minutes"] and p["minutes"] > 0.8 * limit:
            warnings.append(f"{p['target']}: ~{p['minutes']} min is close to or above the stack's {limit} min "
                            "CloudFormation wait limit (instance.creationpolicy_timeout); use fewer episodes or actions")
    known = [p["usd"] for p in per if p["usd"] is not None]
    return dict(
        schema=PLAN_SCHEMA, id=plan_id, condition=f"bench-{plan_id}", created_utc=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        region=region, answers=selection["answers"], benchmark=selection["benchmark"], suite=selection["suite"],
        candidates=selection["candidates"], excluded=selection["excluded"],
        actions=[{k: a[k] for k in ("id", "suite", "task_id", "task", "env_name", "instruction", "group", "action_type")
                  if k in a} for a in actions],
        episodes=episodes, seed=int(seed),
        estimate=dict(per_target=per, total_usd=round(sum(known), 2) if known else None,
                      wall_minutes=max((p["minutes"] or 0) for p in per) if per else None,
                      note="EC2 On-Demand only, from bench/catalog.yaml figures; stack create/delete, NAT, EBS and "
                           "S3 not included; targets run in parallel"),
        warnings=warnings, disclaimer=DISCLAIMER)


def _model_instance(target: str) -> dict:
    """instance: block of models/<target>.yaml ({} when unreadable)."""
    try:
        import yaml  # noqa: PLC0415
        return (yaml.safe_load((BASE_DIR / "models" / f"{target}.yaml").read_text()) or {}).get("instance") or {}
    except Exception:  # noqa: BLE001 - only used for a warning
        return {}


def load_plan(path) -> dict:
    import yaml  # noqa: PLC0415
    plan = yaml.safe_load(Path(path).read_text())
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"{path}: not a bench plan ({PLAN_SCHEMA})")
    if not re.fullmatch(r"[0-9a-f]{6}", str(plan.get("id", ""))):
        raise ValueError(f"{path}: id must be 6 hex characters")
    return plan


def write_plan(plan: dict, path) -> None:
    import yaml  # noqa: PLC0415
    Path(path).write_text("# Bench plan written by `python vlasim.py bench plan` (docs/bench.md). " + DISCLAIMER + "\n"
                          + yaml.safe_dump(plan, sort_keys=False, allow_unicode=False, width=120))


def print_plan(plan: dict, out=sys.stdout) -> None:
    p = lambda s="": print(s, file=out)  # noqa: E731
    p(f"Bench plan {plan['id']} (condition {plan['condition']}, region {plan['region']})")
    p(f"  {DISCLAIMER}")
    p("Candidates:")
    for c in plan["candidates"]:
        p(f"  + {c['target']}: {c['reason']}")
    if plan["excluded"]:
        p("Excluded:")
        for x in plan["excluded"]:
            p(f"  - {x['target']}: {x['reason']}")
    p(f"Actions ({plan['episodes']} episodes each, seed {plan['seed']}):")
    for a in plan["actions"]:
        p(f"  {a['id']:<18} [{a['group']}] {a['instruction']}")
    p("Estimate (EC2 On-Demand, catalog figures):")
    for e in plan["estimate"]["per_target"]:
        usd = "?" if e["usd"] is None else f"${e['usd']:.2f}"
        mins = "?" if e["minutes"] is None else f"~{e['minutes']} min"
        p(f"  {e['target']:<18} {e['instance']:<12} {mins:>10}  {usd:>8}   ({e['basis']})")
    tot = plan["estimate"]["total_usd"]
    p(f"  total {'?' if tot is None else f'${tot:.2f}'}, wall time ~{plan['estimate']['wall_minutes']} min (parallel)")
    for w in plan.get("warnings") or []:
        p(f"  [warn] {w}")


def _ask(prompt: str, options: list, inp=input, out=sys.stdout, default: int | None = None) -> int:
    print(prompt, file=out)
    for i, label in enumerate(options, 1):
        print(f"  {i}. {label}", file=out)
    while True:
        raw = inp(f"choice [1-{len(options)}]{f' (default {default})' if default else ''}: ").strip()
        if not raw and default:
            return default - 1
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print("  please answer with one of the numbers above", file=out)


def ask_answers(inp=input, out=sys.stdout) -> dict:
    answers = {}
    for key, prompt, opts in QUESTIONS:
        answers[key] = opts[_ask(prompt, [label for _, label in opts], inp, out)][0]
        print(file=out)
    return answers


def parse_action_ids(raw, actions: list) -> list:
    """'1,3' (numbers in the printed list) or ids ('libero_spatial:0') -> the picked actions, 1-3 of them."""
    items = raw if isinstance(raw, list) else [x.strip() for x in str(raw).split(",") if x.strip()]
    by_id = {a["id"]: a for a in actions}
    picked = []
    for it in items:
        s = str(it)
        if s.isdigit() and 1 <= int(s) <= len(actions):
            a = actions[int(s) - 1]
        elif s in by_id:
            a = by_id[s]
        else:
            raise ValueError(f"unknown action {s!r} (not valid for every candidate)")
        if a not in picked:
            picked.append(a)
    if not 1 <= len(picked) <= 3:
        raise ValueError("pick 1 to 3 actions")
    return picked


def _config_region() -> str:
    try:
        import yaml  # noqa: PLC0415
        return (yaml.safe_load((BASE_DIR / "simulator-config.yaml").read_text()).get("deployment") or {}).get(
            "region") or "us-east-1"
    except Exception:  # noqa: BLE001 - best effort default
        return "us-east-1"


def cmd_plan(args, inp=input, out=None) -> int:
    out = out or sys.stdout
    import seeds  # noqa: PLC0415

    catalog = load_catalog(args.catalog)
    region = args.region or _config_region()
    given = {}
    if args.answers:
        import yaml  # noqa: PLC0415
        given = yaml.safe_load(Path(args.answers).read_text()) or {}
        answers = {k: given.get(k) for k in ANSWER_VALUES}
    else:
        answers = ask_answers(inp, out)
    episodes = int(args.episodes or given.get("episodes") or DEFAULT_EPISODES)
    only = args.targets or given.get("targets")
    if isinstance(only, str):
        only = [x.strip() for x in only.split(",") if x.strip()]
    sel = select(catalog, answers, region, episodes, only=only)
    if not args.answers:
        for c in sel["candidates"]:
            print(f"  + {c['target']}: {c['reason']}", file=out)
        for x in sel["excluded"]:
            print(f"  - {x['target']}: {x['reason']}", file=out)
    if not sel["candidates"]:
        print("[bench] No target fits these answers. Relax one (license, GPU size or edge), or see "
              "docs/bench.md for the targets that can be benchmarked.", file=sys.stderr)
        return 1
    if len(sel["candidates"]) == 1:
        print("[bench] [warn] only one candidate fits: the report will show it alone (nothing to compare).", file=out)
    acts = actions_for(catalog, sel)
    raw_actions = args.actions or given.get("actions")
    if raw_actions is None:
        print(f"\nActions valid for every candidate ({sel['benchmark']}, {len(acts)}):", file=out)
        for i, a in enumerate(acts, 1):
            print(f"  {i:>2}. [{GROUP_LABELS.get(a['group'], a['group'])}] {a['id']}: {a['instruction']}", file=out)
        while True:
            raw_actions = inp("pick 1-3 actions (numbers, comma-separated): ")
            try:
                picked = parse_action_ids(raw_actions, acts)
                break
            except ValueError as e:
                print(f"  {e}", file=out)
        if not args.episodes and "episodes" not in given:
            raw = inp(f"episodes per action [{DEFAULT_EPISODES}]: ").strip()
            episodes = int(raw) if raw.isdigit() and int(raw) > 0 else DEFAULT_EPISODES
    else:
        picked = parse_action_ids(raw_actions, acts)
    seed_arg = str(args.seed or given.get("seed") or "auto")
    seed = seeds.resolve(seed_arg)["value"]
    if seed is None:
        raise ValueError("--seed must be auto or a number")
    plan = make_plan(catalog, sel, picked, episodes, seed, region)
    print(file=out)
    print_plan(plan, out)
    if not args.yes:
        if inp(f"\nwrite {args.out} ? [y/N]: ").strip().lower() not in ("y", "yes"):
            print("[bench] not written.", file=out)
            return 1
    write_plan(plan, args.out)
    print(f"[bench] wrote {args.out}. Next: python vlasim.py bench run --plan {shlex.quote(str(args.out))} --dry-run",
          file=out)
    return 0


# ───────────────────────────── run ─────────────────────────────

def tasks_spec(plan: dict, target: str, catalog: dict) -> dict:
    """The --tasks-file of one target: the plan's actions in that target's own tasks: format."""
    fam = catalog["targets"][target]["family"]
    tasks = []
    for a in plan["actions"]:
        common = dict(seed=plan["seed"], condition=plan["condition"],
                      description=f"bench {plan['id']}: {a['instruction']}")
        if fam == "lerobot-libero":
            tasks.append(dict(suite=a["suite"], task_ids=[a["task_id"]], env_name=a["env_name"],
                              n_episodes=plan["episodes"], **common))
        elif fam == "pi":
            tasks.append(dict(suite=a["suite"], task_ids=[a["task_id"]], num_episodes=plan["episodes"], **common))
        elif fam == "robotwin":
            tasks.append(dict(setting=a["suite"], robotwin_tasks=[a["task"]], n_episodes=plan["episodes"], **common))
        else:
            raise ValueError(f"{target}: family {fam!r} is not supported by bench")
    return dict(name=plan["condition"], description=f"vlasim bench {plan['id']} ({len(tasks)} action(s) x "
                                                    f"{plan['episodes']} ep, seed {plan['seed']})", tasks=tasks)


def run_dir(plan_path) -> Path:
    p = Path(plan_path).resolve()
    return p.parent / f"{p.stem}.d"


def deploy_commands(plan: dict, plan_path, catalog: dict, email: str = "", no_gallery: bool = False) -> list:
    """[(target, tasks file path, argv)] — argv runs from the repo root."""
    out = []
    d = run_dir(plan_path)
    for c in plan["candidates"]:
        t = c["target"]
        f = d / f"tasks-{t}.yaml"
        argv = [sys.executable, "deploy.py", "--vla", t, "--tasks-file", str(f), "--seed", str(plan["seed"]),
                "--region", plan["region"]]
        if email:
            argv += ["--email", email]
        if no_gallery:
            argv.append("--no-gallery")
        out.append((t, f, argv))
    return out


def write_tasks_files(plan: dict, plan_path, catalog: dict) -> list:
    import yaml  # noqa: PLC0415
    d = run_dir(plan_path)
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for c in plan["candidates"]:
        f = d / f"tasks-{c['target']}.yaml"
        f.write_text(yaml.safe_dump(tasks_spec(plan, c["target"], catalog), sort_keys=False))
        paths.append(f)
    return paths


def render_check(target: str, tasks_file: Path, seed: int, outdir: Path) -> str:
    """Render the target's user data with the tasks file (generate.py, no AWS) and bash -n it; '' = ok."""
    outdir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, "generate.py", "--vla", target, "--tasks-file", str(tasks_file),  # nosec B603
                        "--seed", str(seed), "--output-dir", str(outdir)], cwd=str(BASE_DIR),
                       capture_output=True, text=True)
    if r.returncode:
        return (r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout).strip() else "generate failed"
    b = subprocess.run(["bash", "-n", str(outdir / f"{target}.sh")], capture_output=True, text=True)  # nosec B603,B607
    return b.stderr.strip() if b.returncode else ""


def cmd_run(args, out=None) -> int:
    out = out or sys.stdout
    catalog = load_catalog(args.catalog)
    plan = load_plan(args.plan)
    write_tasks_files(plan, args.plan, catalog)
    cmds = deploy_commands(plan, args.plan, catalog, email=args.email or "", no_gallery=args.no_gallery)
    d = run_dir(args.plan)
    print(f"[bench] plan {plan['id']}: {len(cmds)} target(s), condition {plan['condition']}, tasks files in {d}",
          file=out)
    if args.dry_run:
        bad = 0
        for t, f, argv in cmds:
            print(" ".join(shlex.quote(x) for x in ["python", *argv[1:]]), file=out)
            if not args.no_render:
                err = render_check(t, f, plan["seed"], d / "userdata")
                print(f"  render + bash -n {t}: {'ok' if not err else 'FAIL: ' + err}", file=out)
                bad += bool(err)
        print("[bench] dry run: nothing deployed. Without --dry-run the commands above run in parallel "
              f"(logs: {d}/deploy-<target>.log).", file=out)
        return 1 if bad else 0
    if not args.no_gallery:
        # Create the shared gallery once before the parallel deploys (each deploy.py would race to create it).
        import gallery  # noqa: PLC0415
        email = args.email or _config_email()
        gallery.ensure_gallery(plan["region"], email)
    started = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    procs = []
    for t, _, argv in cmds:
        log = open(d / f"deploy-{t}.log", "w")  # noqa: SIM115 - closed after wait()
        procs.append((t, log, subprocess.Popen(argv, cwd=str(BASE_DIR), stdout=log, stderr=subprocess.STDOUT)))  # nosec B603
        print(f"[bench] started {t} (log {d}/deploy-{t}.log)", file=out)
    rcs = {}
    for t, log, p in procs:
        rcs[t] = p.wait()
        log.close()
        print(f"[bench] {t}: deploy.py exit {rcs[t]}", file=out)
    (d / "run.json").write_text(json.dumps(dict(plan=plan["id"], started_utc=started, exit=rcs), indent=1))
    print(f"[bench] the runs continue on the instances; when the e-mails arrive: python vlasim.py bench report "
          f"--plan {shlex.quote(str(args.plan))}", file=out)
    return 1 if any(rcs.values()) else 0


def _config_email() -> str:
    import yaml  # noqa: PLC0415
    return (yaml.safe_load((BASE_DIR / "simulator-config.yaml").read_text()).get("deployment") or {}).get(
        "notify_email", "")


def cmd_destroy(args, out=None) -> int:
    out = out or sys.stdout
    plan = load_plan(args.plan)
    rc = 0
    for c in plan["candidates"]:
        argv = [sys.executable, "destroy.py", "--vla", c["target"], "--region", plan["region"], "--keep-data", "--yes"]
        print(" ".join(shlex.quote(x) for x in ["python", *argv[1:]]), file=out)
        if not args.dry_run:
            rc |= subprocess.run(argv, cwd=str(BASE_DIR)).returncode  # nosec B603
    return rc


# ───────────────────────────── report ─────────────────────────────

class LocalS3:
    """results.S3 / leaderboard.S3-compatible reader over a directory tree <root>/<bucket>/<key> (tests and
    offline reports from `aws s3 sync` copies)."""

    def __init__(self, root):
        self.root = Path(root)

    def _p(self, bucket, key=""):
        return self.root / bucket / key

    def objects(self, region, bucket, prefix):
        base = self._p(bucket)
        out = []
        for f in sorted(base.rglob("*")):
            if f.is_file():
                key = f.relative_to(base).as_posix()
                if key.startswith(prefix):
                    out.append(dict(Key=key, LastModified=dt.datetime.fromtimestamp(f.stat().st_mtime, dt.timezone.utc)))
        return out

    def keys(self, region, bucket, prefix):
        return [o["Key"] for o in self.objects(region, bucket, prefix)]

    def prefixes(self, region, bucket, prefix):
        d = self._p(bucket, prefix)
        return sorted(f"{prefix}{x.name}/" for x in d.iterdir() if x.is_dir()) if d.is_dir() else []

    def text(self, region, bucket, key, head=None):
        f = self._p(bucket, key)
        try:
            data = f.read_bytes()
        except OSError:
            return None
        return (data[:head] if head else data).decode("utf-8", "replace")

    def buckets(self):
        return sorted(x.name for x in self.root.iterdir() if x.is_dir())


def _stack_of(target: str) -> str:
    from destroy import STACK_NAMES  # noqa: PLC0415
    return STACK_NAMES[target].lower()


def collect_bench(plan: dict, s3, bucket_names: list) -> dict:
    """{target: [run dicts]} — runs (leaderboard.read_run) of each candidate's bucket that have at least one task
    with the plan's condition; only those tasks are kept."""
    import leaderboard  # noqa: PLC0415

    by_stack = {}
    for b in bucket_names:
        m = results.BUCKET_RE.match(b)
        if m and m.group("region") == plan["region"]:
            by_stack[m.group("stack")] = b
            by_stack.setdefault(m.group("stack") + "-demo", b)  # long names drop "-demo" (deploy.results_bucket_name)
    since = plan["created_utc"][:10]
    out = {}
    for c in plan["candidates"]:
        t = c["target"]
        bucket = by_stack.get(_stack_of(t))
        out[t] = []
        if not bucket:
            continue
        runs = sorted(p.rstrip("/") for p in s3.prefixes(plan["region"], bucket, "") if results.RUN_RE.match(p.rstrip("/")))
        for run in runs:
            if run < since:
                continue
            r = leaderboard.read_run(s3, bucket, _stack_of(t), plan["region"], run)
            r["tasks"] = [x for x in r["tasks"] if x.get("condition") == plan["condition"]]
            if r["tasks"]:
                out[t].append(r)
    return out


def _action_of(plan: dict, task: dict, index: int):
    """Plan action of one task-N: by env_name / robotwin task name when present, else by position."""
    name = task.get("task_name") or ""
    for a in plan["actions"]:
        if name and name in (a.get("env_name"), a.get("task")):
            return a
    m = re.search(r"task-(\d+)$", task.get("task", ""))
    i = int(m.group(1)) if m else index
    return plan["actions"][i] if 0 <= i < len(plan["actions"]) else None


def summarize(plan: dict, runs_by_target: dict, catalog: dict) -> list:
    """One row per candidate: pooled k/n, Wilson CI, per-action k/n, wall time and cost of the run(s), videos."""
    import leaderboard  # noqa: PLC0415

    rows = []
    for c in plan["candidates"]:
        t = c["target"]
        runs = runs_by_target.get(t) or []
        per_action = {a["id"]: dict(k=0, n=0, videos=[]) for a in plan["actions"]}
        k = n = 0
        wall = usd = 0.0
        wall_known = usd_known = False
        instance = None
        pending = 0
        for r in runs:
            cost = leaderboard.run_cost(r)
            if cost["wall_s"]:
                wall += cost["wall_s"]
                wall_known = True
            if cost["usd"] is not None:
                usd += cost["usd"]
                usd_known = True
            instance = cost["instance_type"] or instance
            for i, task in enumerate(r["tasks"]):
                a = _action_of(plan, task, i)
                if task["status"] != "done" or not task.get("n"):
                    pending += 1
                    continue
                k += task["k"] or 0
                n += task["n"]
                if a:
                    pa = per_action[a["id"]]
                    pa["k"] += task["k"] or 0
                    pa["n"] += task["n"]
                    vids = [o["Key"] for o in r["objects"] if o["Key"].startswith(task["prefix"] + "videos/")
                            and o["Key"].endswith(".mp4")]
                    pa["videos"] += [dict(bucket=r["bucket"], key=v, run=r["run"]) for v in sorted(vids)[:2]]
        lo, hi = results.wilson(k, n)
        e = catalog["targets"].get(t, {})
        est = next((x for x in plan["estimate"]["per_target"] if x["target"] == t), {})
        rows.append(dict(target=t, k=k, n=n, rate=(k / n) if n else None, lo=lo, hi=hi, per_action=per_action,
                         runs=[r["run"] for r in runs], pending=pending, wall_s=wall if wall_known else None,
                         usd=round(usd, 2) if usd_known else None, instance=instance or e.get("instance"),
                         est_usd=est.get("usd"), est_minutes=est.get("minutes"), commercial_ok=e.get("commercial_ok"),
                         license=e.get("license")))
    return rows


def rank(rows: list) -> dict:
    """Ranked rows + recommendation. Rank by success rate; a candidate whose 95% CI overlaps the leader's is a
    tie with it, and a tie is broken by license (commercial use OK first) then cost (measured, else estimate)."""
    have = [r for r in rows if r["n"]]
    if not have:
        return dict(order=rows, tie=[], text="No finished bench task yet: run the plan and wait for the e-mails.")
    have.sort(key=lambda r: (-(r["rate"] or 0), r["target"]))
    lead = have[0]
    tie = [r for r in have if r["hi"] >= lead["lo"]]

    def cost(r):
        v = r["usd"] if r["usd"] is not None else r["est_usd"]
        return v if v is not None else 1e9

    if len(tie) > 1:
        tie.sort(key=lambda r: (0 if r["commercial_ok"] is True else 1, cost(r), r["target"]))
        pick = tie[0]
        text = (f"Tie: the 95% intervals of {', '.join(r['target'] for r in tie)} overlap, so these runs do not "
                f"separate them. Pick by cost and license: {pick['target']} "
                f"({'commercial use OK' if pick['commercial_ok'] is True else 'review the license'}, "
                f"{'$%.2f' % cost(pick) if cost(pick) < 1e9 else 'cost unknown'}). More episodes narrow the intervals.")
    else:
        pick = lead
        text = (f"{pick['target']} leads: {pick['k']}/{pick['n']} with a 95% interval "
                f"{pick['lo']:.2f}-{pick['hi']:.2f} that does not overlap the others.")
    rest = [r for r in rows if r not in have]
    order = [pick] + [r for r in tie if r is not pick] + [r for r in have if r not in tie] + rest
    return dict(order=order, tie=[r["target"] for r in tie] if len(tie) > 1 else [], pick=pick["target"], text=text)


def bar_chart_svg(rows: list) -> str:
    """Horizontal bars of the success rate with the 95% Wilson interval as a whisker (inline SVG)."""
    left, width, bar_h, gap = 150, 420, 22, 14
    h = 30 + len(rows) * (bar_h + gap)
    e = html.escape
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{left + width + 80}" height="{h}" role="img" '
             f'aria-label="Success rate with 95% confidence interval per model" font-family="sans-serif" font-size="12">']
    for x in (0, 0.25, 0.5, 0.75, 1.0):
        px = left + x * width
        parts.append(f'<line x1="{px:.1f}" y1="10" x2="{px:.1f}" y2="{h - 14}" stroke="#ddd"/>')
        parts.append(f'<text x="{px:.1f}" y="{h - 2}" text-anchor="middle" fill="#666">{int(x * 100)}%</text>')
    for i, r in enumerate(rows):
        y = 14 + i * (bar_h + gap)
        parts.append(f'<text x="{left - 8}" y="{y + bar_h * 0.7:.1f}" text-anchor="end">{e(r["target"])}</text>')
        if r["rate"] is None:
            parts.append(f'<text x="{left + 4}" y="{y + bar_h * 0.7:.1f}" fill="#888">no result yet</text>')
            continue
        parts.append(f'<rect x="{left}" y="{y}" width="{r["rate"] * width:.1f}" height="{bar_h}" fill="#4a78b5">'
                     f'<title>{e(r["target"])}: {r["k"]}/{r["n"]}</title></rect>')
        x1, x2, cy = left + r["lo"] * width, left + r["hi"] * width, y + bar_h / 2
        parts.append(f'<line x1="{x1:.1f}" y1="{cy}" x2="{x2:.1f}" y2="{cy}" stroke="#222" stroke-width="2"/>')
        for xx in (x1, x2):
            parts.append(f'<line x1="{xx:.1f}" y1="{y + 4}" x2="{xx:.1f}" y2="{y + bar_h - 4}" stroke="#222" stroke-width="2"/>')
        parts.append(f'<text x="{left + width + 6}" y="{y + bar_h * 0.7:.1f}">{r["rate"] * 100:.0f}% ({r["k"]}/{r["n"]})</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_html(plan: dict, rows: list, ranking: dict, video_url=None, generated: str | None = None) -> str:
    """One self-contained HTML page (inline CSS + SVG, no external JS). video_url(bucket, key, run, target) ->
    URL or None."""
    e = html.escape
    generated = generated or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    labels = {k: dict(opts) for k, _, opts in QUESTIONS}
    ans = "".join(f"<li>{e(prompt)} <b>{e(labels[k].get(plan['answers'].get(k), str(plan['answers'].get(k))))}</b></li>"
                  for k, prompt, _ in QUESTIONS)
    cands = "".join(f"<li><b>{e(c['target'])}</b>: {e(c['reason'])}</li>" for c in plan["candidates"])
    excl = "".join(f"<li>{e(x['target'])}: {e(x['reason'])}</li>" for x in plan["excluded"]) or "<li>none</li>"
    by_t = {r["target"]: r for r in rows}

    def fmt_rate(k, n):
        if not n:
            return "-"
        lo, hi = results.wilson(k, n)
        return f"{k}/{n} = {k / n:.2f} [{lo:.2f}-{hi:.2f}]"

    head = "".join(f"<th>{e(r['target'])}</th>" for r in rows)
    act_rows = ""
    for a in plan["actions"]:
        cells = "".join(f"<td>{e(fmt_rate(by_t[r['target']]['per_action'][a['id']]['k'], by_t[r['target']]['per_action'][a['id']]['n']))}</td>"
                        for r in rows)
        act_rows += f"<tr><td>{e(a['id'])}</td><td>{e(GROUP_LABELS.get(a['group'], a['group']))}</td><td>{e(a['instruction'])}</td>{cells}</tr>"
    vid_rows = ""
    for a in plan["actions"]:
        cells = ""
        for r in rows:
            vids = by_t[r["target"]]["per_action"][a["id"]]["videos"]
            cell = ""
            for v in vids[:2]:
                url = video_url(v["bucket"], v["key"], v["run"], r["target"]) if video_url else None
                name = v["key"].rsplit("/", 1)[-1]
                ok = "success=True" in name
                tag = "success" if ok else ("failure" if "success=False" in name else "")
                if url:
                    cell += (f'<figure><video src="{e(url)}" controls preload="none" width="240"></video>'
                             f'<figcaption>{e(tag or name)}</figcaption></figure>')
                else:
                    cell += f"<div>s3://{e(v['bucket'])}/{e(v['key'])}</div>"
            cells += f"<td>{cell or 'no video'}</td>"
        vid_rows += f"<tr><td>{e(a['instruction'])}</td>{cells}</tr>"
    cost_rows = ""
    for r in rows:
        wall = "-" if r["wall_s"] is None else f"{r['wall_s'] / 60:.0f} min"
        usd = "-" if r["usd"] is None else f"${r['usd']:.2f}"
        est = "-" if r["est_usd"] is None else f"${r['est_usd']:.2f} / ~{r['est_minutes']} min"
        cost_rows += (f"<tr><td>{e(r['target'])}</td><td>{e(r['instance'] or '-')}</td><td>{wall}</td><td>{usd}</td>"
                      f"<td>{est}</td><td>{e(', '.join(r['runs']) or '-')}</td><td>{r['pending']}</td></tr>")
    rank_rows = "".join(
        f"<tr><td>{i}</td><td>{e(r['target'])}</td><td>{e(fmt_rate(r['k'], r['n']))}</td>"
        f"<td>{'yes' if r['commercial_ok'] is True else ('no' if r['commercial_ok'] is False else 'review')}</td>"
        f"<td>{'tie' if r['target'] in ranking.get('tie', []) else ''}</td></tr>"
        for i, r in enumerate(ranking["order"], 1))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>RFM bench {e(plan['id'])}</title>
<style>
body{{font-family:-apple-system,"Segoe UI",sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}}
.disclaimer{{border:1px solid #c9a227;background:#fff8e1;padding:10px 14px;border-radius:4px}}
table{{border-collapse:collapse;margin:8px 0 20px}}td,th{{border:1px solid #ccc;padding:4px 8px;vertical-align:top;font-size:13px}}
th{{background:#f3f3f3}}figure{{margin:0 0 6px}}figcaption{{font-size:11px;color:#555}}.rec{{font-size:15px}}
</style></head><body>
<h1>RFM benchmark {e(plan['id'])}</h1>
<p class="disclaimer"><b>{e(DISCLAIMER)}</b> Condition <code>{e(plan['condition'])}</code>, region {e(plan['region'])},
{plan['episodes']} episodes per action, seed {plan['seed']}. Generated {e(generated)} by <code>python vlasim.py bench report</code>.</p>
<h2>Recommendation</h2><p class="rec">{e(ranking['text'])}</p>
<table><tr><th>#</th><th>Model</th><th>k/n = rate [95% CI]</th><th>Commercial use</th><th></th></tr>{rank_rows}</table>
<p>Tie rule: when the 95% Wilson intervals overlap, the runs do not separate the models; choose by license, then cost.</p>
<h2>Success rate (95% Wilson interval)</h2>
{bar_chart_svg(rows)}
<h2>Per action</h2>
<table><tr><th>Action</th><th>Group</th><th>Instruction</th>{head}</tr>{act_rows}</table>
<h2>Rollouts side by side</h2>
<table><tr><th>Action</th>{head}</tr>{vid_rows}</table>
<h2>Wall time and cost</h2>
<table><tr><th>Model</th><th>Instance</th><th>Wall time (measured)</th><th>EC2 cost (measured)</th><th>Plan estimate</th><th>Runs</th><th>Tasks not done</th></tr>{cost_rows}</table>
<p>Measured = RUN_ID to the last upload of the run at the pinned On-Demand price (leaderboard.py); stack create/delete, NAT, EBS and S3 not included.</p>
<h2>Your answers</h2><ul>{ans}</ul>
<h2>Chosen models and why</h2><ul>{cands}</ul>
<h2>Not chosen</h2><ul>{excl}</ul>
</body></html>
"""


def cmd_report(args, out=None) -> int:
    out = out or sys.stdout
    catalog = load_catalog(args.catalog)
    plan = load_plan(args.plan)
    gallery_info = None
    if args.from_dir:
        s3 = LocalS3(args.from_dir)
        names = s3.buckets()
        root = Path(args.from_dir).resolve()

        def video_url(bucket, key, run, target):
            return (root / bucket / key).as_uri()
    else:
        import boto3  # noqa: PLC0415

        import leaderboard  # noqa: PLC0415
        s3 = leaderboard.S3(boto3, plan["region"])
        names = [b for b, _, _ in leaderboard.list_buckets(argparse.Namespace(vla=None, region=plan["region"]), boto3)]
        if not args.no_gallery:
            try:
                import gallery  # noqa: PLC0415
                gallery_info = gallery.lookup(plan["region"], args.email or _config_email())
            except Exception as ex:  # noqa: BLE001 - gallery is optional; fall back to presigned URLs
                print(f"[bench] gallery not available ({ex}); writing the report locally", file=out)
        client = boto3.client("s3", region_name=plan["region"])
        prefixes = {}

        def video_url(bucket, key, run, target):
            if gallery_info:  # the GalleryPublisher copies videos to g/<id>/<target>/<key>; the page sits in g/<id>/bench/
                if target not in prefixes:
                    import gallery  # noqa: PLC0415
                    from destroy import STACK_NAMES  # noqa: PLC0415
                    outs = gallery.stack_outputs(plan["region"], STACK_NAMES[target]) or {}
                    prefixes[target] = outs.get("GalleryPrefix") or f"g/{gallery_info['id']}/{target}/"
                rel = prefixes[target].split("/", 2)[-1] if prefixes[target].startswith("g/") else prefixes[target]
                return f"../{rel}{key}"
            return client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key},
                                                 ExpiresIn=7 * 24 * 3600)
    runs = collect_bench(plan, s3, names)
    rows = summarize(plan, runs, catalog)
    ranking = rank(rows)
    page = render_html(plan, rows, ranking, video_url=video_url)
    dest = Path(args.out or f"bench-report-{plan['id']}.html")
    dest.write_text(page)
    print(f"[bench] {ranking['text']}", file=out)
    print(f"[bench] wrote {dest}", file=out)
    if gallery_info:
        import boto3  # noqa: PLC0415
        key = f"g/{gallery_info['id']}/bench/{plan['id']}.html"
        boto3.client("s3", region_name=plan["region"]).put_object(
            Bucket=gallery_info["bucket"], Key=key, Body=page.encode(), ContentType="text/html; charset=utf-8",
            CacheControl="no-cache")
        print(f"[bench] published s3://{gallery_info['bucket']}/{key}", file=out)
        print(f"[bench] open {gallery_info['url']}bench/{plan['id']}.html (gallery password: python vlasim.py gallery)",
              file=out)
    return 0


# ───────────────────────────── CLI ─────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vlasim bench", description=__doc__.split("\n\n")[1])
    p.add_argument("--catalog", default=str(CATALOG_PATH), help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan", help="answer 5 questions, pick actions, write a plan file")
    pl.add_argument("--answers", metavar="YAML", help="answers (robot, action, commercial, gpu, edge; optional "
                    "actions, episodes, seed) instead of the interactive questions")
    pl.add_argument("--targets", help="comma-separated targets to compare (must fit the answers; default: the "
                    f"{MAX_CANDIDATES} cheapest that do)")
    pl.add_argument("--actions", help="comma-separated action ids or list numbers (1-3)")
    pl.add_argument("--episodes", type=int, help=f"episodes per action (default {DEFAULT_EPISODES})")
    pl.add_argument("--seed", help="auto (default: draw one) or a number; every target gets the same seed")
    pl.add_argument("--region", help="AWS region (default: simulator-config.yaml)")
    pl.add_argument("--out", default="bench-plan.yaml", help="plan file (default bench-plan.yaml)")
    pl.add_argument("--yes", "-y", action="store_true", help="write the plan without asking")
    ru = sub.add_parser("run", help="deploy every target of a plan (deploy.py --tasks-file), in parallel")
    ru.add_argument("--plan", required=True)
    ru.add_argument("--dry-run", action="store_true", help="write the tasks files, print the deploy commands, render "
                    "each target's user data (no AWS call) and stop")
    ru.add_argument("--no-render", action="store_true", help="--dry-run: skip the render + bash -n check")
    ru.add_argument("--email", help="notification e-mail (default: simulator-config.yaml)")
    ru.add_argument("--no-gallery", action="store_true", help="deploy without the video history gallery")
    rp = sub.add_parser("report", help="one HTML report of the plan's runs (condition bench-<id>)")
    rp.add_argument("--plan", required=True)
    rp.add_argument("--out", help="HTML file (default bench-report-<id>.html)")
    rp.add_argument("--from-dir", metavar="DIR", help="read <DIR>/<results bucket>/<RUN_ID>/... instead of S3 "
                    "(an `aws s3 sync` copy); no AWS call")
    rp.add_argument("--email", help="gallery e-mail (default: simulator-config.yaml)")
    rp.add_argument("--no-gallery", action="store_true", help="do not publish to the gallery; videos as presigned URLs")
    de = sub.add_parser("destroy", help="destroy.py --keep-data for every target of the plan")
    de.add_argument("--plan", required=True)
    de.add_argument("--dry-run", action="store_true", help="print the commands only")
    a = p.parse_args(argv)
    try:
        return {"plan": cmd_plan, "run": cmd_run, "report": cmd_report, "destroy": cmd_destroy}[a.cmd](a)
    except (ValueError, OSError) as ex:
        print(f"[bench] error: {ex}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
