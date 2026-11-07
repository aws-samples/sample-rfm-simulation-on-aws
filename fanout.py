#!/usr/bin/env python3
"""
fanout.py — run one benchmark as N shard stacks over regions and instance types (docs/shards.md).

    python vlasim.py deploy robolab-pi05 --robolab-variant full-shard --shards 16 --runs 2 \\
        --regions us-east-1,us-west-2,eu-central-1 --max-instances 16 --instance-types g6e.2xlarge,g6e.4xlarge \\
        --dry-run

Every (run, shard) unit is one `deploy.py --shard K/N [--shard-run R]` = one stack (<base>-s03of16 or
<base>-r2-s03of16) with its own instance and results bucket. The scheduler:

  * keeps at most --max-instances units alive and launches the next one when one finishes;
  * tries the candidates of a unit in a fixed order: every --regions entry for the first --instance-types
    entry, then the next type (the first region rotates with the shard index to spread the load). An
    InsufficientInstanceCapacity / quota failure moves the unit to its next candidate and caps that
    (region, type) at the number of units still running there; any other failure is retried once;
  * --spot: launch Spot first (On-Demand after it inside the same stack); a Spot interruption re-queues
    only that unit, and after two interruptions the unit runs On-Demand;
  * --auto-destroy: delete each unit's stack (results bucket kept) as soon as its run is done;
  * keeps its state in <run dir>/fanout-state.json (written after every step). Re-running the same command
    resumes; a file named STOP in the run dir (or --max-usd reached) stops new launches;
  * when every unit is done, downloads the shard results and merges them per run into
    <run dir>/merged/run-<r>/ (the same summary.txt / robolab_results.json one run writes), with the actual
    cost: instance-hours from EC2 (launch to termination) x On-Demand price (or the Spot price history).

--dry-run prints every unit's deploy command, region and instance type, renders each shard's user data
(generate.py + bash -n) and touches nothing in AWS.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import signal
import subprocess  # nosec B404 - fixed argv lists only
import sys
import time
from pathlib import Path

import shards

BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = "fanout-state.json"
RESULT_FILE = "fanout-result.json"
STATE_VERSION = 1
TERMINAL = ("done", "failed")
ALIVE = ("launching", "running", "cleanup")  # states that hold a --max-instances slot
COLLECT_TRIES = 10
DEFAULT_POLL_S = 120
CAPACITY_BACKOFF_S = 20 * 60
CAPACITY_ROUNDS = 3
OTHER_RETRIES = 1
SPOT_INTERRUPTIONS_BEFORE_ON_DEMAND = 2


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def iso(t: dt.datetime | None) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ") if t else ""


# --- plan ------------------------------------------------------------------------------------------------

def base_stack(vla: str) -> str:
    from destroy import STACK_NAMES  # noqa: PLC0415 - one stack-name map (destroy.py)

    return STACK_NAMES[vla]


def candidates(regions: list[str], types: list[str], k: int) -> list[dict]:
    """Ordered (region, instance type) list of shard k: all regions for the first type, then the next type.
    The first region rotates with k so the shards of one type spread over the regions."""
    off = (k - 1) % len(regions)
    rot = regions[off:] + regions[:off]
    return [dict(region=r, instance_type=t) for t in types for r in rot]


def build_plan(vla: str, config: dict, n: int, runs: int, regions: list[str], types: list[str], *,
               max_instances: int, spot: bool = False, auto_destroy: bool = False, variant: str = "",
               robotwin_protocol: str = "subset", run_dir: Path, passthrough: list[str] | None = None,
               max_usd: float | None = None) -> dict:
    """Plan dict (state file 'plan' + 'units'). Raises ValueError for a target that cannot be split."""
    if not regions:
        raise ValueError("--regions needs at least one region")
    if not types:
        raise ValueError("--instance-types needs at least one instance type")
    parts = shards.split_plan(vla, config, n, robotwin_protocol=robotwin_protocol)
    base = base_stack(vla)
    units = {}
    for r in range(1, runs + 1):
        for p in parts:
            tok = shards.shard_token(p["k"], n, r if runs > 1 else None)
            units[tok] = dict(token=tok, run=r, k=p["k"], n=n, kind=p["kind"], stack=shards.stack_name(base, tok),
                              units=p["units"], n_units=p["n_units"], episodes=p["episodes"],
                              tasks_file=str(run_dir / "tasks" / f"tasks-s{p['k']:02d}of{n:02d}.yaml")
                              if p["kind"] == "tasks-file" else "",
                              robotwin_shard=p.get("robotwin_shard", ""), cands=candidates(regions, types, p["k"]),
                              cand=0, state="queued", attempts=[], interruptions=0, other_failures=0, rounds=0)
    plan = dict(vla=vla, shards=n, runs=runs, regions=regions, instance_types=types, max_instances=max_instances,
                spot=spot, auto_destroy=auto_destroy, variant=variant, robotwin_protocol=robotwin_protocol,
                passthrough=list(passthrough or []), max_usd=max_usd, base_stack=base)
    return dict(
        plan=plan,
        parts=[{k: v for k, v in p.items() if k != "units"} for p in parts],
        units=units,
        fingerprint=fingerprint(vla, config, parts, plan),
    )


def plan_key(plan: dict) -> dict:
    """The plan fields a resumed run must agree on (the rest may change between invocations)."""
    return {k: plan[k] for k in ("vla", "shards", "runs", "variant", "robotwin_protocol")}


def deploy_argv(state: dict, unit: dict, cand: dict, spot: bool, fanout_id: str) -> list[str]:
    """deploy.py command of one unit attempt (runs from the repo root)."""
    plan = state["plan"]
    argv = [sys.executable, "deploy.py", "--vla", plan["vla"], "--shard", f"{unit['k']}/{unit['n']}"]
    if plan["runs"] > 1:
        argv += ["--shard-run", str(unit["run"])]
    argv += ["--region", cand["region"], "--instance-types", cand["instance_type"], "--repeats", "1",
             "--fanout-id", fanout_id]
    if plan["variant"]:
        argv += ["--robolab-variant", plan["variant"]]
    if unit["kind"] == "tasks-file":
        argv += ["--tasks-file", unit["tasks_file"]]
    if unit["kind"] == "robotwin":
        argv += ["--robotwin-protocol", plan["robotwin_protocol"], "--robotwin-shard", unit["robotwin_shard"]]
    if spot:
        argv.append("--spot")
    return argv + list(plan["passthrough"])


def fanout_id_of(run_dir: Path) -> str:
    return Path(run_dir).name


# --- state ----------------------------------------------------------------------------------------------

def load_state(run_dir: Path) -> dict | None:
    p = Path(run_dir) / STATE_FILE
    return json.loads(p.read_text()) if p.is_file() else None


def save_state(run_dir: Path, state: dict) -> None:
    p = Path(run_dir) / STATE_FILE
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
    os.replace(tmp, p)


# --- scheduler ------------------------------------------------------------------------------------------

class Scheduler:
    """One step = retry pending stack deletions, poll running units, collect finished ones, launch into free
    slots, save. Backend does the I/O (AwsBackend below; tests pass a fake).

    Unit states: queued -> launching (intent saved before the process is spawned) -> running -> collecting
    (compute done, results not yet confirmed) -> done; a failed / interrupted attempt and --auto-destroy go
    through cleanup (stack deletion pending; holds a slot until CloudFormation confirms the deletion)."""

    def __init__(self, run_dir: Path, state: dict, backend, log=print):
        self.run_dir = Path(run_dir)
        self.state = state
        self.b = backend
        self.log = log
        self.state.setdefault("caps", {})
        self.state.setdefault("events", [])
        self.state.setdefault("version", STATE_VERSION)
        self.reconcile()

    # helpers
    @property
    def plan(self) -> dict:
        return self.state["plan"]

    def save(self) -> None:
        save_state(self.run_dir, self.state)

    def units(self, *states) -> list[dict]:
        us = sorted(self.state["units"].values(), key=lambda u: (u["run"], u["k"]))
        return [u for u in us if not states or u["state"] in states]

    def alive(self) -> list[dict]:
        """Units that hold a slot: a deploy in flight or a stack whose deletion is not confirmed yet."""
        return self.units(*ALIVE)

    def event(self, msg: str) -> None:
        line = f"{iso(self.b.now())} {msg}"
        self.state["events"].append(line)
        self.log(f"[fanout] {msg}")

    def running_on(self, region: str, itype: str, exclude: dict | None = None) -> int:
        return sum(1 for u in self.alive() if u is not exclude
                   and u["attempts"][-1]["region"] == region and u["attempts"][-1]["instance_type"] == itype)

    def stop_requested(self) -> str:
        if (self.run_dir / "STOP").exists():
            return "STOP file"
        cap = self.plan.get("max_usd")
        if cap is not None and self.cost_so_far() >= cap:
            return f"--max-usd {cap} reached"
        return ""

    def cost_so_far(self) -> float:
        usd = 0.0
        for u in self.state["units"].values():
            for a in u["attempts"]:
                if a.get("cost"):
                    usd += a["cost"]["usd"]
                elif a.get("started") and not a.get("ended") and a.get("phase") != "not-started":
                    p, _ = self.b.price(a["instance_type"], a["region"], "on-demand")
                    h = shards.instance_hours(a["started"], iso(self.b.now()))
                    usd += (p or 0) * (h or 0)
        return round(usd, 4)

    # resume
    def reconcile(self) -> None:
        """Resume after a crash: an attempt whose launch intent was saved but whose process identity was not is
        matched against its receipt file and CloudFormation; it is never relaunched blindly. Units that stopped
        at result collection are collected again."""
        now = self.b.now()
        changed = False
        for u in self.units("launching"):
            a = u["attempts"][-1]
            pid = self.b.receipt(a)
            if pid:
                a.update(pid=pid, phase="launched")
                u["state"] = "running"
                self.event(f"{u['token']}: resume: deploy process {pid} found from its receipt")
            elif self.b.stack_status(u["stack"], a["region"]):
                a.update(pid=None, phase="launched")
                u["state"] = "running"
                self.event(f"{u['token']}: resume: no receipt but {u['stack']} exists; following CloudFormation")
            else:
                a.update(phase="not-started", ended=iso(now), rc=None, reason="not-started")
                u["state"] = "queued"
                self.event(f"{u['token']}: resume: launch intent without process or stack; re-queued")
            changed = True
        for u in self.units("failed"):
            if u.get("fail_reason") == "results-missing":
                u.update(state="collecting", collect_tries=0)
                self.event(f"{u['token']}: resume: retrying result collection")
                changed = True
        if changed:
            self.save()

    # step
    def step(self) -> str:
        now = self.b.now()
        collecting = self.units("collecting")  # units that reach it in this step were collected once already
        for u in self.units("cleanup"):
            self._cleanup(u, now)
        for u in self.units("running"):
            self._poll(u, now)
        for u in collecting:
            if u["state"] == "collecting":
                self._collect(u, now)
        stop = self.stop_requested()
        if stop and not self.state.get("stopped"):
            self.state["stopped"] = iso(now)
            self.event(f"stop: {stop}; no new launches (running units continue)")
        if not stop:
            self.state.pop("stopped", None)
            self._launch(now)
        self.save()
        if all(u["state"] in TERMINAL for u in self.units()):
            return "finished"
        if stop and not self.alive() and not self.units("collecting"):
            return "stopped"
        return "running"

    # instances owned by one attempt (immutable ids; each instance belongs to exactly one attempt)
    def _own_instances(self, u: dict, a: dict) -> None:
        owned = {i for b in u["attempts"] if b is not a for i in b.get("instance_ids") or []}
        t0 = shards.parse_time(a["started"]) - dt.timedelta(minutes=5)
        mine = set(a.get("instance_ids") or [])
        for i in self.b.instances(a["region"], stack=u["stack"]):
            lt = shards.parse_time(i.get("launch"))
            if i["instance_id"] not in owned and lt and lt >= t0:
                mine.add(i["instance_id"])
        a["instance_ids"] = sorted(mine)

    def _poll(self, u: dict, now: dt.datetime) -> None:
        a = u["attempts"][-1]
        self._own_instances(u, a)
        if a.get("spot") and a["instance_ids"]:
            insts = self.b.instances(a["region"], ids=a["instance_ids"])
            if any(i.get("interrupted") for i in insts):
                self.b.kill(a)
                a.update(ended=iso(now), rc=None, reason="spot-interruption")
                self._record_cost(a, now)
                u["interruptions"] += 1
                if u["interruptions"] >= SPOT_INTERRUPTIONS_BEFORE_ON_DEMAND:
                    u["no_spot"] = True
                self._to_cleanup(u, "queued", now, f"{u['token']}: Spot interruption in {a['region']} "
                                 f"({a['instance_type']}); re-queued" + (" On-Demand" if u.get("no_spot") else ""))
                return
        rc = self.b.poll(a)
        if rc is None:
            return
        a.update(ended=iso(now), rc=rc)
        self._record_cost(a, now)
        if rc == 0:
            u.update(state="collecting", collect_tries=0)
            self.event(f"{u['token']}: compute done in {a['region']} ({a['instance_type']}); collecting results")
            self._collect(u, now)
            return
        why = shards.classify_failure(self.b.log_tail(a))
        a["reason"] = why
        if why in ("capacity", "quota"):
            key = f"{a['region']}|{a['instance_type']}"
            self.state["caps"][key] = min(self.state["caps"].get(key, 10 ** 6),
                                          self.running_on(a["region"], a["instance_type"], exclude=u))
            u["cand"] += 1
            if u["cand"] >= len(u["cands"]):
                u["rounds"] += 1
                if u["rounds"] >= CAPACITY_ROUNDS:
                    u["fail_reason"] = why
                    self._to_cleanup(u, "failed", now, f"{u['token']}: {why} in every region/type for "
                                     f"{CAPACITY_ROUNDS} rounds; failed")
                    return
                u["cand"] = 0
                u["not_before"] = iso(now + dt.timedelta(seconds=CAPACITY_BACKOFF_S))
                self.state["caps"] = {}  # capacity moves: start the next round without caps
            nxt = u["cands"][u["cand"]]
            self._to_cleanup(u, "queued", now, f"{u['token']}: {why} in {a['region']} ({a['instance_type']}); "
                             f"next candidate {nxt['region']} {nxt['instance_type']}")
            return
        u["other_failures"] += 1
        if u["other_failures"] > OTHER_RETRIES:
            u["fail_reason"] = "deploy-failed"
            self._to_cleanup(u, "failed", now, f"{u['token']}: deploy failed (rc {rc}) twice; failed (log {a['log']})")
            return
        u["cand"] = (u["cand"] + 1) % len(u["cands"])
        self._to_cleanup(u, "queued", now, f"{u['token']}: deploy failed (rc {rc}, log {a['log']}); "
                         "retry on the next candidate")

    def _collect(self, u: dict, now: dt.datetime) -> None:
        """Download and validate the unit's results; done (or deletion with --auto-destroy) only after that."""
        a = u["attempts"][-1]
        got = self.b.collect(u, a, self.run_dir / "results" / u["token"])
        if got and shards.validate_results(u["kind"], got[0]):
            u["results_dir"], u["run_id"] = str(got[0]), got[1]
            u.pop("fail_reason", None)
            if self.plan["auto_destroy"]:
                self._to_cleanup(u, "done", now, f"{u['token']}: results {u['run_id']} confirmed; --auto-destroy "
                                 f"deleting {u['stack']}")
            else:
                u["state"] = "done"
                self.event(f"{u['token']}: done, results {u['run_id']}")
            return
        u["collect_tries"] = u.get("collect_tries", 0) + 1
        if u["collect_tries"] >= COLLECT_TRIES:
            u.update(state="failed", fail_reason="results-missing")
            self.event(f"{u['token']}: results not found after {COLLECT_TRIES} tries; stack kept, re-run to retry")
        else:
            self.event(f"{u['token']}: results not complete yet (try {u['collect_tries']}); retrying")

    def _to_cleanup(self, u: dict, after: str, now: dt.datetime, msg: str) -> None:
        u["state"] = "cleanup"
        u["after_cleanup"] = after
        self.event(msg)
        self._cleanup(u, now)

    def _cleanup(self, u: dict, now: dt.datetime) -> None:
        """Delete the attempt's stack; the unit keeps its slot until CloudFormation confirms the deletion."""
        a = u["attempts"][-1]
        rc = self.b.destroy(u["stack"], a["region"])
        a["destroy_tries"] = a.get("destroy_tries", 0) + 1
        a["destroy_rc"] = rc
        if rc != 0:
            self.event(f"{u['token']}: deleting {u['stack']} in {a['region']} failed (try {a['destroy_tries']}); "
                       "slot held, retried next step")
            return
        a["destroyed"] = iso(now)
        u["state"] = u.pop("after_cleanup", "queued")
        if u["state"] == "done":
            a["auto_destroyed"] = True
            self.event(f"{u['token']}: --auto-destroy deleted {u['stack']} (results bucket kept)")

    def _record_cost(self, a: dict, now: dt.datetime) -> None:
        """Cost of exactly the instances this attempt owns (by id)."""
        insts = self.b.instances(a["region"], ids=a["instance_ids"]) if a.get("instance_ids") else []
        rows = []
        for i in insts:
            end = i.get("end") or a.get("ended") or iso(now)
            rows.append(dict(i, end=end, end_source="ec2 termination" if i.get("end") else "scheduler"))
        a["instances"] = rows
        a["cost"] = shards.unit_cost(rows, self.b.price)

    def refresh_costs(self) -> None:
        """Re-read, by instance id, the attempts whose termination time EC2 did not report yet."""
        for u in self.state["units"].values():
            for a in u["attempts"]:
                if a.get("ended") and any(r.get("end_source") == "scheduler" for r in a.get("instances") or []):
                    self._record_cost(a, shards.parse_time(a["ended"]))

    def _pick(self, u: dict) -> int | None:
        for j in range(u["cand"], len(u["cands"])):
            c = u["cands"][j]
            cap = self.state["caps"].get(f"{c['region']}|{c['instance_type']}")
            if cap is None or self.running_on(c["region"], c["instance_type"]) < cap:
                return j
        return None

    def _launch(self, now: dt.datetime) -> None:
        for u in self.units("queued"):
            if len(self.alive()) >= self.plan["max_instances"]:
                return
            nb = shards.parse_time(u.get("not_before"))
            if nb and nb > now:
                continue
            j = self._pick(u)
            if j is None:
                continue
            u["cand"] = j
            c = u["cands"][j]
            spot = bool(self.plan["spot"]) and not u.get("no_spot")
            argv = deploy_argv(self.state, u, c, spot, fanout_id_of(self.run_dir))
            n = len(u["attempts"]) + 1
            log = self.run_dir / "logs" / f"{u['token']}-{n}-{c['region']}.log"
            a = dict(n=n, stack=u["stack"], region=c["region"], instance_type=c["instance_type"], spot=spot,
                     started=iso(now), log=str(log), rc_file=str(log.with_suffix(".rc")), argv=argv,
                     phase="launching", pid=None, instance_ids=[])
            u["attempts"].append(a)
            u["state"] = "launching"
            self.save()  # intent first: a crash after the spawn is reconciled, not relaunched
            a["pid"] = self.b.launch(u, a)
            a["phase"] = "launched"
            u["state"] = "running"
            self.save()
            self.event(f"{u['token']}: launch {u['stack']} in {c['region']} ({c['instance_type']}"
                       f"{', spot first' if spot else ''})")


# --- merge -----------------------------------------------------------------------------------------------

def unit_wall(u: dict) -> tuple[dt.datetime | None, dt.datetime | None]:
    starts = [shards.parse_time(a.get("started")) for a in u["attempts"]]
    ends = [shards.parse_time(a.get("ended")) for a in u["attempts"]]
    starts = [s for s in starts if s]
    ends = [e for e in ends if e]
    return (min(starts) if starts else None), (max(ends) if ends else None)


def merge(run_dir: Path, state: dict, config: dict) -> dict:
    """Merge the downloaded shard results per run and write merged/run-<r>/ + fanout-result.json."""
    plan = state["plan"]
    out = dict(vla=plan["vla"], shards=plan["shards"], runs=[], generated=iso(utcnow()),
               price_note="instance-hours from EC2 LaunchTime to termination x On-Demand $/h "
                          "(leaderboard.PRICES, AWS Price List API) or the EC2 Spot price history; "
                          "stack create/delete, EBS, NAT gateway and data transfer are not included")
    for r in range(1, plan["runs"] + 1):
        us = [u for u in state["units"].values() if u["run"] == r]
        us.sort(key=lambda u: u["k"])
        done = [u for u in us if u["state"] == "done" and u.get("results_dir")]
        missing = [u["token"] for u in us if u not in done]
        dest = Path(run_dir) / "merged" / f"run-{r}"
        inst_h = sum((a.get("cost") or {}).get("instance_hours", 0) for u in us for a in u["attempts"])
        usd = sum((a.get("cost") or {}).get("usd", 0) for u in us for a in u["attempts"])
        unpriced = [x for u in us for a in u["attempts"] for x in (a.get("cost") or {}).get("unpriced", [])]
        walls = [unit_wall(u) for u in us]
        t0 = min((w[0] for w in walls if w[0]), default=None)
        t1 = max((w[1] for w in walls if w[1]), default=None)
        cost = dict(instance_hours=round(inst_h, 3), usd=round(usd, 2), wall_clock_s=int((t1 - t0).total_seconds())
                    if t0 and t1 else None, started=iso(t0), ended=iso(t1), unpriced_instances=unpriced)
        meta = dict(fanout_id=fanout_id_of(run_dir), fanout_shards=plan["shards"], fanout_run=r,
                    fanout_complete=not missing, instance_hours=cost["instance_hours"], cost_usd=cost["usd"])
        summary = {}
        if done:
            dirs = [Path(u["results_dir"]) for u in done]
            if done[0]["kind"] == "robolab":
                tasks, per_task = shards.robolab_tasks(config, plan["vla"])
                summary = shards.merge_robolab(dirs, dest, tasks, per_task, meta)
            else:
                summary = shards.merge_tasks(dirs, dest, meta)
        n_eps = summary.get("n_episodes") or 0
        cost["usd_per_episode"] = round(cost["usd"] / n_eps, 4) if n_eps and cost["usd"] else None
        row = dict(run=r, merged_dir=str(dest) if done else "", complete=not missing, missing_shards=missing,
                   n_episodes=summary.get("n_episodes"), n_success=summary.get("n_success"),
                   success_rate=summary.get("success_rate"), ci95_wilson=summary.get("ci95_wilson"), cost=cost,
                   shards=[dict(token=u["token"], state=u["state"], run_id=u.get("run_id", ""),
                                attempts=[dict(region=a["region"], instance_type=a["instance_type"], spot=a["spot"],
                                               started=a["started"], ended=a.get("ended", ""), rc=a.get("rc"),
                                               reason=a.get("reason", ""), cost=a.get("cost"))
                                          for a in u["attempts"]]) for u in us])
        if done:
            for name in ("robolab_results.json", "summary.json"):
                p = dest / name
                if p.is_file():
                    data = json.loads(p.read_text())
                    data["cost"] = cost
                    p.write_text(json.dumps(data, indent=1))
            with open(dest / "fanout-cost.json", "w") as f:
                json.dump(row, f, indent=1)
        out["runs"].append(row)
    (Path(run_dir) / RESULT_FILE).write_text(json.dumps(out, indent=1))
    return out


# --- AWS backend -----------------------------------------------------------------------------------------

class AwsBackend:
    """Real I/O: deploy.py subprocesses (detached, so they survive the scheduler), CloudFormation, EC2, S3."""

    def __init__(self, run_dir: Path):
        import boto3  # noqa: PLC0415

        self.boto3 = boto3
        self.run_dir = Path(run_dir)
        self._account = None

    def now(self) -> dt.datetime:
        return utcnow()

    def launch(self, unit: dict, a: dict) -> int:
        Path(a["log"]).parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        # concurrent `npx cdk` calls sharing one npm cache race on the same _npx entry (bench.child_env)
        env["npm_config_cache"] = str(self.run_dir / "npm-cache" / unit["token"])
        # the receipt (<rc file>.pid) is written by the child itself before deploy.py starts, so a scheduler that
        # dies between spawn and save finds the process on resume (Scheduler.reconcile)
        cmd = (f"echo $$ > {shlex.quote(a['rc_file'] + '.pid')}; {shlex.join(a['argv'])} > {shlex.quote(a['log'])} 2>&1; "
               f"echo $? > {shlex.quote(a['rc_file'])}")
        p = subprocess.Popen(["bash", "-c", cmd], cwd=str(BASE_DIR), env=env, start_new_session=True,  # nosec B603,B607
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return p.pid

    def receipt(self, a: dict):
        """pid from the attempt's receipt file when that process is still alive, else None."""
        p = Path(a["rc_file"] + ".pid")
        try:
            pid = int(p.read_text().strip())
            os.kill(pid, 0)
            return pid
        except (OSError, ValueError):
            return None

    def poll(self, a: dict):
        rcf = Path(a["rc_file"])
        if rcf.is_file() and rcf.read_text().strip():
            try:
                return int(rcf.read_text().strip())
            except ValueError:
                return 1
        try:
            os.kill(int(a["pid"]), 0)
            return None
        except (OSError, TypeError, ValueError):
            pass
        # deploy.py is gone without an rc (scheduler host rebooted): CloudFormation has the answer
        status = self.stack_status(a.get("stack", ""), a["region"])
        return {"CREATE_COMPLETE": 0, "UPDATE_COMPLETE": 0}.get(status, None if status and "IN_PROGRESS" in status
                                                                 else 1)

    def stack_status(self, stack: str, region: str) -> str:
        if not stack:
            return ""
        try:
            return self.boto3.client("cloudformation", region_name=region).describe_stacks(
                StackName=stack)["Stacks"][0]["StackStatus"]
        except Exception:  # noqa: BLE001 - absent stack
            return ""

    def kill(self, a: dict) -> None:
        try:
            os.killpg(int(a["pid"]), signal.SIGTERM)
        except (OSError, TypeError, ValueError):
            pass

    def log_tail(self, a: dict, n: int = 400_000) -> str:
        p = Path(a["log"])
        return p.read_text(errors="replace")[-n:] if p.is_file() else ""

    def destroy(self, stack: str, region: str) -> int:
        """Delete the compute stack (results bucket RETAINed). 0 = gone."""
        cfn = self.boto3.client("cloudformation", region_name=region)
        if not self.stack_status(stack, region):
            return 0
        try:
            cfn.delete_stack(StackName=stack)
            cfn.get_waiter("stack_delete_complete").wait(StackName=stack, WaiterConfig={"Delay": 15, "MaxAttempts": 160})
            return 0
        except Exception as e:  # noqa: BLE001 - reported in the state, the run goes on
            print(f"[fanout] [warn] delete {stack} in {region}: {e}", file=sys.stderr)
            return 1

    def instances(self, region: str, stack: str = "", ids: list | None = None) -> list[dict]:
        """Instances tagged StackName=<stack> (discovery), or exactly the given instance ids."""
        ec2 = self.boto3.client("ec2", region_name=region)
        out = []
        try:
            kw = {"InstanceIds": list(ids)} if ids else {"Filters": [{"Name": "tag:StackName", "Values": [stack]}]}
            pages = ec2.get_paginator("describe_instances").paginate(**kw)
            for page in pages:
                for res in page["Reservations"]:
                    for i in res["Instances"]:
                        state = i["State"]["Name"]
                        end = shards.termination_time(i.get("StateTransitionReason", "")) \
                            if state in ("shutting-down", "terminated") else None
                        reason = (i.get("StateReason") or {}).get("Code", "")
                        out.append(dict(instance_id=i["InstanceId"], instance_type=i["InstanceType"], region=region,
                                        az=i["Placement"]["AvailabilityZone"], launch=iso(i["LaunchTime"]),
                                        end=iso(end) if end else "", state=state,
                                        market="spot" if i.get("InstanceLifecycle") == "spot" else "on-demand",
                                        interrupted=reason == "Server.SpotInstanceTermination"))
        except Exception as e:  # noqa: BLE001
            print(f"[fanout] [warn] describe instances of {stack} in {region}: {e}", file=sys.stderr)
        return out

    def price(self, itype: str, region: str, market: str = "on-demand", az: str = "", start: str = "", end: str = ""):
        import leaderboard  # noqa: PLC0415

        od = leaderboard.price_per_hour(itype, region)
        if market == "spot" and az:
            try:
                t0, t1 = shards.parse_time(start), shards.parse_time(end)
                ec2 = self.boto3.client("ec2", region_name=region)
                # one day before launch so the price in effect at launch is included; every page
                points = []
                for page in ec2.get_paginator("describe_spot_price_history").paginate(
                        InstanceTypes=[itype], AvailabilityZone=az, ProductDescriptions=["Linux/UNIX"],
                        StartTime=t0 - dt.timedelta(days=1), EndTime=t1):
                    points += [(h["Timestamp"], float(h["SpotPrice"])) for h in page["SpotPriceHistory"]]
                got = shards.spot_cost(points, t0, t1)
                if got:
                    usd, rate = got
                    return rate, (f"EC2 Spot price history {az}, integrated over the instance lifetime "
                                  f"({len(points)} point(s), USD {usd:.4f})")
            except Exception:  # noqa: BLE001 - fall back to On-Demand as the upper bound
                pass
            return od, f"On-Demand upper bound (no Spot history), leaderboard.PRICES {leaderboard.PRICES_CHECKED}"
        if od is None:
            return None, f"no price for {itype} in {region} in leaderboard.PRICES"
        return od, f"On-Demand, AWS Price List API via leaderboard.PRICES (checked {leaderboard.PRICES_CHECKED})"

    def _account_id(self, region: str) -> str:
        if not self._account:
            self._account = self.boto3.client("sts", region_name=region).get_caller_identity()["Account"]
        return self._account

    def collect(self, u: dict, a: dict, dest: Path):
        """Download the unit's run (newest RUN_ID started after the attempt) -> (dir, run_id) or None."""
        import yaml  # noqa: PLC0415

        region = a["region"]
        try:
            outputs = {o["OutputKey"]: o["OutputValue"] for o in self.boto3.client(
                "cloudformation", region_name=region).describe_stacks(StackName=u["stack"])["Stacks"][0].get("Outputs", [])}
        except Exception:  # noqa: BLE001
            outputs = {}
        prefix = (yaml.safe_load((BASE_DIR / "simulator-config.yaml").read_text()).get("deployment") or {}).get(
            "s3_results_prefix", "vla-sim-results")
        bucket = outputs.get("S3BucketName") or shards.bucket_name(prefix, u["stack"], region, self._account_id(region),
                                                                    u["token"])
        s3 = self.boto3.client("s3", region_name=region)
        runs = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Delimiter="/"):
            runs += [p["Prefix"].rstrip("/") for p in page.get("CommonPrefixes", [])]
        t0 = shards.parse_time(a["started"]) - dt.timedelta(minutes=5)
        cands = []
        for r in runs:
            try:
                t = dt.datetime.strptime(r, "%Y-%m-%d-%H%M").replace(tzinfo=dt.timezone.utc)
            except ValueError:
                continue
            if t >= t0:
                cands.append(r)
        if not cands:
            return None
        run_id = sorted(cands)[-1]
        dest = Path(dest)
        n = 0
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{run_id}/"):
            for o in page.get("Contents", []):
                rel = o["Key"][len(run_id) + 1:]
                keep = (rel.endswith("summary.txt") or rel.endswith("robolab_results.json")
                        or rel.endswith("episode_results.jsonl") or rel.endswith("protocol.yaml"))
                if not keep or "/videos/" in rel:
                    continue
                # RoboLab: <run>/task-0/... is the shard dir; other targets keep task-N/ under the shard dir
                target = dest / (rel[len("task-0/"):] if u["kind"] == "robolab" and rel.startswith("task-0/") else rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, o["Key"], str(target))
                n += 1
        (dest / "source.json").write_text(json.dumps(dict(bucket=bucket, region=region, run_id=run_id, objects=n)))
        return dest, run_id


# --- CLI -------------------------------------------------------------------------------------------------

def _csv(v: str) -> list[str]:
    return [x.strip() for x in (v or "").split(",") if x.strip()]


def parse_args(argv):
    p = argparse.ArgumentParser(prog="vlasim deploy <target> --shards N", allow_abbrev=False,
                                description="Run one benchmark as N shard stacks (docs/shards.md).")
    p.add_argument("target", nargs="?", help="target (same as --vla)")
    p.add_argument("--vla", help="target")
    p.add_argument("--shards", type=int, required=True, metavar="N", help="number of shards (1..256)")
    p.add_argument("--runs", default="1", metavar="R|auto",
                   help="independent runs of the whole benchmark, each split into N shards (R x N stacks); "
                        "auto = the target's --repeats auto (2 for RoboLab). Default 1")
    p.add_argument("--regions", default="", metavar="LIST", help="comma-separated regions (default: config region)")
    p.add_argument("--instance-types", default="", metavar="LIST",
                   help="comma-separated instance types, tried in order (default: the target's preferred list)")
    p.add_argument("--max-instances", type=int, default=0, metavar="M",
                   help="at most M shard stacks alive at once (default: N x R)")
    p.add_argument("--spot", action="store_true", help="Spot first; an interruption re-runs only that shard")
    p.add_argument("--auto-destroy", action="store_true",
                   help="delete each shard stack when its run is done (results bucket kept)")
    p.add_argument("--max-usd", type=float, default=None, help="stop new launches once the spend reaches this")
    p.add_argument("--robolab-variant", default="", metavar="NAME",
                   help="robolab-*: model.variant for every shard (e.g. full-shard = RoboLab-120 on one GPU per shard)")
    p.add_argument("--robotwin-protocol", default="subset", help="RoboTwin targets: protocol to split")
    p.add_argument("--run-dir", default="", help="state + results directory (default fanout/<target>-<variant>-<N>x<R>)")
    p.add_argument("--gallery", action="store_true",
                   help="publish shard videos to the video history gallery (default off: 2N stacks would race on it)")
    p.add_argument("--dry-run", action="store_true", help="print the plan and the deploy commands; no AWS call")
    p.add_argument("--no-render", action="store_true", help="--dry-run: skip generate.py + bash -n per shard")
    p.add_argument("--poll", type=int, default=DEFAULT_POLL_S, help="seconds between scheduler steps")
    p.add_argument("--task-set", default="", metavar="NAME",
                   help="split task_sets.NAME of models/<vla>.yaml instead of its tasks: list")
    p.add_argument("--tasks-file", default="", metavar="PATH",
                   help="split the tasks: list of this file (shape of one task_sets entry) instead of the model's")
    p.add_argument("--email", "-e", default="", help="notification e-mail (passed to every deploy.py)")
    p.add_argument("--seed", default="", help="--seed of every shard (default: the model's seeds)")
    p.add_argument("--skip-doctor", action="store_true", help=argparse.SUPPRESS)
    args, rest = p.parse_known_args(argv)
    args.vla = args.vla or args.target
    if not args.vla:
        p.error("give the target: vlasim deploy <target> --shards N")
    if args.target and args.vla != args.target:
        p.error("give the target once (positional or --vla)")
    return p, args, rest


def load_config(vla: str, variant: str = "", task_set: str = "", tasks_file: str = "") -> dict:
    """Merged model config with --robolab-variant / --task-set / --tasks-file applied before the split. The
    selected set is kept under `_shard_set` (name, description, model overrides) for the per-shard files."""
    import yaml  # noqa: PLC0415

    import generate  # noqa: PLC0415
    import task_sets  # noqa: PLC0415

    cfg = generate._load_merged_config(BASE_DIR / "simulator-config.yaml", vla)
    if variant:
        if variant not in (cfg.get("variants") or {}):
            raise ValueError(f"--robolab-variant {variant}: models/{vla}.yaml variants are "
                             f"{', '.join(sorted(cfg.get('variants') or {})) or 'none'}")
        cfg["model"] = dict(cfg.get("model") or {}, variant=variant)
    if task_set and tasks_file:
        raise ValueError("--tasks-file and --task-set both replace the tasks: list; give one")
    if task_set or tasks_file:
        if shards.kind(vla, cfg) != "tasks-file":
            raise ValueError(f"--task-set / --tasks-file with --shards apply to targets with a tasks: list; {vla} is "
                             f"split as {shards.kind(vla, cfg)}")
        if tasks_file:
            try:
                spec = yaml.safe_load(Path(tasks_file).read_text())
            except (OSError, yaml.YAMLError) as e:
                raise ValueError(f"--tasks-file: {e}") from None
            cfg = task_sets.apply_file(cfg, spec, tasks_file)
            overrides = dict((spec or {}).get("model") or {})
            desc = str((spec or {}).get("description", ""))
        else:
            cfg = task_sets.apply(cfg, task_set, vla)
            st = (cfg.get("task_sets") or {}).get(task_set) or {}
            overrides, desc = dict(st.get("model") or {}), str(st.get("description", ""))
        cfg["_shard_set"] = dict(name=cfg["task_set"], description=desc, model=overrides)
    return cfg


# deploy.py flags the scheduler sets per shard: a passthrough copy would override them (argparse keeps the last)
SCHEDULER_OWNED = ("--vla", "--shard", "--shard-run", "--region", "-r", "--instance-types", "--repeats",
                   "--fanout-id", "--robolab-variant", "--tasks-file", "--task-set", "--robotwin-shard",
                   "--robotwin-protocol", "--spot", "--judge", "--run", "--bridge", "--collect", "--libero-variant",
                   "--libero-objects")
# passthrough flags that do not change what is evaluated (excluded from the resume fingerprint)
OPERATIONAL_FLAGS = {"--email": 1, "-e": 1, "--no-gallery": 0, "--update-gallery": 0}


def owned_flags(rest: list[str]) -> list[str]:
    return [t for t in rest if any(t == f or t.startswith(f + "=") for f in SCHEDULER_OWNED)]


def benchmark_passthrough(passthrough: list[str]) -> list[str]:
    out, skip = [], 0
    for t in passthrough:
        if skip:
            skip -= 1
            continue
        flag = t.split("=", 1)[0]
        if flag in OPERATIONAL_FLAGS:
            skip = 0 if "=" in t else OPERATIONAL_FLAGS[flag]
            continue
        out.append(t)
    return out


def fingerprint(vla: str, config: dict, parts: list[dict], plan: dict) -> dict:
    """What makes the merged result one benchmark: effective task units per shard (and the shard task entries),
    episodes, seed and every benchmark-affecting option. A resume must match it; operational settings
    (regions, instance types, --max-instances, --spot, --auto-destroy, --max-usd, e-mail) may change."""
    import hashlib  # noqa: PLC0415

    model = config.get("model") or {}
    var = (config.get("variants") or {}).get(str(model.get("variant", ""))) or {}
    doc = dict(vla=vla, shards=plan["shards"], runs=plan["runs"], variant=plan["variant"],
               robotwin_protocol=plan["robotwin_protocol"], model=model, variant_config=var,
               protocol=(config.get("protocols") or {}).get(var.get("protocol")) or {},
               eval=config.get("eval") or {}, task_set=config.get("_shard_set") or {},
               passthrough=benchmark_passthrough(plan["passthrough"]),
               parts=[dict(k=p["k"], units=p["units"], episodes=p["episodes"], tasks=p.get("tasks")) for p in parts])
    blob = json.dumps(doc, sort_keys=True, default=str)
    return dict(sha=hashlib.sha256(blob.encode()).hexdigest(), doc=json.loads(blob))


def fingerprint_diff(a: dict, b: dict) -> list[str]:
    da, db = (a or {}).get("doc") or {}, (b or {}).get("doc") or {}
    return sorted(k for k in set(da) | set(db) if da.get(k) != db.get(k))


def default_types(vla: str, config: dict) -> list[str]:
    """The target's instance list, same rule as resolveInstanceCfg() in the CDK stack (variant first)."""
    inst = dict(config.get("instance") or {})
    var = (config.get("variants") or {}).get(str((config.get("model") or {}).get("variant", "")))
    if var and var.get("instance"):
        inst.update(var["instance"])
    return list(inst.get("preferred") or [])


def render_check(vla: str, unit: dict, state: dict, outdir: Path) -> str:
    """generate.py for this shard into outdir + bash -n; '' = ok."""
    outdir.mkdir(parents=True, exist_ok=True)
    plan = state["plan"]
    argv = [sys.executable, "generate.py", "--vla", vla, "--output-dir", str(outdir), "--output-suffix", unit["token"],
            "--repeats", "1"]
    if unit["kind"] == "robolab":
        argv += ["--robolab-shard", f"{unit['k']}/{unit['n']}"]
    if plan["variant"]:
        argv += ["--robolab-variant", plan["variant"]]
    if unit["kind"] == "tasks-file":
        argv += ["--tasks-file", unit["tasks_file"]]
    if unit["kind"] == "robotwin":
        argv += ["--robotwin-protocol", plan["robotwin_protocol"], "--robotwin-shard", unit["robotwin_shard"]]
    r = subprocess.run(argv, cwd=str(BASE_DIR), capture_output=True, text=True)  # nosec B603
    if r.returncode:
        lines = (r.stderr or r.stdout).strip().splitlines()
        return lines[-1] if lines else "generate failed"
    b = subprocess.run(["bash", "-n", str(outdir / f"{vla}-{unit['token']}.sh")], capture_output=True, text=True)  # nosec
    return b.stderr.strip() if b.returncode else ""


def write_tasks_files(state: dict, config: dict) -> None:
    import yaml  # noqa: PLC0415

    seen = set()
    for u in state["units"].values():
        if u["kind"] != "tasks-file" or u["tasks_file"] in seen:
            continue
        seen.add(u["tasks_file"])
        part = shards.tasks_file_shard(config["tasks"], u["k"], u["n"])
        f = Path(u["tasks_file"])
        f.parent.mkdir(parents=True, exist_ok=True)
        sset = config.get("_shard_set") or {}
        tag = f"s{u['k']:02d}of{u['n']:02d}"
        spec = dict(name=f"{sset['name']}-{tag}"[:64] if sset.get("name") else f"shard-{tag}",
                    description=f"vlasim deploy --shards {u['n']}: shard {u['k']}/{u['n']}"
                                + (f" of {sset['name']} ({sset['description']})" if sset.get("name") else ""),
                    tasks=part)
        if sset.get("model"):
            spec["model"] = sset["model"]  # the set's model overrides travel with every shard file
        f.write_text(yaml.safe_dump(spec, sort_keys=False))


def print_plan(state: dict, out=None) -> None:
    out = out or sys.stdout
    plan = state["plan"]
    us = sorted(state["units"].values(), key=lambda u: (u["run"], u["k"]))
    print(f"[fanout] {plan['vla']}{' variant ' + plan['variant'] if plan['variant'] else ''}: {plan['shards']} shard(s)"
          f" x {plan['runs']} run(s) = {len(us)} stack(s), at most {plan['max_instances']} alive", file=out)
    print(f"[fanout] regions {','.join(plan['regions'])}; instance types {','.join(plan['instance_types'])}"
          f"{'; Spot first' if plan['spot'] else ''}{'; auto-destroy' if plan['auto_destroy'] else ''}"
          f"{'; max USD ' + str(plan['max_usd']) if plan['max_usd'] is not None else ''}", file=out)
    eps = [u["episodes"] for u in us if u["run"] == 1]
    print(f"[fanout] split: {sum(u['n_units'] for u in us if u['run'] == 1)} task unit(s), "
          f"{min(u['n_units'] for u in us)}-{max(u['n_units'] for u in us)} per shard"
          + (f", {min(eps)}-{max(eps)} episodes per shard, {sum(eps)} per run" if all(eps) else ""), file=out)


def main(argv=None) -> int:
    p, args, rest = parse_args(sys.argv[1:] if argv is None else argv)
    import yaml  # noqa: PLC0415

    sim = yaml.safe_load((BASE_DIR / "simulator-config.yaml").read_text()).get("deployment") or {}
    bad = owned_flags(rest)
    if bad:
        print(f"[error] {' '.join(bad)}: set per shard by the scheduler; use the fanout flags (--regions, "
              "--instance-types, --robolab-variant, --task-set, --tasks-file, --spot) instead", file=sys.stderr)
        return 2
    try:
        config = load_config(args.vla, args.robolab_variant, args.task_set, args.tasks_file)
        if args.runs == "auto":
            import repeatability  # noqa: PLC0415

            runs, _ = repeatability.resolve_repeats(args.vla, "auto")
        else:
            runs = int(args.runs)
        if not 1 <= runs <= shards.MAX_RUNS:
            raise ValueError(f"--runs must be 1..{shards.MAX_RUNS}")
        if not 1 <= args.shards <= shards.MAX_SHARDS:
            raise ValueError(f"--shards must be 1..{shards.MAX_SHARDS}")
        regions = _csv(args.regions) or [sim.get("region", "us-east-1")]
        types = _csv(args.instance_types) or default_types(args.vla, config)
        variant = args.robolab_variant
        run_dir = Path(args.run_dir or BASE_DIR / "fanout" /
                       f"{args.vla}{'-' + variant if variant else ''}-{args.shards}x{runs}").resolve()
        passthrough = list(rest) + ([] if args.gallery else ["--no-gallery"])
        passthrough += (["--email", args.email] if args.email else []) + (["--seed", args.seed] if args.seed else [])
        state = build_plan(args.vla, config, args.shards, runs, regions, types,
                           max_instances=args.max_instances or args.shards * runs, spot=args.spot,
                           auto_destroy=args.auto_destroy, variant=variant, robotwin_protocol=args.robotwin_protocol,
                           run_dir=run_dir, passthrough=passthrough, max_usd=args.max_usd)
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2
    bad_region = [r for r in regions if not re.match(r"^[a-z]{2,3}(-[a-z]+)+-\d+$", r)]
    if bad_region:
        print(f"[error] invalid region(s): {', '.join(bad_region)}", file=sys.stderr)
        return 2
    acct = "123456789012"
    prefix = sim.get("s3_results_prefix", "vla-sim-results")
    for u in state["units"].values():
        for c in u["cands"]:
            try:
                shards.bucket_name(prefix, u["stack"], c["region"], acct, u["token"])
            except ValueError as e:
                print(f"[error] {e}", file=sys.stderr)
                return 2

    prev = load_state(run_dir)
    if prev and plan_key(prev["plan"]) != plan_key(state["plan"]):
        print(f"[error] {run_dir}/{STATE_FILE} belongs to another plan ({plan_key(prev['plan'])}); "
              "use another --run-dir or remove that directory", file=sys.stderr)
        return 2
    if prev and (prev.get("fingerprint") or {}).get("sha") != state["fingerprint"]["sha"]:
        diff = fingerprint_diff(prev.get("fingerprint"), state["fingerprint"]) or ["fingerprint missing"]
        print(f"[error] {run_dir}/{STATE_FILE}: the benchmark changed since the shards there ran ({', '.join(diff)}); "
              "a resume must evaluate the same task units, episodes, seed and options. Use another --run-dir.",
              file=sys.stderr)
        return 2
    if prev:
        # resume: keep units, attempts, caps; take the flags that may change (regions, types, M, spot, ...)
        for tok, u in state["units"].items():
            old = prev["units"].get(tok)
            if old:
                if old["state"] == "queued" and old["cands"] != u["cands"]:
                    old["cands"], old["cand"] = u["cands"], 0
                state["units"][tok] = old
        state = dict(prev, plan=state["plan"], parts=state["parts"], units=state["units"],
                     fingerprint=state["fingerprint"])

    print_plan(state)
    write_tasks_files(state, config)
    if args.dry_run:
        bad = 0
        for u in sorted(state["units"].values(), key=lambda u: (u["run"], u["k"])):
            c = u["cands"][u["cand"]]
            spot = state["plan"]["spot"] and not u.get("no_spot")
            argv = ["python", *deploy_argv(state, u, c, spot, fanout_id_of(run_dir))[1:]]
            what = f"{u['n_units']} task(s)" + (f", {u['episodes']} ep" if u["episodes"] else "")
            print(f"\n# {u['token']}  stack {u['stack']}  {c['region']} {c['instance_type']}  [{u['state']}]  {what}")
            if u["kind"] == "robolab":
                print(f"#   tasks: {' '.join(u['units'])}")
            print(f"#   fallback: {' -> '.join(x['region'] + ':' + x['instance_type'] for x in u['cands'])}")
            print(shlex.join(argv))
            if not args.no_render and u["run"] == 1:
                err = render_check(args.vla, u, state, run_dir / "userdata")
                print(f"#   render + bash -n: {'ok' if not err else 'FAIL: ' + err}")
                bad += bool(err)
        print(f"\n[fanout] dry run: nothing deployed. State dir: {run_dir}. Without --dry-run the scheduler runs "
              f"until every shard is done (Ctrl-C is safe: the deploys keep running; re-run to resume; "
              f"touch {run_dir}/STOP to stop new launches).")
        return 1 if bad else 0

    run_dir.mkdir(parents=True, exist_ok=True)
    sched = Scheduler(run_dir, state, AwsBackend(run_dir))
    try:
        while True:
            status = sched.step()
            if status != "running":
                break
            time.sleep(args.poll)
    except KeyboardInterrupt:
        save_state(run_dir, sched.state)
        print(f"\n[fanout] interrupted; the deploys keep running. Re-run the same command to resume ({run_dir}).")
        return 130
    if status == "stopped":
        print(f"[fanout] stopped ({sched.state.get('stopped')}); remove {run_dir}/STOP and re-run to continue.")
        return 3
    sched.refresh_costs()
    save_state(run_dir, sched.state)
    res = merge(run_dir, sched.state, config)
    for r in res["runs"]:
        c = r["cost"]
        print(f"[fanout] run {r['run']}: {r['n_success']}/{r['n_episodes']} "
              f"({'complete' if r['complete'] else 'missing ' + ','.join(r['missing_shards'])}), "
              f"wall {c['wall_clock_s']} s, {c['instance_hours']} instance-h, USD {c['usd']}"
              f" -> {r['merged_dir']}")
    print(f"[fanout] {run_dir / RESULT_FILE}")
    return 0 if all(r["complete"] for r in res["runs"]) else 1


if __name__ == "__main__":
    sys.exit(main())
