"""vlasim deploy <target> --shards N (shards.py, fanout.py, docs/shards.md).

Offline: the split, the stack / bucket names, the scheduler state machine against a fake backend (capacity ->
next region, quota caps, STOP, resume, Spot interruption, auto-destroy), the merge of shard results and the cost
accounting. Nothing is launched; the dry run renders user data through generate.py only.
"""

import copy
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

yaml = pytest.importorskip("yaml")
pytest.importorskip("jinja2")
import fanout  # noqa: E402
import generate  # noqa: E402
import shards  # noqa: E402

REPORT = ROOT / "assets" / "robolab" / "robolab_report.py"


def _cfg(vla, variant=None):
    cfg = copy.deepcopy(generate._load_merged_config(ROOT / "simulator-config.yaml", vla))
    if variant:
        cfg["model"]["variant"] = variant
    return cfg


# --- split -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 7, 16, 120])
def test_split_partitions_and_is_stable(n):
    items = [f"t{i:03d}" for i in range(120)]
    parts = shards.split(items, n)
    flat = [x for p in parts for x in p]
    assert sorted(flat) == items and len(flat) == len(set(flat))  # union = full set, disjoint
    assert parts == shards.split(list(items), n)  # stable
    assert parts[0][0] == "t000" and all(p[0] == items[k] for k, p in enumerate(parts))  # shard k starts at unit k
    assert max(map(len, parts)) - min(map(len, parts)) <= 1


def test_split_refuses_more_shards_than_units():
    with pytest.raises(ValueError, match="cannot be split into 5"):
        shards.split([1, 2, 3], 5)


def test_robolab_120_list_matches_the_pinned_commit():
    commit, names = shards.robolab_120()
    assert len(names) == 120 == len(set(names)) and names == sorted(names)
    for vla in shards.ROBOLAB_TARGETS:
        assert _cfg(vla)["model"]["robolab_commit"] == commit


@pytest.mark.parametrize("vla,variant", [("robolab-pi05", "full-shard"), ("robolab-cosmos3", "edge-full-shard"),
                                         ("robolab-pi05", "full"), ("robolab-pi05", "subset")])
def test_robolab_shards_cover_the_protocol(vla, variant):
    cfg = _cfg(vla, variant)
    tasks, per_task = shards.robolab_tasks(cfg, vla)
    n = 16 if len(tasks) >= 16 else 4
    parts = [shards.robolab_shard(cfg, vla, k, n) for k in range(1, n + 1)]
    assert sorted(t for p in parts for t in p["tasks"]) == sorted(tasks)
    assert sorted(i for p in parts for i in p["task_ids"]) == list(range(len(tasks)))
    assert all(p["tasks"] == [tasks[i] for i in p["task_ids"]] for p in parts)
    assert per_task == (10 if "full" in variant else 5)


def test_robolab_list_out_of_date_is_an_error(monkeypatch):
    cfg = _cfg("robolab-pi05", "full-shard")
    cfg["model"]["robolab_commit"] = "0" * 40
    with pytest.raises(ValueError, match="regenerate the file"):
        shards.robolab_tasks(cfg, "robolab-pi05")


def test_tasks_file_shards_regroup_task_ids():
    tasks = [dict(suite="libero_spatial", task_ids=[0, 1, 2, 3], n_episodes=5),
             dict(suite="libero_goal", task_ids=[7, 8], n_episodes=5), dict(env_name="x/OtherTask", n_episodes=3)]
    parts = [shards.tasks_file_shard(tasks, k, 3) for k in (1, 2, 3)]
    units = sorted((e.get("suite") or e["env_name"], t) for p in parts for e in p for t in e.get("task_ids") or [None])
    assert units == sorted([("libero_spatial", i) for i in range(4)] + [("libero_goal", 7), ("libero_goal", 8),
                                                                        ("x/OtherTask", None)])
    # units in order: spatial#0..3, goal#7, goal#8, OtherTask -> shard 1 = units 0, 3, 6
    assert parts[0] == [dict(suite="libero_spatial", task_ids=[0, 3], n_episodes=5), dict(env_name="x/OtherTask", n_episodes=3)]
    assert parts[1] == [dict(suite="libero_spatial", task_ids=[1]), dict(suite="libero_goal", task_ids=[7])] or \
        parts[1] == [dict(suite="libero_spatial", task_ids=[1], n_episodes=5), dict(suite="libero_goal", task_ids=[7],
                                                                                   n_episodes=5)]
    assert tasks[0]["task_ids"] == [0, 1, 2, 3]  # input untouched


def test_split_plan_kinds():
    p = shards.split_plan("smolvla", dict(tasks=[dict(suite="libero_spatial", task_ids=[0, 1], n_episodes=5)]), 2)
    assert [x["kind"] for x in p] == ["tasks-file"] * 2 and [x["episodes"] for x in p] == [5, 5]
    rt = shards.split_plan("smolvla-robotwin", _cfg("smolvla-robotwin"), 5)
    assert [x["robotwin_shard"] for x in rt] == [f"{k}/5" for k in range(1, 6)]
    with pytest.raises(ValueError, match="cannot be sharded"):
        shards.split_plan("openvla-oft", _cfg("openvla-oft"), 2)  # tasks by suite (dict): nothing to enumerate


# --- names -------------------------------------------------------------------------------------------------

def test_tokens_and_stack_names():
    assert shards.shard_token(3, 16) == "s03of16" and shards.shard_token(3, 16, run=2) == "r2-s03of16"
    assert shards.shard_token(7, 120) == "s007of120"
    assert shards.parse_token("r2-s03of16") == (2, 3, 16) and shards.parse_token("s03of16") == (None, 3, 16)
    assert shards.stack_name("RoboLab-Pi05-Demo", "s03of16") == "RoboLab-Pi05-Demo-s03of16"
    assert shards.parse_shard("3/16") == (3, 16)
    for bad in ("0/16", "17/16", "3", "a/b"):
        with pytest.raises(ValueError):
            shards.parse_shard(bad)


# The same literals are asserted in cdk/test/cdk.test.ts (resultsBucketName) so the two rules cannot drift.
BUCKETS = [
    ("RoboLab-Pi05-Demo", "us-east-1", "", "vla-sim-results-robolab-pi05-demo-us-east-1-123456789012"),
    ("RoboLab-Pi05-Demo-s03of16", "us-east-1", "s03of16", "vla-sim-results-robolab-pi05-s03of16-us-east-1-123456789012"),
    ("RoboLab-Cosmos3-Demo-r2-s03of16", "eu-central-1", "r2-s03of16",
     "vla-sim-results-robolab-co-r2-s03of16-eu-central-1-123456789012"),
    ("OpenArm-Lift-ACT-Demo", "ap-northeast-2", "", "vla-sim-results-openarm-lift-act-ap-northeast-2-123456789012"),
]


@pytest.mark.parametrize("stack,region,token,want", BUCKETS)
def test_bucket_names(stack, region, token, want):
    got = shards.bucket_name("vla-sim-results", stack, region, "123456789012", token)
    assert got == want and len(got) <= 63


def test_every_shard_bucket_fits():
    from destroy import STACK_NAMES

    for base in STACK_NAMES.values():
        for region in ("us-east-1", "ap-northeast-2", "eu-central-1"):
            for tok in (shards.shard_token(16, 16), shards.shard_token(16, 16, 20), shards.shard_token(256, 256, 20)):
                name = shards.bucket_name("vla-sim-results", shards.stack_name(base, tok), region, "123456789012", tok)
                assert len(name) <= 63 and tok in name


def test_failure_classes():
    assert shards.classify_failure("Received response status [FAILED] from custom resource. Message returned: "
                                   "No capacity available for any instance type: ['g6e.2xlarge/us-east-1a']") == "capacity"
    assert shards.classify_failure("An error occurred (VcpuLimitExceeded) when calling the RunInstances") == "quota"
    assert shards.classify_failure("npm ERR! code ENOTFOUND") == "other"


# --- scheduler ---------------------------------------------------------------------------------------------

T0 = dt.datetime(2026, 11, 8, 0, 0, tzinfo=dt.timezone.utc)


class FakeBackend:
    """script[(token, attempt n)] = dict(rc=0|1, log=str, after=steps, interrupt=bool); default rc 0 after 1 step.

    Every attempt launches one instance (id i-<token>-<n>) that stays visible under its stack tag after it
    terminated, as EC2 does, so attempts on the same stack see each other's instances by tag."""

    def __init__(self, script=None, tmp=None, hours=None):
        self.script = script or {}
        self.t = T0
        self.step_no = 0
        self.launched, self.destroyed, self.killed = [], [], []
        self.records = {}
        self.hours = hours
        self.receipts, self.stacks = {}, {}
        self.destroy_fail = {}   # stack -> number of failing destroy calls left
        self.collect_fail = {}   # token -> number of empty collections left

    def now(self):
        return self.t

    def tick(self, minutes=30):
        self.t += dt.timedelta(minutes=minutes)
        self.step_no += 1

    def _s(self, a):
        return self.script.get((a["token"], a["n"]), {})

    def launch(self, u, a):
        a["token"], a["step"] = u["token"], self.step_no
        self.launched.append((u["token"], a["region"], a["instance_type"], a["spot"]))
        start = shards.parse_time(a["started"])
        iid = f"i-{u['token']}-{a['n']}"
        self.records[iid] = dict(instance_id=iid, stack=u["stack"], instance_type=a["instance_type"], region=a["region"],
                                 az=a["region"] + "a", launch=fanout.iso(start),
                                 end=fanout.iso(start + dt.timedelta(hours=self.hours)) if self.hours else "",
                                 market="spot" if a["spot"] else "on-demand",
                                 interrupted=bool(self._s(a).get("interrupt")))
        return 1000 + len(self.launched)

    def receipt(self, a):
        return self.receipts.get(a["token"] if "token" in a else a["stack"])

    def stack_status(self, stack, region):
        return self.stacks.get(stack, "")

    def poll(self, a):
        s = self._s(a)
        if self.step_no - a["step"] < s.get("after", 1):
            return None
        return s.get("rc", 0)

    def log_tail(self, a):
        return self._s(a).get("log", "")

    def kill(self, a):
        self.killed.append(a["token"])

    def destroy(self, stack, region):
        self.destroyed.append((stack, region))
        if self.destroy_fail.get(stack, 0) > 0:
            self.destroy_fail[stack] -= 1
            return 1
        return 0

    def instances(self, region, stack="", ids=None):
        out = [dict(r) for r in self.records.values() if r["region"] == region
               and ((ids and r["instance_id"] in ids) or (not ids and r["stack"] == stack))]
        for r in out:
            r.pop("stack")
        return out

    def price(self, itype, region, market="on-demand", az="", start="", end=""):
        return {"g6e.2xlarge": 2.0, "g6e.4xlarge": 3.0}.get(itype), "test table"

    def collect(self, u, a, dest):
        if self.collect_fail.get(u["token"], 0) > 0:
            self.collect_fail[u["token"]] -= 1
            return None
        raw = dest / "raw" / "shard-0"
        raw.mkdir(parents=True, exist_ok=True)
        (dest / "summary.txt").write_text("suite: RoboLab/RoboLab-120\n")
        (raw / "episode_results.jsonl").write_text('{"env_name": "T", "success": true}\n')
        return dest, "2026-11-08-0000"


class InstanceBackend(FakeBackend):
    """Instances terminate `hours` after their attempt started."""

    def __init__(self, *a, hours=2.0, **k):
        super().__init__(*a, hours=hours, **k)


def _plan(tmp_path, n=4, runs=1, regions=("us-east-1", "us-west-2"), types=("g6e.2xlarge",), m=2, **kw):
    cfg = _cfg("robolab-pi05", "full-shard")
    return fanout.build_plan("robolab-pi05", cfg, n, runs, list(regions), list(types), max_instances=m,
                             variant="full-shard", run_dir=tmp_path, **kw)


def _run(sched, b, steps=50):
    status = "running"
    for _ in range(steps):
        status = sched.step()
        if status != "running":
            return status
        b.tick()
    return status


def test_scheduler_runs_everything_with_at_most_m(tmp_path):
    b = FakeBackend()
    s = fanout.Scheduler(tmp_path, _plan(tmp_path, n=5, m=2), b, log=lambda *_: None)
    peak = 0
    for _ in range(20):
        status = s.step()
        peak = max(peak, len(s.units("running")))
        if status == "finished":
            break
        b.tick()
    assert status == "finished" and peak == 2 and len(b.launched) == 5
    assert all(u["state"] == "done" for u in s.units())
    # shard k starts in region (k-1) % len(regions)
    assert [r for _, r, _, _ in b.launched] == ["us-east-1", "us-west-2", "us-east-1", "us-west-2", "us-east-1"]
    assert json.loads((tmp_path / fanout.STATE_FILE).read_text())["units"]["s01of05"]["state"] == "done"


def test_capacity_moves_to_next_region_then_type_and_caps(tmp_path):
    script = {("s01of04", 1): dict(rc=1, log="No capacity available for any instance type"),
              ("s03of04", 1): dict(rc=1, log="VcpuLimitExceeded")}
    b = FakeBackend(script)
    st = _plan(tmp_path, n=4, m=4, types=("g6e.2xlarge", "g6e.4xlarge"))
    s = fanout.Scheduler(tmp_path, st, b, log=lambda *_: None)
    assert _run(s, b) == "finished"
    tries = [(t, r, i) for t, r, i, _ in b.launched]
    assert tries[:4] == [("s01of04", "us-east-1", "g6e.2xlarge"), ("s02of04", "us-west-2", "g6e.2xlarge"),
                         ("s03of04", "us-east-1", "g6e.2xlarge"), ("s04of04", "us-west-2", "g6e.2xlarge")]
    assert ("s01of04", "us-west-2", "g6e.2xlarge") in tries[4:]  # next region, same type
    assert ("s03of04", "us-west-2", "g6e.2xlarge") in tries[4:]
    assert ("RoboLab-Pi05-Demo-s01of04", "us-east-1") in b.destroyed  # the failed stack is removed
    assert s.state["units"]["s01of04"]["attempts"][0]["reason"] == "capacity"
    assert s.state["units"]["s03of04"]["attempts"][0]["reason"] == "quota"
    assert "us-east-1|g6e.2xlarge" in s.state["caps"]


def test_capacity_everywhere_fails_after_rounds(tmp_path):
    script = {("s01of02", n): dict(rc=1, log="InsufficientInstanceCapacity") for n in range(1, 20)}
    b = FakeBackend(script)
    s = fanout.Scheduler(tmp_path, _plan(tmp_path, n=2, m=2), b, log=lambda *_: None)
    status = _run(s, b, steps=200)
    u = s.state["units"]["s01of02"]
    assert status == "finished" and u["state"] == "failed"
    assert len(u["attempts"]) == fanout.CAPACITY_ROUNDS * 2  # every candidate, every round
    assert s.state["units"]["s02of02"]["state"] == "done"


def test_other_failure_retried_once(tmp_path):
    script = {("s01of02", 1): dict(rc=1, log="boom"), ("s01of02", 2): dict(rc=1, log="boom")}
    b = FakeBackend(script)
    s = fanout.Scheduler(tmp_path, _plan(tmp_path, n=2, m=2), b, log=lambda *_: None)
    _run(s, b)
    u = s.state["units"]["s01of02"]
    assert u["state"] == "failed" and len(u["attempts"]) == 2 and u["attempts"][1]["region"] == "us-west-2"


def test_stop_file_and_resume(tmp_path):
    b = FakeBackend({("s01of04", 1): dict(after=3)})
    st = _plan(tmp_path, n=4, m=1)
    s = fanout.Scheduler(tmp_path, st, b, log=lambda *_: None)
    s.step()  # launches s01
    (tmp_path / "STOP").write_text("")
    b.tick()
    statuses = []
    for _ in range(6):
        statuses.append(s.step())
        b.tick()
    assert statuses[-1] == "stopped" and len(b.launched) == 1 and s.state.get("stopped")
    assert [u["state"] for u in s.units()] == ["done", "queued", "queued", "queued"]
    # resume: a new scheduler on the saved state file (re-running the command) continues where it stopped
    (tmp_path / "STOP").unlink()
    s2 = fanout.Scheduler(tmp_path, fanout.load_state(tmp_path), b, log=lambda *_: None)
    assert _run(s2, b) == "finished"
    assert [t for t, *_ in b.launched] == ["s01of04", "s02of04", "s03of04", "s04of04"]


def test_spot_interruption_reruns_only_that_shard(tmp_path):
    script = {("s01of02", 1): dict(after=5, interrupt=True), ("s01of02", 2): dict(after=5, interrupt=True)}
    b = InstanceBackend(script)
    s = fanout.Scheduler(tmp_path, _plan(tmp_path, n=2, m=2, spot=True), b, log=lambda *_: None)
    assert _run(s, b) == "finished"
    spots = [(t, sp) for t, _, _, sp in b.launched]
    assert spots == [("s01of02", True), ("s02of02", True), ("s01of02", True), ("s01of02", False)]
    assert b.killed == ["s01of02", "s01of02"] and s.state["units"]["s01of02"]["no_spot"]
    assert s.state["units"]["s02of02"]["interruptions"] == 0


def test_auto_destroy_after_each_shard(tmp_path):
    b = FakeBackend()
    s = fanout.Scheduler(tmp_path, _plan(tmp_path, n=2, m=2, auto_destroy=True), b, log=lambda *_: None)
    assert _run(s, b) == "finished"
    assert sorted(b.destroyed) == [("RoboLab-Pi05-Demo-s01of02", "us-east-1"), ("RoboLab-Pi05-Demo-s02of02", "us-west-2")]


def test_max_usd_stops_new_launches(tmp_path):
    b = InstanceBackend(hours=10)
    s = fanout.Scheduler(tmp_path, _plan(tmp_path, n=3, m=1, max_usd=15.0), b, log=lambda *_: None)
    assert _run(s, b) == "stopped"  # first shard: 10 h x $2 = $20 >= $15
    assert len(b.launched) == 1


def test_runs_expand_run_by_shard(tmp_path):
    st = _plan(tmp_path, n=3, runs=2, m=6)
    assert sorted(st["units"]) == ["r1-s01of03", "r1-s02of03", "r1-s03of03", "r2-s01of03", "r2-s02of03", "r2-s03of03"]
    argv = fanout.deploy_argv(st, st["units"]["r2-s03of03"], st["units"]["r2-s03of03"]["cands"][0], False, "x")
    assert argv[1:] [:8] == ["deploy.py", "--vla", "robolab-pi05", "--shard", "3/3", "--shard-run", "2", "--region"]
    assert "--repeats" in argv and argv[argv.index("--repeats") + 1] == "1"


# --- merge + cost ------------------------------------------------------------------------------------------

def _episodes(tasks, per_task, success_every=3):
    out = []
    for t in tasks:
        for e in range(per_task):
            out.append(dict(env_name=t, task_name=t, run=0, env_id=e, success=(e % success_every == 0), score=0.5))
    return out


def _shard_dir(root, token, tasks, per_task):
    d = root / token
    raw = d / "raw" / "shard-0"
    raw.mkdir(parents=True)
    (raw / "episode_results.jsonl").write_text("\n".join(json.dumps(e) for e in _episodes(tasks, per_task)) + "\n")
    (d / "summary.txt").write_text("model: robolab-pi05 (pi0.5-DROID)\ncheckpoint: gs://x\nsuite: RoboLab/RoboLab-120\n"
                                   f"n_episodes: {len(tasks) * per_task}\nshards: 1\ninstance: g6e.2xlarge\n"
                                   f"fanout_shard: {token}\nstack_robolab: 9cce00ca\n")
    return d


def test_merge_equals_a_single_run(tmp_path):
    cfg = _cfg("robolab-pi05", "full-shard")
    tasks, per_task = shards.robolab_tasks(cfg, "robolab-pi05")
    parts = [shards.robolab_shard(cfg, "robolab-pi05", k, 16)["tasks"] for k in range(1, 17)]
    dirs = [_shard_dir(tmp_path / "sh", shards.shard_token(k, 16), p, per_task) for k, p in enumerate(parts, 1)]
    merged = shards.merge_robolab(dirs, tmp_path / "merged", tasks, per_task, dict(fanout_shards=16))
    # one run over all tasks on one instance: the same report script over one episode file
    single = tmp_path / "single" / "run"
    single.mkdir(parents=True)
    (single / "episode_results.jsonl").write_text("\n".join(json.dumps(e) for e in _episodes(tasks, per_task)) + "\n")
    out = tmp_path / "single-upload"
    subprocess.run([sys.executable, str(REPORT), "--run-dir", str(single), "--upload-dir", str(out), "--label",
                    "RoboLab/RoboLab-120", "--tasks", " ".join(tasks), "--episodes-per-task", str(per_task)],
                   check=True, capture_output=True)
    ref = json.loads((out / "robolab_results.json").read_text())
    for key in ("n_episodes", "n_success", "success_rate", "ci95_wilson", "tasks_complete", "n_tasks", "n_expected",
                "per_task", "suite"):
        assert merged[key] == ref[key], key
    assert merged["n_episodes"] == 1200 and merged["tasks_complete"] == 120
    kv = shards.read_kv(tmp_path / "merged" / "summary.txt")
    assert kv["n_episodes"] == "1200" and kv["suite"] == "RoboLab/RoboLab-120" and kv["fanout_shards"] == "16"
    assert "fanout_shard" not in kv and "instance" not in kv and kv["stack_robolab"] == "9cce00ca"


def test_unit_cost_is_hours_times_price():
    price = lambda t, r, m, *a: ({"g6e.2xlarge": 2.24208}.get(t), "table")  # noqa: E731
    c = shards.unit_cost([dict(instance_id="i-1", instance_type="g6e.2xlarge", region="us-east-1",
                               launch="2026-11-08T00:00:00Z", end="2026-11-08T04:30:00Z"),
                          dict(instance_id="i-2", instance_type="p9.huge", region="us-east-1",
                               launch="2026-11-08T00:00:00Z", end="2026-11-08T01:00:00Z")], price)
    assert c["instance_hours"] == 4.5 and c["usd"] == round(4.5 * 2.24208, 4) and c["unpriced"] == ["i-2"]
    assert shards.termination_time("User initiated (2026-11-07 16:43:50 GMT)") == \
        dt.datetime(2026, 11, 7, 16, 43, 50, tzinfo=dt.timezone.utc)


def test_end_to_end_merge_writes_cost_per_run(tmp_path):
    cfg = _cfg("robolab-pi05", "full-shard")
    tasks, per_task = shards.robolab_tasks(cfg, "robolab-pi05")

    class B(InstanceBackend):
        def collect(self, u, a, dest):
            part = shards.robolab_shard(cfg, "robolab-pi05", u["k"], u["n"])["tasks"]
            d = _shard_dir(tmp_path / "dl", u["token"], part, per_task)
            return d, "2026-11-08-0000"

    b = B(hours=4.0)
    st = _plan(tmp_path, n=4, runs=2, m=8)
    s = fanout.Scheduler(tmp_path, st, b, log=lambda *_: None)
    assert _run(s, b) == "finished"
    res = fanout.merge(tmp_path, s.state, cfg)
    assert [r["run"] for r in res["runs"]] == [1, 2] and all(r["complete"] for r in res["runs"])
    r1 = res["runs"][0]
    assert r1["n_episodes"] == 1200 and r1["cost"]["instance_hours"] == 16.0 and r1["cost"]["usd"] == 32.0
    # wall clock = first deploy start to last shard done (scheduler clock: one 30 min step here)
    assert r1["cost"]["usd_per_episode"] == round(32.0 / 1200, 4) and r1["cost"]["wall_clock_s"] == 1800
    merged = json.loads((tmp_path / "merged" / "run-1" / "robolab_results.json").read_text())
    assert merged["cost"]["usd"] == 32.0 and merged["n_episodes"] == 1200
    assert json.loads((tmp_path / fanout.RESULT_FILE).read_text())["runs"][1]["cost"]["usd"] == 32.0


# --- CLI ---------------------------------------------------------------------------------------------------

def test_dry_run_prints_every_unit(tmp_path, capsys):
    rc = fanout.main(["robolab-pi05", "--robolab-variant", "full-shard", "--shards", "16", "--runs", "2", "--regions",
                      "us-east-1,us-west-2,eu-central-1", "--max-instances", "16", "--instance-types",
                      "g6e.2xlarge,g6e.4xlarge", "--run-dir", str(tmp_path / "rd"), "--dry-run", "--no-render"])
    out = capsys.readouterr().out
    assert rc == 0
    cmds = [line for line in out.splitlines() if line.startswith("python deploy.py")]
    assert len(cmds) == 32 and len(set(cmds)) == 32
    assert "--shard 16/16 --shard-run 2 --region us-east-1 --instance-types g6e.2xlarge" in out
    assert "1200 per run" in out and not (tmp_path / "rd" / fanout.STATE_FILE).exists()


def test_cli_refuses_a_target_that_cannot_be_split(tmp_path, capsys):
    assert fanout.main(["openvla-oft", "--shards", "2", "--run-dir", str(tmp_path), "--dry-run"]) == 2
    assert "cannot be sharded" in capsys.readouterr().err


def test_generate_renders_one_shard(tmp_path):
    r = subprocess.run([sys.executable, "generate.py", "--vla", "robolab-pi05", "--robolab-variant", "full-shard",
                        "--robolab-shard", "3/16", "--output-suffix", "s03of16", "--repeats", "1", "--output-dir",
                        str(tmp_path)], cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    text = (tmp_path / "robolab-pi05-s03of16.sh").read_text()
    cfg = _cfg("robolab-pi05", "full-shard")
    part = shards.robolab_shard(cfg, "robolab-pi05", 3, 16)
    assert f"TASKS=({' '.join(part['tasks'])} )" in text
    assert '--meta "fanout_shard=3/16"' in text and f'--meta "task_ids=[{",".join(map(str, part["task_ids"]))}]"' in text


# --- audit fixes (F1..F8) ----------------------------------------------------------------------------------

def test_f1_interruption_only_counts_the_current_attempts_instances(tmp_path):
    # attempt 1 is interrupted; attempt 2 runs on the same stack in the same region, where EC2 still lists the
    # interrupted instance under the stack tag. Attempt 2 must not be killed for it.
    script = {("s01of01", 1): dict(after=3, interrupt=True), ("s01of01", 2): dict(after=3)}
    b = InstanceBackend(script)
    s = fanout.Scheduler(tmp_path, _plan(tmp_path, n=1, m=1, regions=("us-east-1",), spot=True), b, log=lambda *_: None)
    assert _run(s, b) == "finished"
    u = s.state["units"]["s01of01"]
    assert b.killed == ["s01of01"] and u["interruptions"] == 1 and u["state"] == "done"
    assert [a["instance_ids"] for a in u["attempts"]] == [["i-s01of01-1"], ["i-s01of01-2"]]


def test_f2_each_instance_is_costed_once_across_retries(tmp_path):
    script = {("s01of01", 1): dict(rc=1, log="boom")}
    b = InstanceBackend(script, hours=2.0)
    s = fanout.Scheduler(tmp_path, _plan(tmp_path, n=1, m=1, regions=("us-east-1",)), b, log=lambda *_: None)
    assert _run(s, b) == "finished"
    s.refresh_costs()
    atts = s.state["units"]["s01of01"]["attempts"]
    assert [a["cost"]["usd"] for a in atts] == [4.0, 4.0]  # 2 h x $2 each, not 4 + 8
    assert [[r["instance_id"] for r in a["instances"]] for a in atts] == [["i-s01of01-1"], ["i-s01of01-2"]]
    assert fanout.merge(tmp_path, s.state, _cfg("robolab-pi05", "full-shard"))["runs"][0]["cost"]["usd"] == 8.0


def test_f3_spot_cost_is_integrated_over_the_lifetime():
    t = lambda m: T0 + dt.timedelta(minutes=m)  # noqa: E731
    # $1/h from before launch until minute 59, then $3/h for the last minute; points unsorted
    usd, rate = shards.spot_cost([(t(59), 3.0), (t(-300), 1.0), (t(-600), 9.0)], t(0), t(60))
    assert abs(usd - (59 / 60 + 3 / 60)) < 1e-9 and abs(rate - usd) < 1e-9  # ~1.033, not the mean 2.0
    usd, _ = shards.spot_cost([(t(-5), 2.0), (t(30), 4.0), (t(90), 8.0)], t(0), t(60))  # change after end clipped
    assert abs(usd - 3.0) < 1e-9
    assert shards.spot_cost([], t(0), t(60)) is None


class CrashOnLaunch(FakeBackend):
    def launch(self, u, a):
        super().launch(u, a)
        raise KeyboardInterrupt  # the scheduler dies right after the spawn, before it can record the pid


@pytest.mark.parametrize("found", ["receipt", "stack", "nothing"])
def test_f4_launch_intent_is_reconciled_not_relaunched(tmp_path, found):
    b = CrashOnLaunch()
    s = fanout.Scheduler(tmp_path, _plan(tmp_path, n=1, m=1, regions=("us-east-1",)), b, log=lambda *_: None)
    with pytest.raises(KeyboardInterrupt):
        s.step()
    saved = fanout.load_state(tmp_path)
    assert saved["units"]["s01of01"]["state"] == "launching"  # the intent was on disk before the spawn
    b2 = FakeBackend()
    b2.launched = list(b.launched)
    if found == "receipt":
        b2.receipts["RoboLab-Pi05-Demo-s01of01"] = 4242
    elif found == "stack":
        b2.stacks["RoboLab-Pi05-Demo-s01of01"] = "CREATE_IN_PROGRESS"
    s2 = fanout.Scheduler(tmp_path, saved, b2, log=lambda *_: None)
    u = s2.state["units"]["s01of01"]
    if found == "nothing":
        assert u["state"] == "queued" and u["attempts"][0]["phase"] == "not-started"
        s2.step()
        assert len(b2.launched) == 2  # only now, after reconciliation found no process and no stack
    else:
        assert u["state"] == "running" and len(b2.launched) == 1
        assert u["attempts"][0]["pid"] == (4242 if found == "receipt" else None)


def test_f5_failed_deletion_holds_the_slot_and_blocks_the_replacement(tmp_path):
    script = {("s01of02", 1): dict(rc=1, log="InsufficientInstanceCapacity")}
    b = FakeBackend(script)
    b.destroy_fail["RoboLab-Pi05-Demo-s01of02"] = 2
    s = fanout.Scheduler(tmp_path, _plan(tmp_path, n=2, m=1), b, log=lambda *_: None)
    s.step()  # launches s01
    b.tick()
    s.step()  # s01 fails, deletion fails: cleanup holds the only slot
    u = s.state["units"]["s01of02"]
    assert u["state"] == "cleanup" and len(b.launched) == 1
    b.tick()
    s.step()  # second deletion fails too
    assert u["state"] == "cleanup" and len(b.launched) == 1
    b.tick()
    s.step()  # deletion confirmed -> re-queued, then relaunched in the next region
    assert len(b.launched) == 2 and b.launched[1][:2] == ("s01of02", "us-west-2")
    assert _run(s, b) == "finished"


def test_f5_auto_destroy_reported_only_after_confirmed_deletion(tmp_path):
    b = FakeBackend()
    b.destroy_fail["RoboLab-Pi05-Demo-s01of01"] = 1
    s = fanout.Scheduler(tmp_path, _plan(tmp_path, n=1, m=1, auto_destroy=True), b, log=lambda *_: None)
    s.step()
    b.tick()
    s.step()
    u = s.state["units"]["s01of01"]
    assert u["state"] == "cleanup" and not any("--auto-destroy deleted" in e for e in s.state["events"])
    b.tick()
    assert s.step() == "finished" and u["attempts"][-1]["auto_destroyed"]
    assert any("--auto-destroy deleted" in e for e in s.state["events"])


def test_f6_tasks_file_is_split_and_owned_flags_are_rejected(tmp_path, capsys):
    tf = tmp_path / "custom.yaml"
    tf.write_text(yaml.safe_dump(dict(name="custom", tasks=[dict(suite="libero_spatial", task_ids=[0, 1, 2, 3],
                                                                n_episodes=2, seed=7)])))
    rd = tmp_path / "rd"
    rc = fanout.main(["smolvla", "--shards", "2", "--tasks-file", str(tf), "--run-dir", str(rd), "--dry-run",
                      "--no-render"])
    out = capsys.readouterr().out
    assert rc == 0
    cmds = [line for line in out.splitlines() if line.startswith("python deploy.py")]
    assert len(cmds) == 2 and all(c.count("--tasks-file") == 1 and str(tf) not in c for c in cmds)
    ids = [yaml.safe_load((rd / "tasks" / f"tasks-s0{k}of02.yaml").read_text())["tasks"][0]["task_ids"] for k in (1, 2)]
    assert ids == [[0, 2], [1, 3]]
    assert yaml.safe_load((rd / "tasks" / "tasks-s01of02.yaml").read_text())["name"] == "custom-s01of02"
    for owned in (["--region", "us-east-1"], ["--repeats", "2"], ["--shard=1/2"]):
        assert fanout.main(["smolvla", "--shards", "2", "--run-dir", str(rd), "--dry-run", *owned]) == 2
        assert "set per shard by the scheduler" in capsys.readouterr().err
    assert fanout.main(["robolab-pi05", "--shards", "2", "--task-set", "x", "--run-dir", str(rd), "--dry-run"]) == 2


def test_f7_resume_rejects_a_different_benchmark_but_not_operational_changes(tmp_path, capsys):
    rd = tmp_path / "rd"
    base = ["robolab-pi05", "--robolab-variant", "full-shard", "--shards", "4", "--run-dir", str(rd), "--dry-run",
            "--no-render", "--seed", "1"]
    cfg = fanout.load_config("robolab-pi05", "full-shard")
    st = fanout.build_plan("robolab-pi05", cfg, 4, 1, ["us-east-1"], ["g6e.2xlarge"], max_instances=4,
                           variant="full-shard", run_dir=rd, passthrough=["--no-gallery", "--seed", "1"])
    rd.mkdir()
    fanout.save_state(rd, st)
    assert fanout.main(base) == 0  # same benchmark
    assert fanout.main(base + ["--max-instances", "2", "--regions", "us-west-2,eu-central-1", "--email",
                               "a@example.com"]) == 0  # operational knobs may change
    capsys.readouterr()
    assert fanout.main([x if x != "1" else "2" for x in base]) == 2  # --seed 2
    assert "benchmark changed" in capsys.readouterr().err and "passthrough" in fanout.fingerprint_diff(
        st["fingerprint"], fanout.build_plan("robolab-pi05", cfg, 4, 1, ["us-east-1"], ["g6e.2xlarge"], max_instances=4,
                                             variant="full-shard", run_dir=rd,
                                             passthrough=["--no-gallery", "--seed", "2"])["fingerprint"])
    cfg2 = copy.deepcopy(cfg)
    cfg2["protocols"]["full"]["num_envs"] = 5  # episode protocol changed
    st2 = fanout.build_plan("robolab-pi05", cfg2, 4, 1, ["us-east-1"], ["g6e.2xlarge"], max_instances=4,
                            variant="full-shard", run_dir=rd, passthrough=["--no-gallery", "--seed", "1"])
    assert st2["fingerprint"]["sha"] != st["fingerprint"]["sha"]


def test_f8_collection_is_retried_and_destroy_waits_for_results(tmp_path):
    b = FakeBackend()
    b.collect_fail["s01of01"] = 2
    s = fanout.Scheduler(tmp_path, _plan(tmp_path, n=1, m=1, auto_destroy=True), b, log=lambda *_: None)
    s.step()
    b.tick()
    s.step()  # compute done, first collection empty
    u = s.state["units"]["s01of01"]
    assert u["state"] == "collecting" and b.destroyed == []
    b.tick()
    s.step()  # second collection empty
    assert u["state"] == "collecting" and b.destroyed == []
    b.tick()
    assert s.step() == "finished" and u["state"] == "done" and u["results_dir"]
    assert b.destroyed == [("RoboLab-Pi05-Demo-s01of01", "us-east-1")]


def test_f8_results_missing_is_retried_on_resume(tmp_path):
    b = FakeBackend()
    b.collect_fail["s01of01"] = fanout.COLLECT_TRIES
    s = fanout.Scheduler(tmp_path, _plan(tmp_path, n=1, m=1), b, log=lambda *_: None)
    assert _run(s, b) == "finished"
    u = s.state["units"]["s01of01"]
    assert u["state"] == "failed" and u["fail_reason"] == "results-missing" and not u.get("results_dir")
    s2 = fanout.Scheduler(tmp_path, fanout.load_state(tmp_path), b, log=lambda *_: None)
    assert s2.state["units"]["s01of01"]["state"] == "collecting"
    assert _run(s2, b) == "finished" and s2.state["units"]["s01of01"]["state"] == "done"
    assert not shards.validate_results("robolab", tmp_path / "nope")


def test_second_scheduler_on_the_same_run_dir_is_refused(tmp_path):
    # another process holds the lock (POSIX locks are per process, so use a child)
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, time; sys.path.insert(0, sys.argv[1]); import fanout; from pathlib import Path; "
         "f = fanout.acquire_lock(Path(sys.argv[2])); print('locked' if f else 'no', flush=True); time.sleep(30)",
         str(Path(fanout.__file__).parent), str(tmp_path)],
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "locked"
        assert fanout.acquire_lock(tmp_path) is None
    finally:
        holder.kill()
        holder.wait()
    f = fanout.acquire_lock(tmp_path)  # released with the process
    assert f is not None
    f.close()


def test_real_run_without_email_fails_before_any_launch(tmp_path, capsys):
    rc = fanout.main(["robolab-pi05", "--robolab-variant", "full-shard", "--shards", "2",
                      "--regions", "us-west-2", "--run-dir", str(tmp_path / "r"), "--skip-doctor"])
    assert rc == 2
    assert "Email not set" in capsys.readouterr().err
    assert not (tmp_path / "r" / fanout.STATE_FILE).exists()


def test_launches_are_paced_by_deploys_in_flight(tmp_path):
    # each deploy.py takes 2 steps; at most 3 in flight, so launches come in waves of 3
    b = FakeBackend(script={(f"s{k:02d}of07", 1): dict(after=2) for k in range(1, 8)})
    st = _plan(tmp_path, n=7, m=7)
    st["launch_batch"] = 3
    s = fanout.Scheduler(tmp_path, st, b, log=lambda *_: None)
    seen, peak = [], 0
    for _ in range(12):
        status = s.step()
        peak = max(peak, len(s.units("running")))
        seen.append(len(b.launched))
        if status == "finished":
            break
        b.tick()
    assert peak == 3 and seen[0] == 3 and status == "finished" and len(b.launched) == 7


def test_deploy_is_detached_once_the_instance_is_up(tmp_path):
    # the stack is followed in CloudFormation; the slot for a new deploy.py opens at once
    b = FakeBackend(script={(f"s{k:02d}of04", 1): dict(after=3) for k in range(1, 5)})
    b.detach_when_up = True
    st = _plan(tmp_path, n=4, m=4)
    st["launch_batch"] = 1
    s = fanout.Scheduler(tmp_path, st, b, log=lambda *_: None)
    s.step()
    assert len(b.launched) == 1 and b.killed == []
    b.tick()
    s.step()  # poll: the instance is up -> detach s01, which frees the deploy slot for s02
    assert b.killed == ["s01of04"] and len(b.launched) == 2
    assert s.state["units"]["s01of04"]["attempts"][-1]["detached"]
    assert s.state["units"]["s01of04"]["state"] == "running"
    for _ in range(12):
        if s.step() == "finished":
            break
        b.tick()
    assert all(u["state"] == "done" for u in s.units())


def test_queued_unit_goes_back_to_an_earlier_candidate_with_room(tmp_path):
    # B1 run 2026-11-08: r2-s12 hit the quota in us-west-2, moved to its last candidate (cap 0) and sat
    # queued for hours although us-west-2 had room again once other shards finished.
    b = FakeBackend()
    st = _plan(tmp_path, n=2, m=4, types=("g6e.2xlarge", "g6e.4xlarge"))
    s = fanout.Scheduler(tmp_path, st, b, log=lambda *_: None)
    u = s.state["units"]["s01of02"]
    last = len(u["cands"]) - 1
    u["cand"] = last
    for i, c in enumerate(u["cands"]):
        s.state["caps"][f"{c['region']}|{c['instance_type']}"] = 0
    early = u["cands"][1]
    s.state["caps"][f"{early['region']}|{early['instance_type']}"] = 3
    assert s._pick(u) == 1
    s.state["caps"][f"{early['region']}|{early['instance_type']}"] = 0
    assert s._pick(u) is None
