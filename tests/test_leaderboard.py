"""Offline tests for leaderboard.py (`vlasim results --matrix`): benchmark labels, target rows,
cell selection and pooling, stack-difference marks, cost / time / latency, docs block.

S3 is an in-memory fake with LastModified times, so no AWS credentials or network are needed.
"""

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import leaderboard as lb  # noqa: E402
import results  # noqa: E402

ACCT = "123456789012"
REGION = "us-east-1"


def bucket(stack, region=REGION):
    return f"vla-sim-results-{stack}-{region}-{ACCT}"


def ts(run, minutes):
    return lb.run_started(run) + dt.timedelta(minutes=minutes)


# --- labels -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("suite,condition,label", [
    ("libero_spatial", "S0", "LIBERO-Spatial"),
    ("libero_10", "-", "LIBERO-10"),
    ("libero_long", "-", "LIBERO-10"),
    ("libero_spatial", "LIBERO-Plus/camera", "LIBERO-Plus/camera (Spatial)"),
    ("libero_goal", "LIBERO-Plus/light", "LIBERO-Plus/light (Goal)"),
    ("robolab", "RoboLab/subset12", "RoboLab/subset12"),
    ("-", "RoboLab/subset12", "RoboLab/subset12"),
    ("beat_block_hammer", "RoboTwin2/demo_clean", "RoboTwin2/demo_clean"),
    ("RoboTwin2/clean", "subset", "RoboTwin2/clean (subset)"),
    ("RoboTwin2/randomized", "randomized", "RoboTwin2/randomized"),
    ("RoboTwin2/clean", "S2a", "RoboTwin2"),
    ("simpler_bridge", "S0", "SimplerEnv-Bridge"),
    ("simpler_env_widowx/widowx_spoon_on_towel", "-", "SimplerEnv-Bridge"),
    ("libero_sim/LIVING_ROOM_SCENE2_put_both", "-", "LIBERO scenes (GR00T)"),
    ("robocasa_panda_omron/OpenSingleDoor_PandaOmron_Env", "-", "RoboCasa-Kitchen"),
    ("gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA", "-", "GR1-Tabletop"),
    ("my_new_suite", "S0", "my_new_suite"),
])
def test_benchmark_label(suite, condition, label):
    assert lb.benchmark_label(suite, condition) == label


@pytest.mark.parametrize("condition,baseline", [
    ("S0", True), ("-", True), ("", True), ("none", True), ("LIBERO-Plus/camera", True),
    ("RoboLab/subset12", True), ("RoboTwin2/demo_randomized", True),
    ("S2a", False), ("P0", False), ("smoke-knobs", False), ("replan", False),
])
def test_is_baseline(condition, baseline):
    assert lb.is_baseline(condition) is baseline


@pytest.mark.parametrize("condition,suite,baseline", [
    ("subset", "RoboTwin2/clean", True), ("full", "RoboTwin2/randomized", True),
    ("subset", "libero_spatial", False), ("clean", None, False),
])
def test_is_baseline_protocol(condition, suite, baseline):
    assert lb.is_baseline(condition, suite) is baseline


def test_column_order_known_first():
    cols = ["zzz", "RoboTwin2/demo_clean", "LIBERO-10", "SimplerEnv-Bridge", "LIBERO-Spatial",
            "LIBERO-Plus/camera (Spatial)", "RoboLab/subset12"]
    assert sorted(cols, key=lb.column_sort_key) == [
        "LIBERO-Spatial", "LIBERO-10", "LIBERO-Plus/camera (Spatial)", "SimplerEnv-Bridge",
        "RoboLab/subset12", "RoboTwin2/demo_clean", "zzz"]


TARGETS = ["gr00t", "gr00t-g1", "pi", "openvla-oft", "rldx", "rldx-simpler", "smolvla", "lingbot-va"]


@pytest.mark.parametrize("stack,model,target", [
    ("gr00t-bridge-demo", None, "gr00t"),
    ("gr00t-g1-demo", None, "gr00t-g1"),
    ("rldx-simpler-demo", "rldx-1", "rldx-simpler"),   # model field is not a target id -> stack prefix
    ("lingbot-t3", "lingbot-va", "lingbot-va"),
    ("oft-spatial-h", None, "openvla-oft"),            # alias
    ("smolvla-robotwin", "smolvla", "smolvla"),
    ("robolab-pi05-demo", None, "robolab-pi05"),       # unknown: stack name minus -demo
    ("mystack", "pi", "pi"),                           # model field wins when it is a target id
])
def test_target_of(stack, model, target):
    assert lb.target_of(stack, model, TARGETS) == target


# --- instance type, price, wall time -----------------------------------------------------------------

@pytest.mark.parametrize("text,it", [
    ("instance: g6e.xlarge", "g6e.xlarge"),
    ("Instance: i-0123456789abcdef0 (g6.2xlarge)", "g6.2xlarge"),
    ("instance: i-0123 (p5.4xlarge)", "p5.4xlarge"),
    ("Instance: unknown", None),
    ("", None),
])
def test_instance_type_in(text, it):
    assert lb.instance_type_in(text) == it


def test_prices_pinned_and_positive():
    assert lb.price_per_hour("g6.xlarge", "us-east-1") == 0.8048
    assert lb.price_per_hour("g5.2xlarge", "ap-northeast-2") == 1.4903
    assert lb.price_per_hour("g6.xlarge", "mars-east-1") is None
    assert lb.price_per_hour(None, "us-east-1") is None
    assert all(v > 0 for r in lb.PRICES.values() for v in r.values())


def test_upload_span_ignores_judge_and_captions():
    run = "2026-10-19-0558"
    objs = [
        dict(Key=f"{run}/task-0/summary.txt", LastModified=ts(run, 30)),
        dict(Key=f"{run}/task-0/videos/a_success.mp4", LastModified=ts(run, 31)),
        dict(Key=f"{run}/task-0/videos/a_success.txt", LastModified=ts(run, 60 * 24 * 9)),  # judge badge
        dict(Key=f"{run}/judge.json", LastModified=ts(run, 60 * 24 * 9)),
        dict(Key=f"{run}/judge-diag-fail/x.log", LastModified=ts(run, 60 * 24 * 9)),
    ]
    assert lb.upload_span_s(run, objs) == 31 * 60
    assert lb.upload_span_s(run, objs[2:]) is None              # nothing but late writes
    assert lb.upload_span_s(run, [dict(Key=f"{run}/x", LastModified=ts(run, 60 * 49))]) is None  # > 48 h
    assert lb.upload_span_s("scratch", objs) is None


@pytest.mark.parametrize("kv,n,expected", [
    ({}, 10, 10),
    ({"num_episodes": "5"}, None, 5),
    ({"n_episodes_per_task": "5", "task_ids": "[0,1]"}, None, 10),
    ({"num_episodes_per_task": "2", "n_tasks": "10"}, None, 20),
    ({"num_trials_per_task": "5"}, None, None),               # task count unknown
    ({"n_episodes_per_task": "5", "task_ids": "ALL"}, None, None),
])
def test_task_episodes(kv, n, expected):
    assert lb.task_episodes(dict(kv=kv, n=n)) == expected


# --- fake S3 / boto3 ------------------------------------------------------------------------------

class FakeS3:
    """leaderboard.S3 surface over {bucket: {key: (body, LastModified)}}."""

    def __init__(self, objects):
        self.store = objects
        self.gets = []

    def prefixes(self, region, bucket, prefix):
        out = set()
        for k in self.store.get(bucket, {}):
            if k.startswith(prefix) and "/" in k[len(prefix):]:
                out.add(prefix + k[len(prefix):].split("/", 1)[0] + "/")
        return sorted(out)

    def objects(self, region, bucket, prefix):
        return [dict(Key=k, LastModified=v[1]) for k, v in sorted(self.store.get(bucket, {}).items())
                if k.startswith(prefix)]

    def text(self, region, bucket, key, head=None):
        self.gets.append(key)
        v = self.store.get(bucket, {}).get(key)
        return None if v is None else (v[0][:head] if head else v[0])


class _FakeClient:
    def __init__(self, buckets):
        self.buckets = buckets

    def get_caller_identity(self):
        return {"Account": ACCT}

    def list_buckets(self):
        return {"Buckets": [{"Name": b} for b in self.buckets]}


class _FakeBoto3:
    def __init__(self, buckets):
        self.buckets = buckets

    def client(self, name, region_name=None):
        return _FakeClient(self.buckets)


def summary(**kv):
    return "".join(f"{k}: {v}\n" for k, v in kv.items())


def put(objs, stack, run, rel, body, minutes, region=REGION):
    objs.setdefault(bucket(stack, region), {})[f"{run}/{rel}"] = (body, ts(run, minutes))


def fixture_objects():
    """One fixture per label kind plus pooling / condition / stack-difference cases."""
    o = {}
    # smolvla: LIBERO-Spatial baseline with stack_mujoco 3.8.1, then LIBERO-Plus (5 dims) in one run.
    r = "2026-10-27-0514"
    put(o, "smolvla-demo", r, "task-0/summary.txt", summary(
        model="smolvla", suite="libero_spatial", condition="S0", n_success=8, n_episodes=10, success_rate=0.8,
        stack_mujoco="3.8.1", instance_type="g6.xlarge", episode_wall_s=14.0, policy_ms_per_step=41.5), 20)
    r = "2026-10-30-1424"
    for i, dim in enumerate(["camera", "robot", "language", "light", "background"]):
        put(o, "smolvla-demo", r, f"task-{i}/summary.txt", summary(
            model="smolvla", suite="libero_spatial", condition=f"LIBERO-Plus/{dim}", n_success=i, n_episodes=10,
            success_rate=i / 10, stack_mujoco="3.8.1"), 10 + 5 * i)
        put(o, "smolvla-demo", r, f"task-{i}/eval_info.json",
            json.dumps({"overall": {"eval_ep_s": 10.0 + i, "n_episodes": 10}}), 10 + 5 * i)
    put(o, "smolvla-demo", r, "task-0/videos/x--eval_episode_0.txt", "Instance: i-0abc (g6.xlarge)\n", 40)
    # pi: LIBERO-Spatial baseline on a different MuJoCo -> both LIBERO-Spatial cells marked; caption
    # gives the instance type; an S2a run is excluded by default.
    r = "2026-10-22-1412"
    put(o, "pi-demo", r, "task-0/summary.txt", summary(
        suite="libero_spatial", condition="S0", success_rate=0.9, n_episodes=20, n_success=18,
        stack_mujoco="3.2.3"), 60)
    put(o, "pi-demo", r, "task-0/videos/rollout_a_success.txt", "Instance: i-0def (g5.xlarge)\n", 61)
    put(o, "pi-demo", "2026-10-23-0100", "task-0/summary.txt", summary(
        suite="libero_spatial", condition="S2a", success_rate=0.1, n_episodes=20, n_success=2,
        stack_mujoco="3.2.3"), 30)
    # gr00t: SimplerEnv-Bridge; instance from userdata.log; latency from the rollout log.
    r = "2026-10-30-1425"
    for i in range(2):
        put(o, "gr00t-bridge-demo", r, f"task-{i}/summary.txt", summary(
            env_name="simpler_env_widowx/widowx_spoon_on_towel", suite="simpler_bridge", condition="S0",
            n_episodes=10, n_success=5, success_rate=0.5, stack_sapien="2.2.2"), 20 + i)
        put(o, "gr00t-bridge-demo", r, f"task-{i}/rollout.log", "Collecting 10 episodes took 120.0 seconds\n", 20)
    put(o, "gr00t-bridge-demo", r, "userdata.log", "[x] Instance: i-0aaa (g6.2xlarge)\n", 25)
    # gr00t LIBERO scenes: env_name per task, one column.
    r = "2026-10-26-1921"
    for i, scene in enumerate(["LIVING_ROOM_SCENE2_a", "KITCHEN_SCENE3_b"]):
        put(o, "gr00t-demo", r, f"task-{i}/summary.txt", summary(
            env_name=f"libero_sim/{scene}", condition="S0", n_episodes=5, n_success=5 - i, success_rate=1.0), 30)
    # RoboLab and RoboTwin (other branches): labels only through `condition`.
    r = "2026-10-30-1500"
    put(o, "robolab-pi05-demo", r, "task-0/summary.txt", summary(
        model="pi", suite="robolab", condition="RoboLab/subset12", n_episodes=12, n_success=6,
        success_rate=0.5, instance="g6e.2xlarge", run_wall_s=1800), 31)
    put(o, "smolvla-robotwin", r, "task-0/summary.txt", summary(
        model="smolvla", suite="beat_block_hammer", condition="RoboTwin2/demo_clean", n_episodes=10,
        n_success=3, success_rate=0.3), 40)
    # lingbot-va shards: ten stacks, disjoint task_ids, same minute -> pooled; an older rerun of task 0
    # in the same bucket is not pooled.
    for i in range(3):
        r = f"2026-10-15-170{i}"
        put(o, f"lingbot-t{i}", r, "task-0/summary.txt", summary(
            model="lingbot-va", checkpoint="lerobot/lingbot_va_libero_long@cf56f4c6", suite="libero_10",
            task_ids=f"[{i}]", condition="S0", n_episodes=5, n_success=5 - (i == 2), success_rate=1.0), 100)
        put(o, f"lingbot-t{i}", r, "task-0/videos/a.txt", "Instance: i-1 (g6e.2xlarge)\n", 101)
    put(o, "lingbot-t0", "2026-10-14-0900", "task-0/summary.txt", summary(
        model="lingbot-va", checkpoint="lerobot/lingbot_va_libero_long@cf56f4c6", suite="libero_10",
        task_ids="[0]", condition="S0", n_episodes=5, n_success=0, success_rate=0.0), 100)
    # rate-only summary without fingerprint or condition -> marked; still running task ignored.
    r = "2026-10-14-0642"
    put(o, "xvla-demo", r, "task-0/summary.txt", summary(
        model="xvla", suite="libero_spatial", task_ids="[0,1]", n_episodes_per_task=5, success_rate=0.9), 15)
    put(o, "xvla-demo", r, "task-1/eval.log", "running", 16)
    return o


TARGETS_FX = ["gr00t", "pi", "smolvla", "lingbot-va", "xvla"]


@pytest.fixture
def data(monkeypatch):
    monkeypatch.setattr(lb, "known_targets", lambda: TARGETS_FX)
    objs = fixture_objects()
    fake = FakeS3(objs)
    args = argparse.Namespace(region=None, vla=None, since=None)
    runs = lb.collect_runs(args, _FakeBoto3(list(objs)), s3=fake, workers=2)
    return runs, fake


def by_cell(d):
    return {(c["target"], c["benchmark"]): c for c in d["cells"]}


def test_matrix_rows_columns_and_cells(data):
    runs, _ = data
    d = lb.build(lb.select_cells(runs, targets=TARGETS_FX))
    assert d["rows"] == ["gr00t", "lingbot-va", "pi", "smolvla", "xvla"]
    assert d["columns"] == [
        "LIBERO-Spatial", "LIBERO-10",
        "LIBERO-Plus/background (Spatial)", "LIBERO-Plus/camera (Spatial)", "LIBERO-Plus/language (Spatial)",
        "LIBERO-Plus/light (Spatial)", "LIBERO-Plus/robot (Spatial)",
        "LIBERO scenes (GR00T)", "SimplerEnv-Bridge", "RoboLab/subset12", "RoboTwin2/demo_clean"]
    c = by_cell(d)
    assert (c[("pi", "LIBERO-Spatial")]["k"], c[("pi", "LIBERO-Spatial")]["n"]) == (18, 20)  # S2a excluded
    assert (c[("gr00t", "LIBERO scenes (GR00T)")]["k"], c[("gr00t", "LIBERO scenes (GR00T)")]["n"]) == (9, 10)
    assert (c[("gr00t", "SimplerEnv-Bridge")]["k"], c[("gr00t", "SimplerEnv-Bridge")]["n"]) == (10, 20)
    assert c[("pi", "RoboLab/subset12")]["n"] == 12                   # model: pi on a robolab stack
    assert c[("smolvla", "RoboTwin2/demo_clean")]["k"] == 3
    assert c[("smolvla", "LIBERO-Plus/robot (Spatial)")]["k"] == 1
    lo, hi = results.wilson(18, 20)
    assert (c[("pi", "LIBERO-Spatial")]["ci95_lo"], c[("pi", "LIBERO-Spatial")]["ci95_hi"]) == (lo, hi)
    x = c[("xvla", "LIBERO-Spatial")]
    assert (x["k"], x["n"], x["rate"], x["no_fingerprint"]) == (None, None, 0.9, True)


def test_shards_pooled_reruns_not(data):
    runs, _ = data
    c = by_cell(lb.build(lb.select_cells(runs, targets=TARGETS_FX)))
    lv = c[("lingbot-va", "LIBERO-10")]
    assert (lv["k"], lv["n"], lv["runs"]) == (14, 15, 3)               # rerun 0/5 of task 0 not pooled
    assert all("2026-10-14-0900" not in s for s in lv["sources"])
    c0 = by_cell(lb.build(lb.select_cells(runs, pool_hours=0, targets=TARGETS_FX)))
    assert c0[("lingbot-va", "LIBERO-10")]["runs"] == 1


def test_stack_differences_marked(data):
    runs, _ = data
    d = lb.build(lb.select_cells(runs, targets=TARGETS_FX))
    assert d["stack_differences"] == {
        "LIBERO-Spatial": {"stack_mujoco": {"pi": "3.2.3", "smolvla": "3.8.1"}}}
    c = by_cell(d)
    assert c[("pi", "LIBERO-Spatial")]["stacks_differ"] == ["stack_mujoco"]
    assert c[("xvla", "LIBERO-Spatial")]["stacks_differ"] == []        # no fingerprint -> ‡, not †
    md = lb.render_markdown(d)
    assert "0.90 (18/20) [0.70-0.97]†" in md and "0.90 (rate only)‡" in md
    assert "- LIBERO-Spatial: `stack_mujoco`: `pi`=3.2.3, `smolvla`=3.8.1" in md


def test_conditions_rows(data):
    runs, _ = data
    c = by_cell(lb.build(lb.select_cells(runs, all_conditions=True, targets=TARGETS_FX)))
    assert c[("pi [S2a]", "LIBERO-Spatial")]["k"] == 2
    only = by_cell(lb.build(lb.select_cells(runs, condition="S2a", targets=TARGETS_FX)))
    assert list(only) == [("pi", "LIBERO-Spatial")]


def test_cost_time_latency(data):
    runs, _ = data
    c = by_cell(lb.build(lb.select_cells(runs, targets=TARGETS_FX)))
    s = c[("smolvla", "LIBERO-Spatial")]                     # explicit summary fields
    assert (s["instance_type"], s["wall_s"], s["wall_src"]) == ("g6.xlarge", 20 * 60, "upload span")
    assert s["usd_per_run"] == pytest.approx(20 / 60 * 0.8048)
    assert s["usd_per_episode"] == pytest.approx(20 / 60 * 0.8048 / 10)
    assert (s["episode_wall_s"], s["policy_ms_per_step"], s["timing_src"]) == (14.0, 41.5, "summary")
    p = c[("smolvla", "LIBERO-Plus/robot (Spatial)")]        # eval_info.json; caption -> instance
    assert (p["instance_type"], p["episode_wall_s"], p["timing_src"]) == ("g6.xlarge", 11.0, "eval_info.json")
    assert p["usd_per_episode"] == pytest.approx(30 / 60 * 0.8048 / 50)  # 50 episodes; late caption ignored
    g = c[("gr00t", "SimplerEnv-Bridge")]                    # userdata.log + rollout log
    assert (g["instance_type"], g["episode_wall_s"], g["timing_src"]) == ("g6.2xlarge", 12.0, "rollout.log")
    rl = c[("pi", "RoboLab/subset12")]                       # run_wall_s beats the upload span
    assert (rl["wall_s"], rl["wall_src"], rl["instance_type"]) == (1800, "run_wall_s", "g6e.2xlarge")
    assert rl["usd_per_run"] == pytest.approx(0.5 * 2.24208)
    lv = c[("lingbot-va", "LIBERO-10")]
    assert lv["wall_s"] == 3 * 100 * 60 and lv["usd_per_episode"] == pytest.approx(3 * 100 / 60 * 2.24208 / 15)
    rt = c[("smolvla", "RoboTwin2/demo_clean")]              # no instance type anywhere
    assert (rt["instance_type"], rt["usd_per_run"], rt["usd_per_episode"]) == ("?", None, None)
    x = c[("xvla", "LIBERO-Spatial")]                        # rate-only: episodes from per-task x task_ids
    assert x["usd_per_episode"] is None and x["instance_type"] == "?"


def test_no_cost_skips_timing_reads(data):
    runs, fake = data
    fake.gets.clear()
    d = lb.build(lb.select_cells(runs, targets=TARGETS_FX), with_cost=False)
    assert fake.gets == [] and "usd_per_run" not in d["cells"][0]
    assert "Cost and time" not in lb.render_markdown(d, with_cost=False)


# --- docs block and CLI -----------------------------------------------------------------------------

def test_replace_block():
    text = f"# R\n\nintro\n{lb.DOCS_BEGIN}\nold\n{lb.DOCS_END}\ntail\n"
    out = lb.replace_block(text, "new table\n")
    assert out == f"# R\n\nintro\n{lb.DOCS_BEGIN}\nnew table\n{lb.DOCS_END}\ntail\n"
    assert lb.replace_block(out, "new table") == out                # idempotent
    with pytest.raises(ValueError):
        lb.replace_block("no markers", "x")


def test_docs_results_has_markers():
    text = (ROOT / "docs" / "results.md").read_text()
    assert text.count(lb.DOCS_BEGIN) == 1 and text.count(lb.DOCS_END) == 1
    assert text.index(lb.DOCS_BEGIN) < text.index(lb.DOCS_END)


def test_cli_matrix_csv_json_write_docs(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lb, "known_targets", lambda: TARGETS_FX)
    objs = fixture_objects()
    monkeypatch.setattr(lb, "S3", lambda boto3, region: FakeS3(objs))
    monkeypatch.setitem(sys.modules, "boto3", _FakeBoto3(list(objs)))
    doc = tmp_path / "results.md"
    doc.write_text(f"head\n{lb.DOCS_BEGIN}\n{lb.DOCS_END}\nfoot\n")
    out_csv, out_json = tmp_path / "m.csv", tmp_path / "m.json"
    rc = results.main(["--matrix", "--csv", str(out_csv), "--json", str(out_json), "--write-docs", str(doc)])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("| Target | LIBERO-Spatial | LIBERO-10 |")
    assert "| `lingbot-va` |  | 0.93 (14/15) [0.70-0.99]‡ |" in out  # no stack_* keys -> ‡
    rows = list(csv.DictReader(out_csv.open()))
    assert {r["benchmark"] for r in rows} >= {"RoboLab/subset12", "RoboTwin2/demo_clean",
                                              "LIBERO-Plus/camera (Spatial)", "SimplerEnv-Bridge"}
    j = json.loads(out_json.read_text())
    assert j["prices_checked"] == lb.PRICES_CHECKED and len(j["cells"]) == len(rows)
    text = doc.read_text()
    assert text.startswith(f"head\n{lb.DOCS_BEGIN}\n_Generated ") and text.endswith(f"{lb.DOCS_END}\nfoot\n")
    assert "| `pi` |" in text
    doc.write_text("no markers\n")
    assert results.main(["--write-docs", str(doc)]) == 2
    assert doc.read_text() == "no markers\n"


def test_matrix_no_results(monkeypatch, capsys):
    monkeypatch.setattr(lb, "S3", lambda boto3, region: FakeS3({}))
    monkeypatch.setitem(sys.modules, "boto3", _FakeBoto3([]))
    assert results.main(["--matrix"]) == 0
    assert "No comparable results" in capsys.readouterr().out


def test_policy_stack_keys_are_not_compared():
    kvs = [{"stack_eval_args": "--policy.path=a", "stack_server": "86e1ee52", "stack_sapien": "3.0.0b1"}]
    assert lb.fingerprint(kvs) == {"stack_sapien": "3.0.0b1"}


def test_protocol_card_clears_the_unknown_comparability_mark(monkeypatch):
    """A cell whose every finished task wrote protocol.yaml is no longer marked ‡, even without a stack fingerprint."""
    monkeypatch.setattr(lb, "known_targets", lambda: TARGETS_FX)
    objs = fixture_objects()
    r = sorted({k.split("/")[0] for k in objs[bucket("xvla-demo")] if results.RUN_RE.match(k.split("/")[0])})[-1]
    put(objs, "xvla-demo", r, "task-0/protocol.yaml",
        "schema: rfm-simulation/protocol-card/v1\ntarget: 'xvla'\ncondition: 'unknown'\nk: unknown\nn: unknown\n", 15)
    args = argparse.Namespace(region=None, vla=None, since=None)
    runs = lb.collect_runs(args, _FakeBoto3(list(objs)), s3=FakeS3(objs), workers=2)
    x = by_cell(lb.build(lb.select_cells(runs, targets=TARGETS_FX)))[("xvla", "LIBERO-Spatial")]
    assert x["protocol_card"] is True and x["no_fingerprint"] is False
    plain = by_cell(lb.build(lb.select_cells(lb.collect_runs(args, _FakeBoto3(list(fixture_objects())),
                                                              s3=FakeS3(fixture_objects()), workers=2),
                                             targets=TARGETS_FX)))[("xvla", "LIBERO-Spatial")]
    assert plain["protocol_card"] is False and plain["no_fingerprint"] is True


def test_unreadable_protocol_card_is_ignored():
    class V:
        def text(self, region, bucket, key):
            return {"a/protocol.yaml": "schema: something-else\n", "b/protocol.yaml": ": : not yaml ["}.get(key)
    assert results.read_protocol_card(V(), "r", "b", "a/") is None
    assert results.read_protocol_card(V(), "r", "b", "b/") is None
    assert results.read_protocol_card(V(), "r", "b", "c/") is None
