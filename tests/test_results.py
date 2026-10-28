"""Offline unit tests for results.py: Wilson interval, summary.txt parsing and aggregation.

S3 is replaced by an in-memory fake, so no AWS credentials or network are needed.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import results  # noqa: E402

ACCT = "123456789012"
BUCKET = f"vla-sim-results-openvla-oft-libero-10-us-east-1-{ACCT}"


# --- Wilson interval ------------------------------------------------------------------------------

def test_wilson_known_value():
    lo, hi = results.wilson(48, 50)
    assert (round(lo, 2), round(hi, 2)) == (0.87, 0.99)


def test_wilson_no_episodes():
    assert results.wilson(0, 0) == (None, None)


@pytest.mark.parametrize("k,n", [(0, 10), (10, 10), (1, 3), (5, 10), (37, 100), (1, 1)])
def test_wilson_bounds_contain_point_estimate(k, n):
    lo, hi = results.wilson(k, n)
    assert 0.0 <= lo <= k / n <= hi <= 1.0


def test_wilson_symmetric_and_narrows_with_n():
    lo, hi = results.wilson(5, 10)
    assert math.isclose(lo, 1 - hi)
    lo2, hi2 = results.wilson(50, 100)
    assert hi2 - lo2 < hi - lo


def test_wilson_edges_are_not_degenerate():
    # Unlike the normal approximation, 0/n and n/n still give a non-zero-width interval.
    assert results.wilson(0, 10)[1] > 0
    assert results.wilson(10, 10)[0] < 1


# --- regexes ---------------------------------------------------------------------------------------

def test_bucket_re():
    m = results.BUCKET_RE.match(BUCKET)
    assert m and m.group("stack") == "openvla-oft-libero-10"
    assert m.group("region") == "us-east-1" and m.group("acct") == ACCT
    assert results.BUCKET_RE.match("vla-sim-results-x-us-east-1-123") is None


def test_run_re():
    assert results.RUN_RE.match("2026-10-19-0558")
    assert not results.RUN_RE.match("2026-10-19")


# --- fake S3 ---------------------------------------------------------------------------------------

class FakeS3:
    """Just enough of results.S3: prefixes / keys / text over a dict of key -> body."""

    def __init__(self, objects):
        self.objects = objects

    def prefixes(self, region, bucket, prefix):
        out = set()
        for k in self.objects.get(bucket, {}):
            if k.startswith(prefix) and "/" in k[len(prefix):]:
                out.add(prefix + k[len(prefix):].split("/", 1)[0] + "/")
        return sorted(out)

    def keys(self, region, bucket, prefix):
        return [k for k in self.objects.get(bucket, {}) if k.startswith(prefix)]

    def text(self, region, bucket, key):
        return self.objects.get(bucket, {}).get(key)


def test_task_row_counts_from_summary():
    s3 = FakeS3({BUCKET: {"R/task-0/summary.txt": (
        "suite: libero_10\ncondition: replan k=4\nsuccess_rate: 0.8\nn_episodes: 10\nn_success: 8\n"
        "stack_step_scale: 1.0\n")}})
    r = results.task_row(s3, "us-east-1", BUCKET, "R", "R/task-0/")
    assert r["status"] == "done" and r["task"] == "task-0"
    assert (r["k"], r["n"], r["rate"]) == (8, 10, 0.8)
    assert r["suite"] == "libero_10" and r["condition"] == "replan"
    assert r["stack"] == {"stack_step_scale": "1.0"}


def test_task_row_counts_from_videos():
    s3 = FakeS3({BUCKET: {
        "R/task-1/summary.txt": "env_name: libero_goal\n",
        "R/task-1/videos/ep0--success=True.mp4": "",
        "R/task-1/videos/ep1--success=False.mp4": "",
        "R/task-1/videos/ep2--success=True.mp4": "",
        "R/task-1/videos/notes.txt": "",
    }})
    r = results.task_row(s3, "us-east-1", BUCKET, "R", "R/task-1/")
    assert (r["k"], r["n"]) == (2, 3)
    assert math.isclose(r["rate"], 2 / 3)
    assert r["suite"] == "libero_goal" and r["condition"] == "-"


def test_task_row_rate_only_derives_k():
    s3 = FakeS3({BUCKET: {"R/task-2/summary.txt": "success_rate: 0.5\nn_episodes: 4\n"}})
    r = results.task_row(s3, "us-east-1", BUCKET, "R", "R/task-2/")
    assert (r["k"], r["n"]) == (2, 4)


def test_task_row_missing_summary_is_running_not_zero():
    r = results.task_row(FakeS3({BUCKET: {}}), "us-east-1", BUCKET, "R", "R/task-3/")
    assert r == {"run": "R", "task": "task-3", "status": "running/failed"}


def test_aggregate_pools_counts_and_tracks_not_done():
    base = dict(bucket_stack="s", region="us-east-1", run="R", suite="libero_10", condition="-")
    rows = [
        dict(base, task="task-0", status="done", k=8, n=10, rate=0.8),
        dict(base, task="task-1", status="done", k=9, n=10, rate=0.9),
        dict(base, task="task-2", status="running/failed"),
        dict(base, task="task-3", status="done", k=None, n=None, rate=0.7, suite="libero_goal"),
    ]
    agg = results.aggregate(rows)
    g = agg[("s", "us-east-1", "R", "libero_10", "-")]
    assert (g["k"], g["n"], g["tasks"], g["not_done"]) == (17, 20, 3, 1)
    g2 = agg[("s", "us-east-1", "R", "libero_goal", "-")]
    assert g2["rates"] == [0.7] and g2["n"] == 0


# --- collect() with a fake boto3 -------------------------------------------------------------------

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


def test_collect_latest_run_and_filters(monkeypatch):
    other_acct = "vla-sim-results-pi-us-east-1-999999999999"
    objects = {BUCKET: {
        "2026-10-18-0100/task-0/summary.txt": "success_rate: 0.1\nn_episodes: 10\nn_success: 1\n",
        "2026-10-19-0558/task-0/summary.txt": "success_rate: 0.9\nn_episodes: 10\nn_success: 9\n",
        "2026-10-19-0558/task-1/summary.txt": "success_rate: 1.0\nn_episodes: 10\nn_success: 10\n",
        "scratch/task-0/summary.txt": "success_rate: 0.0\n",
    }, other_acct: {"2026-10-19-0558/task-0/summary.txt": "success_rate: 0.5\n"}}
    monkeypatch.setattr(results, "S3", lambda boto3, region: FakeS3(objects))
    args = argparse.Namespace(region=None, vla=None, since=None, all_runs=False)
    rows = results.collect(args, _FakeBoto3([BUCKET, other_acct]))
    assert {r["run"] for r in rows} == {"2026-10-19-0558"}  # latest run only, other account skipped
    assert sorted((r["task"], r["k"]) for r in rows) == [("task-0", 9), ("task-1", 10)]
    assert all(r["bucket_stack"] == "openvla-oft-libero-10" for r in rows)

    args.all_runs = True
    assert len(results.collect(args, _FakeBoto3([BUCKET]))) == 3
    args.vla = "pi"
    assert results.collect(args, _FakeBoto3([BUCKET])) == []


# --- post-run judge (judge.json next to the run) ------------------------------------------------------

JUDGE_JSON = json.dumps({
    "judge": "flora-rewarder-outcome", "model": "nvidia/Flora-Rewarder",
    "model_revision": "031299e46e452fc0cedb4e0975eb9fa95fc37783", "success_threshold": 4,
    "summary": {"agreement": 0.8, "n_agree": 8, "n_compared": 10, "n_scored": 10, "n_episodes": 10,
                "mean_score_sim_success": 4.6, "mean_score_sim_failure": 1.8,
                "confusion": {"sim_success_flora_success": 5, "sim_success_flora_failure": 1,
                              "sim_failure_flora_success": 1, "sim_failure_flora_failure": 3}},
    "episodes": [],
})


def test_judge_summary_and_line():
    s3 = FakeS3({BUCKET: {"R/judge.json": JUDGE_JSON, "B/judge.json": "{not json"}})
    j = results.judge_summary(s3, "us-east-1", BUCKET, "R")
    assert (j["agreement"], j["n_agree"], j["model_revision"]) == (0.8, 8, "031299e4")
    line = results.judge_line(j)
    assert "agreement 0.80 (8/10)" in line and "sim-success 4.6 / sim-failure 1.8" in line and ">= 4" in line
    assert results.judge_summary(s3, "us-east-1", BUCKET, "none") is None
    assert results.judge_line(results.judge_summary(s3, "us-east-1", BUCKET, "B")) == "unreadable judge.json"


def test_collect_attaches_judge_to_judged_runs(monkeypatch, capsys):
    objects = {BUCKET: {
        "2026-10-19-0558/task-0/summary.txt": "success_rate: 0.9\nn_episodes: 10\nn_success: 9\n",
        "2026-10-19-0558/judge.json": JUDGE_JSON,
        "2026-10-18-0100/task-0/summary.txt": "success_rate: 0.1\nn_episodes: 10\nn_success: 1\n",
    }}
    monkeypatch.setattr(results, "S3", lambda boto3, region: FakeS3(objects))
    args = argparse.Namespace(region=None, vla=None, since=None, all_runs=True)
    rows = results.collect(args, _FakeBoto3([BUCKET]))
    judged = {r["run"]: bool(r.get("judge")) for r in rows}
    assert judged == {"2026-10-19-0558": True, "2026-10-18-0100": False}

    monkeypatch.setitem(sys.modules, "boto3", _FakeBoto3([BUCKET]))
    assert results.main(["--all-runs"]) == 0
    out = capsys.readouterr().out
    assert "post-run judge (judge.json):" in out and "2026-10-19-0558: agreement 0.80 (8/10)" in out
