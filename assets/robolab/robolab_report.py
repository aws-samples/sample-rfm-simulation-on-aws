#!/usr/bin/env python3
"""Aggregate a RoboLab run (one or more shards) into this project's result layout.

RoboLab writes, per shard, ``<shard>/<folder>/episode_results.jsonl`` (one JSON object per episode:
env_name, task_name, run, env_id, success, score, ...) and per task ``<shard>/<folder>/<env_name>/*.mp4``
named ``<instruction>_<run>_env<env_id>[_viewport].mp4`` (``<instruction>_<run>[_viewport].mp4`` when
num_envs == 1). This script (stdlib only, runs on the host after the containers exit):

  * copies the videos to ``<upload_dir>/videos/<task>_run<r>_env<e>_success=<True|False>[_viewport].mp4``
    (the ``success=`` token is what results.py counts when summary.txt has no k/n) with one caption
    ``.txt`` each (the gallery shows it under the video),
  * writes ``<upload_dir>/summary.txt`` (key: value, read by ``vlasim results``) and
    ``<upload_dir>/robolab_results.json`` (per task / per difficulty breakdown, 95% Wilson intervals).

Usage:
  robolab_report.py --run-dir shard-0/run --run-dir shard-1/run --upload-dir out \\
      --label RoboLab/subset12 --tasks "A B C" --episodes-per-task 5 [--task-metadata task_metadata.json] \\
      [--max-videos-per-task 10] [--meta key=value ...]

Prints ``<n_success>/<n_episodes>`` on the last line; exit 0 even when episodes are missing (the caller
decides, from n_episodes, whether the run failed).
"""

import argparse
import json
import math
import os
import re
import shutil
import sys

VIDEO_RE = re.compile(r"_(?P<run>\d+)(?:_env(?P<env>\d+))?(?P<view>_viewport)?\.mp4$")
SAFE_RE = re.compile(r"[^A-Za-z0-9_.=-]+")


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def load_episodes(run_dir):
    path = os.path.join(run_dir, "episode_results.jsonl")
    out = []
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ep = json.loads(line)
            except ValueError:
                continue
            if isinstance(ep, dict) and "env_name" in ep:
                out.append(ep)
    return out


def dedupe(episodes):
    """Resumed runs can append the same (env, run, env_id) twice; keep the last record."""
    by_key = {}
    for ep in episodes:
        by_key[(ep.get("env_name"), ep.get("run"), ep.get("env_id"))] = ep
    return list(by_key.values())


def task_of(ep):
    return ep.get("task_name") or ep.get("env_name")


def find_videos(run_dir):
    """[(env_name, run, env_id, is_viewport, path)] for every mp4 under <run_dir>/<env_name>/."""
    found = []
    if not os.path.isdir(run_dir):
        return found
    for env_name in sorted(os.listdir(run_dir)):
        sub = os.path.join(run_dir, env_name)
        if not os.path.isdir(sub):
            continue
        for fn in sorted(os.listdir(sub)):
            m = VIDEO_RE.search(fn)
            if not m:
                continue
            env_id = int(m.group("env")) if m.group("env") is not None else 0
            found.append((env_name, int(m.group("run")), env_id, bool(m.group("view")), os.path.join(sub, fn)))
    return found


def aggregate(episodes, tasks, episodes_per_task, metadata):
    per_task = {}
    for ep in episodes:
        t = task_of(ep)
        g = per_task.setdefault(t, {"task": t, "n": 0, "k": 0, "scores": []})
        g["n"] += 1
        g["k"] += 1 if ep.get("success") is True else 0
        if isinstance(ep.get("score"), (int, float)):
            g["scores"].append(float(ep["score"]))
    rows = []
    names = list(tasks) if tasks else sorted(per_task)
    for t in names + sorted(set(per_task) - set(names)):
        g = per_task.get(t, {"task": t, "n": 0, "k": 0, "scores": []})
        meta = metadata.get(t, {})
        rows.append({
            "task": t,
            "n": g["n"],
            "k": g["k"],
            "success_rate": round(g["k"] / g["n"], 4) if g["n"] else None,
            "mean_score": round(sum(g["scores"]) / len(g["scores"]), 4) if g["scores"] else None,
            "expected": episodes_per_task,
            "complete": g["n"] >= episodes_per_task,
            "difficulty": meta.get("difficulty_label"),
            "attributes": meta.get("attributes"),
        })
    by_diff = {}
    for r in rows:
        d = r["difficulty"] or "unknown"
        g = by_diff.setdefault(d, {"n": 0, "k": 0})
        g["n"] += r["n"]
        g["k"] += r["k"]
    for g in by_diff.values():
        g["success_rate"] = round(g["k"] / g["n"], 4) if g["n"] else None
        g["ci95"] = wilson(g["k"], g["n"])
    return rows, by_diff


def load_metadata(path):
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except ValueError:
        return {}
    items = data if isinstance(data, list) else list(data.values()) if isinstance(data, dict) else []
    return {x["task_name"]: x for x in items if isinstance(x, dict) and "task_name" in x}


def caption(ep, task, run, env_id, viewport, meta, label):
    lines = [
        "=== RoboLab Simulation Context ===",
        f"Benchmark:  {label} (NVIDIA RoboLab, Isaac Lab, DROID Franka)",
        f"Task:       {task}",
        f"Instruction: {ep.get('instruction', '') if ep else ''}",
        f"Episode:    run {run}, env {env_id}{' (viewport)' if viewport else ' (policy cameras)'}",
        f"Success:    {ep.get('success') if ep else 'unknown'}",
    ]
    if ep and ep.get("score") is not None:
        lines.append(f"Score:      {ep.get('score')} (subtask progress)")
    if ep and ep.get("reason"):
        lines.append(f"Reason:     {ep.get('reason')}")
    lines.append("")
    for k, v in meta.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", action="append", default=[], help="RoboLab output folder of one shard (repeat)")
    p.add_argument("--upload-dir", required=True)
    p.add_argument("--label", required=True, help="suite label, e.g. RoboLab/subset12")
    p.add_argument("--tasks", default="", help="space-separated expected task names (empty = all found)")
    p.add_argument("--episodes-per-task", type=int, required=True)
    p.add_argument("--n-tasks", type=int, default=0, help="expected task count when --tasks is empty (all)")
    p.add_argument("--task-metadata", default="", help="RoboLab robolab/tasks/_metadata/task_metadata.json")
    p.add_argument("--max-videos-per-task", type=int, default=10)
    p.add_argument("--meta", action="append", default=[], help="key=value copied into summary/captions")
    a = p.parse_args(argv)

    meta = {}
    for kv in a.meta:
        k, _, v = kv.partition("=")
        if k:
            meta[k.strip()] = v.strip()
    tasks = a.tasks.split()
    metadata = load_metadata(a.task_metadata)

    episodes = []
    videos = []
    for rd in a.run_dir:
        episodes += load_episodes(rd)
        videos += find_videos(rd)
    episodes = dedupe(episodes)
    index = {(ep.get("env_name"), ep.get("run"), ep.get("env_id")): ep for ep in episodes}

    rows, by_diff = aggregate(episodes, tasks, a.episodes_per_task, metadata)
    n = sum(r["n"] for r in rows)
    k = sum(r["k"] for r in rows)
    n_tasks = len(tasks) or a.n_tasks or len(rows)
    expected = n_tasks * a.episodes_per_task

    vid_dir = os.path.join(a.upload_dir, "videos")
    os.makedirs(vid_dir, exist_ok=True)
    per_task_count = {}
    n_videos = 0
    for env_name, run, env_id, viewport, path in sorted(videos, key=lambda v: (v[0], v[1], v[2], not v[3])):
        ep = index.get((env_name, run, env_id))
        task = task_of(ep) if ep else env_name
        key = (task, viewport)
        if per_task_count.get(key, 0) >= a.max_videos_per_task:
            continue
        per_task_count[key] = per_task_count.get(key, 0) + 1
        succ = ep.get("success") if ep else None
        tag = "True" if succ is True else "False" if succ is False else "unknown"
        base = SAFE_RE.sub("_", f"{task}_run{run}_env{env_id}_success={tag}{'_viewport' if viewport else ''}")
        shutil.copyfile(path, os.path.join(vid_dir, base + ".mp4"))
        with open(os.path.join(vid_dir, base + ".txt"), "w") as f:
            f.write(caption(ep, task, run, env_id, viewport, meta, a.label))
        n_videos += 1

    rate = k / n if n else None
    report = {
        "benchmark": "RoboLab-120",
        "suite": a.label,
        "n_episodes": n,
        "n_success": k,
        "n_expected": expected,
        "success_rate": round(rate, 4) if rate is not None else None,
        "ci95_wilson": wilson(k, n),
        "tasks_complete": sum(1 for r in rows if r["complete"]),
        "n_tasks": n_tasks,
        "by_difficulty": by_diff,
        "per_task": rows,
        "n_videos_uploaded": n_videos,
        "meta": meta,
    }
    with open(os.path.join(a.upload_dir, "robolab_results.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(a.upload_dir, "summary.txt"), "w") as f:
        for key in ("model", "checkpoint", "code"):
            if key in meta:
                f.write(f"{key}: {meta[key]}\n")
        f.write(f"suite: {a.label}\n")
        f.write(f"n_episodes: {n}\n")
        f.write(f"n_success: {k}\n")
        f.write(f"success_rate: {'' if rate is None else f'{rate:.4f}'}\n")
        f.write(f"n_expected: {expected}\n")
        f.write(f"tasks_complete: {report['tasks_complete']}/{n_tasks}\n")
        for key, v in meta.items():
            if key not in ("model", "checkpoint", "code"):
                f.write(f"{key}: {v}\n")
    print(f"{k}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
