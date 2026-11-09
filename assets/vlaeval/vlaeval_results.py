#!/usr/bin/env python3
"""Turn one vla-eval benchmark result into this project's result layout (stdlib only).

vla-eval (allenai/vla-evaluation-harness) writes, per benchmark entry of its run config, into ``output_dir``:

  * ``<Benchmark>_<subname>_aggregate.json`` — ``tasks[]`` with ``task`` (instruction), ``episodes[]``
    (``task_idx``, ``episode_id``, ``metrics.success``, ``steps``, optional ``failure_reason`` /
    ``failure_detail``), ``num_episodes``, ``mean_success``; top level ``benchmark``, ``harness_version``,
    ``config`` (the benchmark entry as run), ``server_info``, ``seed``, ``mean_success``,
    ``num_episodes_total``, ``num_errors`` (only when some episode errored).
  * ``episodes/<Benchmark>_<subname>/task<TTTT>_ep<EEEE>_<success|fail|error>.mp4`` (+ ``.jsonl`` step traces)
    when ``recording.record_video`` is on.
  * ``recording-<uuid>.sqlite`` (the harness's own recording database).

This script, run on the host after ``vla-eval run`` exits:

  * copies each episode video to ``<video_dir>/<suite>_t<TT>--episode_<E>-<success|failure>.mp4`` (outcome
    from the aggregate, which is what the harness counts; an errored episode is a failure) with a caption
    ``.txt`` next to it (the gallery shows it under the video),
  * writes ``summary.txt`` (``key: value`` lines read by ``vlasim results`` and the protocol card): suite,
    env_name, k/n, success_rate, errors, seed, condition, action_type and every ``--meta key=value`` given
    (pins, stack fingerprint).

Usage:
  vlaeval_results.py --output-dir <vla-eval output_dir> --summary <summary.txt> --video-dir <dir> \\
      --suite libero_spatial [--meta key=value ...]

Prints ``<n_success>/<n_episodes>`` on the last line. Exit 2 when there is no aggregate JSON or it does not
parse (the caller fails the run), 0 otherwise.
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys

VIDEO_RE = re.compile(r"^task(\d+)_ep(\d+)_(success|fail|error)\.mp4$")
SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def find_aggregate(output_dir):
    """The single ``*_aggregate.json`` of one benchmark entry (newest if a re-run left several)."""
    found = sorted(glob.glob(os.path.join(output_dir, "*_aggregate.json")), key=os.path.getmtime)
    return found[-1] if found else None


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("tasks"), list):
        raise ValueError(f"{path}: not a vla-eval aggregate (no tasks list)")
    return data


def episodes(agg):
    """Flat list of dict(task_idx, episode_id, task, success, error, steps), in the aggregate's order."""
    out = []
    for t_pos, task in enumerate(agg.get("tasks") or []):
        for ep in task.get("episodes") or []:
            tidx = ep.get("task_idx", ep.get("task_id", t_pos))
            out.append(dict(
                task_idx=int(tidx), episode_id=int(ep.get("episode_id", ep.get("episode_idx", 0))),
                task=str(task.get("task") or ep.get("name") or ""),
                success=bool((ep.get("metrics") or {}).get("success")),
                error=str(ep.get("failure_reason") or ""), steps=ep.get("steps")))
    return out


def counts(agg):
    eps = episodes(agg)
    k = sum(e["success"] for e in eps)
    return k, len(eps), sum(bool(e["error"]) for e in eps)


def video_name(suite, task_idx, episode_id, success):
    return f"{SAFE_RE.sub('_', suite)}_t{task_idx:02d}--episode_{episode_id}-{'success' if success else 'failure'}.mp4"


def collect_videos(output_dir, agg, suite, video_dir, caption=None):
    """Copy every episode mp4 under a gallery name; returns (copied, mismatched file-vs-aggregate outcomes)."""
    by_key = {(e["task_idx"], e["episode_id"]): e for e in episodes(agg)}
    os.makedirs(video_dir, exist_ok=True)
    copied, mismatch = 0, 0
    for path in sorted(glob.glob(os.path.join(output_dir, "episodes", "*", "*.mp4"))):
        m = VIDEO_RE.match(os.path.basename(path))
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)))
        ep = by_key.get(key)
        if ep is None:  # a video the aggregate does not list (should not happen); keep it, outcome from its name
            ep = dict(task_idx=key[0], episode_id=key[1], task="", success=m.group(3) == "success", error="",
                      steps=None)
        elif (m.group(3) == "success") != ep["success"]:
            mismatch += 1
        name = video_name(suite, ep["task_idx"], ep["episode_id"], ep["success"])
        dest = os.path.join(video_dir, name)
        shutil.copyfile(path, dest)
        copied += 1
        lines = [f"Task:     {ep['task'] or '?'} (task {ep['task_idx']}, episode {ep['episode_id']})",
                 f"Outcome:  {'success' if ep['success'] else 'failure'}"
                 + (f" (episode error: {ep['error']})" if ep["error"] else ""),
                 f"Steps:    {ep['steps'] if ep['steps'] is not None else '?'}",
                 f"Source:   {os.path.relpath(path, output_dir)} (vla-eval)"]
        with open(dest[: -len(".mp4")] + ".txt", "w", encoding="utf-8") as f:
            f.write("\n".join(([caption, ""] if caption else []) + lines) + "\n")
    return copied, mismatch


def summary_lines(agg, suite, meta):
    k, n, errors = counts(agg)
    cfg = agg.get("config") or {}
    params = cfg.get("params") or {}
    server = agg.get("server_info") or {}
    out = [
        ("suite", suite),
        ("env_name", suite),
        ("vla_eval_benchmark", agg.get("benchmark", "")),  # not `benchmark:` (results.py reads that as LIBERO-Plus)
        ("n_tasks", len(agg.get("tasks") or [])),
        ("n_episodes_per_task", cfg.get("episodes_per_task", "")),
        ("success_rate", f"{k / n:.3f}" if n else "unknown"),
        ("n_success", k),
        ("n_episodes", n),
        ("n_errors", errors),
        ("harness_mean_success", agg.get("mean_success", "")),
        ("seed", agg.get("seed", params.get("seed", ""))),
        ("max_steps", cfg.get("max_steps") or "suite default"),
        ("init_states", "fixed (benchmark init-state files, episode i = init state i)"),
        ("stack_vla_eval_harness_version", agg.get("harness_version", "")),
        ("stack_vla_eval_model_server", server.get("model_server", "")),
        ("stack_vla_eval_protocol_version", server.get("protocol_version", "")),
        ("stack_vla_eval_mode", agg.get("mode", "")),
        ("stack_vla_eval_render", (agg.get("render") or {}).get("requested", "")),
    ]
    have = {key for key, _ in out}
    for item in meta:
        key, _, value = item.partition("=")
        key = key.strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise ValueError(f"--meta {item!r}: key must be [a-z][a-z0-9_]*")
        if key in have:  # a value the aggregate already gives wins; meta only adds
            continue
        out.append((key, " ".join(value.split())))
        have.add(key)
    return [f"{key}: {value}" for key, value in out]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--output-dir", required=True, help="vla-eval output_dir of one benchmark entry")
    ap.add_argument("--summary", required=True, help="summary.txt to write")
    ap.add_argument("--video-dir", required=True, help="directory for the renamed episode videos")
    ap.add_argument("--suite", required=True, help="suite label (LIBERO suite name)")
    ap.add_argument("--caption", default="", help="first line of every video caption")
    ap.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE", help="extra summary.txt line")
    a = ap.parse_args(argv)
    path = find_aggregate(a.output_dir)
    if not path:
        print(f"[vlaeval_results] no *_aggregate.json in {a.output_dir}", file=sys.stderr)
        return 2
    try:
        agg = load(path)
        lines = summary_lines(agg, a.suite, a.meta)
    except (OSError, ValueError) as e:
        print(f"[vlaeval_results] {e}", file=sys.stderr)
        return 2
    copied, mismatch = collect_videos(a.output_dir, agg, a.suite, a.video_dir, a.caption)
    lines.append(f"n_videos: {copied}")
    with open(a.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if mismatch:
        print(f"[vlaeval_results] warning: {mismatch} video file outcome(s) differ from the aggregate "
              "(the aggregate wins)", file=sys.stderr)
    k, n, _ = counts(agg)
    print(f"aggregate: {os.path.basename(path)}  videos: {copied}")
    print(f"{k}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
