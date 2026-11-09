#!/usr/bin/env python3
"""Turn one vla-eval echo smoke into this project's summary.txt (stdlib only; the echo half of the results adapter).

A ``kind: echo`` task of the vlaeval target runs the harness's own smoke test of a benchmark image,
``vla-eval test -c <run config>`` (``cli/smoke.py`` ``run_benchmark_test``): an in-process ``EchoModelServer``
answers every observation with a zero action of the entry's ``action_dim`` (default 7), the benchmark container
runs 1 task x 1 episode capped at 50 steps, in a temporary output dir that the harness deletes afterwards. There is
no aggregate JSON and no video; the only result is the test's report on stdout (``cli/main.py`` ``_record`` and
``cli/smoke.py`` ``print_report``; rich prints plain text when stdout is not a TTY)::

      ✓ benchmark/eval-task-2: success_rate=0% (41.3s)
    ...
    Results: 1 passed, 0 failed, 0 skipped    total: 41.3s

This script reads that log and writes ``summary.txt`` (``key: value`` lines, the same reader as
``vlaeval_results.py``): suite, ``kind: echo_smoke``, ``echo_result: pass|fail|skip|unknown``, the test's message and
duration, a ``note`` that this is a plumbing check of the image and not a policy score, and every ``--meta`` given.
It deliberately writes **no** ``success_rate`` / ``n_success`` / ``n_episodes``: ``vlasim results`` then lists the
task without a rate and the protocol card has k / n ``unknown``. A zero-action episode says nothing about a policy.

``vla-eval test`` exits 0 on a skip (image not present, docker missing) as on a pass, so only a report of exactly
``1 passed, 0 failed, 0 skipped`` counts: the last line is then ``echo pass`` and the exit code 0. Anything else
still writes the summary and exits 3 (the caller marks the task failed); exit 2 when the log cannot be read.

Usage:
  vlaeval_echo.py --echo-log <vla-eval test log> --summary <summary.txt> --suite vlabench_echo \\
      [--init-states TEXT] [--meta key=value ...]
  (--output-dir / --video-dir / --caption are accepted and ignored, so the task loop calls both halves alike)
"""

import argparse
import re
import sys

LINE_RE = re.compile(r"^\s*\S+\s+benchmark/(\S+?):\s+(.*?)(?:\s+\((\d+(?:\.\d+)?)s\))?\s*$")
TOTALS_RE = re.compile(r"Results:\s*(\d+) passed,\s*(\d+) failed,\s*(\d+) skipped")
NOTE = ("echo smoke (vla-eval test: EchoModelServer, zero actions, 1 task x 1 episode, at most 50 steps): "
        "a plumbing check of the benchmark image, not a policy score")


def report(text):
    """(status, message, seconds, (passed, failed, skipped) or None) of a `vla-eval test` log. status: pass / fail /
    skip / unknown. The message is the last per-test progress line; the status comes from the report's totals."""
    message, seconds = "", ""
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if m:
            message, seconds = m.group(2).strip(), m.group(3) or ""
    totals = TOTALS_RE.findall(text)
    if not totals:
        return "unknown", message or "no vla-eval test report in the log", seconds, None
    passed, failed, skipped = (int(x) for x in totals[-1])
    if failed:
        status = "fail"
    elif skipped:
        status = "skip"
    elif passed == 1:
        status = "pass"
    else:
        status = "unknown"
    return status, message, seconds, (passed, failed, skipped)


def summary_lines(text, suite, meta, init_states=""):
    status, message, seconds, totals = report(text)
    out = [
        ("suite", suite),
        ("env_name", suite),
        ("kind", "echo_smoke"),
        ("echo_result", status),
        ("echo_message", " ".join(message.split()) or "none"),
        ("echo_seconds", seconds or "unknown"),
        ("echo_tests", "unknown" if totals is None else "{} passed, {} failed, {} skipped".format(*totals)),
        ("note", NOTE),
        ("n_tasks", 1),
        ("n_episodes_per_task", 1),
        ("max_steps", "50 (vla-eval test cap)"),
        ("init_states", init_states or "the benchmark's first task, episode 0"),
        ("model", "EchoModelServer (vla-eval test, zero actions)"),
        ("checkpoint", "none (echo smoke)"),
        ("n_action_steps", "1 (echo server)"),
    ]
    have = {key for key, _ in out}
    for item in meta:
        key, _, value = item.partition("=")
        key = key.strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise ValueError(f"--meta {item!r}: key must be [a-z][a-z0-9_]*")
        if key in have:  # the echo values above win; meta only adds (as in vlaeval_results.py)
            continue
        out.append((key, " ".join(value.split())))
        have.add(key)
    return status, [f"{key}: {value}" for key, value in out]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--echo-log", required=True, help="stdout + stderr of `vla-eval test -c <run config>`")
    ap.add_argument("--summary", required=True, help="summary.txt to write")
    ap.add_argument("--suite", required=True, help="suite label")
    ap.add_argument("--init-states", default="", help="how the episode start state is chosen")
    ap.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE", help="extra summary.txt line")
    for ignored in ("--output-dir", "--video-dir", "--caption"):
        ap.add_argument(ignored, default="", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    try:
        with open(a.echo_log, encoding="utf-8", errors="replace") as f:
            text = f.read()
        status, lines = summary_lines(text, a.suite, a.meta, a.init_states)
    except (OSError, ValueError) as e:
        print(f"[vlaeval_echo] {e}", file=sys.stderr)
        return 2
    lines.append("n_videos: 0")
    with open(a.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if status != "pass":
        print(f"[vlaeval_echo] echo smoke {status}: not counted as a pass", file=sys.stderr)
        print(f"echo smoke: {status}")
        return 3
    print("echo pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
