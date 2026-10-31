#!/usr/bin/env python3
"""
results.py — collect evaluation results from every vla-sim-results-* bucket.

Every target writes <RUN_ID>/task-<N>/summary.txt (key: value lines). This
script reads them all and prints one row per (stack, suite, condition) with
k/n successes and a 95% Wilson interval, so runs under different knobs
(`condition`, `replan`, `step_scale`, ...) can be compared side by side.

    python vlasim.py results                        # all buckets, latest run per stack
    python vlasim.py results --all-runs             # every run, not just the latest
    python vlasim.py results --vla openvla-oft --since 2026-10-19
    python vlasim.py results --csv out.csv --json out.json
    python vlasim.py results --robustness           # LIBERO-Plus per-dimension drop vs the LIBERO baseline
    python vlasim.py results --matrix               # target x benchmark matrix with cost and latency
    python vlasim.py results --write-docs           # refresh the leaderboard block in docs/results.md
    python vlasim.py results --compare openvla-oft pi   # A>B / B>A / tie per suite and setting (compare.py)

Episode count, in order of preference: n_episodes (+ n_success) in summary.txt;
otherwise the per-episode videos (success=True/False) in task-N/videos/;
otherwise the rate is shown without k/n. A task with no summary.txt is
reported as "running/failed", never as 0.

LIBERO-Plus sweeps (deploy --libero-variant plus) write one task per perturbation dimension with
condition LIBERO-Plus/<dimension>, so they show up as one row per dimension. --robustness puts the
latest sweep of each stack next to its LIBERO baseline (the latest run of the same stack and suite
with the default condition) and prints the drop in percentage points:

    python vlasim.py results --robustness --vla smolvla

When a run was scored by the post-run judge (python vlasim.py judge --vla ...),
<RUN_ID>/judge.json is read too and its agreement with the simulator's success
flag is printed below the table.
"""
import argparse
import collections
import csv
import json
import math
import re
import sys
from pathlib import Path

BUCKET_RE = re.compile(r"^vla-sim-results-(?P<stack>.+)-(?P<region>[a-z]{2}-[a-z]+-\d)-(?P<acct>\d{12})$")
KV_RE = re.compile(r"^([A-Za-z_][\w]*): ?(.*)$", re.M)
RUN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{4}$")
PLUS_PREFIX = "LIBERO-Plus/"
BASELINE_CONDITIONS = ("-", "S0")  # the default stack (no knob) of every target
DOCS_RESULTS = Path(__file__).resolve().parent / "docs" / "results.md"


def wilson(k, n, z=1.96):
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _float(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


class S3:
    def __init__(self, boto3, region):
        self.boto3 = boto3
        self.clients = {}
        self.default_region = region

    def c(self, region):
        if region not in self.clients:
            self.clients[region] = self.boto3.client("s3", region_name=region)
        return self.clients[region]

    def prefixes(self, region, bucket, prefix):
        out = []
        for page in self.c(region).get_paginator("list_objects_v2").paginate(
                Bucket=bucket, Prefix=prefix, Delimiter="/"):
            out += [p["Prefix"] for p in page.get("CommonPrefixes", [])]
        return out

    def keys(self, region, bucket, prefix):
        out = []
        for page in self.c(region).get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            out += [o["Key"] for o in page.get("Contents", [])]
        return out

    def text(self, region, bucket, key):
        try:
            body = self.c(region).get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception:
            return None
        return body.decode("utf-8", "replace")


def task_row(s3, region, bucket, run, task_prefix, episodes=False):
    summ = s3.text(region, bucket, task_prefix + "summary.txt")
    task = task_prefix.rstrip("/").rsplit("/", 1)[-1]
    if summ is None:
        return dict(run=run, task=task, status="running/failed")
    kv = dict(KV_RE.findall(summ))
    rate = _float(kv.get("success_rate"))
    n = _int(kv.get("n_episodes"))
    k = _int(kv.get("n_success"))
    if n is None:
        vids = [x for x in s3.keys(region, bucket, task_prefix + "videos/") if x.endswith(".mp4")]
        ok = sum("success=True" in x for x in vids)
        bad = sum("success=False" in x for x in vids)
        if ok + bad:
            k, n = ok, ok + bad
    if k is None and n and rate is not None:
        k = round(rate * n)
    if rate is None and n:
        rate = k / n
    stack_fp = {x: v for x, v in kv.items() if x.startswith("stack_")}
    row = dict(run=run, task=task, status="done" if rate is not None else "no-rate",
               suite=kv.get("suite") or kv.get("env_name") or kv.get("task_suite") or "-",
               condition=(kv.get("condition") or "-").split()[0], k=k, n=n, rate=rate, stack=stack_fp)
    if kv.get("benchmark"):
        row.update(benchmark=kv["benchmark"], perturbation=kv.get("perturbation", "-"))
    if episodes:
        import compare
        eps = compare.episodes_from_keys(s3.keys(region, bucket, task_prefix + "videos/"))
        if eps and len(eps) == n:
            row["episodes"] = eps
    return row


def read_protocol_card(s3, region, bucket, task_prefix):
    """<task>/protocol.yaml (evaluation protocol card), parsed; None when absent or unreadable."""
    text = s3.text(region, bucket, task_prefix + "protocol.yaml")
    if not text:
        return None
    try:
        import yaml
        card = yaml.safe_load(text)
    except Exception:
        return None
    return card if isinstance(card, dict) and str(card.get("schema", "")).startswith("rfm-simulation/protocol-card/") \
        else None


def judge_summary(s3, region, bucket, run):
    """Compact view of <run>/judge.json (post-run judge), or None when the run was not judged."""
    text = s3.text(region, bucket, f"{run}/judge.json")
    if not text:
        return None
    try:
        report = json.loads(text)
        summ = report["summary"]
    except (ValueError, KeyError, TypeError):
        return {"error": "unreadable judge.json"}
    return dict(judge=report.get("judge", "judge"), model=report.get("model", ""),
                model_revision=str(report.get("model_revision", ""))[:8],
                threshold=report.get("success_threshold"), agreement=summ.get("agreement"),
                n_agree=summ.get("n_agree"), n_compared=summ.get("n_compared"), n_scored=summ.get("n_scored"),
                n_episodes=summ.get("n_episodes"), mean_score_sim_success=summ.get("mean_score_sim_success"),
                mean_score_sim_failure=summ.get("mean_score_sim_failure"), confusion=summ.get("confusion"))


def judge_line(j):
    if j.get("error"):
        return j["error"]
    agree = "n/a" if j["agreement"] is None else f"{j['agreement']:.2f} ({j['n_agree']}/{j['n_compared']})"
    return (f"agreement {agree}, scored {j['n_scored']}/{j['n_episodes']}, mean score sim-success "
            f"{j['mean_score_sim_success']} / sim-failure {j['mean_score_sim_failure']} "
            f"[{j['model'] or j['judge']}@{j['model_revision']}, success = score >= {j['threshold']}]")


def collect(args, boto3):
    s3 = S3(boto3, args.region)
    acct = boto3.client("sts").get_caller_identity()["Account"]
    buckets = []
    for b in boto3.client("s3").list_buckets()["Buckets"]:
        m = BUCKET_RE.match(b["Name"])
        if not m or m.group("acct") != acct:
            continue
        if args.vla and not m.group("stack").startswith(args.vla):
            continue
        if args.region and m.group("region") != args.region:
            continue
        buckets.append((b["Name"], m.group("stack"), m.group("region")))
    rows = []
    for bucket, stack, region in sorted(buckets):
        try:
            runs = sorted(p.rstrip("/") for p in s3.prefixes(region, bucket, "") if RUN_RE.match(p.rstrip("/")))
        except Exception as e:
            print(f"# skip {bucket}: {e.__class__.__name__}", file=sys.stderr)
            continue
        if args.since:
            runs = [r for r in runs if r >= args.since]
        if not args.all_runs:
            runs = runs[-1:]
        for run in runs:
            judge = None
            for t in s3.prefixes(region, bucket, run + "/"):
                if re.search(r"/task-\d+/$", t):
                    if judge is None:
                        judge = judge_summary(s3, region, bucket, run) or {}
                    r = task_row(s3, region, bucket, run, t, episodes=getattr(args, "compare", None) is not None)
                    r.update(bucket_stack=stack, region=region)
                    if judge:
                        r["judge"] = judge
                    rows.append(r)
    return rows


def aggregate(rows):
    agg = collections.OrderedDict()
    for r in sorted(rows, key=lambda x: (x["bucket_stack"], x["region"], x["run"])):
        key = (r["bucket_stack"], r["region"], r["run"], r.get("suite", "-"), r.get("condition", "-"))
        g = agg.setdefault(key, dict(k=0, n=0, rates=[], tasks=0, not_done=0, stack={}))
        g["tasks"] += 1
        if r["status"] != "done":
            g["not_done"] += 1
            continue
        g["stack"].update(r.get("stack") or {})
        if r.get("n"):
            g["k"] += r["k"] or 0
            g["n"] += r["n"]
        else:
            g["rates"].append(r["rate"])
    return agg


def _pool(rows):
    k = sum(r["k"] or 0 for r in rows if r.get("n"))
    n = sum(r["n"] for r in rows if r.get("n"))
    return k, n


def robustness(rows):
    """LIBERO-Plus view: per (stack, region, suite), the latest LIBERO-Plus run split by perturbation
    dimension, next to the latest baseline run (condition '-' or 'S0') of the same stack and suite.

    Returns a list of dicts (one per dimension + one 'all' row per sweep). k/n are pooled episode
    counts; drop_pp = (baseline rate - dimension rate) x 100. 'all' pools every dimension of the sweep:
    with the same number of variants per dimension it equals the mean over dimensions.
    """
    done = [r for r in rows if r.get("status") == "done"]
    groups = collections.OrderedDict()
    for r in sorted(done, key=lambda x: (x["bucket_stack"], x["region"], x.get("suite", "-"), x["run"])):
        groups.setdefault((r["bucket_stack"], r["region"], r.get("suite", "-")), []).append(r)
    out = []
    for (stack, region, suite), rs in groups.items():
        plus = [r for r in rs if str(r.get("condition", "")).startswith(PLUS_PREFIX)]
        if not plus:
            continue
        plus_run = max(r["run"] for r in plus)
        plus = [r for r in plus if r["run"] == plus_run]
        base = [r for r in rs if r.get("condition") in BASELINE_CONDITIONS]
        base_run = max((r["run"] for r in base), default=None)
        base = [r for r in base if r["run"] == base_run]
        bk, bn = _pool(base)
        brate = bk / bn if bn else None
        dims = collections.OrderedDict()
        for r in plus:
            dims.setdefault(r["condition"][len(PLUS_PREFIX):], []).append(r)
        for dim, drs in [*dims.items(), ("all", plus)]:
            k, n = _pool(drs)
            rate = k / n if n else None
            lo, hi = wilson(k, n)
            out.append(dict(
                stack=stack, region=region, suite=suite, perturbation=dim, run=plus_run,
                k_n=f"{k}/{n}" if n else "-", rate="" if rate is None else f"{rate:.2f}",
                ci95="" if lo is None else f"{lo:.2f}-{hi:.2f}",
                baseline="-" if brate is None else f"{brate:.2f} ({bk}/{bn})",
                baseline_run=base_run or "-",
                drop_pp="" if rate is None or brate is None else f"{(brate - rate) * 100:+.1f}"))
    return out


def _print_table(table, cols):
    def cell(t, c):
        v = str(t[c])
        return v if len(v) <= 48 else v[:45] + "..."
    width = {c: max(len(c), *(len(cell(t, c)) for t in table)) for c in cols}
    print("  ".join(c.ljust(width[c]) for c in cols))
    for t in table:
        print("  ".join(cell(t, c).ljust(width[c]) for c in cols))


def run_compare(a, boto3=None):
    import compare
    (sa, ca), (sb, cb) = (compare.parse_side(x) for x in a.compare)
    null = None
    if a.from_csv:
        A, B = compare.cells_from_csv(a.from_csv, sa), compare.cells_from_csv(a.from_csv, sb)
        if a.episodes:
            compare.attach_episodes(A, a.episodes, sa)
            compare.attach_episodes(B, a.episodes, sb)
    else:
        if boto3 is None:
            try:
                import boto3
            except ImportError:
                print("vlasim results: boto3 is not installed — run `python vlasim.py init` first.", file=sys.stderr)
                return 2
        rows = []
        for stack in dict.fromkeys((sa, sb)):
            a.vla = stack
            rows += collect(a, boto3)
        A, ra = compare.cells_from_rows(rows, sa)
        B, rb = compare.cells_from_rows(rows, sb)
        if a.all_runs:
            null = compare.rerun_null(ra, a.alpha) + (compare.rerun_null(rb, a.alpha) if sb != sa else [])
    if not A or not B:
        print(f"No finished cells for {'A' if not A else 'B'} ({sa if not A else sb}).")
        return 1
    rows = compare.compare(A, B, ca, cb, alpha=a.alpha, raw_p=a.raw_p)
    if not rows:
        print("No suite/condition row exists on both sides.")
        return 1
    compare.print_rows(a.compare[0], a.compare[1], rows, null, a.alpha, a.raw_p)
    if a.csv:
        cols = ["suite", "condition_a", "condition_b", "k_a", "n_a", "k_b", "n_b", "p", "q", "verdict",
                "paired_n", "only_a", "only_b", "p_mcnemar"]
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(a=a.compare[0], b=a.compare[1], alpha=a.alpha, raw_p=a.raw_p, rows=rows, rerun_null=null),
                      f, indent=1)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="vlasim results", description=__doc__.split("\n\n")[1])
    p.add_argument("--vla", help="only buckets whose stack name starts with this (e.g. openvla-oft)")
    p.add_argument("--region", help="only buckets in this region")
    p.add_argument("--since", help="only runs with RUN_ID >= this (e.g. 2026-10-19 or 2026-10-19-0558)")
    p.add_argument("--all-runs", action="store_true", help="every run, not just the latest per bucket")
    p.add_argument("--csv", help="write the aggregated table as CSV")
    p.add_argument("--json", help="write per-task rows and the aggregate as JSON")
    p.add_argument("--robustness", action="store_true",
                   help="LIBERO-Plus view: success per perturbation dimension of the latest sweep vs the latest "
                        "LIBERO baseline of the same stack (reads every run; --since still applies)")
    m = p.add_argument_group("matrix (leaderboard.py)", "target x benchmark matrix: latest comparable run per "
                             "cell, k/n, Wilson 95% CI, estimated cost and time per run, latency when recorded")
    m.add_argument("--matrix", action="store_true", help="print the matrix instead of the per-run table "
                   "(--csv / --json then write one row per cell)")
    m.add_argument("--write-docs", nargs="?", const=str(DOCS_RESULTS), metavar="PATH",
                   help="also replace the leaderboard block of PATH (default docs/results.md) with the matrix")
    m.add_argument("--all-conditions", action="store_true", help="add one row per non-baseline stack condition "
                   "(default: baseline conditions S0 / - / <Benchmark>/<setting> only)")
    m.add_argument("--condition", help="only this condition (exact label), one row per target")
    m.add_argument("--pool-hours", type=float, default=6.0, help="pool shards (other stacks, same condition and "
                   "fingerprint) started within this many hours of the latest run (default 6; 0 = no pooling)")
    m.add_argument("--no-cost", action="store_true", help="skip the cost / time / latency table (fewer S3 reads)")
    c = p.add_argument_group("compare (compare.py)", "two policies, row by row: k/n, two-sided Fisher exact, "
                             "Benjamini-Hochberg q over the rows, verdict A>B / B>A / tie; paired McNemar when "
                             "per-episode outcomes exist; re-run null with --all-runs")
    c.add_argument("--compare", nargs=2, metavar=("A", "B"), help="<stack>[:<condition>] for each side")
    c.add_argument("--from-csv", metavar="PATH", help="offline input: CSV with model,suite,condition,successes,episodes "
                   "(A and B are model names); no AWS access")
    c.add_argument("--episodes", metavar="PATH", help="with --from-csv: per-episode outcomes JSON for the paired test")
    c.add_argument("--alpha", type=float, default=0.05, help="significance level for the verdict (default 0.05)")
    c.add_argument("--raw-p", action="store_true", help="decide on the uncorrected Fisher p instead of the BH q")
    a = p.parse_args(argv)
    if a.compare:
        return run_compare(a)
    if a.robustness:
        a.all_runs = True
    try:
        import boto3
    except ImportError:
        print("vlasim results: boto3 is not installed — run `python vlasim.py init` first.", file=sys.stderr)
        return 2
    if a.matrix or a.write_docs:
        import leaderboard
        return leaderboard.run(a, boto3)
    rows = collect(a, boto3)
    if not rows:
        print("No results found (no vla-sim-results-* bucket with a RUN_ID/task-N/ prefix).")
        return 0
    if a.robustness:
        table = robustness(rows)
        if not table:
            print("No LIBERO-Plus runs found (deploy with --libero-variant plus, then re-run).")
            return 0
        cols = ["stack", "region", "suite", "perturbation", "run", "k_n", "rate", "ci95", "baseline", "drop_pp"]
        _print_table(table, cols)
        print("\nbaseline = latest run of the same stack and suite with the default condition "
              "(baseline_run in --csv / --json); drop_pp = baseline - rate, in percentage points.")
        if a.csv:
            with open(a.csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols + ["baseline_run"])
                w.writeheader()
                w.writerows(table)
        if a.json:
            with open(a.json, "w") as f:
                json.dump(dict(rows=rows, robustness=table), f, indent=1, default=str)
        return 0
    agg = aggregate(rows)
    table = []
    for (stack, region, run, suite, cond), g in agg.items():
        if g["n"]:
            rate = g["k"] / g["n"]
            lo, hi = wilson(g["k"], g["n"])
            kn = f"{g['k']}/{g['n']}"
        elif g["rates"]:
            rate, lo, hi, kn = sum(g["rates"]) / len(g["rates"]), None, None, "-"
        else:
            rate, lo, hi, kn = None, None, None, "-"
        table.append(dict(stack=stack, region=region, run=run, suite=suite, condition=cond, k_n=kn,
                          rate="" if rate is None else f"{rate:.2f}",
                          ci95="" if lo is None else f"{lo:.2f}-{hi:.2f}",
                          tasks=g["tasks"], not_done=g["not_done"],
                          fingerprint=" ".join(f"{k}={v}" for k, v in sorted(g["stack"].items()))))
    cols = ["stack", "region", "run", "suite", "condition", "k_n", "rate", "ci95", "tasks", "not_done"]
    _print_table(table, cols)
    fps = collections.OrderedDict()
    for t in table:
        if t["fingerprint"]:
            fps.setdefault(f"{t['stack']}@{t['region']} {t['run']}", t["fingerprint"])
    if fps:
        print("\nstack fingerprints:")
        for k, v in fps.items():
            print(f"  {k}: {v}")
    judges = collections.OrderedDict()
    for r in sorted(rows, key=lambda x: (x["bucket_stack"], x["region"], x["run"])):
        if r.get("judge"):
            judges.setdefault(f"{r['bucket_stack']}@{r['region']} {r['run']}", r["judge"])
    if judges:
        print("\npost-run judge (judge.json):")
        for k, j in judges.items():
            print(f"  {k}: {judge_line(j)}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols + ["fingerprint"])
            w.writeheader()
            w.writerows(table)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(rows=rows, table=table), f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
