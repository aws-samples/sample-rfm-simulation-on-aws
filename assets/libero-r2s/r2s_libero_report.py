"""Score a LIBERO real2sim swap run: public scene vs reconstructed objects (runs on the instance; stdlib).

  python3 r2s_libero_report.py count <eval log>
      Print "<successes> <episodes>" parsed from one evaluation log (OpenVLA-OFT run_libero_eval.py
      log file or openpi examples/libero/main.py output: "Task: <instruction>" ... "Success: True|False").
  python3 r2s_libero_report.py final --out <dir> --cond public=<log> --cond collision=<log> ...
                                     [--tasks tasks.json] [--objects objects.json] [--check check.json]
                                     [--suite libero_object] [--model name]
      Write <dir>/results.json and <dir>/summary.txt: per condition k/n, rate and 95% Wilson interval,
      drop vs the public scene in percentage points, and per-object (= per task, the task's object of
      interest) success, plus the object build manifest and the physics check.
"""
import json
import math
import os
import re
import sys

TASK_RE = re.compile(r"(?:^|[\s|:])Task: (.+?)\s*$")
SUCCESS_RE = re.compile(r"(?:^|[\s|:])Success: (True|False)\b")


def wilson(k, n, z=1.96):
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def parse_log(path):
    """[(task instruction, success bool)] in episode order."""
    out, task = [], None
    with open(path, errors="replace") as f:
        for line in f:
            m = TASK_RE.search(line)
            if m:
                task = m.group(1).strip()
                continue
            m = SUCCESS_RE.search(line)
            if m:
                out.append((task or "?", m.group(1) == "True"))
    return out


def tally(eps):
    k = sum(1 for _, s in eps if s)
    return k, len(eps)


def fmt(k, n):
    if not n:
        return "-"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.2f} [{lo:.2f}, {hi:.2f}]"


def final(out_dir, conds, tasks=None, objects=None, check=None, suite="-", model="-"):
    per_task_obj = {}
    for t in tasks or []:
        per_task_obj[" ".join(t["language"].split()).lower()] = ",".join(t["objects"])
    res = dict(version="r2s-libero-report-v1", suite=suite, model=model, conditions={}, per_object={})
    base = None
    for name, log in conds:
        eps = parse_log(log) if os.path.exists(log) else []
        k, n = tally(eps)
        lo, hi = wilson(k, n)
        row = dict(k=k, n=n, rate=round(k / n, 4) if n else None,
                   ci95=[round(lo, 4), round(hi, 4)] if n else None, log=os.path.basename(log))
        if name == "public":
            base = row
        res["conditions"][name] = row
        for task, ok in eps:
            obj = per_task_obj.get(task.lower(), task)
            cell = res["per_object"].setdefault(obj, {}).setdefault(name, dict(k=0, n=0))
            cell["k"] += int(ok)
            cell["n"] += 1
    for name, row in res["conditions"].items():
        if base and base["n"] and row["n"] and name != "public":
            row["drop_pp_vs_public"] = round((base["k"] / base["n"] - row["k"] / row["n"]) * 100, 1)
    if objects:
        res["objects"] = objects
    if check:
        res["physics_check"] = dict(unstable=check.get("unstable", []), steps=check.get("steps"),
                                    limits=check.get("limits"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    lines = [f"suite: {suite}", f"model: {model}", "benchmark: libero_r2s"]
    for name, row in res["conditions"].items():
        lines.append(f"{name}: {fmt(row['k'], row['n'])}"
                     + (f" (drop {row['drop_pp_vs_public']:+.1f} pp)" if "drop_pp_vs_public" in row else ""))
    lines.append("per_object:")
    for obj, cells in sorted(res["per_object"].items()):
        lines.append(f"  {obj}: " + "; ".join(f"{c} {cells[c]['k']}/{cells[c]['n']}" for c in res["conditions"]
                                              if c in cells))
    if objects:
        lines.append("objects (cluster, scale, extent ratio vs stock):")
        for name, o in sorted(objects.get("objects", {}).items()):
            lines.append(f"  {name}: cluster {o['cluster']}, scale {o['scale']}, extent {o['extent_ratio']}, "
                         f"mass {o['mass_kg']} kg, gt chamfer {o.get('gt_chamfer_m')} m")
        if objects.get("kept_stock"):
            lines.append("  kept stock (unstable in the physics check): " + ", ".join(objects["kept_stock"]))
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return res


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    if argv[0] == "count":
        k, n = tally(parse_log(argv[1]))
        print(k, n)
        return 0
    if argv[0] != "final":
        raise SystemExit(__doc__)
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--cond", action="append", default=[], help="name=log (repeat; first public = reference)")
    p.add_argument("--tasks")
    p.add_argument("--objects")
    p.add_argument("--check")
    p.add_argument("--suite", default="-")
    p.add_argument("--model", default="-")
    a = p.parse_args(argv[1:])
    conds = [tuple(c.split("=", 1)) for c in a.cond]

    def load(x):
        return json.load(open(x)) if x and os.path.exists(x) else None

    res = final(a.out, conds, load(a.tasks), load(a.objects), load(a.check), a.suite, a.model)
    print(open(os.path.join(a.out, "summary.txt")).read(), end="")
    return 0 if any(r["n"] for r in res["conditions"].values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
