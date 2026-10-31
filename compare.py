"""
compare.py — is policy A better than policy B, or is it a tie? (`vlasim results --compare A B`)

Two policies are compared row by row (one row per suite and evaluation setting). Each row gets the k/n of
both sides, a two-sided Fisher exact p, a Benjamini-Hochberg q over all rows of the comparison, and a
verdict: A>B, B>A or tie (q < alpha decides; --raw-p uses p instead). Pure Python, no SciPy.

    python vlasim.py results --compare openvla-oft pi                 # latest run of each, matched on suite + condition
    python vlasim.py results --compare openvla-oft:S0 openvla-oft:S3b # same policy, two settings, matched on suite
    python vlasim.py results --compare oft pi --from-csv cells.csv    # offline: a table of model,suite,condition,successes,episodes

A side is `<stack>[:<condition>]`. Without a condition on either side, rows are matched on (suite, condition);
with a condition, that side is restricted to it and rows are matched on suite.

Paired test: when both sides of a row have per-episode outcomes (video names `episode=<i>--success=<bool>--task=<t>`,
or --episodes JSON), the episodes are paired by (task, k-th occurrence of that task) — the start-state index in
LIBERO — and an exact McNemar p is printed next to Fisher. Only same-start-state pairs are counted.

Re-run null (--all-runs): when one stack has two or more finished runs with the same suite, condition and stack
fingerprint, those runs are compared with the same test. That is the noise floor: a verdict between A and B is
only meaningful if re-running the same stack does not produce verdicts of its own.
"""
import collections
import csv
import json
import math
import re

EPISODE_RE = re.compile(r"episode=(\d+)--success=(True|False)--task=([^/]+?)\.mp4$")
REL_TOL = 1 + 1e-7  # same tolerance as R's fisher.test / SciPy's binomtest for "as or less likely" tables


def _logc(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_two_sided(k1, n1, k2, n2):
    """Two-sided Fisher exact p for successes k1/n1 vs k2/n2 (sum of tables no more likely than the observed one)."""
    if n1 <= 0 or n2 <= 0:
        return 1.0
    s, n = k1 + k2, n1 + n2
    lo, hi = max(0, s - n2), min(s, n1)
    logs = [_logc(n1, x) + _logc(n2, s - x) - _logc(n, s) for x in range(lo, hi + 1)]
    obs = logs[k1 - lo]
    p = sum(math.exp(v) for v in logs if v <= obs + math.log(REL_TOL))
    return min(1.0, p)


def binom_two_sided(k, n, p0=0.5):
    """Two-sided exact binomial p (SciPy binomtest default): sum of outcomes no more likely than k."""
    if n <= 0:
        return 1.0
    logs = [_logc(n, x) + x * math.log(p0) + (n - x) * math.log(1 - p0) for x in range(n + 1)]
    obs = logs[k]
    return min(1.0, sum(math.exp(v) for v in logs if v <= obs + math.log(REL_TOL)))


def mcnemar_exact(only_a, only_b):
    """Exact McNemar p from the discordant counts (A succeeded and B failed, B succeeded and A failed)."""
    return binom_two_sided(only_a, only_a + only_b) if only_a + only_b else 1.0


def bh(pvals):
    """Benjamini-Hochberg adjusted q-values, in input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [1.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, pvals[i] * m / rank)
        q[i] = min(1.0, prev)
    return q


def verdict(k1, n1, k2, n2, decide, alpha):
    if decide >= alpha:
        return "tie"
    return "A>B" if k1 / n1 > k2 / n2 else "B>A"


def parse_side(spec):
    stack, _, cond = spec.partition(":")
    return stack, (cond or None)


def pair_episodes(a, b):
    """a, b: lists of dicts (task, ep, success). Pair by (task, k-th occurrence of the task). Returns (n, only_a, only_b)."""
    def keyed(eps):
        seen, out = collections.Counter(), {}
        for e in sorted(eps, key=lambda e: (e["task"], e["ep"])):
            seen[e["task"]] += 1
            out[(e["task"], seen[e["task"]])] = bool(e["success"])
        return out
    ka, kb = keyed(a), keyed(b)
    common = set(ka) & set(kb)
    return (len(common), sum(1 for x in common if ka[x] and not kb[x]), sum(1 for x in common if kb[x] and not ka[x]))


def episodes_from_keys(keys):
    out = []
    for key in keys:
        m = EPISODE_RE.search(key)
        if m:
            out.append(dict(ep=int(m.group(1)), success=m.group(2) == "True", task=m.group(3)))
    return out


def compare(a_cells, b_cells, a_cond=None, b_cond=None, alpha=0.05, raw_p=False):
    """a_cells / b_cells: {(suite, condition): dict(k, n[, episodes])}. Returns the comparison rows, BH over all rows."""
    def pick(cells, cond):
        if cond is None:
            return {key: v for key, v in cells.items()}
        return {(s, c): v for (s, c), v in cells.items() if c == cond}
    A, B = pick(a_cells, a_cond), pick(b_cells, b_cond)
    by_suite = a_cond is not None or b_cond is not None
    rows = []
    for (s, c), va in sorted(A.items()):
        if by_suite:
            matches = [(key, v) for key, v in B.items() if key[0] == s]
            if len(matches) != 1:
                continue
            (_, cb), vb = matches[0]
        else:
            vb, cb = B.get((s, c)), c
            if vb is None:
                continue
        if not va.get("n") or not vb.get("n"):
            continue
        row = dict(suite=s, condition_a=c, condition_b=cb, k_a=va["k"], n_a=va["n"], k_b=vb["k"], n_b=vb["n"],
                   p=fisher_two_sided(va["k"], va["n"], vb["k"], vb["n"]))
        if va.get("episodes") and vb.get("episodes"):
            n, oa, ob = pair_episodes(va["episodes"], vb["episodes"])
            if n:
                row.update(paired_n=n, only_a=oa, only_b=ob, p_mcnemar=mcnemar_exact(oa, ob))
        rows.append(row)
    for r, q in zip(rows, bh([r["p"] for r in rows])):
        r["q"] = q
        r["verdict"] = verdict(r["k_a"], r["n_a"], r["k_b"], r["n_b"], r["p"] if raw_p else q, alpha)
    return rows


def rerun_null(runs, alpha=0.05):
    """runs: {(suite, condition, fingerprint): [(run_id, k, n), ...]} for ONE stack. Consecutive runs are compared."""
    out = []
    for (s, c, fp), lst in sorted(runs.items()):
        lst = sorted(lst)
        for (r1, k1, n1), (r2, k2, n2) in zip(lst, lst[1:]):
            if n1 and n2:
                p = fisher_two_sided(k1, n1, k2, n2)
                out.append(dict(suite=s, condition=c, run_1=r1, run_2=r2, k_1=k1, n_1=n1, k_2=k2, n_2=n2, p=p,
                                differs=p < alpha))
    return out


def cells_from_csv(path, model):
    """Offline input: CSV with model,suite,condition,successes,episodes (the study table format)."""
    cells = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["model"] == model:
                cells[(r["suite"], r["condition"])] = dict(k=int(r["successes"]), n=int(r["episodes"]))
    return cells


def attach_episodes(cells, path, model):
    """--episodes JSON: list of dict(model, suite, cond, episodes=[{task, ep|sub, success}])."""
    for r in json.load(open(path)):
        key = (r["suite"], r["cond"])
        if r.get("model") == model and key in cells:
            eps = [dict(task=e["task"], ep=(e.get("sub", 0), e.get("ep", 0)), success=e["success"]) for e in r["episodes"]]
            if len(eps) == cells[key]["n"]:
                cells[key]["episodes"] = eps
    return cells


def cells_from_rows(rows, stack_prefix):
    """S3 rows (results.collect) of every bucket whose stack starts with stack_prefix -> latest run per (suite, condition).
    Returns (cells, reruns) where reruns feeds rerun_null."""
    groups = collections.defaultdict(lambda: dict(k=0, n=0, episodes=[], fp={}, ok=True))
    for r in rows:
        if not r["bucket_stack"].startswith(stack_prefix):
            continue
        key = (r.get("suite", "-"), r.get("condition", "-"), r["bucket_stack"], r["region"], r["run"])
        g = groups[key]
        if r["status"] != "done" or not r.get("n"):
            g["ok"] = False
            continue
        g["k"] += r["k"] or 0
        g["n"] += r["n"]
        g["fp"].update(r.get("stack") or {})
        g["episodes"].extend(dict(e, task=f"{r['task']}/{e['task']}") for e in r.get("episodes") or [])
    cells, reruns = {}, collections.defaultdict(list)
    for (s, c, stack, region, run), g in sorted(groups.items(), key=lambda x: x[0][4]):
        if not g["ok"]:
            continue
        fp = " ".join(f"{k}={v}" for k, v in sorted(g["fp"].items()))
        reruns[(s, c, f"{stack}@{region} {fp}")].append((run, g["k"], g["n"]))
        cell = dict(k=g["k"], n=g["n"], run=f"{stack}@{region} {run}")
        if g["episodes"] and len(g["episodes"]) == g["n"]:
            cell["episodes"] = g["episodes"]
        cells[(s, c)] = cell  # sorted by run id -> the latest run wins
    return cells, reruns


def print_rows(a, b, rows, null, alpha, raw_p, out=print):
    def fmt(x):
        return f"{x:.4f}" if x >= 1e-4 else f"{x:.1e}"
    cols = ["suite", "condition", f"A={a}", f"B={b}", "fisher_p", "bh_q", "verdict", "paired (only A / only B, p)"]
    table = []
    for r in rows:
        cond = r["condition_a"] if r["condition_a"] == r["condition_b"] else f"{r['condition_a']} vs {r['condition_b']}"
        paired = (f"{r['paired_n']} ({r['only_a']}/{r['only_b']}, p={fmt(r['p_mcnemar'])})" if "paired_n" in r else "-")
        table.append([r["suite"], cond, f"{r['k_a']}/{r['n_a']}", f"{r['k_b']}/{r['n_b']}", fmt(r["p"]),
                      fmt(r["q"]), r["verdict"], paired])
    w = [max(len(str(x)) for x in [c] + [t[i] for t in table]) for i, c in enumerate(cols)]
    out("  ".join(c.ljust(w[i]) for i, c in enumerate(cols)))
    for t in table:
        out("  ".join(str(x).ljust(w[i]) for i, x in enumerate(t)))
    v = collections.Counter(r["verdict"] for r in rows)
    out(f"\n{len(rows)} rows: A>B {v['A>B']} · B>A {v['B>A']} · tie {v['tie']} "
        f"(two-sided Fisher exact; verdict = {'p' if raw_p else 'Benjamini-Hochberg q'} < {alpha} over these rows)")
    if null is not None:
        if null:
            d = sum(1 for x in null if x["differs"])
            out(f"re-run null: {len(null)} pairs of runs with the same stack, suite, condition and fingerprint; "
                f"{d} differ at p < {alpha} (max |k_1 - k_2| = {max(abs(x['k_1'] - x['k_2']) for x in null)})")
        else:
            out("re-run null: none (needs two finished runs of one stack with the same suite, condition and "
                "fingerprint; use --all-runs)")
