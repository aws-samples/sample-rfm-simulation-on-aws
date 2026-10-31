#!/usr/bin/env python3
"""rlinf_report.py — results of an RLinf evaluate -> RL -> re-evaluate run (rlinf-oft-libero target).

Runs on the instance host with the system python3 (stdlib only; TensorBoard event files are read with the
`tensorboard` package inside the RLinf image, see `curve`). Subcommands:

  success <eval.log>                         print the last eval/success_once value (or nothing)
  curve <tb_dir> <out_prefix>                (inside the image) training curve -> <out_prefix>.csv / .json
  phase <out.json> --phase baseline|rl ...   one phase record (rate, n, k, Wilson interval, checkpoint)
  report <baseline.json> <rl.json> <out_dir> results.json + summary.txt (before / after, delta, intervals)

The per-task summary.txt files that `vlasim results` reads are written from the phase records
(`phase ... --summary <path>`), so the two phases show up as two rows: condition `baseline` and
`rl-grpo-<N>steps`.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

# RLinf prints its metric dict at the end of an eval; accept "eval/success_once: 0.52", "'eval/success_once': 0.52",
# "eval/success_once=0.52" and "eval/success_once | 0.52" (rich table), optionally wrapped in tensor(...) or, as
# RLinf v0.4 prints it, numpy's array(0.5, dtype=float32).
SUCCESS_RE = re.compile(r"eval/success_once['\"]?\s*[:=|]\s*(?:tensor\(|array\()?\s*([0-9]*\.?[0-9]+(?:[eE]-?\d+)?)")
CURVE_TAGS = ("env/success_once", "env/return", "env/episode_len", "rollout/rewards", "train/actor/policy_loss",
              "train/actor/approx_kl", "train/actor/grad_norm", "time/step")


def wilson(k, n, z=1.96):
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def parse_success(text):
    vals = [float(m.group(1)) for m in SUCCESS_RE.finditer(text or "")]
    vals = [v for v in vals if 0.0 <= v <= 1.0]
    return vals[-1] if vals else None


def phase_record(phase, rate, n, checkpoint, condition, extra=None):
    k = None if rate is None or not n else int(round(rate * n))
    lo, hi = wilson(k, n) if k is not None else (None, None)
    rec = dict(phase=phase, condition=condition, success_rate=rate, n_episodes=n if rate is not None else None,
               n_success=k, ci95_lo=lo, ci95_hi=hi, checkpoint=checkpoint)
    rec.update(extra or {})
    return rec


def summary_lines(rec, common):
    """key: value lines for task-<N>/summary.txt (results.py KV format)."""
    out = dict(common)
    out.update(condition=rec["condition"], phase=rec["phase"], checkpoint=rec["checkpoint"])
    if rec["success_rate"] is not None:
        out.update(success_rate=f"{rec['success_rate']:.4f}", n_episodes=rec["n_episodes"], n_success=rec["n_success"])
    else:
        out["success_rate"] = "unknown"
    for key in ("rl_steps", "rl_steps_requested", "rl_profile", "train_config", "eval_config", "eval_num_envs", "eval_seed"):
        if rec.get(key) is not None:
            out[key] = rec[key]
    return "".join(f"{k}: {v}\n" for k, v in out.items())


def _fmt(rec):
    if rec.get("success_rate") is None:
        return "unknown"
    s = f"{rec['success_rate'] * 100:.1f}% ({rec['n_success']}/{rec['n_episodes']}"
    if rec.get("ci95_lo") is not None:
        s += f", 95% CI {rec['ci95_lo'] * 100:.1f}-{rec['ci95_hi'] * 100:.1f}%"
    return s + ")"


def report(baseline, rl, meta=None):
    meta = dict(meta or {})
    delta = None
    if baseline.get("success_rate") is not None and rl.get("success_rate") is not None:
        delta = rl["success_rate"] - baseline["success_rate"]
    data = dict(meta, rows=[baseline, rl], delta_pp=None if delta is None else round(delta * 100, 2))
    lines = [f"RLinf evaluate -> RL -> re-evaluate ({meta.get('suite', '?')}, profile {meta.get('rl_profile', '?')}, "
             f"{rl.get('rl_steps', '?')} GRPO steps)",
             f"baseline (SFT {baseline.get('checkpoint', '?')}): {_fmt(baseline)}",
             f"rl       ({rl.get('checkpoint', '?')}): {_fmt(rl)}"]
    if delta is not None:
        lines.append(f"delta: {delta * 100:+.1f} points")
        if baseline.get("ci95_hi") is not None and rl.get("ci95_lo") is not None:
            sep = rl["ci95_lo"] > baseline["ci95_hi"] or rl["ci95_hi"] < baseline["ci95_lo"]
            lines.append("intervals: " + ("disjoint" if sep else "overlap (difference not resolved at this n)"))
    if meta.get("published_sft") is not None:
        lines.append(f"published (RLinf model card, 500 episodes): SFT {meta['published_sft'] * 100:.2f}% -> "
                     f"GRPO {meta['published_rl'] * 100:.2f}%")
    return data, "\n".join(lines) + "\n"


def tb_scalars(tb_dir):
    """{tag: [(step, value), ...]} from every event file under tb_dir (needs the tensorboard package)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: PLC0415
    out = {}
    for ev in sorted(Path(tb_dir).rglob("events.out.tfevents.*")):
        acc = EventAccumulator(str(ev), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            out.setdefault(tag, []).extend((s.step, s.value) for s in acc.Scalars(tag))
    return out


def curve_rows(scalars, tags=CURVE_TAGS):
    """One row per step with the CURVE_TAGS present (last value wins per step)."""
    keep = [t for t in tags if t in scalars] + sorted(t for t in scalars if t.startswith("env/") and t not in tags)
    steps = {}
    for tag in keep:
        for step, val in scalars[tag]:
            steps.setdefault(int(step), {})[tag] = float(val)
    return keep, [dict(step=s, **steps[s]) for s in sorted(steps)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("success")
    p.add_argument("log")
    p = sub.add_parser("curve")
    p.add_argument("tb_dir")
    p.add_argument("out_prefix")
    p = sub.add_parser("phase")
    p.add_argument("out")
    p.add_argument("--phase", choices=("baseline", "rl"), required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--meta", default="{}", help="JSON object merged into the record")
    p.add_argument("--common", default="{}", help="JSON object of summary.txt lines shared by both phases")
    p.add_argument("--summary", help="also write a task summary.txt here")
    p = sub.add_parser("report")
    p.add_argument("baseline")
    p.add_argument("rl")
    p.add_argument("out_dir")
    p.add_argument("--meta", default="{}")
    a = ap.parse_args(argv)

    if a.cmd == "success":
        rate = parse_success(Path(a.log).read_text(errors="replace"))
        if rate is not None:
            print(f"{rate:.6f}")
        return 0
    if a.cmd == "curve":
        keep, rows = curve_rows(tb_scalars(a.tb_dir))
        Path(a.out_prefix + ".json").write_text(json.dumps(dict(tags=keep, rows=rows), indent=1))
        with open(a.out_prefix + ".csv", "w") as fh:
            fh.write(",".join(["step", *keep]) + "\n")
            for r in rows:
                fh.write(",".join([str(r["step"]), *("" if t not in r else f"{r[t]:.6g}" for t in keep)]) + "\n")
        print(f"curve: {len(rows)} steps, tags {', '.join(keep) or 'none'}")
        return 0 if rows else 1
    if a.cmd == "phase":
        log = Path(a.log)
        rate = parse_success(log.read_text(errors="replace")) if log.exists() else None
        rec = phase_record(a.phase, rate, a.n, a.checkpoint, a.condition, json.loads(a.meta))
        Path(a.out).write_text(json.dumps(rec, indent=1))
        if a.summary:
            Path(a.summary).write_text(summary_lines(rec, json.loads(a.common)))
        print(f"{a.phase}: {_fmt(rec)}")
        return 0 if rate is not None else 1
    if a.cmd == "report":
        data, text = report(json.loads(Path(a.baseline).read_text()), json.loads(Path(a.rl).read_text()),
                            json.loads(a.meta))
        out = Path(a.out_dir)
        (out / "results.json").write_text(json.dumps(data, indent=1))
        (out / "summary.txt").write_text(text)
        sys.stdout.write(text)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
