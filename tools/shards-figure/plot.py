"""Figure for docs/shards.md: one RoboLab-120 benchmark as 16 one-GPU shards vs one 8-GPU instance.

    python tools/shards-figure/plot.py extract <16-shard fanout-result.json> <1-shard fanout-result.json>
    python tools/shards-figure/plot.py render

`extract` writes docs/data/shards-robolab120.json (only the numbers the figure uses); `render` turns that file
into docs/images/shards-robolab120.svg. No plotting library: the SVG is written by hand so the figure can be
rebuilt anywhere the repo's Python runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "data" / "shards-robolab120.json"
SVG = ROOT / "docs" / "images" / "shards-robolab120.svg"
US_G6E_48XL = 30.13118  # On-Demand $/h, us-east-1 / us-west-2 (AWS Price List API, 2026-10-30)

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SERIES = {"16": "#2a78d6", "1": "#eb6834"}  # categorical slots 1 and 2 (validated pair)


def extract(sixteen: str, one: str) -> None:
    out = []
    for path in (sixteen, one):
        d = json.loads(Path(path).read_text())
        for r in d["runs"]:
            att = [a for s in r["shards"] for a in s["attempts"] if a.get("rc") == 0]
            out.append(dict(
                layout=str(d["shards"]), run=r["run"], k=r["n_success"], n=r["n_episodes"],
                ci=r["ci95_wilson"], usd=r["cost"]["usd"], instance_h=r["cost"]["instance_hours"],
                wall_h=round(r["cost"]["wall_clock_s"] / 3600, 2),
                instance_type=att[0]["instance_type"], regions=sorted({a["region"] for a in att})))
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(dict(
        benchmark="RoboLab-120 full protocol (120 tasks x 10 episodes), policy pi0.5-DROID (robolab-pi05)",
        measured="2026-11-07/08", runs=out), indent=1) + "\n")
    print(DATA)


def _t(x, y, s, size=12, fill=INK2, anchor="start", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def render() -> None:
    d = json.loads(DATA.read_text())
    runs = d["runs"]
    for r in runs:
        r["usd_us"] = round(r["instance_h"] * US_G6E_48XL, 2) if r["instance_type"] == "g6e.48xlarge" else r["usd"]
    W, H = 1260, 480
    pw, ph, top, gap, left = 250, 250, 96, 110, 180
    rows = [f"{'16 x g6e.4xlarge' if r['layout'] == '16' else '1 x g6e.48xlarge'} · run {r['run']}" for r in runs]
    panels = [
        ("Success rate (95% Wilson)", "%", 0.20, 0.32, lambda r: r["k"] / r["n"] * 100,
         lambda v, r: f"{r['k']}/{r['n']}"),
        ("Cost per run", "USD", 0.0, 300.0, lambda r: r["usd"], lambda v, r: f"${v:.0f}"),
        ("Wall clock per run", "hours", 0.0, 12.0, lambda r: r["wall_h"], lambda v, r: f"{v:.1f} h"),
    ]
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif" role="img" '
         f'aria-label="RoboLab-120 as 16 one-GPU shards versus one 8-GPU instance: success, cost, wall clock">',
         f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
         _t(24, 30, "One RoboLab-120 benchmark (1200 episodes, pi0.5-DROID): 16 one-GPU shards vs one 8-GPU instance",
            15, INK, weight="600"),
         _t(24, 52, "Same seed and episodes, two runs each. Measured On-Demand cost from EC2 instance lifetimes.", 12)]
    # legend
    lx = 24
    for key, name in (("16", "16 shards x g6e.4xlarge (1 L40S each)"), ("1", "1 x g6e.48xlarge (8 L40S)")):
        s.append(f'<rect x="{lx}" y="64" width="12" height="12" rx="2" fill="{SERIES[key]}"/>')
        s.append(_t(lx + 18, 74.5, name, 12, INK2))
        lx += 290
    band = ph / len(runs)
    for pi, (title, unit, lo, hi, val, lab) in enumerate(panels):
        x0 = left + pi * (pw + gap)
        w = pw - (40 if pi == 0 else 0)
        sx = lambda v: x0 + (v - lo) / (hi - lo) * w  # noqa: E731
        s.append(_t(x0, top + 6, title, 13, INK, weight="600"))
        for g in range(5):
            gx = x0 + w * g / 4
            s.append(f'<line x1="{gx:.1f}" y1="{top + 16}" x2="{gx:.1f}" y2="{top + 16 + ph}" stroke="{GRID}"/>')
            tick = (lo + (hi - lo) * g / 4) * (100 if pi == 0 else 1)
            s.append(_t(gx, top + ph + 32, f"{tick:.0f}", 11, MUTED, "middle"))
        s.append(_t(x0 + w / 2, top + ph + 48, unit, 11, MUTED, "middle"))
        if pi == 0:  # published reference for the same policy and protocol
            rx = x0 + (0.28 - lo) / (hi - lo) * w
            s.append(f'<line x1="{rx:.1f}" y1="{top + 16}" x2="{rx:.1f}" y2="{top + 16 + ph}" stroke="{INK2}" '
                     f'stroke-dasharray="4 3"/>')
            s.append(_t(rx + 4, top + 26, "leaderboard 28.0%", 10.5, INK2))
        for i, r in enumerate(runs):
            cy = top + 16 + band * i + band / 2
            if pi == 0:
                s.append(_t(x0 - 8, cy + 4, rows[i], 11, INK2, "end"))
            v = val(r)
            c = SERIES[r["layout"]]
            if pi == 0:
                a, b = (x * 100 for x in r["ci"])
                lsc = lambda p: x0 + (p / 100 - lo) / (hi - lo) * w  # noqa: E731
                s.append(f'<g><title>{rows[i]}: {r["k"]}/{r["n"]} = {v:.1f}% (95% {a:.1f}-{b:.1f}%)</title>'
                         f'<line x1="{lsc(a):.1f}" y1="{cy}" x2="{lsc(b):.1f}" y2="{cy}" stroke="{c}" stroke-width="2"/>'
                         f'<circle cx="{lsc(v):.1f}" cy="{cy}" r="5" fill="{c}" stroke="{SURFACE}" stroke-width="2"/></g>')
                s.append(_t(lsc(b) + 8, cy + 4, lab(v, r), 11, INK2))
            else:
                bw = sx(v) - x0
                s.append(f'<g><title>{rows[i]}: {lab(v, r)}</title><rect x="{x0}" y="{cy - 9}" width="{bw:.1f}" '
                         f'height="18" rx="4" fill="{c}"/></g>')
                txt = lab(v, r)
                if pi == 1 and r["usd_us"] != r["usd"]:
                    xu = sx(r["usd_us"])
                    s.append(f'<line x1="{xu:.1f}" y1="{cy - 12}" x2="{xu:.1f}" y2="{cy + 12}" stroke="{INK}" '
                             f'stroke-width="2"/>')
                    txt += f" · at US price ${r['usd_us']:.0f}"
                s.append(_t(x0 + bw + 6, cy + 4, txt, 11, INK2))
    notes = [
        "Cost: the 8-GPU runs ran in ap-northeast-2 (no g6e.48xlarge capacity in us-west-2, eu-central-1, us-east-1); "
        "the black tick is the same instance-hours at the US price.",
        "Wall clock includes what happened: 16-shard runs include scheduler restarts and one shard queued 3.8 h "
        "(bug fixed); 8-GPU runs include 1.3 h of capacity search.",
        "Instance-hours per run: 16 shards 39.0 / 39.9 (16 setups), one instance 6.5 / 6.4. Leaderboard reference "
        "for pi0.5: 28.0% (336/1200).",
    ]
    for i, n in enumerate(notes):
        s.append(_t(24, H - 44 + i * 16, n, 10.5, INK2))
    s.append("</svg>")
    SVG.parent.mkdir(parents=True, exist_ok=True)
    SVG.write_text("\n".join(s) + "\n")
    print(SVG)


if __name__ == "__main__":
    if sys.argv[1:2] == ["extract"] and len(sys.argv) == 4:
        extract(sys.argv[2], sys.argv[3])
    elif sys.argv[1:] == ["render"]:
        render()
    else:
        sys.exit(__doc__)
