"""Collect the stage outputs of one real2sim run into results.json, summary.txt and one caption per video.

summary.txt is the `key: value` file every target writes (results.py reads it): success_rate /
n_episodes / n_success are the floating-hand grasps in the reconstructed (`recon`) world, one episode
per graspable object; the baseline and reference worlds and the reconstruction quality ride along.

usage: python r2s_report.py --work <run dir> --videos <dir with the mp4s> --label <suite label>
       [--meta key=value ...]
  <run dir> holds (all optional): capture_info.json, metrics.json (ns-eval), train.json, frame.json
  (ns_to_world), segment/segment.json, grasp/grasp.json, grasp/grasp_ur5e.json
"""

import argparse
import json
import os

WORLDS = ("recon", "scenehull", "gt")
WORLD_TEXT = {"recon": "per-object collision meshes from the reconstruction (this pipeline)",
              "scenehull": "one static convex hull around all objects (baseline)",
              "gt": "ground-truth meshes at ground-truth poses (reference)"}


def _load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def collect(work):
    seg = _load(os.path.join(work, "segment", "segment.json")) or {}
    metrics = (_load(os.path.join(work, "metrics.json")) or {}).get("results", {})
    out = {
        "version": "r2s-report-v1",
        "capture": _load(os.path.join(work, "capture_info.json")),
        "splat": {"metrics": {k: metrics[k] for k in ("psnr", "ssim", "lpips") if k in metrics} or None,
                  "train": _load(os.path.join(work, "train.json")),
                  "frame_check": _load(os.path.join(work, "frame.json"))},
        "segment": {k: seg.get(k) for k in ("n_objects", "scene_hull_volume_m3", "usd", "params")} if seg else None,
        "segment_gt": {k: v for k, v in seg["gt"].items() if k != "per_object"} if seg.get("gt") else None,
        "grasp_hand": _load(os.path.join(work, "grasp", "grasp.json")),
        "grasp_ur5e": _load(os.path.join(work, "grasp", "grasp_ur5e.json")),
    }
    return out


def summary_lines(rep, label, meta):
    hand = (rep.get("grasp_hand") or {}).get("summary", {})
    ur5e = (rep.get("grasp_ur5e") or {}).get("summary", {})
    rec = hand.get("recon", {})
    n, k = rec.get("n"), rec.get("success")
    lines = {"model": "real2sim", "suite": label}
    lines.update(meta)
    lines["n_episodes"] = n if n is not None else ""
    lines["n_success"] = k if k is not None else ""
    lines["success_rate"] = "%.3f" % (k / n) if n else "unknown"
    for w in WORLDS:
        if w in hand:
            lines[f"hand_{w}"] = f"{hand[w]['success']}/{hand[w]['n']}"
        if w in ur5e:
            lines[f"ur5e_{w}"] = f"{ur5e[w]['success']}/{ur5e[w]['n']}"
    m = (rep.get("splat") or {}).get("metrics") or {}
    for key in ("psnr", "ssim", "lpips"):
        if key in m:
            lines[key] = "%.4f" % m[key]
    seg = rep.get("segment") or {}
    if seg.get("n_objects") is not None:
        lines["n_objects"] = seg["n_objects"]
    gt = rep.get("segment_gt") or {}
    for key in ("n_gt", "matched", "merged", "collision_volume_ratio_per_object"):
        if key in gt:
            lines["gt_" + key if not key.startswith("n_") else key] = gt[key]
    return [f"{a}: {b}" for a, b in lines.items()]


def captions(rep, videos):
    """{video file name: caption text} for the grasp videos (per target and world) and the splat orbit."""
    out = {}
    for kind, key in (("floating Robotiq 2F-85", "grasp_hand"), ("UR5e + Robotiq 2F-85", "grasp_ur5e")):
        for row in (rep.get(key) or {}).get("per_target", []):
            for w in WORLDS:
                r = row.get(w) or {}
                if not r.get("video"):
                    continue
                flags = ", ".join(f"{a}={r.get(a)}" for a in ("reached", "lifted", "undisturbed"))
                out[r["video"]] = (f"Real2Sim grasp replay ({kind})\nworld: {w} - {WORLD_TEXT[w]}\n"
                                   f"target cluster: {row['target']} {('(' + ', '.join(row['names']) + ')') if row.get('names') else ''}\n"
                                   f"success={bool(r.get('success'))} ({flags})\n"
                                   f"reach_err_m={r.get('reach_err_m')} lift_dz_m={r.get('lift_dz_m')} pushed_m={r.get('pushed_m')}\n")
    m = (rep.get("splat") or {}).get("metrics") or {}
    if os.path.isfile(os.path.join(videos, "splat_orbit.mp4")):
        out["splat_orbit.mp4"] = ("Real2Sim 3D Gaussian Splatting reconstruction (nerfstudio splatfacto), "
                                  "rendered along the training camera path\n"
                                  + "".join(f"{k}: {m[k]:.4f}\n" for k in ("psnr", "ssim", "lpips") if k in m))
    return {k: v for k, v in out.items() if os.path.isfile(os.path.join(videos, k))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--meta", nargs="*", default=[], help="extra key=value lines for summary.txt")
    a = ap.parse_args()
    rep = collect(a.work)
    meta = dict(kv.split("=", 1) for kv in a.meta if "=" in kv)
    rep["meta"] = meta
    json.dump(rep, open(os.path.join(a.work, "results.json"), "w"), indent=1, default=float)
    open(os.path.join(a.work, "summary.txt"), "w").write("\n".join(summary_lines(rep, a.label, meta)) + "\n")
    for name, text in captions(rep, a.videos).items():
        open(os.path.join(a.videos, os.path.splitext(name)[0] + ".txt"), "w").write(text)
    print(open(os.path.join(a.work, "summary.txt")).read(), end="")


if __name__ == "__main__":
    main()
