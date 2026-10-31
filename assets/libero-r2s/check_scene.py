"""Physics check of the swapped LIBERO scenes before any policy runs (no rendering, CPU is enough).

usage: python check_scene.py --suite libero_object --tasks 0,1,5 --dirs stable_hope_objects,stable_hope_objects_r2s_full
                             [--objects milk,cream_cheese] [--steps 40] [--out check.json]
                             [--unstable-out unstable.txt] [--warn-only]

For every asset directory and task: build the LIBERO env (robosuite, no renderer), load the task's first
fixed initial state, hold the robot still (zero action, gripper open) for --steps control steps and
compare where every HOPE object comes to rest with where the stock object rests from the same initial
state (stock dir first; without it, against the pose after the first step). A replaced object that rests
lower/higher, falls over or slides away would make the evaluation measure the asset, not the policy:
exit 1 (unless --warn-only). Needs hope_objects.py patched by r2s_select.py patch-hope.
"""
import argparse
import json
import math
import os
import sys

DRIFT_M = 0.03     # horizontal + vertical displacement of a resting object
SINK_M = 0.01      # drop below its initial height
TILT_DEG = 20.0
STOCK = "stable_hope_objects"


def quat_to_mat(q):
    """3x3 rotation matrix (columns = the body's local axes in world) of a MuJoCo w x y z quaternion."""
    import numpy as np
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--dirs", required=True)
    p.add_argument("--objects", default="", help="replaced objects (checked strictly); default: every HOPE object")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--out")
    p.add_argument("--warn-only", action="store_true")
    p.add_argument("--unstable-out", help="write the unstable replaced objects (comma-separated) to this file")
    a = p.parse_args(argv)

    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    from libero.libero.envs.objects import hope_objects

    if not hasattr(hope_objects, "R2S_HOPE_DIR"):
        raise SystemExit("check_scene: hope_objects.py is not patched (r2s_select.py patch-hope)")
    strict = {x for x in a.objects.split(",") if x}
    suite = benchmark.get_benchmark_dict()[a.suite]()
    dirs = [x for x in a.dirs.split(",") if x]
    dirs.sort(key=lambda d: d != STOCK)  # stock first: its resting poses are the reference
    rest = {}  # (task, object) -> (pos, rotation) of the stock object at the end of the hold
    report, bad = [], []
    for d in dirs:
        hope_objects.R2S_HOPE_DIR = d
        for tid in [int(x) for x in a.tasks.split(",")]:
            task = suite.get_task(tid)
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = ControlEnv(bddl_file_name=bddl, has_renderer=False, has_offscreen_renderer=False,
                             use_camera_obs=False, camera_heights=64, camera_widths=64)
            env.seed(7)
            env.reset()
            env.set_init_state(suite.get_task_init_states(tid)[0])
            sim = env.sim
            hope = {n: o for n, o in env.env.objects_dict.items() if type(o).__module__ == hope_objects.__name__}

            def pose(o):
                bid = sim.model.body_name2id(o.root_body)
                return np.array(sim.data.body_xpos[bid]), quat_to_mat(sim.data.body_xquat[bid])

            # Right after set_init_state the body poses still hold the reset() placement, and LIBERO's initial
            # states release the objects slightly above the table (measured: stock milk z 0.035 -> 0.086 ->
            # 0.125 -> rests at 0.070 by step 6). The fallback reference is therefore the pose after one step.
            env.step([0.0] * 6 + [-1.0])
            first = {n: pose(o) for n, o in hope.items()}
            for _ in range(a.steps):
                env.step([0.0] * 6 + [-1.0])
            for n, o in hope.items():
                p1, r1 = pose(o)
                if d == STOCK:
                    rest[(tid, n)] = (p1, r1)
                ref = rest.get((tid, n))
                against = "stock" if ref is not None else "first_step"
                p0, r0 = ref if ref is not None else first[n]
                up = int(np.argmax(np.abs(r0[2])))  # local axis closest to world z
                tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(r0[:, up], r1[:, up]))))))
                cat = n.rsplit("_", 1)[0]
                rec = dict(dir=d, task=tid, object=n, category=cat, against=against,
                           drift_m=round(float(np.linalg.norm((p1 - p0)[:2])), 4),
                           sink_m=round(float(p0[2] - p1[2]), 4), tilt_deg=round(tilt, 1))
                rec["ok"] = rec["drift_m"] <= DRIFT_M and abs(rec["sink_m"]) <= SINK_M and rec["tilt_deg"] <= TILT_DEG
                report.append(rec)
                if not rec["ok"] and (not strict or cat in strict) and d != STOCK:
                    bad.append(rec)
            env.close()
            worst = max((r["drift_m"] for r in report if r["dir"] == d and r["task"] == tid), default=0)
            print(f"check_scene {d} task {tid}: {len(hope)} HOPE objects, worst drift {worst:.4f} m")
    unstable = sorted({r["category"] for r in bad})
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(steps=a.steps, limits=dict(drift_m=DRIFT_M, sink_m=SINK_M, tilt_deg=TILT_DEG),
                           objects=report, unstable=bad, unstable_objects=unstable), f, indent=1)
    for r in bad:
        print(f"check_scene UNSTABLE: {r}")
    if a.unstable_out:
        with open(a.unstable_out, "w") as f:
            f.write(",".join(unstable) + "\n")
    if bad and not a.warn_only:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
