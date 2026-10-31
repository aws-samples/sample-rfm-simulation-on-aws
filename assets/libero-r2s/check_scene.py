"""Physics check of the swapped LIBERO scenes before any policy runs (no rendering, CPU is enough).

usage: python check_scene.py --suite libero_object --tasks 0,1,5 --dirs stable_hope_objects,stable_hope_objects_r2s_full
                             [--objects milk,cream_cheese] [--steps 40] [--out check.json] [--warn-only]

For every asset directory and task: build the LIBERO env (robosuite, no renderer), load the task's first
fixed initial state, hold the robot still (zero action, gripper open) for --steps control steps and
measure every HOPE object's drift and tilt. A replaced object that sinks into the table, falls over or
slides away means its alignment is wrong and the evaluation would measure the asset, not the policy:
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


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--dirs", required=True)
    p.add_argument("--objects", default="", help="replaced objects (checked strictly); default: every HOPE object")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--out")
    p.add_argument("--warn-only", action="store_true")
    a = p.parse_args(argv)

    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    from libero.libero.envs.objects import hope_objects

    if not hasattr(hope_objects, "R2S_HOPE_DIR"):
        raise SystemExit("check_scene: hope_objects.py is not patched (r2s_select.py patch-hope)")
    strict = {x for x in a.objects.split(",") if x}
    suite = benchmark.get_benchmark_dict()[a.suite]()
    report, bad = [], []
    for d in [x for x in a.dirs.split(",") if x]:
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
                q = sim.data.body_xquat[bid]  # w x y z
                w, x, y, z = q
                zz = [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)]  # local axes' world z
                return np.array(sim.data.body_xpos[bid]), np.array(zz)

            start = {n: pose(o) for n, o in hope.items()}
            for _ in range(a.steps):
                env.step([0.0] * 6 + [-1.0])
            for n, o in hope.items():
                p0, z0 = start[n]
                p1, z1 = pose(o)
                up = int(np.argmax(np.abs(z0)))
                tilt = math.degrees(math.acos(max(-1.0, min(1.0, abs(z1[up]) / max(abs(z0[up]), 1e-9)))))
                cat = n.rsplit("_", 1)[0]
                rec = dict(dir=d, task=tid, object=n, category=cat, drift_m=round(float(np.linalg.norm(p1 - p0)), 4),
                           sink_m=round(float(p0[2] - p1[2]), 4), tilt_deg=round(tilt, 1))
                rec["ok"] = rec["drift_m"] <= DRIFT_M and rec["sink_m"] <= SINK_M and rec["tilt_deg"] <= TILT_DEG
                report.append(rec)
                if not rec["ok"] and (not strict or cat in strict) and d != "stable_hope_objects":
                    bad.append(rec)
            env.close()
            worst = max((r["drift_m"] for r in report if r["dir"] == d and r["task"] == tid), default=0)
            print(f"check_scene {d} task {tid}: {len(hope)} HOPE objects, worst drift {worst:.4f} m")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(steps=a.steps, limits=dict(drift_m=DRIFT_M, sink_m=SINK_M, tilt_deg=TILT_DEG),
                           objects=report, unstable=bad), f, indent=1)
    for r in bad:
        print(f"check_scene UNSTABLE: {r}")
    if bad and not a.warn_only:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
