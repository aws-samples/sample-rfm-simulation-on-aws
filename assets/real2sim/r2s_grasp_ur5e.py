"""Grasp replay through a UR5e arm (MuJoCo Menagerie UR5e + Robotiq 2F-85), same worlds as r2s_grasp_mj.

The floating-hand replay (r2s_grasp_mj) gives the clean collision answer. This one drives the same
top-down pick through an arm, so the video shows a robot: the arm stands on a pedestal at the table
edge on the target's outward side, every waypoint is solved by damped-least-squares IK on the `g_pinch`
site (position + orientation), and the arm's position servos track the joint path with gravity
compensation (qfrc_bias on the arm dofs, as the UR controller does).

Phases: home -> high (TCP 25 cm above, 1.0 s) -> pre (8 cm above, straight down 0.8 s) -> descend 1.0 s
-> settle 0.6 s -> close 0.8 s -> lift 10 cm 1.0 s -> hold 0.5 s. Scored as r2s_grasp_mj.

usage: MUJOCO_GL=egl python r2s_grasp_ur5e.py --seg <segment dir> --menagerie <dir> --out <dir>
       [--gt-scene .. --gt-meshes ..] [--targets 1,3] [--video]
"""

import argparse
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2s_grasp_mj import (  # noqa: E402
    M_OURS_TO_MOUNT,
    Recorder,
    build,
    grasp_for,
    load_scene,
    score,
    summarise,
    write_video,
)

ARM = ["ur_shoulder_pan_joint", "ur_shoulder_lift_joint", "ur_elbow_joint",
       "ur_wrist_1_joint", "ur_wrist_2_joint", "ur_wrist_3_joint"]
HOME = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
REACH_BACK = 0.42            # base distance from the target, along the outward direction


def rot_err(Rc, Rt):
    return 0.5 * (np.cross(Rc[:, 0], Rt[:, 0]) + np.cross(Rc[:, 1], Rt[:, 1]) + np.cross(Rc[:, 2], Rt[:, 2]))


class Arm:
    def __init__(self, m):
        self.m = m
        self.q_adr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ARM])
        self.dof = np.array([m.jnt_dofadr[m.joint(j).id] for j in ARM])
        self.act = np.array([m.actuator(j.replace("_joint", "")).id for j in ARM])
        self.grip = m.actuator("ur_g_fingers_actuator").id
        self.site = m.site("ur_g_pinch").id
        self.dI = mujoco.MjData(m)

    def ik(self, p, R, q0, iters=200):
        m, d = self.m, self.dI
        q = q0.copy()
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        e = np.zeros(6)
        for _ in range(iters):
            d.qpos[self.q_adr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            e = np.concatenate([p - d.site_xpos[self.site], rot_err(d.site_xmat[self.site].reshape(3, 3), R)])
            if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(e[3:]) < 2e-3:
                break
            mujoco.mj_jacSite(m, d, jp, jr, self.site)
            J = np.vstack([jp[:, self.dof], jr[:, self.dof]])
            q = q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        return q, float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))


def plan(arm, R, p_tcp):
    """Joint waypoints for high -> pre -> grasp -> lift; tries both closing-axis signs, keeps the one nearer HOME."""
    best = None
    for flip in (False, True):
        Rf = R @ np.diag([-1, -1, 1]) if flip else R
        q, ep, er = arm.ik(p_tcp + [0, 0, 0.08], Rf, HOME)
        if ep > 0.005 or er > 0.05:
            continue
        cost = float(np.abs(q - HOME).sum())
        if best is None or cost < best[0]:
            best = (cost, Rf, q)
    if best is None:
        return None
    _, Rf, q_pre = best
    # Come in from above: a joint-space sweep from HOME straight to `pre` drags the hand through the scene.
    # Solve the high point from q_pre (same IK branch), then walk back down to pre in a straight line.
    up_path, q, ep_h = [], q_pre, 0.0
    for s in np.linspace(0, 1, 30)[1:]:
        q, ep_h, _ = arm.ik(p_tcp + [0, 0, 0.08 + 0.17 * s], Rf, q)
        up_path.append(q)
    if ep_h > 0.005:
        return None
    q_high = up_path[-1]
    over = up_path[::-1][1:] + [q_pre]
    down, up, ep = [], [], 0.0
    for s in np.linspace(0, 1, 30)[1:]:
        q, ep, _ = arm.ik(p_tcp + [0, 0, 0.08 * (1 - s)], Rf, q)
        down.append(q)
    for s in np.linspace(0, 1, 30)[1:]:
        q, _, _ = arm.ik(p_tcp + [0, 0, 0.10 * s], Rf, q)
        up.append(q)
    return q_high, np.array(over), np.array(down), np.array(up), ep


def replay(m, arm, body, wp, p_tcp, rec):
    d = mujoco.MjData(m)
    q_high, over, down, up, _ = wp
    d.qpos[arm.q_adr] = HOME
    d.ctrl[arm.act] = HOME
    mujoco.mj_forward(m, d)
    k = 0

    def run(seconds, target_of_t, grip=0.0):
        nonlocal k
        n = int(seconds / m.opt.timestep)
        for i in range(n):
            d.ctrl[arm.act] = target_of_t((i + 1) / n)
            d.ctrl[arm.grip] = grip(i / n) if callable(grip) else grip
            d.qfrc_applied[arm.dof] = d.qfrc_bias[arm.dof]   # gravity compensation, as the UR controller does
            mujoco.mj_step(m, d)
            rec.step(m, d, k)
            k += 1

    def lerp(a, b):
        return lambda t: a + (b - a) * (3 * t * t - 2 * t ** 3)

    def path(P):
        return lambda t: P[min(len(P) - 1, int(t * (len(P) - 1) + 0.5))]

    oid = m.body(body).id if body else None
    run(0.3, lambda t: HOME)
    c0 = d.xipos[oid].copy() if oid is not None else None
    run(1.0, lerp(HOME, q_high))
    run(0.8, path(over))
    run(1.0, path(down))
    run(0.6, lambda t: down[-1])                      # let the servos settle before measuring / closing
    reach_err = float(np.linalg.norm(d.site_xpos[arm.site] - p_tcp))
    pushed = float(np.linalg.norm(d.xipos[oid] - c0)) if oid is not None else 0.0
    run(0.8, lambda t: down[-1], grip=lambda t: 255 * min(1.0, t * 4))
    run(1.0, path(up), grip=255)
    run(0.5, lambda t: up[-1], grip=255)
    dz = float(d.xipos[oid][2] - c0[2]) if oid is not None else 0.0
    return score(reach_err, pushed, dz)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seg", required=True)
    ap.add_argument("--menagerie", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-scene")
    ap.add_argument("--gt-meshes")
    ap.add_argument("--targets", help="comma-separated cluster ids (default: all)")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--max-videos", type=int, default=0, help="with --video: record only the first N graspable targets")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rep, recon, hull, gt, names = load_scene(a.seg, a.gt_scene, a.gt_meshes)
    if not recon:
        raise SystemExit("segment.json has no objects: nothing to grasp")
    gc = np.array([mm.centroid for _, mm in gt]) if gt else None
    centre = np.mean([mm.centroid[:2] for mm in recon], axis=0)
    targets = [int(t) for t in a.targets.split(",")] if a.targets else list(range(len(recon)))
    results, n_recorded = [], 0
    for t in targets:
        grasp, info = grasp_for(recon[t])
        row = {"target": t, "names": names.get(t, []), "grasp": info}
        if grasp is None:
            results.append(row)
            continue
        video = a.video and (a.max_videos <= 0 or n_recorded < a.max_videos)
        n_recorded += int(video)
        R_ours, p_tcp = grasp
        u = p_tcp[:2] - centre
        u = u / (np.linalg.norm(u) + 1e-9)
        base = np.array([p_tcp[0] + REACH_BACK * u[0], p_tcp[1] + REACH_BACK * u[1], 0.0])
        armcfg = {"base": base.tolist(), "yaw": float(np.arctan2(-u[1], -u[0]))}
        worlds = {
            "recon": lambda: build(a.menagerie, rep["table"], [("obj_%02d" % i, mm) for i, mm in enumerate(recon)],
                                   arm=armcfg),
            "scenehull": lambda: build(a.menagerie, rep["table"], [], static_hull=hull,
                                       ghosts=[("g%02d" % i, mm) for i, mm in enumerate(recon)], arm=armcfg),
        }
        if gt:
            worlds["gt"] = lambda: build(a.menagerie, rep["table"], [("gt_" + n, mm) for n, mm in gt], arm=armcfg)
        for w, mk in worlds.items():
            m = mk()
            arm = Arm(m)
            wp = plan(arm, R_ours @ M_OURS_TO_MOUNT, p_tcp)
            if wp is None:
                row[w] = {"success": False, "reason": "IK has no solution from this base"}
                continue
            body = {"recon": "obj_%02d" % t, "scenehull": None,
                    "gt": "gt_" + gt[int(np.argmin(np.linalg.norm(gc - recon[t].centroid, axis=1)))][0] if gt else None}[w]
            look = [p_tcp[0] * 0.75 + base[0] * 0.25, p_tcp[1] * 0.75 + base[1] * 0.25, 0.12]
            az = float(np.degrees(np.arctan2(u[0], -u[1]))) - 35     # side view, the pedestal does not block the table
            rec = Recorder(m, look, video, azimuth=az)
            if video:
                rec.cam.distance, rec.cam.elevation = 1.25, -40
            res = replay(m, arm, body, wp, p_tcp, rec)
            rec.close()
            res["ik_err_m"] = round(wp[4], 5)
            if w == "scenehull":
                res["lifted"] = False
                res["success"] = False
            if video and rec.frames:
                path = os.path.join(a.out, "ur5e_t%02d_%s.mp4" % (t, w))
                write_video(path, rec.frames)
                res["video"] = os.path.basename(path)
            row[w] = res
        results.append(row)
        print(json.dumps({"target": t, "names": row["names"]} | {w: row[w]["success"] for w in worlds if w in row}),
              flush=True)
    worlds_seen = [w for w in ("recon", "scenehull", "gt") if any(w in r for r in results)]
    json.dump({"version": "r2s-grasp-ur5e-v1", "summary": summarise(results, worlds_seen), "per_target": results},
              open(os.path.join(a.out, "grasp_ur5e.json"), "w"), indent=1, default=float)
    print(json.dumps(summarise(results, worlds_seen)))


if __name__ == "__main__":
    main()
