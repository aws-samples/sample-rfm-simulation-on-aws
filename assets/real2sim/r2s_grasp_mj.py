"""Grasp replay: the same scripted top-down pick in up to three collision worlds (MuJoCo, floating Robotiq 2F-85).

  recon      per-object collision meshes from r2s_segment (obj_XX.obj) as free bodies   <- this pipeline
  scenehull  one static convex hull around every object (a single-mesh scene export)   <- baseline
  gt         ground-truth meshes at their GT poses as free bodies (HOPE-Video only)      <- reference

Phases: open, TCP 8 cm above the grasp -> descend 0.6 s -> settle 0.25 s -> close 0.8 s -> lift 10 cm 1.0 s
-> hold 0.5 s. Scored axes: reached (TCP within 1 cm of the grasp point), lifted (object COM rose >= 5 cm),
undisturbed (< 2 cm push before closing); success = all three. The scene hull is static, so nothing can be
lifted in that world by construction. Everything runs in the table frame (z up, origin on the table top).

The gripper (robotiq_2f85) and arm (universal_robots_ur5e) come from MuJoCo Menagerie
(https://github.com/google-deepmind/mujoco_menagerie, fetched at a pinned commit at run time).

usage: MUJOCO_GL=egl python r2s_grasp_mj.py --seg <segment out dir> --menagerie <dir> --out <dir>
       [--gt-scene <hope scene dir> --gt-meshes <dir>] [--targets 1,3,4] [--video]
"""

import argparse
import json
import os
import sys

import mujoco
import numpy as np
import trimesh

M_OURS_TO_MOUNT = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
MAX_OPEN = 0.075                     # the 2F-85 opens 85 mm; leave room for the pads
COLORS = [(0.86, 0.33, 0.25), (0.25, 0.55, 0.85), (0.95, 0.70, 0.20), (0.35, 0.70, 0.40),
          (0.60, 0.40, 0.75), (0.90, 0.50, 0.70), (0.45, 0.75, 0.80), (0.70, 0.55, 0.35)]


def table_frame(table):
    """World -> table rotation (rows = table axes) and a point on the table top."""
    if "axes" in table:
        R = np.array(table["axes"], dtype=float)
    else:                                        # segment.json written before the axes field existed
        z = np.array(table["normal"], dtype=float)
        z /= np.linalg.norm(z)
        helper = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        x = np.cross(helper, z)
        x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(z, x), z])
    p0 = np.array(table["center"], dtype=float) + R[2] * table["size"][2] / 2
    return R, p0


def to_table(mesh, R, p0):
    m = mesh.copy()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ p0
    m.apply_transform(T)
    return m


def grasp_for(mesh):
    """Top-down grasp over the object's xy box; the closing axis is the narrow side if the gripper fits."""
    V = mesh.vertices
    xy = V[:, :2] - V[:, :2].mean(0)
    _, U = np.linalg.eigh(np.cov(xy.T))
    proj = xy @ U
    ext = proj.max(0) - proj.min(0)
    mid = V[:, :2].mean(0) + U @ ((proj.max(0) + proj.min(0)) / 2)
    axis = next((int(i) for i in np.argsort(ext) if ext[i] < MAX_OPEN), None)
    top = float(V[:, 2].max())
    z = max(0.015, top - min(0.035, 0.45 * top))
    if axis is None:
        return None, {"width_m": [round(float(e), 4) for e in ext], "reason": "wider than the gripper on both axes"}
    c = np.array([U[0, axis], U[1, axis], 0.0])
    zc = np.array([0.0, 0.0, -1.0])
    R_ours = np.stack([np.cross(c, zc), c, zc], axis=1)
    return (R_ours, np.array([mid[0], mid[1], z])), {"width_m": round(float(ext[axis]), 4), "top_m": round(top, 4),
                                                       "tcp_z_m": round(z, 4)}


def build(menagerie, table, bodies, static_hull=None, ghosts=(), arm=None):
    """arm = None -> floating hand on a mocap weld; arm = {'base': xyz, 'yaw': rad} -> UR5e on a pedestal."""
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.option.impratio = 10                                   # what 2f85.xml asks for; attach keeps the parent's
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST   # UR5e asks for it (stiff position servos)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720
    wb = spec.worldbody
    lt = wb.add_light()
    lt.pos, lt.dir, lt.diffuse = [0.3, -0.4, 1.2], [-0.3, 0.4, -1.0], [0.8, 0.8, 0.8]
    lt2 = wb.add_light()
    lt2.pos, lt2.dir, lt2.diffuse = [-0.5, 0.5, 1.0], [0.5, -0.5, -1.0], [0.35, 0.35, 0.35]
    fl = wb.add_geom()
    fl.type, fl.size, fl.pos, fl.rgba = mujoco.mjtGeom.mjGEOM_PLANE, [2, 2, 0.1], [0, 0, -0.75], [0.82, 0.83, 0.85, 1]
    tb = wb.add_geom()
    tb.name, tb.type = "table", mujoco.mjtGeom.mjGEOM_BOX
    tb.size = [table["size"][0] / 2 + 0.1, table["size"][1] / 2 + 0.1, 0.02]
    tb.pos, tb.rgba, tb.friction = [0, 0, -0.02], [0.62, 0.50, 0.38, 1], [1.0, 0.005, 0.0001]

    def add_mesh(name, mesh):
        ms = spec.add_mesh()
        ms.name = name
        ms.uservert = mesh.vertices.astype(np.float32).reshape(-1).tolist()
        ms.userface = mesh.faces.astype(np.int32).reshape(-1).tolist()
        return name

    for i, (name, mesh) in enumerate(bodies):
        b = wb.add_body()
        b.name = name
        b.add_freejoint()
        g = b.add_geom()
        g.type, g.meshname = mujoco.mjtGeom.mjGEOM_MESH, add_mesh("m_" + name, mesh)
        g.density, g.friction, g.condim = 500, [1.0, 0.01, 0.0002], 4
        g.rgba = list(COLORS[i % len(COLORS)]) + [1]
    if static_hull is not None:
        g = wb.add_geom()
        g.name, g.type = "scene_hull", mujoco.mjtGeom.mjGEOM_MESH
        g.meshname, g.rgba = add_mesh("m_hull", static_hull), [0.55, 0.60, 0.70, 0.45]
    for i, (name, mesh) in enumerate(ghosts):                  # visual only: where the real objects are
        g = wb.add_geom()
        g.type, g.meshname = mujoco.mjtGeom.mjGEOM_MESH, add_mesh("m_" + name, mesh)
        g.contype, g.conaffinity, g.rgba = 0, 0, list(COLORS[i % len(COLORS)]) + [1]

    gripper = mujoco.MjSpec.from_file(os.path.join(menagerie, "robotiq_2f85", "2f85.xml"))
    if arm is not None:
        ur = mujoco.MjSpec.from_file(os.path.join(menagerie, "universal_robots_ur5e", "ur5e.xml"))
        ur.site("attachment_site").attach_body(gripper.worldbody.first_body(), "g_", "")
        bx, by, bz = arm["base"]
        ped = wb.add_geom()
        ped.type, ped.size = mujoco.mjtGeom.mjGEOM_CYLINDER, [0.07, (bz + 0.75) / 2, 0]
        ped.pos, ped.rgba = [bx, by, (bz - 0.75) / 2], [0.25, 0.27, 0.30, 1]
        fr = wb.add_frame()
        fr.pos, fr.quat = [bx, by, bz], [np.cos(arm["yaw"] / 2), 0, 0, np.sin(arm["yaw"] / 2)]
        fr.attach_body(ur.worldbody.first_body(), "ur_", "")
        return spec.compile()
    mocap = wb.add_body()
    mocap.name, mocap.mocap = "target", True
    hand = wb.add_body()
    hand.name = "hand"
    hand.add_freejoint()
    hg = hand.add_geom()
    hg.type, hg.size, hg.mass, hg.contype, hg.conaffinity = mujoco.mjtGeom.mjGEOM_SPHERE, [0.005, 0, 0], 0.05, 0, 0
    hand.add_frame().attach_body(gripper.worldbody.first_body(), "g_", "")
    eq = spec.add_equality()
    eq.type, eq.name1, eq.name2 = mujoco.mjtEq.mjEQ_WELD, "target", "hand"
    eq.objtype, eq.solref, eq.data = mujoco.mjtObj.mjOBJ_BODY, [0.01, 1], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    return spec.compile()


class Recorder:
    """One live Renderer at a time: with OSMesa a second live context renders garbage / flipped frames."""

    def __init__(self, m, look, on, azimuth=140.0, w=640, h=480, fps=30):
        self.on, self.frames, self.every = on, [], max(1, int(round(1 / fps / m.opt.timestep)))
        if on:
            self.r = mujoco.Renderer(m, h, w)
            self.cam = mujoco.MjvCamera()
            self.cam.lookat[:] = look
            self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.75, azimuth, -38

    def close(self):
        if self.on:
            self.r.close()

    def step(self, m, d, k):
        if self.on and k % self.every == 0:
            self.r.update_scene(d, self.cam)
            self.frames.append(self.r.render().copy())


def replay(m, body, grasp, rec):
    """Run the scripted pick on a floating hand; `body` = name of the object to score (None = none)."""
    d = mujoco.MjData(m)
    R_ours, p_tcp = grasp
    R = R_ours @ M_OURS_TO_MOUNT
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    hand = m.body("hand").id
    hq = m.jnt_qposadr[m.body("hand").jntadr[0]]
    mujoco.mj_forward(m, d)
    pinch = d.xmat[hand].reshape(3, 3).T @ (d.site_xpos[m.site("g_pinch").id] - d.xpos[hand])

    def origin(tcp):
        return tcp - R @ pinch

    start, goal = origin(p_tcp + np.array([0, 0, 0.08])), origin(p_tcp)
    d.qpos[hq:hq + 3], d.qpos[hq + 3:hq + 7] = start, q
    d.mocap_pos[0], d.mocap_quat[0] = start, q
    for _ in range(150):                                   # let the free objects settle on the table first
        mujoco.mj_step(m, d)
    oid = m.body(body).id if body else None
    c0 = d.xipos[oid].copy() if oid is not None else None
    k = 0

    def step():
        nonlocal k
        mujoco.mj_step(m, d)
        rec.step(m, d, k)
        k += 1

    n = int(0.6 / m.opt.timestep)
    for i in range(n):
        d.mocap_pos[0] = start + (goal - start) * (i + 1) / n
        step()
    for _ in range(int(0.25 / m.opt.timestep)):
        step()
    reach_err = float(np.linalg.norm(d.site_xpos[m.site("g_pinch").id] - p_tcp))
    pushed = float(np.linalg.norm(d.xipos[oid] - c0)) if oid is not None else 0.0
    for i in range(int(0.8 / m.opt.timestep)):
        d.ctrl[0] = 255 * min(1.0, (i + 1) / 200)
        step()
    n = int(1.0 / m.opt.timestep)
    for i in range(n):
        d.mocap_pos[0] = goal + np.array([0, 0, 0.10]) * (i + 1) / n
        step()
    for _ in range(int(0.5 / m.opt.timestep)):
        step()
    dz = float(d.xipos[oid][2] - c0[2]) if oid is not None else 0.0
    return score(reach_err, pushed, dz)


def score(reach_err, pushed, dz):
    res = {"reach_err_m": round(reach_err, 4), "pushed_m": round(pushed, 4), "lift_dz_m": round(dz, 4)}
    res.update(reached=reach_err < 0.01, lifted=dz >= 0.05, undisturbed=pushed < 0.02)
    res["success"] = bool(res["reached"] and res["lifted"] and res["undisturbed"])
    return res


def load_scene(seg_dir, gt_scene=None, gt_meshes=None):
    """(segment report, recon meshes, scene hull, gt [(name, mesh)], {cluster: [gt names]}) in the table frame."""
    rep = json.load(open(os.path.join(seg_dir, "segment.json")))
    R, p0 = table_frame(rep["table"])
    recon = [to_table(trimesh.load(os.path.join(seg_dir, "obj_%02d.obj" % o["id"]), force="mesh"), R, p0)
             for o in rep["objects"]]
    hull = trimesh.util.concatenate(recon).convex_hull if recon else None
    gt = []
    if gt_scene and gt_meshes:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from r2s_segment import hope_gt  # noqa: PLC0415

        gt = [(n, to_table(mm, R, p0)) for n, mm in hope_gt(gt_scene, gt_meshes)]
    names = {}
    for r in rep.get("gt", {}).get("per_object", []):
        for i in r["matched"] or []:
            names.setdefault(i, []).append(r["gt"])
    return rep, recon, hull, gt, names


def write_video(path, frames):
    import imageio.v2 as iio  # noqa: PLC0415

    iio.mimwrite(path, frames, fps=30, codec="libx264", quality=8, macro_block_size=16)


def summarise(results, worlds):
    return {w: {"n": sum(1 for r in results if w in r), "success": sum(1 for r in results if r.get(w, {}).get("success"))}
            for w in worlds}


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
    targets = [int(t) for t in a.targets.split(",")] if a.targets else list(range(len(recon)))
    worlds = {"recon": build(a.menagerie, rep["table"], [("obj_%02d" % i, mm) for i, mm in enumerate(recon)]),
              "scenehull": build(a.menagerie, rep["table"], [], static_hull=hull,
                                 ghosts=[("g%02d" % i, mm) for i, mm in enumerate(recon)])}
    if gt:
        worlds["gt"] = build(a.menagerie, rep["table"], [("gt_" + n, mm) for n, mm in gt])
    gc = np.array([mm.centroid for _, mm in gt]) if gt else None
    centre = np.mean([mm.centroid[:2] for mm in recon], axis=0)
    results, n_recorded = [], 0
    for t in targets:
        grasp, info = grasp_for(recon[t])
        row = {"target": t, "names": names.get(t, []), "grasp": info}
        if grasp is None:
            results.append(row)
            continue
        video = a.video and (a.max_videos <= 0 or n_recorded < a.max_videos)
        n_recorded += int(video)
        for w, m in worlds.items():
            body = {"recon": "obj_%02d" % t, "scenehull": None,
                    "gt": "gt_" + gt[int(np.argmin(np.linalg.norm(gc - recon[t].centroid, axis=1)))][0] if gt else None}[w]
            # camera on the far side of the target from the scene centre, so the other objects stay behind it
            out_dir = grasp[1][:2] - centre
            az = float(np.degrees(np.arctan2(-out_dir[1], -out_dir[0])))
            rec = Recorder(m, [grasp[1][0], grasp[1][1], 0.05], video, azimuth=az)
            res = replay(m, body, grasp, rec)
            rec.close()
            if w == "scenehull":                         # the hull is static: nothing can be lifted by construction
                res["lifted"] = False
                res["success"] = False
            if video and rec.frames:
                path = os.path.join(a.out, "hand_t%02d_%s.mp4" % (t, w))
                write_video(path, rec.frames)
                res["video"] = os.path.basename(path)
            if w == "gt":
                res["gt_object"] = body[3:]
            row[w] = res
        results.append(row)
        print(json.dumps({"target": t, "names": row["names"]} | {w: row[w]["success"] for w in worlds if w in row}),
              flush=True)
    summ = summarise(results, worlds)
    out = {"version": "r2s-grasp-mj-v1", "summary": summ, "per_target": results,
           "scene_hull_volume_m3": round(float(hull.volume), 6)}
    json.dump(out, open(os.path.join(a.out, "grasp.json"), "w"), indent=1, default=float)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
