"""Tabletop point cloud -> table plane + per-object collision meshes -> USD (UsdPhysics) + fidelity json.

v1 = geometric separation: RANSAC fits the table plane, DBSCAN clusters the points above it, each cluster
gets its footprint on the table added (the cameras never see an object's bottom, so its hull would
float) and becomes a convex hull (or a CoACD decomposition). Objects that touch each other merge into one
cluster; with ground truth the json reports that as `merged`.

The table frame is (u, v, n): n = plane normal pointing at the cameras (--cameras, a nerfstudio
transforms.json) or, without cameras, towards +z; u, v span the plane. The table extent and the
"above the table" box are measured in that frame, so the capture's world frame does not need z up.
All thresholds are metres: the capture poses must be metric.

usage: python r2s_segment.py --cloud scene.ply --out out/ [--cameras transforms.json]
       [--gt-scene <hope scene dir> --gt-meshes <dir>] [--voxel 0.003] [--eps 0.008] [--decomp hull|coacd]
"""

import argparse
import json
import os
import sys

import numpy as np
import trimesh

VERSION = "r2s-segment-v1"


def plane_basis(n):
    """(u, v) spanning the plane with normal n. u = helper x n with helper = +y (or +x when n ~ +-y)."""
    n = np.asarray(n, dtype=float) / np.linalg.norm(n)
    helper = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(helper, n)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def orient_normal(n, d, up_points=None):
    """Flip (n, d) so that `up_points` (camera centres) lie above the plane; without them, n_z >= 0."""
    n, d = np.asarray(n, dtype=float), float(d)
    if up_points is not None and len(up_points):
        if float(np.mean(np.asarray(up_points) @ n + d)) < 0:
            n, d = -n, -d
    elif n[2] < 0:
        n, d = -n, -d
    return n, d


def camera_centres(transforms_json):
    meta = json.load(open(transforms_json))
    return np.array([np.asarray(f["transform_matrix"], dtype=float)[:3, 3] for f in meta["frames"]])


def complete_to_table(pts, n, d):
    """Add the object's footprint on the table (the lowest 20% of its points, projected onto the plane)."""
    h = pts @ n + d
    low = pts[h < np.percentile(h, 20)]
    return np.vstack([pts, low - np.outer(low @ n + d, n)]) if len(low) else pts


def object_mesh(pts, mode):
    hull = trimesh.points.PointCloud(pts).convex_hull
    if mode == "hull":
        return [hull]
    import coacd  # noqa: PLC0415 - optional dependency (--decomp coacd)

    parts = coacd.run_coacd(coacd.Mesh(hull.vertices, hull.faces), threshold=0.05)
    return [trimesh.Trimesh(v, f) for v, f in parts]


def segment_points(pcd, plane_dist=0.01, above=0.006, max_height=0.40, eps=0.008, min_points=200,
                   decomp="hull", up_points=None):
    """Open3D point cloud (already down-sampled) -> (table dict, [object dict]). Pure apart from RANSAC."""
    (a, b, c, d), inl = pcd.segment_plane(distance_threshold=plane_dist, ransac_n=3, num_iterations=2000)
    n, d = orient_normal(np.array([a, b, c]) / np.linalg.norm([a, b, c]), d / np.linalg.norm([a, b, c]), up_points)
    u, v = plane_basis(n)
    B = np.stack([u, v], 1)                                  # world -> in-plane coordinates
    P = np.asarray(pcd.points)
    tab_uv = P[np.asarray(inl)] @ B
    lo, hi = np.percentile(tab_uv, 1, 0), np.percentile(tab_uv, 99, 0)
    uv, h = P @ B, P @ n + d
    keep = (h > above) & (h < max_height) & np.all((uv > lo) & (uv < hi), 1)
    obj = pcd.select_by_index(np.where(keep)[0])
    Q = np.asarray(obj.points)
    lab = np.asarray(obj.cluster_dbscan(eps=eps, min_points=10)) if len(Q) else np.zeros(0, dtype=int)
    objs = []
    for k in range(lab.max() + 1 if len(lab) else 0):
        pts = Q[lab == k]
        if len(pts) < min_points:
            continue
        full = complete_to_table(pts, n, d)
        parts = object_mesh(full, decomp)
        objs.append({"points": pts, "parts": parts, "volume_m3": float(sum(p.volume for p in parts)),
                     "centroid": full.mean(0)})
    mid = (lo + hi) / 2
    centre = u * mid[0] + v * mid[1] - n * d - n * 0.02      # table slab 4 cm thick, its top on the plane
    table = {"normal": n.tolist(), "offset": d, "axes": [u.tolist(), v.tolist(), n.tolist()],
             "center": centre.tolist(), "size": [float(hi[0] - lo[0]), float(hi[1] - lo[1]), 0.04],
             "n_inliers": int(len(inl))}
    return table, objs


def surface_samples(mesh, n=20000):
    return np.asarray(mesh.sample(n))


def chamfer_f(a, b, tau=0.005):
    import open3d as o3d  # noqa: PLC0415

    pa = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(a))
    pb = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(b))
    da = np.asarray(pa.compute_point_cloud_distance(pb))
    db = np.asarray(pb.compute_point_cloud_distance(pa))
    prec, rec = (da < tau).mean(), (db < tau).mean()
    f = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return float(da.mean() + db.mean()) / 2, float(f)


def hope_gt(scene_dir, mesh_dir):
    """HOPE-Video GT in world metres (dataset README: extrinsics w2c in m, object poses and meshes in cm)."""
    a = json.load(open(os.path.join(scene_dir, "0000.json")))
    c2w = np.linalg.inv(np.array(a["camera"]["extrinsics"]))
    out = []
    for o in a["objects"]:
        P = np.array(o["pose"], dtype=float)
        P[:3, 3] /= 100
        m = trimesh.load(os.path.join(mesh_dir, o["class"] + ".obj"), force="mesh")
        # the textured OBJs duplicate vertices along UV / normal seams, so trimesh sees them as open
        # (is_volume False) although the surface is closed; welding the seams restores a closed mesh
        # with the same geometry (volume, centroid and surface samples are unchanged)
        m.merge_vertices(merge_tex=True, merge_norm=True)
        m.apply_scale(0.01)
        m.apply_transform(c2w @ P)
        out.append((o["class"], m))
    return out


def gt_report(objs, gt, scene_hull_volume):
    gc = np.array([m.centroid for _, m in gt])
    claimed = {}
    for i, o in enumerate(objs):
        j = int(np.argmin(np.linalg.norm(gc - o["centroid"], axis=1)))
        claimed.setdefault(j, []).append(i)
    per = []
    for j, (name, m) in enumerate(gt):
        ids = claimed.get(j, [])
        if not ids:
            per.append({"gt": name, "matched": None})
            continue
        pred = trimesh.util.concatenate([p for i in ids for p in objs[i]["parts"]])
        cd, f5 = chamfer_f(surface_samples(pred), surface_samples(m))
        coll = sum(objs[i]["volume_m3"] for i in ids)      # same definition as the scene-level ratio below
        per.append({"gt": name, "matched": ids, "chamfer_m": round(cd, 4), "fscore_5mm": round(f5, 3),
                    "centroid_err_m": round(float(np.linalg.norm(pred.centroid - m.centroid)), 4),
                    "collision_volume_m3": round(float(coll), 7),
                    "gt_volume_m3": round(float(m.volume), 7) if m.is_volume else None,
                    # collision volume of the matched cluster(s) / GT mesh volume; None only when the
                    # GT mesh is not closed (its volume is then undefined)
                    "volume_ratio": round(float(coll / m.volume), 3) if m.is_volume else None})
    # a GT object is `merged` when its nearest cluster is also the nearest cluster of another GT object
    near = {j: int(np.argmin([np.linalg.norm(o["centroid"] - gc[j]) for o in objs])) for j in range(len(gt))} if objs else {}
    for j, r in enumerate(per):
        r["merged"] = sum(1 for k in near if near[k] == near.get(j)) > 1
    # scene level: collision volume of every cluster (matched or not) / volume of all GT meshes
    gt_vol = sum(m.volume for _, m in gt)
    return {"n_gt": len(gt), "per_object": per,
            "matched": sum(1 for r in per if r["matched"]), "merged": sum(1 for r in per if r["merged"]),
            "collision_volume_ratio_per_object": round(sum(o["volume_m3"] for o in objs) / gt_vol, 3),
            "collision_volume_ratio_scene_hull": round(scene_hull_volume / gt_vol, 3) if scene_hull_volume else None}


def _quat_wxyz(R):
    q = np.empty(4)
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q[:] = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def write_usd(path, table, objs, density):
    """USD stage with UsdPhysics collision: a static table slab and one rigid body per object."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt  # noqa: PLC0415 - usd-core is optional

    st = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/World")
    st.SetDefaultPrim(root.GetPrim())
    UsdPhysics.Scene.Define(st, "/World/physicsScene")
    t = UsdGeom.Cube.Define(st, "/World/table")
    t.GetSizeAttr().Set(1.0)
    x = UsdGeom.Xformable(t)
    x.AddTranslateOp().Set(Gf.Vec3d(*table["center"]))
    w, i, j, k = _quat_wxyz(np.array(table["axes"], dtype=float).T)
    x.AddOrientOp().Set(Gf.Quatf(float(w), float(i), float(j), float(k)))
    x.AddScaleOp().Set(Gf.Vec3f(*table["size"]))
    UsdPhysics.CollisionAPI.Apply(t.GetPrim())
    for oi, o in enumerate(objs):
        xf = UsdGeom.Xform.Define(st, "/World/obj_%02d" % oi)
        UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
        UsdPhysics.MassAPI.Apply(xf.GetPrim()).CreateMassAttr(float(o["volume_m3"] * density))
        for pj, part in enumerate(o["parts"]):
            m = UsdGeom.Mesh.Define(st, "/World/obj_%02d/col_%02d" % (oi, pj))
            m.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in part.vertices]))
            m.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(part.faces)))
            m.GetFaceVertexIndicesAttr().Set(Vt.IntArray(part.faces.reshape(-1).tolist()))
            UsdPhysics.CollisionAPI.Apply(m.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
    st.Save()


def run(cloud_pcd, out, args, up_points=None):
    """Segment an Open3D cloud and write obj_XX.obj, scene_collision.usda and segment.json into `out`."""
    import open3d as o3d  # noqa: PLC0415

    o3d.utility.random.seed(args.seed)
    os.makedirs(out, exist_ok=True)
    pcd = cloud_pcd.voxel_down_sample(args.voxel)
    pcd, _ = pcd.remove_statistical_outlier(20, 2.0)
    table, objs = segment_points(pcd, plane_dist=args.plane_dist, above=args.above, max_height=args.max_height,
                                 eps=args.eps, min_points=args.min_points, decomp=args.decomp, up_points=up_points)
    scene_hull = trimesh.points.PointCloud(np.vstack([o["points"] for o in objs])).convex_hull if objs else None
    rep = {"version": VERSION, "seed": args.seed, "decomp": args.decomp, "n_objects": len(objs), "table": table,
           "params": {k: getattr(args, k) for k in ("voxel", "eps", "min_points", "above", "max_height", "plane_dist")},
           "up_hint": "cameras" if up_points is not None else "+z",
           "scene_hull_volume_m3": float(scene_hull.volume) if scene_hull else None,
           "objects": [{"id": i, "n_points": int(len(o["points"])), "n_parts": len(o["parts"]),
                        "volume_m3": o["volume_m3"], "centroid": o["centroid"].round(4).tolist()}
                       for i, o in enumerate(objs)]}
    if args.gt_scene and args.gt_meshes:
        rep["gt"] = gt_report(objs, hope_gt(args.gt_scene, args.gt_meshes), rep["scene_hull_volume_m3"])
    try:
        write_usd(os.path.join(out, "scene_collision.usda"), table, objs, args.density)
        rep["usd"] = "scene_collision.usda"
    except ImportError:
        print("[r2s_segment] [warn] usd-core (pxr) not installed: no scene_collision.usda", file=sys.stderr)
        rep["usd"] = None
    for i, o in enumerate(objs):
        trimesh.util.concatenate(o["parts"]).export(os.path.join(out, "obj_%02d.obj" % i))
    json.dump(rep, open(os.path.join(out, "segment.json"), "w"), indent=1)
    return rep


def parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cameras", help="nerfstudio transforms.json: the table normal points at these cameras")
    ap.add_argument("--gt-scene")
    ap.add_argument("--gt-meshes")
    ap.add_argument("--voxel", type=float, default=0.003)
    ap.add_argument("--eps", type=float, default=0.008)
    ap.add_argument("--min-points", type=int, default=200)
    ap.add_argument("--above", type=float, default=0.006, help="ignore points closer than this to the table (m)")
    ap.add_argument("--max-height", type=float, default=0.40, help="ignore points higher than this (m)")
    ap.add_argument("--plane-dist", type=float, default=0.01)
    ap.add_argument("--decomp", default="hull", choices=["hull", "coacd"])
    ap.add_argument("--density", type=float, default=500.0, help="kg/m^3 for the USD masses")
    ap.add_argument("--seed", type=int, default=0, help="RANSAC seed (unseeded runs drift run to run)")
    return ap


def main():
    import open3d as o3d  # noqa: PLC0415

    a = parser().parse_args()
    src = o3d.io.read_point_cloud(a.cloud)
    if not src.has_points():                               # a mesh input (.glb / .obj): sample its surface
        src = o3d.io.read_triangle_mesh(a.cloud).sample_points_uniformly(2_000_000)
    rep = run(src, a.out, a, up_points=camera_centres(a.cameras) if a.cameras else None)
    short = {k: rep[k] for k in ("n_objects", "scene_hull_volume_m3")}
    if "gt" in rep:
        short["gt"] = {k: v for k, v in rep["gt"].items() if k != "per_object"}
    print(json.dumps(short, indent=1))


if __name__ == "__main__":
    main()
