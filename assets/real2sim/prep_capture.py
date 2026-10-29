"""Turn a capture into a nerfstudio data directory (images/ + transforms.json [+ sparse_pc.ply]).

Two inputs:

  --hope <scene dir>   a HOPE-Video scene (NNNN.json + NNNN_rgb.jpg + scene.ply). HOPE extrinsics are
                       world->camera in OpenCV axes with the translation in metres (see the dataset README
                       note); nerfstudio wants camera->world in OpenGL axes, so each matrix is inverted and
                       its y / z columns are negated. The CascadeStereo cloud (scene.ply) is down-sampled into
                       sparse_pc.ply, which gives splatfacto a far better start than a random init.
  --capture <zip>      your own capture: a zip with images/ and a nerfstudio transforms.json (camera->world,
                       OpenGL, **metric** poses — the segmentation thresholds are in metres). An optional
                       `ply_file_path` cloud in the zip is kept.

Sanity check written to capture_info.json: how many cameras look at the cloud centre (their -z axis
within ~37 deg of it). A wrong pose convention shows up as ~0 here instead of ~all frames.

usage: python prep_capture.py (--hope DIR | --capture ZIP) --out DATA_DIR [--voxel 0.004] [--max-frames N]
"""

import argparse
import json
import os
import shutil
import sys
import zipfile

import numpy as np


def w2c_opencv_to_c2w_opengl(w2c):
    """4x4 world->camera (OpenCV: x right, y down, z forward) -> camera->world (OpenGL: y up, z back)."""
    c2w = np.linalg.inv(np.asarray(w2c, dtype=float))
    c2w[:3, 1:3] *= -1
    return c2w


def cameras_facing(c2w_list, centre, cos_min=0.8):
    """Number of OpenGL camera->world poses whose viewing direction (-z) points at `centre`."""
    n = 0
    for c2w in c2w_list:
        c2w = np.asarray(c2w, dtype=float)
        d = np.asarray(centre, dtype=float) - c2w[:3, 3]
        norm = np.linalg.norm(d)
        if norm > 0 and float((-c2w[:3, 2]) @ d / norm) > cos_min:
            n += 1
    return n


def hope_frames(scene_dir, max_frames=0):
    """[(annotation id, rgb file, frame dict)] for a HOPE-Video scene, in frame order."""
    ids = sorted(f[:-5] for f in os.listdir(scene_dir) if f.endswith(".json") and f[:-5].isdigit())
    if max_frames and len(ids) > max_frames:
        ids = [ids[int(i)] for i in np.linspace(0, len(ids) - 1, max_frames)]
    out = []
    for k in ids:
        cam = json.load(open(os.path.join(scene_dir, k + ".json")))["camera"]
        K = cam["intrinsics"]
        rgb = k + "_rgb.jpg"
        out.append((k, rgb, {
            "file_path": "images/" + rgb,
            "transform_matrix": w2c_opencv_to_c2w_opengl(cam["extrinsics"]).tolist(),
            "w": int(cam["width"]), "h": int(cam["height"]),
            "fl_x": float(K[0][0]), "fl_y": float(K[1][1]), "cx": float(K[0][2]), "cy": float(K[1][2]),
        }))
    return out


def validate_transforms(meta, root):
    """Raise ValueError when a user transforms.json cannot be trained on."""
    frames = meta.get("frames") or []
    if not frames:
        raise ValueError("transforms.json has no frames")
    for f in frames:
        m = np.asarray(f.get("transform_matrix"), dtype=float)
        if m.shape != (4, 4) or not np.all(np.isfinite(m)):
            raise ValueError(f"bad transform_matrix for {f.get('file_path')}")
        if not os.path.isfile(os.path.join(root, f["file_path"])):
            raise ValueError(f"image missing: {f['file_path']}")
        if not all(k in f or k in meta for k in ("fl_x", "fl_y", "cx", "cy", "w", "h")):
            raise ValueError(f"intrinsics (fl_x, fl_y, cx, cy, w, h) missing for {f['file_path']}")
    return len(frames)


def sparse_cloud(src, dst, voxel):
    import open3d as o3d  # noqa: PLC0415 - only the cloud step needs open3d

    pc = o3d.io.read_point_cloud(src).voxel_down_sample(voxel)
    pc, _ = pc.remove_statistical_outlier(20, 2.0)
    o3d.io.write_point_cloud(dst, pc)
    return np.asarray(pc.points)


def prep_hope(scene_dir, out, voxel, max_frames):
    frames = hope_frames(scene_dir, max_frames)
    if not frames:
        raise SystemExit(f"no NNNN.json frames in {scene_dir}")
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    for _, rgb, _ in frames:
        shutil.copyfile(os.path.join(scene_dir, rgb), os.path.join(out, "images", rgb))
    meta = {"camera_model": "PINHOLE", "frames": [f for _, _, f in frames]}
    info = {"source": "hope", "scene_dir": os.path.abspath(scene_dir), "frames": len(frames)}
    ply = os.path.join(scene_dir, "scene.ply")
    if os.path.isfile(ply):
        pts = sparse_cloud(ply, os.path.join(out, "sparse_pc.ply"), voxel)
        meta["ply_file_path"] = "sparse_pc.ply"
        info["init_points"] = int(len(pts))
        info["cameras_facing_cloud"] = cameras_facing([f["transform_matrix"] for f in meta["frames"]], pts.mean(0))
    return meta, info


def prep_zip(capture, out):
    with zipfile.ZipFile(capture) as z:
        names = z.namelist()
        tj = next((n for n in names if n.rstrip("/").split("/")[-1] == "transforms.json"), None)
        if tj is None:
            raise SystemExit("capture zip has no transforms.json (expected images/ + transforms.json)")
        prefix = tj[: -len("transforms.json")]
        for n in names:
            rel = n[len(prefix):] if n.startswith(prefix) else None
            if not rel or n.endswith("/"):
                continue
            if rel.startswith("/") or ".." in rel.split("/"):
                raise SystemExit(f"unsafe path in capture zip: {n}")
            dst = os.path.join(out, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with z.open(n) as s, open(dst, "wb") as d:
                shutil.copyfileobj(s, d)
    meta = json.load(open(os.path.join(out, "transforms.json")))
    try:
        n = validate_transforms(meta, out)
    except ValueError as e:
        raise SystemExit(f"capture zip: {e}") from e
    info = {"source": "capture", "frames": n}
    ply = meta.get("ply_file_path")
    if ply and os.path.isfile(os.path.join(out, ply)):
        import open3d as o3d  # noqa: PLC0415

        pts = np.asarray(o3d.io.read_point_cloud(os.path.join(out, ply)).points)
        info["init_points"] = int(len(pts))
        if len(pts):
            info["cameras_facing_cloud"] = cameras_facing([f["transform_matrix"] for f in meta["frames"]], pts.mean(0))
    else:
        meta.pop("ply_file_path", None)
    return meta, info


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hope")
    src.add_argument("--capture")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--max-frames", type=int, default=0, help="uniformly subsample to N frames (0 = all)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta, info = prep_hope(a.hope, a.out, a.voxel, a.max_frames) if a.hope else prep_zip(a.capture, a.out)
    json.dump(meta, open(os.path.join(a.out, "transforms.json"), "w"), indent=1)
    json.dump(info, open(os.path.join(a.out, "capture_info.json"), "w"), indent=1)
    print(json.dumps(info))
    if "cameras_facing_cloud" in info and info["cameras_facing_cloud"] < 0.5 * info["frames"]:
        print("[prep_capture] [warn] fewer than half of the cameras face the cloud: check the pose convention "
              "(camera->world, OpenGL axes)", file=sys.stderr)


if __name__ == "__main__":
    main()
