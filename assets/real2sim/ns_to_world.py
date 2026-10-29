"""Bring a splatfacto export (splat.ply) back into the metric frame of the capture poses.

nerfstudio trains in a normalised frame: its dataparser re-orients and centres the poses and scales
them (orientation_method=up, center_method=poses, auto_scale_poses) and records that as
dataparser_transforms.json = {"transform": 3x4 [R|t], "scale": s}, i.e. p_ns = s * (R p + t).
`ns-export gaussian-splat` writes the gaussian centres in that normalised frame, so the segmentation
thresholds (metres) would be meaningless on them. This applies the inverse, p = R^T (p_ns / s - t),
keeps only opaque gaussians (sigmoid(opacity) >= --min-opacity, floaters are mostly transparent) and
writes a plain xyz + rgb point cloud.

--check-cloud <ply>: median distance from the converted centres to a reference cloud in the capture
frame (sparse_pc.ply). A few mm means the frames agree; centimetres or more means they do not.

usage: python ns_to_world.py <splat.ply> <dataparser_transforms.json> <out.ply> [--min-opacity 0.5]
       [--check-cloud sparse_pc.ply]
"""

import argparse
import json

import numpy as np

SH_C0 = 0.28209479177387814
PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1", "uint8": "u1",
             "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4", "short": "<i2", "ushort": "<u2"}


def read_ply_vertices(path):
    """Structured array of the vertex element of a binary little-endian PLY (what ns-export writes)."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            header.append(line.decode("ascii", "replace").strip())
            if header[-1] == "end_header":
                break
        if "format binary_little_endian 1.0" not in header:
            raise ValueError(f"{path}: only binary_little_endian PLY is supported")
        n, props, in_vertex = 0, [], False
        for h in header:
            p = h.split()
            if p[:1] == ["element"]:
                in_vertex = p[1] == "vertex"
                if in_vertex:
                    n = int(p[2])
            elif p[:1] == ["property"] and in_vertex:
                if p[1] == "list":
                    raise ValueError(f"{path}: list properties in the vertex element are not supported")
                props.append((p[2], PLY_TYPES[p[1]]))
        dt = np.dtype(props)
        return np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n).copy()


def write_xyzrgb_ply(path, xyz, rgb):
    xyz = np.asarray(xyz, dtype="<f4")
    rgb = np.clip(np.asarray(rgb, dtype=float) * 255.0, 0, 255).astype("u1")
    v = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    v["red"], v["green"], v["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    head = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
            "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n") % len(v)
    with open(path, "wb") as f:
        f.write(head.encode("ascii"))
        f.write(v.tobytes())


def ns_to_world(points, transform, scale):
    """Inverse of nerfstudio's dataparser transform: p_ns = scale * (R p + t)  ->  p."""
    T = np.asarray(transform, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    return (np.asarray(points, dtype=float) / float(scale) - t) @ R  # row vectors: (R^T x)^T = x^T R


def convert(splat, dataparser_json, min_opacity=0.5):
    v = read_ply_vertices(splat)
    dp = json.load(open(dataparser_json))
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(float)
    keep = np.ones(len(v), dtype=bool)
    if "opacity" in v.dtype.names and min_opacity > 0:
        keep = 1.0 / (1.0 + np.exp(-v["opacity"].astype(float))) >= min_opacity
    if all(k in v.dtype.names for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
        rgb = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(float)
    else:
        rgb = np.full((len(v), 3), 0.5)
    return ns_to_world(xyz[keep], dp["transform"], dp["scale"]), rgb[keep], int(len(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("splat")
    ap.add_argument("dataparser_transforms")
    ap.add_argument("out")
    ap.add_argument("--min-opacity", type=float, default=0.5)
    ap.add_argument("--check-cloud")
    a = ap.parse_args()
    xyz, rgb, n_all = convert(a.splat, a.dataparser_transforms, a.min_opacity)
    write_xyzrgb_ply(a.out, xyz, rgb)
    rep = {"gaussians": n_all, "kept_opaque": int(len(xyz)), "min_opacity": a.min_opacity}
    if a.check_cloud and len(xyz):
        import open3d as o3d  # noqa: PLC0415 - only the optional check needs open3d

        ref = o3d.io.read_point_cloud(a.check_cloud)
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        d = np.asarray(pc.compute_point_cloud_distance(ref))
        rep["nn_to_reference_median_m"] = round(float(np.median(d)), 5)
        rep["nn_to_reference_p90_m"] = round(float(np.percentile(d, 90)), 5)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
