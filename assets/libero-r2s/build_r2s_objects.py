"""Build LIBERO object assets from a real2sim stage-1 run (runs on the instance; stdlib only).

usage: python3 build_r2s_objects.py --assets <LIBERO libero/libero/assets> --segment <dir>
                                    [--splat splat_world.ply] [--modes collision,full]
                                    [--scale fit|metric] [--mass match|density] [--map name=cluster,...]
                                    [--manifest out.json]

<dir> holds the stage-1 outputs of `vlasim deploy --vla real2sim` (<run>/real2sim/segment/):
segment.json (table frame, clusters, and for a HOPE-Video scene the cluster <-> ground-truth object
match) and obj_NN.obj (one convex collision hull per cluster, world frame).

LIBERO-Object (and LIBERO-10 / -Goal) use the HOPE grocery objects, registered in
libero/libero/envs/objects/hope_objects.py and loaded from assets/stable_hope_objects/<name>/<name>.xml.
The stage-1 HOPE-Video capture shows the same physical objects, so each reconstructed cluster replaces
the LIBERO object of the same name. For each mode a sibling asset directory
assets/stable_hope_objects_r2s_<mode>/ is written: every object that has a reconstruction gets a new
<name>.xml, every other object is a symlink to the stock one. hope_objects.py is pointed at one of
the directories per evaluation (r2s_select.py), so the stock scene stays available unchanged.

Per replaced object:
  * geometry: the cluster with the most points among those matched to the object (a ground-truth
    object split into several clusters keeps its main fragment; --map overrides the choice);
  * orientation: the object's pose in the real scene is not assumed to be its LIBERO pose (the real
    cookie box stands, LIBERO's lies flat). The cluster is expressed in the table frame (z up), its
    horizontal principal axes are found, and its three axes are matched to the stock object's
    collision box axes by extent order (largest to largest);
  * scale: `fit` (default) scales uniformly so the bounding-box volume equals the stock object's
    (LIBERO's HOPE meshes are scaled 0.5-1.0x of the real objects); the shape (aspect ratios,
    convexity) stays the reconstruction's. `metric` keeps the measured size;
  * placement: horizontal bounding-box centre and bottom face aligned with the stock collision
    geometry, so LIBERO's fixed initial states put the object on the table, not into it;
  * mass: `match` (default) sets the density so the mass equals the stock object's; `density` keeps
    the stock density (100 kg/m^3) on the reconstructed volume;
  * contact parameters (friction, solref, solimp) are copied from the stock geoms.
Modes: `collision` replaces only the collision geoms (the policy still sees the stock textured mesh:
isolates contact geometry); `full` also renders the reconstructed hull, coloured with the mean colour
of the 3D Gaussian Splatting points inside it (what a real capture gives you without texture baking).
"""
import argparse
import json
import math
import os
import re
import shutil
import struct
import sys
import xml.etree.ElementTree as ET

STOCK_DIR = "stable_hope_objects"
MODES = ("collision", "full")
# Local axis of each stock HOPE object that points to world +z in LIBERO's initial states (measured by
# loading every benchmark task that contains the object and reading the settled body orientation).
# Objects not listed (not used by a benchmark task) are centred on all three axes.
UP_AXIS = {"alphabet_soup": 1, "bbq_sauce": 2, "butter": 2, "chocolate_pudding": 2, "cookies": 2,
           "cream_cheese": 2, "ketchup": 1, "milk": 1, "orange_juice": 1, "salad_dressing": 1,
           "tomato_sauce": 1}


def snake(name):
    return "_".join(re.sub(r"([A-Z0-9])", r" \1", name).split()).lower()


# --- small linear algebra (3-vectors / 3x3 as lists) -------------------------------------------------

def sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def matvec(m, v):
    return [dot(m[0], v), dot(m[1], v), dot(m[2], v)]


def quat_mat(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def bbox(pts):
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    return lo, hi


def mesh_volume(verts, faces):
    """Signed-tetrahedron volume of a closed triangle mesh (absolute value)."""
    v = 0.0
    for a, b, c in faces:
        v += dot(verts[a], cross(verts[b], verts[c]))
    return abs(v) / 6.0


# --- IO ----------------------------------------------------------------------------------------------

def read_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) for t in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                for k in range(1, len(idx) - 1):  # fan-triangulate polygons
                    faces.append((idx[0], idx[k], idx[k + 1]))
    if not verts or not faces:
        raise ValueError(f"{path}: no geometry")
    return verts, faces


def write_obj(path, verts, faces):
    with open(path, "w") as f:
        f.write("# real2sim reconstruction (convex collision hull), LIBERO object frame, metres\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


def read_ply_xyzrgb(path):
    """Binary little-endian PLY with float x,y,z + uchar red,green,blue (stage-1 splat_world.ply)."""
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            header += line
        text = header.decode("ascii", "replace")
        if "binary_little_endian" not in text:
            raise ValueError(f"{path}: only binary_little_endian PLY is supported")
        n = int(re.search(r"element vertex (\d+)", text).group(1))
        props = re.findall(r"property (\w+) (\w+)", text.split("element vertex", 1)[1].split("element", 1)[0])
        fmt_map = {"float": "f", "float32": "f", "double": "d", "uchar": "B", "uint8": "B", "int": "i", "uint": "I"}
        fmt = "<" + "".join(fmt_map[t] for t, _ in props)
        names = [p for _, p in props]
        need = ["x", "y", "z", "red", "green", "blue"]
        if any(x not in names for x in need):
            raise ValueError(f"{path}: needs x y z red green blue, has {names}")
        idx = [names.index(x) for x in need]
        size = struct.calcsize(fmt)
        data = f.read(n * size)
    out = []
    for rec in struct.iter_unpack(fmt, data[: (len(data) // size) * size]):
        out.append([rec[i] for i in idx])
    return out


def mean_colour_inside(points, verts, faces, eps=1e-4):
    """Mean RGB (0..1) of the points inside a convex hull (None if no point is inside)."""
    lo, hi = bbox(verts)
    centre = [sum(v[i] for v in verts) / len(verts) for i in range(3)]
    planes = []
    for a, b, c in faces:
        n = cross(sub(verts[b], verts[a]), sub(verts[c], verts[a]))
        ln = math.sqrt(dot(n, n))
        if ln < 1e-12:
            continue
        n = [x / ln for x in n]
        d = dot(n, verts[a])
        if dot(n, centre) - d > 0:  # orient outwards whatever the winding
            n, d = [-x for x in n], -d
        planes.append((n, d))
    acc, k = [0.0, 0.0, 0.0], 0
    for p in points:
        if not (lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1] and lo[2] <= p[2] <= hi[2]):
            continue
        if all(dot(n, p) - d <= eps for n, d in planes):
            acc[0] += p[3]
            acc[1] += p[4]
            acc[2] += p[5]
            k += 1
    if not k:
        return None, 0
    return [acc[0] / k / 255.0, acc[1] / k / 255.0, acc[2] / k / 255.0], k


# --- stock object -------------------------------------------------------------------------------------

def stock_collision(xml_path):
    """(tree, object body, group-0 geoms, corner points in the object frame, collision volume)."""
    tree = ET.parse(xml_path)
    body = None
    for b in tree.getroot().iter("body"):
        if b.get("name") == "object":
            body = b
            break
    if body is None:
        raise ValueError(f"{xml_path}: no <body name=\"object\">")
    geoms = [g for g in body.findall("geom") if g.get("group", "0") == "0"]
    if not geoms:
        raise ValueError(f"{xml_path}: no collision geoms (group 0)")
    pts, vol = [], 0.0
    for g in geoms:
        if g.get("type") != "box":
            raise ValueError(f"{xml_path}: collision geom type {g.get('type')!r} not supported (stock HOPE objects use boxes)")
        pos = [float(x) for x in (g.get("pos") or "0 0 0").split()]
        q = [float(x) for x in (g.get("quat") or "1 0 0 0").split()]
        s = [float(x) for x in g.get("size").split()]
        r = quat_mat(q)
        vol += 8 * s[0] * s[1] * s[2]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    c = matvec(r, [sx * s[0], sy * s[1], sz * s[2]])
                    pts.append([pos[0] + c[0], pos[1] + c[1], pos[2] + c[2]])
    return tree, body, geoms, pts, vol


# --- alignment ----------------------------------------------------------------------------------------

def table_frame(verts, table):
    """World -> table frame (rows of `axes` = u, v, n; origin = table centre)."""
    axes, c = table["axes"], table["center"]
    return [matvec(axes, sub(v, c)) for v in verts]


def horizontal_pca_angle(pts):
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    syy = sum((p[1] - my) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    return 0.5 * math.atan2(2 * sxy, sxx - syy)


def align(recon_tf, stock_pts, scale_mode, up=None):
    """Recon (table frame, z up) -> stock object frame. Returns (verts, info)."""
    th = horizontal_pca_angle(recon_tf)
    c, s = math.cos(-th), math.sin(-th)
    r = [[c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]] for p in recon_tf]  # principal axes on x, y
    rlo, rhi = bbox(r)
    rext = [rhi[i] - rlo[i] for i in range(3)]
    slo, shi = bbox(stock_pts)
    sext = [shi[i] - slo[i] for i in range(3)]
    # Match recon axes to stock axes by extent order: largest -> largest, ...
    r_order = sorted(range(3), key=lambda i: -rext[i])
    s_order = sorted(range(3), key=lambda i: -sext[i])
    perm = [0, 0, 0]  # perm[stock axis] = recon axis
    for ri, si in zip(r_order, s_order):
        perm[si] = ri
    m = [[1.0 if perm[i] == j else 0.0 for j in range(3)] for i in range(3)]
    det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
           + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    if det < 0:  # keep a proper rotation: mirror the recon minor horizontal axis (hulls are near-symmetric)
        r = [[p[0], -p[1], p[2]] for p in r]
    p_obj = [[p[perm[0]], p[perm[1]], p[perm[2]]] for p in r]
    olo, ohi = bbox(p_obj)
    oext = [ohi[i] - olo[i] for i in range(3)]
    if scale_mode == "fit":
        k = (sext[0] * sext[1] * sext[2] / max(oext[0] * oext[1] * oext[2], 1e-12)) ** (1.0 / 3.0)
    elif scale_mode == "metric":
        k = 1.0
    else:
        raise ValueError(f"unknown scale mode {scale_mode!r}")
    # Centre on the stock box; along the object's up axis (its local axis that points to world +z in
    # LIBERO's initial states) align the bottom faces instead, so the object rests on the table.
    oc = [(olo[i] + ohi[i]) / 2 for i in range(3)]
    sc = [(slo[i] + shi[i]) / 2 for i in range(3)]
    if up is not None:
        oc[up] = olo[up]
        sc[up] = slo[up]
    out = [[(p[i] - oc[i]) * k + sc[i] for i in range(3)] for p in p_obj]
    flo, fhi = bbox(out)
    fext = [fhi[i] - flo[i] for i in range(3)]
    info = dict(scale=round(k, 4), up_axis="xyz"[up] if up is not None else None, yaw_deg=round(math.degrees(th), 2), axis_map=perm,
                stock_extent_m=[round(x, 4) for x in sext], recon_extent_m=[round(x, 4) for x in oext],
                final_extent_m=[round(x, 4) for x in fext],
                extent_ratio=[round(fext[i] / sext[i], 3) if sext[i] else None for i in range(3)])
    return out, info


# --- build --------------------------------------------------------------------------------------------

def pick_clusters(seg, explicit):
    """{libero name: (cluster id, gt name, gt record)} from segment.json gt matches (+ --map overrides)."""
    npts = {o["id"]: o.get("n_points", 0) for o in seg.get("objects", [])}
    out = {}
    for rec in (seg.get("gt") or {}).get("per_object", []):
        matched = [m for m in rec.get("matched") or [] if m in npts]
        if not matched:
            continue
        cid = max(matched, key=lambda m: (npts[m], -m))
        out[snake(rec["gt"])] = (cid, rec["gt"], rec)
    for name, cid in (explicit or {}).items():
        if cid not in npts:
            raise SystemExit(f"--map {name}={cid}: no cluster {cid} in segment.json (have {sorted(npts)})")
        out[name] = (cid, out.get(name, (None, name, {}))[1], out.get(name, (None, name, {}))[2])
    return out


def parse_map(s):
    out = {}
    for item in [x for x in (s or "").split(",") if x.strip()]:
        name, _, cid = item.partition("=")
        if not cid.strip().isdigit():
            raise SystemExit(f"--map entries must be name=cluster_id, got {item!r}")
        out[name.strip()] = int(cid)
    return out


def link_or_copy(src, dst):
    # Relative link: the asset tree is also seen under another mount point (openpi's runtime container
    # mounts the checkout at /app), where an absolute host path would dangle.
    try:
        os.symlink(os.path.relpath(src, os.path.dirname(os.path.abspath(dst))), dst)
    except OSError:
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def build(assets, segment_dir, modes=MODES, scale="fit", mass="match", splat=None, explicit=None, keep_stock=()):
    assets = os.path.abspath(assets)
    stock_root = os.path.join(assets, STOCK_DIR)
    if not os.path.isdir(stock_root):
        raise SystemExit(f"{stock_root}: not a LIBERO assets dir")
    seg = json.load(open(os.path.join(segment_dir, "segment.json")))
    table = seg.get("table")
    if not table or "axes" not in table:
        raise SystemExit("segment.json: no table frame (axes) - re-run stage 1 with this repo's real2sim target")
    picks = pick_clusters(seg, explicit)
    stock = sorted(d for d in os.listdir(stock_root) if os.path.isdir(os.path.join(stock_root, d)))
    swap = {n: v for n, v in picks.items() if n in stock}
    skipped = sorted(set(picks) - set(swap))
    kept = sorted(set(swap) & set(keep_stock))
    swap = {n: v for n, v in swap.items() if n not in kept}
    if kept and not swap:
        raise SystemExit(f"every reconstructed LIBERO object is kept stock ({', '.join(kept)}): nothing to swap")
    if not swap:
        raise SystemExit(f"no reconstructed object matches a LIBERO HOPE object (reconstructed: {sorted(picks)}; "
                         f"LIBERO: {stock}). For a non-HOPE capture pass --map <libero_name>=<cluster id>.")
    pts = read_ply_xyzrgb(splat) if splat and "full" in modes else None
    manifest = dict(version="r2s-libero-objects-v1", scale=scale, mass=mass, modes=list(modes),
                    skipped_not_in_libero=skipped, kept_stock=kept, objects={})
    built = {}
    for name in sorted(swap):
        cid, gt_name, gt_rec = swap[name]
        xml_path = os.path.join(stock_root, name, f"{name}.xml")
        verts_w, faces = read_obj(os.path.join(segment_dir, f"obj_{cid:02d}.obj"))
        tree, body, geoms, stock_pts, stock_vol = stock_collision(xml_path)
        verts, info = align(table_frame(verts_w, table), stock_pts, scale, UP_AXIS.get(name))
        vol = mesh_volume(verts, faces)
        dens0 = float(geoms[0].get("density", "1000"))
        stock_mass = dens0 * stock_vol
        dens = stock_mass / vol if mass == "match" else dens0
        rgb, n_in = (mean_colour_inside(pts, verts_w, faces) if pts is not None else (None, 0))
        info.update(cluster=cid, gt=gt_name, gt_chamfer_m=gt_rec.get("chamfer_m"),
                    gt_fscore_5mm=gt_rec.get("fscore_5mm"), gt_clusters=gt_rec.get("matched"),
                    stock_mass_kg=round(stock_mass, 5), mass_kg=round(dens * vol, 5),
                    collision_volume_m3=round(vol, 8), stock_collision_volume_m3=round(stock_vol, 8),
                    density=round(dens, 2), rgb=[round(x, 3) for x in rgb] if rgb else None, splat_points_inside=n_in)
        manifest["objects"][name] = info
        built[name] = (tree, body, geoms, verts, faces, dens, rgb)
    for mode in modes:
        root = os.path.join(assets, f"{STOCK_DIR}_r2s_{mode}")
        if os.path.isdir(root):
            shutil.rmtree(root)
        os.makedirs(root)
        for name in stock:
            src = os.path.join(stock_root, name)
            if name not in built:
                link_or_copy(src, os.path.join(root, name))
                continue
            dst = os.path.join(root, name)
            os.makedirs(dst)
            for f in os.listdir(src):
                if f != f"{name}.xml":
                    link_or_copy(os.path.join(src, f), os.path.join(dst, f))
            write_obj(os.path.join(dst, "r2s_collision.obj"), built[name][3], built[name][4])
            write_xml(os.path.join(src, f"{name}.xml"), os.path.join(dst, f"{name}.xml"), mode, *built[name][5:])
        manifest.setdefault("dirs", {})[mode] = os.path.basename(root)
    return manifest


def write_xml(src_xml, dst_xml, mode, density, rgb):
    tree = ET.parse(src_xml)
    root = tree.getroot()
    body = next(b for b in root.iter("body") if b.get("name") == "object")
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", file="r2s_collision.obj", name="r2s_hull")
    cols = [g for g in body.findall("geom") if g.get("group", "0") == "0"]
    keep = {k: cols[0].get(k) for k in ("solimp", "solref", "friction", "rgba") if cols[0].get(k)}
    for g in cols:
        body.remove(g)
    ET.SubElement(body, "geom", type="mesh", mesh="r2s_hull", density=f"{density:.4f}", group="0", **keep)
    if mode == "full":
        for g in [g for g in body.findall("geom") if g.get("group") == "1"]:
            body.remove(g)
        colour = rgb or [0.7, 0.7, 0.7]
        ET.SubElement(body, "geom", type="mesh", mesh="r2s_hull", group="1", contype="0", conaffinity="0",
                      density="0", rgba=f"{colour[0]:.3f} {colour[1]:.3f} {colour[2]:.3f} 1")
    tree.write(dst_xml)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--assets", required=True, help="LIBERO libero/libero/assets directory")
    p.add_argument("--segment", required=True, help="stage-1 segment dir (segment.json + obj_NN.obj)")
    p.add_argument("--splat", help="stage-1 splat_world.ply (colours for --modes full)")
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--scale", default="fit", choices=["fit", "metric"])
    p.add_argument("--mass", default="match", choices=["match", "density"])
    p.add_argument("--map", default="", help="override: libero_name=cluster_id,...")
    p.add_argument("--manifest", help="write the build manifest (json) here")
    p.add_argument("--keep-stock", default="", help="LIBERO objects to leave stock although reconstructed "
                   "(e.g. the ones check_scene.py found unstable), comma-separated")
    a = p.parse_args(argv)
    modes = [m for m in a.modes.split(",") if m]
    bad = [m for m in modes if m not in MODES]
    if bad or not modes:
        raise SystemExit(f"--modes: choose from {', '.join(MODES)}")
    keep = [x.strip() for x in a.keep_stock.split(",") if x.strip()]
    man = build(a.assets, a.segment, modes, a.scale, a.mass, a.splat, parse_map(a.map), keep_stock=keep)
    text = json.dumps(man, indent=1)
    if a.manifest:
        with open(a.manifest, "w") as f:
            f.write(text + "\n")
    for name, o in man["objects"].items():
        print(f"r2s object {name}: cluster {o['cluster']} scale {o['scale']} extent_ratio {o['extent_ratio']} "
              f"mass {o['mass_kg']} kg rgb {o['rgb']}")
    if man["skipped_not_in_libero"]:
        print("reconstructed but not a LIBERO object (not used): " + ", ".join(man["skipped_not_in_libero"]))
    if man["kept_stock"]:
        print("reconstructed but kept stock (--keep-stock): " + ", ".join(man["kept_stock"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
