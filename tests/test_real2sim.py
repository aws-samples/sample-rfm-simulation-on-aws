"""real2sim stage scripts (assets/real2sim/) on synthetic data, plus the generate.py wiring.

Geometry tests need numpy + trimesh + open3d, the grasp test also mujoco and MuJoCo Menagerie
(robotiq_2f85 + universal_robots_ur5e at the commit pinned in models/real2sim.yaml): set
R2S_MENAGERIE=<dir> to use a local checkout, otherwise it is sparse-fetched once into the pytest cache
and the test is skipped when that is not possible (offline). Nothing here touches AWS.
"""

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets" / "real2sim"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ASSETS))

yaml = pytest.importorskip("yaml")
MODEL = yaml.safe_load((ROOT / "models" / "real2sim.yaml").read_text())


def _need(*mods):
    for m in mods:
        pytest.importorskip(m)


# ───────────────────────────── synthetic scene ─────────────────────────────

BOXES = [  # (centre x, centre y, size x, size y, height) in metres, table frame
    (-0.12, -0.08, 0.05, 0.05, 0.08),
    (0.10, -0.06, 0.04, 0.06, 0.05),
    (0.00, 0.12, 0.06, 0.04, 0.10),
]


def _box_surface(cx, cy, sx, sy, h, step, rng):
    """Points on the 5 visible faces of a box standing on z=0 (no bottom: cameras never see it)."""
    import numpy as np

    xs, ys, zs = np.arange(-sx / 2, sx / 2 + 1e-9, step), np.arange(-sy / 2, sy / 2 + 1e-9, step), np.arange(0, h + 1e-9, step)
    pts = [np.array([[x, y, h] for x in xs for y in ys])]
    pts += [np.array([[x, s * sy / 2, z] for x in xs for z in zs]) for s in (-1, 1)]
    pts += [np.array([[s * sx / 2, y, z] for y in ys for z in zs]) for s in (-1, 1)]
    P = np.vstack(pts) + [cx, cy, 0]
    return P + rng.normal(0, 0.0005, P.shape)


def synthetic_cloud(rot=None, offset=(0.0, 0.0, 0.0)):
    """Table 0.6 x 0.5 m at z=0 + the BOXES, optionally rotated / moved into another world frame.
    Returns (points Nx3, camera centres Kx3) in that frame; cameras sit 0.5 m above the table."""
    import numpy as np

    rng = np.random.default_rng(0)
    g = np.arange(-0.3, 0.3, 0.003)
    gy = np.arange(-0.25, 0.25, 0.003)
    table = np.array([[x, y, 0.0] for x in g for y in gy]) + rng.normal(0, 0.0005, (len(g) * len(gy), 3))
    P = np.vstack([table] + [_box_surface(*b, 0.002, rng) for b in BOXES])
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    cams = np.stack([0.3 * np.cos(ang), 0.3 * np.sin(ang), np.full_like(ang, 0.5)], 1)
    R = np.eye(3) if rot is None else np.asarray(rot)
    return P @ R.T + offset, cams @ R.T + offset


def _pcd(P):
    import open3d as o3d

    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))


# z up -> y up (a nerfstudio / OpenGL style world frame) and moved away from the origin.
Z_TO_Y = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]


@pytest.mark.parametrize("frame", ["z-up", "y-up+offset"])
def test_segment_finds_each_box(frame):
    _need("numpy", "trimesh", "open3d")
    import numpy as np
    import r2s_segment as seg

    rot, off = (None, (0, 0, 0)) if frame == "z-up" else (Z_TO_Y, (0.4, 1.2, -0.7))
    P, cams = synthetic_cloud(rot, off)
    table, objs = seg.segment_points(_pcd(P), up_points=cams)
    assert len(objs) == len(BOXES)
    n = np.array(table["normal"])
    up = np.array([0, 0, 1.0]) if rot is None else np.asarray(rot) @ [0, 0, 1.0]
    assert n @ up > 0.99, "table normal must point at the cameras"
    want = sorted(sx * sy * h for _, _, sx, sy, h in BOXES)
    got = sorted(o["volume_m3"] for o in objs)
    for w, g in zip(want, got):
        assert g == pytest.approx(w, rel=0.25)
    # table axes are orthonormal and n is the third one
    A = np.array(table["axes"])
    assert np.allclose(A @ A.T, np.eye(3), atol=1e-6) and np.allclose(A[2], n)


def test_orient_normal():
    _need("numpy")
    import numpy as np
    from r2s_segment import orient_normal

    n, d = orient_normal([0, 0, -1.0], 0.0, up_points=np.array([[0, 0, 1.0]]))
    assert n[2] == 1.0 and d == 0.0
    n, d = orient_normal([0, 0, 1.0], 0.0, up_points=np.array([[0, 0, -1.0]]))
    assert n[2] == -1.0
    n, _ = orient_normal([0, 0, -1.0], 0.0)
    assert n[2] == 1.0


def test_segment_cli_writes_meshes_usd_and_json(tmp_path):
    _need("numpy", "trimesh", "open3d")
    import r2s_segment as seg

    P, cams = synthetic_cloud()
    args = seg.parser().parse_args(["--cloud", "x", "--out", str(tmp_path)])
    rep = seg.run(_pcd(P), str(tmp_path), args, up_points=cams)
    assert rep["n_objects"] == len(BOXES)
    assert sorted(p.name for p in tmp_path.glob("obj_*.obj")) == ["obj_%02d.obj" % i for i in range(len(BOXES))]
    assert json.loads((tmp_path / "segment.json").read_text())["n_objects"] == len(BOXES)
    try:
        import pxr  # noqa: F401
        assert (tmp_path / "scene_collision.usda").is_file() and rep["usd"] == "scene_collision.usda"
    except ImportError:
        assert rep["usd"] is None


def test_gt_volume_ratio_is_set_for_seam_split_gt_meshes(tmp_path):
    """HOPE GT OBJs duplicate vertices along texture seams (trimesh: not watertight). hope_gt must weld
    them so that the per-object volume_ratio is a number, consistent with the scene-level ratio.
    trimesh needs Pillow to load a textured OBJ (the instance installs it with the sim pins)."""
    _need("numpy", "trimesh", "open3d", "PIL")
    import numpy as np
    import r2s_segment as seg
    import trimesh

    box = trimesh.creation.box(extents=(4.0, 6.0, 8.0))           # cm, like the HOPE meshes
    mesh_dir, scene_dir = tmp_path / "m", tmp_path / "s"
    mesh_dir.mkdir()
    scene_dir.mkdir()
    # every face corner its own vertex + its own UV, as along the texture seams of the HOPE OBJs
    corners = box.vertices[box.faces.reshape(-1)]
    obj = ["v %.6f %.6f %.6f" % tuple(v) for v in corners]
    obj += ["vt %.4f %.4f" % ((k % 7) / 7, (k % 5) / 5) for k in range(len(corners))]
    obj += ["f %d/%d %d/%d %d/%d" % (3 * f + 1, 3 * f + 1, 3 * f + 2, 3 * f + 2, 3 * f + 3, 3 * f + 3)
            for f in range(len(box.faces))]
    (mesh_dir / "Box.obj").write_text("\n".join(obj) + "\n")
    assert not trimesh.load(mesh_dir / "Box.obj", force="mesh").is_volume    # what trimesh sees unwelded
    pose = np.eye(4)
    pose[:3, 3] = [10.0, 0.0, 4.0]                                 # cm
    (scene_dir / "0000.json").write_text(json.dumps({"camera": {"extrinsics": np.eye(4).tolist()},
                                                     "objects": [{"class": "Box", "pose": pose.tolist()}]}))
    gt = seg.hope_gt(str(scene_dir), str(mesh_dir))
    assert gt[0][1].is_volume and gt[0][1].volume == pytest.approx(4 * 6 * 8 * 1e-6, rel=1e-6)

    # two clusters (over-split): halves of a 1.25x larger box
    halves = [trimesh.creation.box(extents=(0.05, 0.06, 0.08)) for _ in range(2)]
    for k, h in enumerate(halves):
        h.apply_scale((0.5, 1.0, 1.0))
        h.apply_translation((0.1 + (k - 0.5) * 0.025, 0.0, 0.04))
    objs = [{"parts": [h], "volume_m3": float(h.volume), "centroid": h.centroid} for h in halves]
    rep = seg.gt_report(objs, gt, scene_hull_volume=None)
    r = rep["per_object"][0]
    assert r["matched"] == [0, 1]
    assert r["gt_volume_m3"] == pytest.approx(192e-6, rel=1e-3)
    assert r["collision_volume_m3"] == pytest.approx(240e-6, rel=1e-3)
    assert r["volume_ratio"] == pytest.approx(1.25, abs=1e-3)
    assert rep["collision_volume_ratio_per_object"] == pytest.approx(r["volume_ratio"], abs=1e-3)


# ───────────────────────────── capture prep / frames ─────────────────────────────

def _look_at_w2c(eye, target=(0, 0, 0)):
    """OpenCV world->camera matrix of a camera at `eye` looking at `target` (world z up)."""
    import numpy as np

    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])                       # rows = camera axes in world
    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = R, -R @ eye
    return w2c


def test_hope_poses_become_opengl_c2w_facing_the_scene(tmp_path):
    _need("numpy")
    import numpy as np
    import prep_capture as pc

    for i, a in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
        eye = [0.4 * np.cos(a), 0.4 * np.sin(a), 0.5]
        cam = {"camera": {"extrinsics": _look_at_w2c(eye).tolist(), "intrinsics": [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
                          "width": 640, "height": 480}, "objects": []}
        (tmp_path / ("%04d.json" % i)).write_text(json.dumps(cam))
        (tmp_path / ("%04d_rgb.jpg" % i)).write_bytes(b"jpg")
    frames = pc.hope_frames(str(tmp_path))
    assert len(frames) == 8
    c2w = [f["transform_matrix"] for _, _, f in frames]
    assert pc.cameras_facing(c2w, [0, 0, 0]) == 8
    # camera centre is preserved by the conversion
    assert np.allclose(np.array(c2w[0])[:3, 3], [0.4, 0.0, 0.5])
    # the un-flipped (OpenCV) matrices would look away from the scene: the flip matters
    cv = [np.linalg.inv(_look_at_w2c([0.4, 0, 0.5]))]
    assert pc.cameras_facing(cv, [0, 0, 0]) == 0
    assert len(pc.hope_frames(str(tmp_path), max_frames=3)) == 3


def _capture_zip(path, frames, extra=None):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("cap/transforms.json", json.dumps({"fl_x": 500, "fl_y": 500, "cx": 320, "cy": 240, "w": 640, "h": 480,
                                                      "frames": frames}))
        for f in frames:
            z.writestr("cap/" + f["file_path"], b"img")
        for name, data in (extra or {}).items():
            z.writestr(name, data)


def test_capture_zip_is_validated(tmp_path):
    _need("numpy")
    import numpy as np
    import prep_capture as pc

    good = [{"file_path": "images/%03d.png" % i, "transform_matrix": np.eye(4).tolist()} for i in range(3)]
    _capture_zip(tmp_path / "ok.zip", good)
    meta, info = pc.prep_zip(str(tmp_path / "ok.zip"), str(tmp_path / "ok"))
    assert info == {"source": "capture", "frames": 3} and "ply_file_path" not in meta
    _capture_zip(tmp_path / "evil.zip", good, extra={"cap/../../evil.txt": b"x"})
    with pytest.raises(SystemExit, match="unsafe path"):
        pc.prep_zip(str(tmp_path / "evil.zip"), str(tmp_path / "evil"))
    bad = [dict(good[0], transform_matrix=[[1, 0], [0, 1]])]
    _capture_zip(tmp_path / "bad.zip", bad)
    with pytest.raises(SystemExit, match="transform_matrix"):
        pc.prep_zip(str(tmp_path / "bad.zip"), str(tmp_path / "bad"))


def test_ns_to_world_inverts_the_dataparser_transform(tmp_path):
    _need("numpy")
    import ns_to_world as nw
    import numpy as np

    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    R = q * np.sign(np.linalg.det(q))
    t, s = rng.normal(size=3), 0.37
    P = rng.normal(size=(50, 3))
    P_ns = s * (P @ R.T + t)
    T = np.hstack([R, t[:, None]])
    assert np.allclose(nw.ns_to_world(P_ns, T, s), P)

    # end to end through a gaussian-splat style ply: transparent gaussians are dropped
    v = np.zeros(50, dtype=[(k, "<f4") for k in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity")])
    v["x"], v["y"], v["z"] = P_ns.T
    v["opacity"] = np.where(np.arange(50) < 30, 4.0, -4.0)      # sigmoid 0.98 / 0.02
    head = "ply\nformat binary_little_endian 1.0\nelement vertex 50\n" + "".join(
        f"property float {k}\n" for k in v.dtype.names) + "end_header\n"
    (tmp_path / "splat.ply").write_bytes(head.encode() + v.tobytes())
    (tmp_path / "dp.json").write_text(json.dumps({"transform": T.tolist(), "scale": s}))
    xyz, rgb, n_all = nw.convert(str(tmp_path / "splat.ply"), str(tmp_path / "dp.json"))
    assert n_all == 50 and len(xyz) == 30 and np.allclose(xyz, P[:30], atol=1e-5)
    nw.write_xyzrgb_ply(str(tmp_path / "out.ply"), xyz, rgb)
    back = nw.read_ply_vertices(str(tmp_path / "out.ply"))
    assert len(back) == 30 and np.allclose(back["x"], xyz[:, 0], atol=1e-6)


# ───────────────────────────── grasp replay ─────────────────────────────

@pytest.fixture(scope="session")
def menagerie(tmp_path_factory):
    local = os.environ.get("R2S_MENAGERIE")
    if local:
        return local
    for cand in (ROOT / ".temp" / "menagerie",):
        if (cand / "robotiq_2f85" / "2f85.xml").is_file():
            return str(cand)
    if not shutil.which("git"):
        pytest.skip("git not available to fetch MuJoCo Menagerie")
    d = tmp_path_factory.mktemp("menagerie")
    sha, repo = MODEL["model"]["menagerie_commit"], MODEL["model"]["menagerie_repo"]
    cmds = [["git", "init", "-q", str(d)], ["git", "-C", str(d), "remote", "add", "origin", repo],
            ["git", "-C", str(d), "sparse-checkout", "set", "robotiq_2f85", "universal_robots_ur5e"],
            ["git", "-C", str(d), "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", sha],
            ["git", "-C", str(d), "checkout", "-q", "FETCH_HEAD"]]
    for c in cmds:
        try:
            if subprocess.run(c, capture_output=True, timeout=300).returncode != 0:
                pytest.skip(f"could not fetch MuJoCo Menagerie @ {sha[:8]} (offline?)")
        except subprocess.TimeoutExpired:
            pytest.skip("MuJoCo Menagerie fetch timed out")
    return str(d)


def test_grasp_replay_lifts_a_reconstructed_box_but_not_the_scene_hull(tmp_path, menagerie):
    _need("numpy", "trimesh", "open3d", "mujoco")
    import numpy as np
    import r2s_grasp_mj as g
    import r2s_segment as seg

    P, cams = synthetic_cloud()
    args = seg.parser().parse_args(["--cloud", "x", "--out", str(tmp_path)])
    seg.run(_pcd(P), str(tmp_path), args, up_points=cams)
    rep, recon, hull, gt, _ = g.load_scene(str(tmp_path))
    assert len(recon) == len(BOXES) and not gt
    # target = the 5 x 5 x 8 cm box (fits the 2F-85 on both axes)
    t = int(np.argmin([np.linalg.norm(m.centroid[:2] - [-0.12, -0.08]) for m in recon]))
    grasp, info = g.grasp_for(recon[t])
    assert grasp is not None and info["width_m"] < g.MAX_OPEN
    worlds = {"recon": g.build(menagerie, rep["table"], [("obj_%02d" % i, m) for i, m in enumerate(recon)]),
              "scenehull": g.build(menagerie, rep["table"], [], static_hull=hull)}
    off = g.Recorder(worlds["recon"], [0, 0, 0], False)
    res = g.replay(worlds["recon"], "obj_%02d" % t, grasp, off)
    assert res["reached"] and res["lifted"] and res["success"], res
    hull_res = g.replay(worlds["scenehull"], None, grasp, g.Recorder(worlds["scenehull"], [0, 0, 0], False))
    assert not hull_res["reached"], "the static scene hull must stop the fingers above the grasp point"


def test_grasp_for_rejects_objects_wider_than_the_gripper():
    _need("numpy", "trimesh")
    import trimesh
    from r2s_grasp_mj import grasp_for

    grasp, info = grasp_for(trimesh.creation.box((0.12, 0.10, 0.05)).apply_translation((0, 0, 0.025)))
    assert grasp is None and "wider" in info["reason"]


# ───────────────────────────── report + generate wiring ─────────────────────────────

def test_report_summary_is_results_py_compatible(tmp_path):
    import r2s_report as rr

    import results

    (tmp_path / "segment").mkdir()
    (tmp_path / "grasp").mkdir()
    (tmp_path / "segment" / "segment.json").write_text(json.dumps({"n_objects": 6, "gt": {"n_gt": 10, "matched": 6}}))
    (tmp_path / "metrics.json").write_text(json.dumps({"results": {"psnr": 24.01, "ssim": 0.855, "lpips": 0.193}}))
    row = {"target": 1, "names": ["Mustard"], "recon": {"success": True, "video": "hand_t01_recon.mp4", "reached": True,
                                                         "lifted": True, "undisturbed": True}}
    (tmp_path / "grasp" / "grasp.json").write_text(json.dumps({"summary": {"recon": {"n": 4, "success": 3},
                                                                           "scenehull": {"n": 4, "success": 0}},
                                                               "per_target": [row]}))
    (tmp_path / "hand_t01_recon.mp4").write_bytes(b"\0")
    rep = rr.collect(str(tmp_path))
    kv = dict(results.KV_RE.findall("\n".join(rr.summary_lines(rep, "hope-video-scene-0000", {"instance": "g5.2xlarge"}))))
    assert kv["success_rate"] == "0.750" and kv["n_episodes"] == "4" and kv["n_success"] == "3"
    assert kv["hand_scenehull"] == "0/4" and kv["suite"] == "hope-video-scene-0000" and kv["psnr"] == "24.0100"
    caps = rr.captions(rep, str(tmp_path))
    assert list(caps) == ["hand_t01_recon.mp4"] and "Mustard" in caps["hand_t01_recon.mp4"]


def _generate():
    pytest.importorskip("jinja2")
    import generate

    return generate


def test_render_embeds_every_stage_script_unchanged():
    gen = _generate()
    cfg = gen._load_merged_config(ROOT / "simulator-config.yaml", "real2sim")
    text = gen.generate_real2sim(cfg, dry_run=True)
    blocks = re.findall(r'base64 -d > "\$\{ASSETS_DIR\}/([\w.]+)" << \'R2S_ASSET_EOF\'\n(.*?)R2S_ASSET_EOF', text, re.S)
    assert [n for n, _ in blocks] == list(gen.REAL2SIM_ASSETS)
    for name, b64 in blocks:
        assert base64.b64decode(b64) == (ASSETS / name).read_bytes(), name
    assert MODEL["model"]["menagerie_commit"] in text and "hope_fetch.py\" --scene 0" in text


@pytest.mark.parametrize("inp,msg", [
    ({"source": "capture", "capture_s3_uri": ""}, "capture_s3_uri"),
    ({"source": "capture", "capture_s3_uri": "s3://b/x.tar"}, "capture_s3_uri"),
    ({"source": "hope", "hope_scene": 12}, "hope_scene"),
    ({"source": "nope"}, "input.source"),
])
def test_bad_inputs_are_rejected(inp, msg, capsys):
    gen = _generate()
    cfg = gen._load_merged_config(ROOT / "simulator-config.yaml", "real2sim")
    with pytest.raises(SystemExit):
        gen.generate_real2sim({**cfg, "input": inp}, dry_run=True)
    assert msg in capsys.readouterr().err


def test_capture_input_renders_the_s3_path():
    gen = _generate()
    cfg = gen._load_merged_config(ROOT / "simulator-config.yaml", "real2sim")
    text = gen.generate_real2sim({**cfg, "input": {"source": "capture", "capture_s3_uri": "s3://my-bucket/caps/desk.zip"}},
                                 dry_run=True)
    assert 'aws s3 cp "s3://my-bucket/caps/desk.zip"' in text and "hope_fetch.py\" --scene" not in text
