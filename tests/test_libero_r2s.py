"""LIBERO-R2S (--libero-objects real2sim): plan, object builder on synthetic meshes, runtime patches,
report, and the generate.py / deploy.py switches.

The builder and patch scripts are stdlib only and run on the instance; here they run on a synthetic
LIBERO asset tree (one stock HOPE object with box collision geoms) and a synthetic stage-1 segment dir
(table frame + one box-shaped hull per cluster), so no LIBERO, MuJoCo or AWS is needed.
"""

import importlib.util
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import libero_r2s  # noqa: E402

ASSETS = ROOT / "assets" / "libero-r2s"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"r2s_{name}", ASSETS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build = _load("build_r2s_objects")
select = _load("r2s_select")
report = _load("r2s_libero_report")

RUN = "s3://example-results/2026-10-29-0142"


# --- plan -------------------------------------------------------------------------------------------

def test_parse_run_accepts_run_and_real2sim_prefix():
    assert libero_r2s.parse_run(RUN) == ("example-results", "2026-10-29-0142/real2sim")
    assert libero_r2s.parse_run(RUN + "/real2sim/") == ("example-results", "2026-10-29-0142/real2sim")
    for bad in ("", "example-results/x", "s3://example-results/", "s3://example-results/../x", "s3://Bad/x"):
        with pytest.raises(ValueError):
            libero_r2s.parse_run(bad)


def test_plan_one_task_per_condition_public_always_first():
    cfg, tasks = libero_r2s.plan({}, RUN, "libero_object", conditions="full", trials=3)
    assert cfg["conditions"] == ["public", "full"]
    assert [t["condition"] for t in tasks] == ["LIBERO-R2S/public", "LIBERO-R2S/full"]
    assert [t["r2s_dir"] for t in tasks] == ["stable_hope_objects", "stable_hope_objects_r2s_full"]
    assert {t["seed"] for t in tasks} == {7} and cfg["trials_per_task"] == 3
    assert cfg["run_uri"] == "s3://example-results/2026-10-29-0142/real2sim"


@pytest.mark.parametrize("block,kw", [
    ({}, dict(suite="libero_99")),
    ({"scale": "huge"}, {}),
    ({"mass": "x"}, {}),
    ({"objects": "Milk=1"}, {}),
    ({}, dict(conditions="public,shiny")),
    ({}, dict(trials=0)),
])
def test_plan_rejects_bad_settings(block, kw):
    suite = kw.pop("suite", "libero_object")
    with pytest.raises(ValueError):
        libero_r2s.plan(block, RUN, suite, **kw)


# --- builder on synthetic meshes ----------------------------------------------------------------------

STOCK_XML = """<mujoco model="textured">
  <asset><mesh file="visual/textured_vis.msh" name="textured_vis" scale="0.0075 0.0075 0.0075" /></asset>
  <worldbody><body><body name="object">
    <geom density="100" friction="0.95 0.3 0.1" type="mesh" mesh="textured_vis" conaffinity="0" contype="0" group="1" />
    <geom density="100" friction="0.95 0.3 0.1" solref="0.001 1" type="box" pos="0 -0.015 0"
          quat="0.70711 0.70711 0 0" size="0.02625 0.02625 0.05475" group="0" />
  </body></body></worldbody>
</mujoco>
"""


def _box_obj(path, lo, hi):
    vs = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    faces = [(1, 3, 4, 2), (5, 6, 8, 7), (1, 2, 6, 5), (3, 7, 8, 4), (1, 5, 7, 3), (2, 4, 8, 6)]
    with open(path, "w") as f:
        for v in vs:
            f.write("v %.5f %.5f %.5f\n" % v)
        for a, b, c, d in faces:  # two triangles per face, outward normals
            f.write(f"f {a} {b} {c}\nf {a} {c} {d}\n")


@pytest.fixture
def scene(tmp_path):
    assets = tmp_path / "assets"
    for name in ("milk", "butter"):
        d = assets / "stable_hope_objects" / name
        (d / "visual").mkdir(parents=True)
        (d / f"{name}.xml").write_text(STOCK_XML)
        (d / "texture_map.png").write_bytes(b"png")
    seg = tmp_path / "segment"
    seg.mkdir()
    # Milk carton standing on the table (z up), 7 x 7 x 20 cm; a small fragment matched to the same
    # object (fewer points, must not be picked); a pineapple LIBERO does not have.
    _box_obj(seg / "obj_00.obj", (0.10, 0.20, 0.0), (0.17, 0.27, 0.20))
    _box_obj(seg / "obj_01.obj", (0.30, 0.30, 0.0), (0.33, 0.33, 0.03))
    _box_obj(seg / "obj_02.obj", (0.50, 0.50, 0.0), (0.60, 0.60, 0.15))
    eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    (seg / "segment.json").write_text(json.dumps({
        "table": {"axes": eye, "center": [0.0, 0.0, 0.0]},
        "objects": [{"id": 0, "n_points": 900}, {"id": 1, "n_points": 50}, {"id": 2, "n_points": 400}],
        "gt": {"per_object": [{"gt": "Milk", "matched": [1, 0], "chamfer_m": 0.004},
                              {"gt": "Pineapple", "matched": [2]}]},
    }))
    return assets, seg


def _verts(path):
    return [[float(x) for x in ln.split()[1:4]] for ln in open(path) if ln.startswith("v ")]


def test_build_replaces_matched_object_fit_scale_bottom_aligned(scene):
    assets, seg = scene
    man = build.build(str(assets), str(seg), modes=("collision", "full"))
    assert sorted(man["objects"]) == ["milk"]
    assert man["skipped_not_in_libero"] == ["pineapple"] and man["kept_stock"] == []
    o = man["objects"]["milk"]
    assert o["cluster"] == 0  # most points among the matched clusters
    # Stock box (object frame): x, z +-0.02625, y from -0.06975 to 0.03975 (up axis y).
    lo = [min(v[i] for v in _verts(assets / "stable_hope_objects_r2s_collision" / "milk" / "r2s_collision.obj"))
          for i in range(3)]
    assert lo[1] == pytest.approx(-0.06975, abs=1e-4)  # bottom face on the stock bottom face
    vol_ratio = o["extent_ratio"][0] * o["extent_ratio"][1] * o["extent_ratio"][2]
    assert vol_ratio == pytest.approx(1.0, rel=3e-3)  # fit: same bounding-box volume as the stock box
    assert o["mass_kg"] == pytest.approx(o["stock_mass_kg"], rel=1e-3)  # mass match
    # Other objects and the non-xml files of the replaced one are relative links into the stock dir.
    butter = assets / "stable_hope_objects_r2s_full" / "butter"
    assert butter.is_symlink() and not os.path.isabs(os.readlink(butter))
    tex = assets / "stable_hope_objects_r2s_full" / "milk" / "texture_map.png"
    assert tex.is_symlink() and tex.read_bytes() == b"png"
    # collision mode keeps the stock visual; full renders the hull instead. Stock dir untouched.
    col = ET.parse(assets / "stable_hope_objects_r2s_collision" / "milk" / "milk.xml").getroot()
    full = ET.parse(assets / "stable_hope_objects_r2s_full" / "milk" / "milk.xml").getroot()
    groups = lambda root: sorted((g.get("group"), g.get("type")) for g in root.iter("geom"))  # noqa: E731
    assert groups(col) == [("0", "mesh"), ("1", "mesh")]
    assert [g.get("mesh") for g in full.iter("geom") if g.get("group") == "1"] == ["r2s_hull"]
    assert [g.get("friction") for g in col.iter("geom") if g.get("group") == "0"] == ["0.95 0.3 0.1"]
    assert (assets / "stable_hope_objects" / "milk" / "milk.xml").read_text() == STOCK_XML


def test_build_keep_stock_and_metric(scene):
    assets, seg = scene
    with pytest.raises(SystemExit, match="kept stock"):
        build.build(str(assets), str(seg), keep_stock=["milk"])
    man = build.build(str(assets), str(seg), modes=("collision",), scale="metric", mass="density")
    o = man["objects"]["milk"]
    assert o["scale"] == 1.0 and sorted(o["final_extent_m"]) == pytest.approx([0.07, 0.07, 0.20], abs=1e-4)
    assert o["density"] == 100.0


# --- runtime patches ---------------------------------------------------------------------------------

HOPE_PY = '''import os
from libero.libero.envs.base_object import register_object


class HopeBaseObject:
    def __init__(self, name, obj_name):
        self.path = os.path.join(
                f"assets/stable_hope_objects/{obj_name}/{obj_name}.xml",
        )
'''


def test_patch_hope_use_idempotent(tmp_path):
    pkg = tmp_path / "libero" / "libero"
    py = pkg / "envs" / "objects" / "hope_objects.py"
    py.parent.mkdir(parents=True)
    py.write_text(HOPE_PY)
    (pkg / "assets" / "stable_hope_objects_r2s_full").mkdir(parents=True)
    assert select.patch_hope(str(py)) == "patched"
    assert select.patch_hope(str(py)) == "already patched"
    select.use(str(py), "stable_hope_objects_r2s_full")
    ns = {}
    exec(compile(py.read_text().replace("from libero.libero.envs.base_object import register_object", ""),
                 str(py), "exec"), ns)
    assert ns["R2S_HOPE_DIR"] == "stable_hope_objects_r2s_full"
    with pytest.raises(SystemExit):
        select.use(str(py), "no_such_dir")


EVAL = "import tqdm\nfor task_id in tqdm.tqdm(range(num_tasks_in_suite)):\n    pass\n"


def test_loop_patch(tmp_path):
    f = tmp_path / "main.py"
    f.write_text(EVAL)
    select.loop(str(f), "0,1,5")
    assert "for task_id in tqdm.tqdm([0,1,5]):" in f.read_text()
    with pytest.raises(SystemExit):
        select.loop(str(f), "0,1,5")  # the stock loop is gone: a second patch must not pass silently
    with pytest.raises(SystemExit):
        select.loop(str(f), "0;rm")


def test_select_tasks_by_object_of_interest(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / "benchmark").mkdir(parents=True)
    (pkg / "benchmark" / "libero_suite_task_map.py").write_text(
        'libero_task_map = {"libero_object": ["pick_milk", "pick_ketchup"]}\n')
    bd = pkg / "bddl_files" / "libero_object"
    bd.mkdir(parents=True)
    for name, interest in (("pick_milk", "milk_1"), ("pick_ketchup", "ketchup_1")):
        (bd / f"{name}.bddl").write_text(
            f"(define (problem p)\n (:language Pick up the {interest[:-2]})\n"
            f" (:objects milk_1 - milk ketchup_1 - ketchup basket_1 - basket)\n"
            f" (:obj_of_interest {interest} basket_1)\n)\n")
    hits = select.select_tasks(str(pkg), "libero_object", ["milk"])
    assert [h["task_id"] for h in hits] == [0] and hits[0]["language"] == "Pick up the milk"
    assert [h["task_id"] for h in select.select_tasks(str(pkg), "libero_object", ["milk"], any_role=True)] == [0, 1]


# --- report ------------------------------------------------------------------------------------------

def test_report_parses_oft_and_openpi_logs(tmp_path):
    pub = tmp_path / "public.log"
    pub.write_text("Task: pick up the milk\nSuccess: True\nSuccess: True\nTask: pick up the soup\nSuccess: False\n")
    full = tmp_path / "full.log"  # docker compose prefix of the openpi client
    full.write_text("runtime-1  | Task: pick up the milk\nruntime-1  | INFO:root:Success: False\r\n"
                    "runtime-1  | INFO:root:Success: True\nruntime-1  | Task: pick up the soup\n"
                    "runtime-1  | INFO:root:Success: False\n")
    tasks = [{"language": "Pick up the milk", "objects": ["milk"]}, {"language": "Pick up the soup",
                                                                     "objects": ["alphabet_soup"]}]
    objects = {"objects": {}, "kept_stock": ["cookies"]}
    res = report.final(str(tmp_path / "out"), [("public", str(pub)), ("full", str(full))], tasks, objects,
                       {"unstable": [], "steps": 40}, suite="libero_object", model="m")
    assert (res["conditions"]["public"]["k"], res["conditions"]["public"]["n"]) == (2, 3)
    assert (res["conditions"]["full"]["k"], res["conditions"]["full"]["n"]) == (1, 3)
    assert res["conditions"]["full"]["drop_pp_vs_public"] == pytest.approx(33.3)
    assert res["per_object"]["milk"]["full"] == {"k": 1, "n": 2}
    lo, hi = res["conditions"]["public"]["ci95"]
    assert 0 < lo < 2 / 3 < hi < 1
    summary = (tmp_path / "out" / "summary.txt").read_text()
    assert "benchmark: libero_r2s" in summary and "kept stock (unstable in the physics check): cookies" in summary


# --- generate.py / deploy.py -------------------------------------------------------------------------

def _run(script, *args):
    return subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT, capture_output=True, text=True,
                          timeout=120)


@pytest.mark.parametrize("vla,extra", [("openvla-oft", ["--libero-suite", "object"]), ("pi", [])])
def test_generate_real2sim_render(tmp_path, vla, extra):
    pytest.importorskip("jinja2")
    p = _run("generate.py", "--vla", vla, *extra, "--output-dir", str(tmp_path), "--libero-objects", "real2sim",
             "--real2sim-run", RUN, "--r2s-conditions", "collision", "--r2s-trials", "2")
    assert p.returncode == 0, p.stderr
    sh = (tmp_path / f"{vla}.sh").read_text()
    assert subprocess.run(["bash", "-n", str(tmp_path / f"{vla}.sh")]).returncode == 0
    assert "s3://example-results/2026-10-29-0142/real2sim/segment/" in sh
    assert '"condition": "LIBERO-R2S/public"' in sh and '"condition": "LIBERO-R2S/collision"' in sh
    assert "LIBERO-R2S/full" not in sh.split("TASKS_JSON=")[-1].split("\n")[0]
    assert "r2s_stabilise" in sh and "--keep-stock" in sh and "libero-r2s/$(basename" in sh


@pytest.mark.parametrize("args,msg", [
    (["--vla", "smolvla", "--libero-objects", "real2sim", "--real2sim-run", RUN], "not available"),
    (["--vla", "pi", "--libero-objects", "real2sim"], "needs --real2sim-run"),
    (["--vla", "pi", "--real2sim-run", RUN], "need --libero-objects real2sim"),
    (["--vla", "openvla-oft", "--libero-objects", "real2sim", "--real2sim-run", RUN, "--r2s-suite", "libero_10"],
     "applies to pi"),
    (["--vla", "pi", "--libero-objects", "real2sim", "--real2sim-run", "s3://example-results/"], "--real2sim-run"),
])
@pytest.mark.parametrize("script", ["generate.py", "deploy.py"])
def test_real2sim_flag_errors(tmp_path, script, args, msg):
    if script == "deploy.py":
        pytest.importorskip("boto3")
    else:
        pytest.importorskip("jinja2")
    p = _run(script, *args, *(["--output-dir", str(tmp_path)] if script == "generate.py" else []))
    assert p.returncode != 0 and msg in p.stderr, p.stderr


def test_deploy_rejects_real2sim_with_bridge_or_plus():
    pytest.importorskip("boto3")
    for extra in (["--bridge"], ["--libero-variant", "plus"]):
        p = _run("deploy.py", "--vla", "pi", "--libero-objects", "real2sim", "--real2sim-run", RUN, *extra)
        assert p.returncode != 0 and "cannot be combined" in p.stderr, p.stderr


def test_matrix_label_and_baseline():
    import leaderboard

    assert leaderboard.benchmark_label("libero_object", "LIBERO-R2S/full") == "LIBERO-R2S/full (Object)"
    assert leaderboard.is_baseline("LIBERO-R2S/public")
