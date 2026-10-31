"""LIBERO-Plus switch (--libero-variant plus): sweep plan, on-instance variant picker, OFT loop patch,
renders, and the results.py robustness view. Offline: the classification file is synthetic."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import libero_plus  # noqa: E402
import results  # noqa: E402

SELECT = ROOT / "assets" / "libero-plus" / "select_variants.py"
PATCH = ROOT / "assets" / "libero-plus" / "oft_task_ids_patch.py"


# --- plan ------------------------------------------------------------------------------------------

def test_parse_perturb_orders_and_validates():
    assert libero_plus.parse_perturb("light,camera,light") == ["camera", "light"]
    assert libero_plus.parse_perturb("all") == list(libero_plus.CATEGORIES)
    assert libero_plus.parse_perturb(["Noise"]) == ["noise"]
    with pytest.raises(ValueError):
        libero_plus.parse_perturb("camera,fog")
    with pytest.raises(ValueError):
        libero_plus.parse_perturb("")


def test_plan_defaults_and_overrides():
    cfg, tasks = libero_plus.plan({"variants_per_category": 4, "seed": 3})
    assert cfg["suite"] == "libero_spatial" and len(tasks) == 7
    assert [t["condition"] for t in tasks][:2] == ["LIBERO-Plus/camera", "LIBERO-Plus/robot"]
    assert all(t["variants"] == 4 and t["variant_seed"] == 3 for t in tasks)
    assert libero_plus.episodes(cfg) == 28
    cfg, tasks = libero_plus.plan({}, perturb="light", n=0, suite="libero_goal")
    assert [t["perturbation_category"] for t in tasks] == ["Light Conditions"]
    assert tasks[0]["suite"] == "libero_goal" and libero_plus.episodes(cfg) is None
    for bad in ({"suite": "libero_90"}, {"variants_per_category": -1}, {"episodes_per_variant": 0}):
        with pytest.raises(ValueError):
            libero_plus.plan(bad)


# --- select_variants.py ------------------------------------------------------------------------------

def _classification(tmp_path):
    cats = list(libero_plus.CATEGORIES.values())
    data = {"libero_spatial": [dict(id=i + 1, name=f"task_{i}", category=cats[i % 7], difficulty_level=1 + i % 5)
                               for i in range(70)]}
    p = tmp_path / "task_classification.json"
    p.write_text(json.dumps(data))
    return p


def _select(tmp_path, *args):
    out = subprocess.run([sys.executable, str(SELECT), str(_classification(tmp_path)), *map(str, args)],
                         capture_output=True, text=True, timeout=60)
    return out


def test_select_is_seeded_zero_based_and_in_category(tmp_path):
    a = _select(tmp_path, "libero_spatial", "Light Conditions", 4, 7, "--manifest", tmp_path / "m.json")
    assert a.returncode == 0, a.stderr
    ids = json.loads(a.stdout)
    assert len(ids) == 4 and ids == sorted(ids)
    assert all(i % 7 == 3 for i in ids)  # category index 3 = Light Conditions, ids are 0-based
    assert json.loads(_select(tmp_path, "libero_spatial", "Light Conditions", 4, 7).stdout) == ids
    assert json.loads(_select(tmp_path, "libero_spatial", "Light Conditions", 4, 8).stdout) != ids
    manifest = json.loads((tmp_path / "m.json").read_text())
    assert [m["task_id"] for m in manifest] == ids and manifest[0]["name"] == f"task_{ids[0]}"


def test_select_all_csv_and_errors(tmp_path):
    out = _select(tmp_path, "libero_spatial", "Sensor Noise", 0, 7, "--format", "csv")
    assert out.returncode == 0 and out.stdout.strip() == ",".join(str(i) for i in range(70) if i % 7 == 5)
    assert _select(tmp_path, "libero_spatial", "Fog", 3, 7).returncode != 0
    assert _select(tmp_path, "libero_goal", "Sensor Noise", 3, 7).returncode != 0


# --- OpenVLA-OFT task-loop patch --------------------------------------------------------------------

def test_oft_patch_rewrites_the_task_loop_once(tmp_path):
    f = tmp_path / "run_libero_eval.py"
    f.write_text("    num_tasks = task_suite.n_tasks\n    for task_id in tqdm.tqdm(range(num_tasks)):\n        pass\n")
    assert subprocess.run([sys.executable, str(PATCH), str(f)], capture_output=True).returncode == 0
    assert "LIBERO_PLUS_TASK_IDS" in f.read_text()
    # second run: the loop is gone → fail loudly instead of silently running every variant
    assert subprocess.run([sys.executable, str(PATCH), str(f)], capture_output=True).returncode != 0


def test_openpi_patch_rewrites_loop_and_video_name_once(tmp_path):
    patch = ROOT / "assets" / "libero-plus" / "openpi_task_ids_patch.py"
    f = tmp_path / "main.py"
    f.write_text("    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):\n"
                 "        p = f\"rollout_{task_segment}_{suffix}.mp4\"\n"
                 "    env_args = {\"bddl_file_name\": task_bddl_file, \"camera_heights\": resolution}\n")
    assert subprocess.run([sys.executable, str(patch), str(f), "4,12"], capture_output=True).returncode == 0
    text = f.read_text()
    assert "tqdm.tqdm([4, 12])" in text and "rollout_{task_id:05d}_{task_segment}" in text
    # LIBERO-Plus' env_wrapper does `"_view_" in bddl_file_name` → TypeError on a pathlib.Path
    assert '"bddl_file_name": str(task_bddl_file)' in text
    # second run: the loop is gone → fail loudly instead of silently running every variant
    assert subprocess.run([sys.executable, str(patch), str(f), "4"], capture_output=True).returncode != 0
    f.write_text("for task_id in tqdm.tqdm(range(num_tasks_in_suite)):\n")  # video site missing
    assert subprocess.run([sys.executable, str(patch), str(f), "4"], capture_output=True).returncode != 0
    f.write_text("for task_id in tqdm.tqdm(range(num_tasks_in_suite)):\n"  # bddl site missing
                 "p = f\"rollout_{task_segment}_{suffix}.mp4\"\n")
    assert subprocess.run([sys.executable, str(patch), str(f), "4"], capture_output=True).returncode != 0


# --- renders -----------------------------------------------------------------------------------------

def _gen(tmp_path, *args):
    return subprocess.run([sys.executable, str(ROOT / "generate.py"), "--output-dir", str(tmp_path), *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("vla,extra", [("smolvla", []), ("openvla-oft", ["--libero-suite", "spatial"]), ("pi", []),
                                       ("xvla", []), ("fastwam", []), ("gr00t", []),
                                       ("vla-adapter", []), ("molmoact2", []),
                                       ("rldx", []), ("lap", [])])
def test_plus_render(vla, extra, tmp_path):
    pytest.importorskip("jinja2")
    out = _gen(tmp_path, "--vla", vla, *extra, "--libero-variant", "plus", "--perturb", "camera,light", "--perturb-n", "3")
    assert out.returncode == 0, out.stderr
    text = (tmp_path / f"{vla}.sh").read_text()
    assert libero_plus.DEFAULTS["repo_commit"] in text and libero_plus.DEFAULTS["assets_sha256"] in text
    tasks = json.loads(text.split("TASKS_JSON='", 1)[1].split("'\n", 1)[0])
    assert [t["condition"] for t in tasks] == ["LIBERO-Plus/camera", "LIBERO-Plus/light"]
    suite = "libero_goal" if vla == "molmoact2" else "libero_spatial"  # molmoact2: suite of its baseline
    assert all(t["variants"] == 3 and t["suite"] == suite for t in tasks)
    if vla in ("smolvla", "xvla", "fastwam", "molmoact2"):
        assert "--env.type=libero_plus" in text and "uv run --active --no-sync lerobot-eval" in text
        assert all(t["n_episodes"] == 1 for t in tasks)
    elif vla == "gr00t":
        assert 'libero_sim/${LP_NAME}' in text and "LP_SETUP_EOF" in text and "LIBERO-PLUS VERIFY OK" in text
        assert "NEEDED = 'libero_spatial'" in text  # checkpoint subfolder follows the sweep suite
        assert all(t["n_episodes"] == 1 and t["embodiment"] == "LIBERO_PANDA" for t in tasks)
    elif vla == "rldx":  # RLDX-1's own LIBERO-Plus path (libero_plus_uv venv, libero_plus_sim/<variant>)
        assert "setup_libero_plus.sh" in text and 'libero_plus_sim/${LP_NAME}' in text
        assert "LIBERO-PLUS VERIFY OK" in text and "LIBERO_PLUS_UV" not in text
        assert "LIBERO_VENV=${RLDX_DIR}/rldx/eval/sim/LIBERO_PLUS/libero_plus_uv/.venv" in text
        assert "run_scripts/eval/libero/setup_libero.sh" not in text
        assert all(t["num_episodes"] == 1 for t in tasks)
    elif vla == "lap":  # pi's Python 3.8 client recipe in LAP's host venv
        assert "lap_lp_task_ids_patch.py" in text and "requirements.lap.txt" in text and "lp_verify.py" in text
        assert 'cp "${LAP_MAIN_PY}.lporig" "${LAP_MAIN_PY}"' in text and "rollout_wrist_{task_id:05d}_" in text
        assert all(t["num_trials_per_task"] == 1 for t in tasks)
    elif vla == "pi":
        assert "pi_lp_task_ids_patch.py" in text and "requirements.openpi.txt" in text
        assert 'rm -rf "${OPENPI_DIR}/third_party/libero"' in text and "lp_verify.py" in text
        assert "hf_dl_guard" not in text and "/root/.libero/config.yaml" not in text
        assert 'LIBERO_PLUS_DIR=${OPENPI_DIR}/third_party/libero' in text and "Cloning LIBERO-Plus" in text
        assert all(t["num_episodes"] == 1 for t in tasks)
    else:
        assert "lp_oft_task_ids_patch.py" in text and "Lifelong-Robot-Learning/LIBERO.git" not in text
        assert all(t["num_trials_per_task"] == 1 for t in tasks)
        if vla == "vla-adapter":  # no stackpatch.py here: the eval script is restored before each patch
            assert 'cp "${LP_EVAL_PY}.lporig" "${LP_EVAL_PY}"' in text and "condition: ${K_COND}" in text
    assert subprocess.run(["bash", "-n", str(tmp_path / f"{vla}.sh")]).returncode == 0


def test_explicit_libero_variant_is_the_default_render(tmp_path):
    pytest.importorskip("jinja2")
    a, b = tmp_path / "a", tmp_path / "b"
    assert _gen(a, "--vla", "smolvla").returncode == 0
    assert _gen(b, "--vla", "smolvla", "--libero-variant", "libero").returncode == 0
    assert (a / "smolvla.sh").read_bytes() == (b / "smolvla.sh").read_bytes()
    assert b"libero_plus" not in (a / "smolvla.sh").read_bytes()


@pytest.mark.parametrize("args", [
    ["--vla", "aspire", "--libero-variant", "plus"],             # no switch in this template
    ["--vla", "cosmos-policy", "--libero-variant", "plus"],      # no switch in this template
    ["--vla", "gr00t-bridge", "--libero-variant", "plus"],       # SimplerEnv, not LIBERO
    ["--vla", "smolvla", "--libero-variant", "plus", "--perturb", "fog"],
    ["--vla", "smolvla", "--perturb", "camera"],                 # --perturb without plus
])
def test_plus_rejects(args, tmp_path):
    pytest.importorskip("jinja2")
    out = _gen(tmp_path, *args)
    assert out.returncode != 0 and "[error]" in out.stderr


# --- results.py --------------------------------------------------------------------------------------

def test_task_row_reads_the_plus_fields():
    from test_results import BUCKET, FakeS3
    s3 = FakeS3({BUCKET: {"R/task-0/summary.txt": (
        "suite: libero_spatial\ncondition: LIBERO-Plus/camera\nsuccess_rate: 0.300\nn_success: 3\nn_episodes: 10\n"
        "benchmark: libero_plus\nperturbation: camera\nperturbation_category: Camera Viewpoints\n")}})
    r = results.task_row(s3, "us-east-1", BUCKET, "R", "R/task-0/")
    assert (r["condition"], r["benchmark"], r["perturbation"], r["k"], r["n"]) == (
        "LIBERO-Plus/camera", "libero_plus", "camera", 3, 10)


def test_robustness_table():
    base = dict(bucket_stack="smolvla-demo", region="us-east-1", suite="libero_spatial", status="done")
    rows = [
        dict(base, run="2026-10-01-0000", task="task-0", condition="S0", k=5, n=10),     # older baseline
        dict(base, run="2026-10-02-0000", task="task-0", condition="S0", k=9, n=10),     # latest baseline
        dict(base, run="2026-10-02-0000", task="task-1", condition="replan", k=1, n=10),  # knob run: not a baseline
        dict(base, run="2026-10-03-0000", task="task-0", condition="LIBERO-Plus/camera", k=2, n=10),
        dict(base, run="2026-10-03-0000", task="task-1", condition="LIBERO-Plus/light", k=8, n=10),
        dict(base, run="2026-09-01-0000", task="task-0", condition="LIBERO-Plus/camera", k=10, n=10),  # older sweep
        dict(base, bucket_stack="pi-demo", run="2026-10-03-0000", task="task-0", condition="S0", k=1, n=1),
    ]
    t = {r["perturbation"]: r for r in results.robustness(rows)}
    assert set(t) == {"camera", "light", "all"}
    assert (t["camera"]["k_n"], t["camera"]["rate"], t["camera"]["drop_pp"]) == ("2/10", "0.20", "+70.0")
    assert t["light"]["drop_pp"] == "+10.0" and t["all"]["k_n"] == "10/20" and t["all"]["drop_pp"] == "+40.0"
    assert t["camera"]["baseline"] == "0.90 (9/10)" and t["camera"]["baseline_run"] == "2026-10-02-0000"
    assert t["all"]["ci95"] == "0.30-0.70"


def test_robustness_without_baseline():
    rows = [dict(bucket_stack="s", region="r", suite="libero_spatial", status="done", run="R", task="task-0",
                 condition="LIBERO-Plus/noise", k=1, n=4)]
    (noise, total) = results.robustness(rows)
    assert noise["baseline"] == "-" and noise["drop_pp"] == "" and total["perturbation"] == "all"
    assert results.robustness([dict(rows[0], condition="S0")]) == []


def test_fork_requirements_filter_and_check_agree():
    """The post-filter check must use the same name boundary as the filter.

    The fork lists ``wandb``; a check without the boundary matched ``wand`` inside it and
    aborted the pi / lap / gr00t plus runs right after the asset download.
    """
    import re
    import subprocess

    fixture = " wandb==0.13.1 \n robosuite==1.4.0\nusd-core>=25.5\nWand\nscikit-image\nnumpy==1.26.4\n"
    root = Path(__file__).resolve().parents[1] / "templates"
    for name in ("pi-userdata.sh.j2", "lap-userdata.sh.j2", "gr00t-userdata.sh.j2"):
        text = (root / name).read_text()
        flt = re.search(r"grep -viE '([^']+)'", text).group(1)
        chk = re.search(r"! grep -qiE '([^']+)'", text).group(1)
        kept = subprocess.run(["grep", "-viE", flt], input=fixture, capture_output=True, text=True).stdout
        assert "wandb" in kept and "robosuite" not in kept, name
        bad = subprocess.run(["grep", "-qiE", chk], input=kept, capture_output=True, text=True).returncode == 0
        assert not bad, f"{name}: check rejects a correctly filtered requirements file"
