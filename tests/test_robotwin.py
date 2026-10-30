"""RoboTwin 2.0 targets (smolvla-robotwin, lingbot-vla2): run plan, rendered userdata, results label.

Offline: renders through generate.py and checks the script text; nothing is launched.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import results  # noqa: E402
import robotwin_plan as rp  # noqa: E402
from tests.test_results import FakeS3  # noqa: E402

pytest.importorskip("jinja2")
yaml = pytest.importorskip("yaml")

TARGETS = ("smolvla-robotwin", "lingbot-vla2")
SUBSET = ["beat_block_hammer", "click_bell", "handover_block", "lift_pot", "stamp_seal"]


# --- plan ----------------------------------------------------------------------------------------

def test_task_tables_agree():
    assert len(rp.TASKS) == 50 == len(set(rp.TASKS))
    assert set(rp.STEP_LIMIT) == set(rp.TASKS)
    assert "open_laptop" not in rp.ALL_TASKS and len(rp.ALL_TASKS) == 49


@pytest.mark.parametrize("vla", TARGETS)
def test_default_subset_from_yaml(vla):
    cfg = yaml.safe_load((ROOT / "models" / f"{vla}.yaml").read_text())
    units = rp.plan(cfg["tasks"])
    assert [u["task"] for u in units] == SUBSET
    assert {u["setting"] for u in units} == {"clean"} and rp.episodes(units) == 25
    assert all(u["seed"] == rp.OFFICIAL_START_SEED for u in units)


def test_official_protocols():
    assert rp.episodes(rp.plan([], "clean")) == 49 * 100
    full = rp.plan([], "full")
    assert len(full) == 98 and rp.episodes(full) == 9800
    assert [u["setting"] for u in full].count("randomized") == 49


@pytest.mark.parametrize("n", [1, 7, 25])
def test_shards_partition_the_plan(n):
    full = rp.plan([], "full")
    shards = [rp.plan([], "full", shard=f"{k}/{n}") for k in range(1, n + 1)]
    flat = [(u["setting"], u["task"]) for s in shards for u in s]
    assert sorted(flat) == sorted((u["setting"], u["task"]) for u in full)
    assert len(flat) == len(set(flat))


@pytest.mark.parametrize("shard", ["0/3", "4/3", "x", "1/0", "1-3"])
def test_bad_shard(shard):
    with pytest.raises(ValueError):
        rp.plan([], "clean", shard=shard)


def test_bad_yaml_entries():
    with pytest.raises(ValueError, match="unknown RoboTwin task"):
        rp.plan([{"setting": "clean", "robotwin_tasks": ["nope"], "n_episodes": 1}])
    with pytest.raises(ValueError, match="setting"):
        rp.plan([{"setting": "hard", "robotwin_tasks": ["click_bell"], "n_episodes": 1}])
    with pytest.raises(ValueError, match="n_episodes"):
        rp.plan([{"setting": "clean", "robotwin_tasks": ["click_bell"], "n_episodes": 0}])
    with pytest.raises(ValueError, match="protocol"):
        rp.plan([], "leaderboard")


def test_worst_case_hours():
    units = rp.plan([{"setting": "clean", "robotwin_tasks": ["click_bell"], "n_episodes": 9}])
    assert rp.worst_case_hours(units, 1.0) == pytest.approx(9 * 400 / 3600)


# --- rendered userdata ---------------------------------------------------------------------------

def _render(tmp_path, vla, *extra):
    proc = subprocess.run([sys.executable, str(ROOT / "generate.py"), "--vla", vla, "--output-dir", str(tmp_path),
                           *extra], cwd=ROOT, capture_output=True, text=True, timeout=120)
    return proc, (tmp_path / f"{vla}.sh")


def _tasks_json(text):
    m = re.search(r"^TASKS_JSON='(.*)'$", text, re.M)
    return json.loads(m.group(1))


@pytest.mark.parametrize("vla", TARGETS)
def test_render_default(vla, tmp_path):
    proc, script = _render(tmp_path, vla)
    assert proc.returncode == 0, proc.stderr
    text = script.read_text()
    cfg = yaml.safe_load((ROOT / "models" / f"{vla}.yaml").read_text())
    tasks = _tasks_json(text)
    assert [t["task"] for t in tasks] == SUBSET
    assert {t["task_config"] for t in tasks} == {"demo_clean"}
    # one lerobot-eval per (setting, task) with the official step limit, results label RoboTwin2/<setting>
    assert "--env.type=robotwin" in text and '--env.task="${RT_TASK}"' in text
    assert 'SUITE="RoboTwin2/${SETTING}"' in text and 'echo "suite: ${SUITE}"' in text
    assert "_eval_step_limit.yml" in text and '--env.episode_length="${EPISODE_LENGTH}"' in text
    assert f"--rename_map={json.dumps(cfg['model']['rename_map'], separators=(', ', ': '))}" in text
    # pins
    assert cfg["model"]["hf_model_revision"] in text and cfg["model"]["lerobot_commit"] in text
    for pin in ("commit", "assets_revision", "curobo_commit"):
        assert rp.PINS[pin] in text
    for r in cfg["model"]["extra_hf_repos"]:
        assert f'hf_pinned "{r["repo"]}" "{r["revision"]}"' in text
    # simulator stack: Vulkan, mplib/sapien without their numpy<2 / opencv-python metadata, CuRobo, no textures
    assert "libvulkan1" in text and "nvidia_icd.json" in text
    assert re.search(r'--no-deps \\\n\s+"sapien==3\.0\.0b1" "mplib==0\.2\.1"', text)
    assert "--constraint /tmp/" in text and "numpy moved" in text
    assert "TORCH_CUDA_ARCH_LIST=\"${GPU_CC}\"" in text
    assert "'background_texture.zip', repo_type='dataset'" not in text
    # pre-model verify imports the whole chain, then a smoke eval before the loop
    for mod in ("lerobot.scripts.lerobot_eval", "curobo.wrap.reacher.motion_gen", "envs.robot.planner",
                f"lerobot.policies.{cfg['model']['policy_module']}.processor_{cfg['model']['policy_module']}"):
        assert f'"{mod}"' in text
    assert text.index("[5/8] Verifying") < text.index("[6/8] Pre-downloading") < text.index("[6.5/8] Smoke eval") \
        < text.index("for TASK_IDX in")
    assert "export HF_HUB_OFFLINE=1" in text
    # nothing re-syncs the venv after the RoboTwin packages went in
    assert not re.search(r"^[^#\n]*\buv run\b", text, re.M)
    # lerobot wrapper patches are asserted, not best-effort
    assert 'LEROBOT_ROBOTWIN_TASK_CONFIG", "demo_clean"' in text and "RFM-UNSTABLE-SEED" in text
    assert "expected 1 match" in text


# Verbatim excerpt of RoboTwin @0aeea2d6 envs/robot/planner.py (the lines the mplib patch rewrites).
_PLANNER_EXCERPT = """import mplib.planner
import mplib
import numpy as np
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld


# ********************** MplibPlanner **********************
class MplibPlanner:
    def __init__(self, urdf_path, srdf_path, move_group, robot_origion_pose, robot_entity,
                 planner_type="mplib_RRT", scene=None):
        if scene is None:
            pass
        else:
            planning_world = SapienPlanningWorld(scene, [robot_entity])
            self.planner = SapienPlanner(planning_world, move_group)

        self.planner_type = planner_type
"""


@pytest.mark.parametrize("vla", TARGETS)
def test_mplib_topp_only_patch(vla, tmp_path):
    # mplib 0.2.1 segfaults on numpy 2 (old pybind11): the template swaps RoboTwin's SapienPlanner for a
    # TOPP-only shim, before the verify step, and the verify asserts it. Run the embedded patch for real.
    proc, script = _render(tmp_path, vla)
    assert proc.returncode == 0, proc.stderr
    text = script.read_text()
    m = re.search(r'"\$\{VENV_PY\}" - "\$\{ROBOTWIN_DIR\}/envs/robot/planner.py" << \'PATCHEOF\'.*?\n(.*?)\nPATCHEOF\n',
                  text, re.S)
    assert m, "RoboTwin planner patch block missing"
    assert text.index("envs/robot/planner.py\" << 'PATCHEOF'") < text.index("[5/8] Verifying")
    assert '"envs.robot.planner is not patched' in text
    body = m.group(1).split("\n", 1)[1]  # drop the `|| die` continuation line
    (tmp_path / "patch.py").write_text(body)
    target = tmp_path / "planner.py"
    target.write_text(_PLANNER_EXCERPT)
    for expect in ("SapienPlanner -> _ToppOnlyPlanner", "already patched"):  # idempotent on re-run
        run = subprocess.run([sys.executable, str(tmp_path / "patch.py"), str(target)], capture_output=True, text=True)
        assert run.returncode == 0 and expect in run.stdout, run.stderr
    out = target.read_text()
    assert "SapienPlanningWorld(scene" not in out
    assert "self.planner = _ToppOnlyPlanner(robot_entity, move_group)" in out
    assert out.index("class _ToppOnlyPlanner") < out.index("class MplibPlanner")
    compile(out, "planner.py", "exec")
    # a planner.py that moved upstream fails loudly instead of silently skipping the patch
    target.write_text(_PLANNER_EXCERPT.replace("SapienPlanner(planning_world", "SapienPlanner(world"))
    run = subprocess.run([sys.executable, str(tmp_path / "patch.py"), str(target)], capture_output=True, text=True)
    assert run.returncode != 0 and "not found exactly once" in run.stderr


def test_render_randomized_shard(tmp_path):
    proc, script = _render(tmp_path, "smolvla-robotwin", "--robotwin-protocol", "full", "--robotwin-shard", "2/25")
    assert proc.returncode == 0, proc.stderr
    text = script.read_text()
    tasks = _tasks_json(text)
    want = rp.plan([], "full", shard="2/25")
    assert [(t["setting"], t["task"]) for t in tasks] == [(u["setting"], u["task"]) for u in want]
    assert {t["n_episodes"] for t in tasks} == {100}
    assert {t["task_config"] for t in tasks} == {rp.SETTINGS[u["setting"]] for u in want}
    assert 'echo "condition: full"' in text
    if any(u["setting"] == "randomized" for u in want):
        assert "'background_texture.zip', repo_type='dataset'" in text


def test_robotwin_flags_rejected_for_other_targets(tmp_path):
    proc, _ = _render(tmp_path, "smolvla", "--robotwin-protocol", "clean")
    assert proc.returncode != 0 and "--robotwin-protocol" in proc.stderr


def test_bad_shard_fails_render(tmp_path):
    proc, _ = _render(tmp_path, "lingbot-vla2", "--robotwin-protocol", "clean", "--robotwin-shard", "9/3")
    assert proc.returncode != 0 and "shard" in proc.stderr


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not on PATH")
@pytest.mark.parametrize("vla", TARGETS)
def test_shellcheck_errors(vla, tmp_path):
    _, script = _render(tmp_path, vla, "--robotwin-protocol", "full", "--robotwin-shard", "1/25")
    out = subprocess.run(["shellcheck", "-S", "error", str(script)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout


# --- vlasim results label ------------------------------------------------------------------------

def test_results_label():
    bucket = "vla-sim-results-smolvla-robotwin-us-east-1-123456789012"
    summ = ("model: smolvla-robotwin\nbenchmark: robotwin2\nsuite: RoboTwin2/clean\nsetting: clean\n"
            "robotwin_task: click_bell\nsuccess_rate: 0.600\nn_success: 3\nn_episodes: 5\ncondition: subset\n"
            "stack_robotwin_commit: 0aeea2d6\n")
    s3 = FakeS3({bucket: {"R/task-0/summary.txt": summ, "R/task-1/summary.txt": summ.replace("n_success: 3", "n_success: 5")}})
    rows = [dict(results.task_row(s3, "us-east-1", bucket, "R", f"R/task-{i}/"), bucket_stack="smolvla-robotwin",
                 region="us-east-1") for i in (0, 1)]
    assert rows[0]["suite"] == "RoboTwin2/clean" and rows[0]["condition"] == "subset"
    g = results.aggregate(rows)[("smolvla-robotwin", "us-east-1", "R", "RoboTwin2/clean", "subset")]
    assert (g["k"], g["n"], g["tasks"]) == (8, 10, 2)
    assert results.BUCKET_RE.match(bucket).group("stack") == "smolvla-robotwin"
