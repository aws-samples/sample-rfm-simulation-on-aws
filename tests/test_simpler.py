"""SimplerEnv targets that run the authors' own evaluation: spatialvla (DelinQu/SimplerEnv-OpenVLA) and magma
(upstream SimplerEnv + the Magma plugin), shared templates/simpler-userdata.sh.j2.

Covers the argv parsing / episode count of generate.simpler_* (argparse abbreviations, grids, episode ranges), the
task validation, the rendered UserData (pins, plugin copy, embedded lock file, task sets), the task loop run through
bash + jq with a stub main_inference that writes SimplerEnv-style video names, and the summary.txt fields results.py
reads.
"""

import base64
import copy
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

yaml = pytest.importorskip("yaml")
pytest.importorskip("jinja2")

from test_results import BUCKET, FakeS3  # noqa: E402

import generate  # noqa: E402
import results  # noqa: E402
import task_sets  # noqa: E402

BASH = shutil.which("bash")
JQ = shutil.which("jq")
TARGETS = ("spatialvla", "magma")


def _model(vla):
    return yaml.safe_load((ROOT / "models" / f"{vla}.yaml").read_text())


def _render(vla, out_dir, *extra, ok=True):
    proc = subprocess.run([sys.executable, str(ROOT / "generate.py"), "--vla", vla, "--output-dir", str(out_dir),
                           *extra], cwd=ROOT, capture_output=True, text=True, timeout=120)
    if ok:
        assert proc.returncode == 0, f"{vla} {extra}:\n{proc.stdout}\n{proc.stderr}"
    return proc


# --- argv parsing and episode count (simpler_env/evaluation/argparse.py + maniskill2_evaluator.py) ---------

def test_flag_resolution_follows_argparse_abbreviations():
    assert generate.simpler_flag("--robot") == "--robot"                      # exact flag wins over prefixes
    assert generate.simpler_flag("--robot-init-x") == "--robot-init-x-range"  # the scripts' abbreviation
    assert generate.simpler_flag("--obj-init-y") == "--obj-init-y-range"
    assert generate.simpler_flag("--robot-init") is None                      # ambiguous
    assert generate.simpler_flag("--nope") is None
    assert generate.simpler_flag("--action-ensemble-temp") is None            # fork-only flag
    assert generate.simpler_flag("--action-ensemble-temp", generate.SIMPLER_FLAGS + ("--action-ensemble-temp",))


@pytest.mark.parametrize("argv,n", [
    ([], 25),                                                                  # defaults: xy mode, 5 x 5 objects
    (["--obj-init-x", "-0.35", "-0.12", "5", "--obj-init-y", "-0.02", "0.42", "5"], 25),
    (["--obj-variation-mode", "episode", "--obj-episode-range", "0", "24"], 24),
    (["--obj-variation-mode", "episode"], 60),                                 # default range 0..60
    (["--obj-variation-mode", "episode", "--obj-episode-range", "10", "20",
      "--robot-init-x", "0.1", "0.3", "3", "--robot-init-y", "0", "0", "2"], 60),
    (["--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "2", "-0.1", "0.1", "3",
      "--obj-init-x-range", "0", "0", "1", "--obj-init-y-range", "0", "0", "1"], 6),
    (["--obj-init-x-range", "-0.08", "-0.02", "3", "--obj-init-y-range", "-0.02", "0.08", "3"], 9),
    (["--obj-episode-range", "0", "5", "--obj-episode-range", "0", "7", "--obj-variation-mode", "episode"], 7),
])
def test_episode_count(argv, n):
    assert generate.simpler_episode_count(argv) == n


@pytest.mark.parametrize("argv,msg", [
    (["0.1"], "before any flag"),
    (["--robot-init"], "not a SimplerEnv main_inference flag"),
    (["--obj-episode-range", "0"], "takes 2 values"),
    (["--obj-variation-mode", "grid"], "xy or episode"),
    (["--obj-variation-mode", "episode", "--obj-episode-range", "5", "5"], "no episodes"),
])
def test_episode_count_errors(argv, msg):
    with pytest.raises(ValueError, match=msg):
        generate.simpler_episode_count(argv)


# --- the shipped yaml ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("vla,task_set,expected", [
    # pick coke can: 5 x 5 grid of one config; move near: episodes 0-19; bridge: 24 episodes each
    ("spatialvla", None, {"google_robot_pick_coke_can": [25], "google_robot_move_near": [20],
                          "widowx_carrot_on_plate": [24], "widowx_put_eggplant_in_basket": [24]}),
    ("spatialvla", "actions", {"google_robot_open_top_drawer": [1] * 9, "google_robot_close_top_drawer": [1] * 9}),
    ("spatialvla", "bridge", {"widowx_spoon_on_towel": [24], "widowx_carrot_on_plate": [24],
                              "widowx_stack_green_on_yellow_block": [24], "widowx_put_eggplant_in_basket": [24]}),
    ("magma", None, {"widowx_carrot_on_plate": [24], "widowx_put_eggplant_in_basket": [24],
                     "google_robot_move_near": [20]}),
    ("magma", "actions", {"google_robot_put_apple_in_top_drawer": [9, 9, 9]}),   # 3 positions x 3 x 3 apple spots
    ("magma", "bridge", {"widowx_carrot_on_plate": [24], "widowx_stack_green_on_yellow_block": [24],
                         "widowx_spoon_on_towel": [24], "widowx_put_eggplant_in_basket": [24]}),
])
def test_shipped_tasks_and_episode_counts(vla, task_set, expected):
    tasks = generate.simpler_tasks(task_sets.apply(_model(vla), task_set, vla), vla)
    assert {t["name"]: t["episodes"] for t in tasks} == expected
    for t in tasks:
        assert t["n_episodes"] == sum(t["episodes"])
        assert t["action_type"] == ("articulated" if task_set == "actions" else "pick-place"), t["name"]
        assert t["policy_setup"] == ("widowx_bridge" if t["name"].startswith("widowx") else "google_robot")


@pytest.mark.parametrize("vla", TARGETS)
def test_pins_are_full_commits(vla):
    m = _model(vla)["model"]
    for key in ("hf_model_revision", "harness_commit", "maniskill_commit"):
        assert re.fullmatch(r"[0-9a-f]{40}", m[key]), key
    assert (ROOT / m["requirements"]).is_file()


def test_requirements_locks_pin_the_simpler_stack():
    for vla in TARGETS:
        lock = (ROOT / _model(vla)["model"]["requirements"]).read_text()
        pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)", lock, re.M))
        # SimplerEnv / ManiSkill2_real2sim needs numpy < 2 and gymnasium < 1; sapien 2.2.2 is its renderer
        assert pins["numpy"] == "1.26.4" and pins["gymnasium"] == "0.29.1" and pins["sapien"] == "2.2.2"
        assert pins["tensorflow"] == "2.15.0"
    spatial = (ROOT / _model("spatialvla")["model"]["requirements"]).read_text()
    assert re.search(r"^transformers==4\.47\.0$", spatial, re.M) and re.search(r"^torch==2\.5\.1$", spatial, re.M)
    magma = (ROOT / _model("magma")["model"]["requirements"]).read_text()
    assert "transformers @ git+https://github.com/jwyang/transformers.git@1708d3ca3608b3c229d7ce6f0351e652436d7d33" in magma
    assert "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/" in magma
    assert re.search(r"^torch==2\.3\.1$", magma, re.M)


def _cfg_with_task(vla="magma", **overrides):
    cfg = copy.deepcopy(_model(vla))
    cfg["tasks"] = [dict(cfg["tasks"][0], **overrides)]
    return cfg


@pytest.mark.parametrize("overrides,msg", [
    (dict(args=["--env-name", "X-v0"]), "set by the template"),
    (dict(args=["--ckpt", "x"]), "set by the template"),            # abbreviation of --ckpt-path
    (dict(variants=[["--logging-dir", "/tmp/x"]]), "set by the template"),
    (dict(args=["--robot", "widowx;rm"]), "unsafe argument tokens"),
    (dict(args=["--robot", "it's"]), "unsafe argument tokens"),
    (dict(args=["--action-ensemble-temp", "-0.8"]), "not a SimplerEnv main_inference flag"),  # magma: upstream argparse
    (dict(policy_setup="franka"), "policy_setup must be one of"),
    (dict(name="a b"), "outside"),
    (dict(env_name=""), "env_name is required"),
    (dict(variants="--robot"), "lists of strings"),
    (dict(action_type="flying"), "action_type"),
])
def test_task_validation_errors(overrides, msg):
    with pytest.raises(ValueError, match=msg):
        generate.simpler_tasks(_cfg_with_task(**overrides), "magma")


def test_duplicate_task_names_are_rejected():
    cfg = copy.deepcopy(_model("magma"))
    cfg["tasks"] = [cfg["tasks"][0], cfg["tasks"][0]]
    with pytest.raises(ValueError, match="duplicate task name"):
        generate.simpler_tasks(cfg, "magma")


def test_spatialvla_may_pass_the_fork_flag():
    tasks = generate.simpler_tasks(_cfg_with_task("spatialvla", args=["--action-ensemble-temp", "-0.8"]), "spatialvla")
    assert tasks[0]["n_episodes"] == 25


# --- render (generate.py CLI) ------------------------------------------------------------------------------

@pytest.mark.skipif(BASH is None, reason="bash not on PATH")
@pytest.mark.parametrize("vla", TARGETS)
@pytest.mark.parametrize("task_set", ["", "actions", "bridge"])
def test_render_pins_and_tasks(vla, task_set, tmp_path):
    extra = ("--task-set", task_set) if task_set else ()
    _render(vla, tmp_path, *extra)
    script = tmp_path / f"{vla}.sh"
    assert subprocess.run([BASH, "-n", str(script)]).returncode == 0
    text = script.read_text()
    m = _model(vla)["model"]
    assert f"git clone {m['harness_repo']}" in text
    assert f'checkout "{m["harness_commit"]}"' in text and f'= "{m["maniskill_commit"]}" ]' in text
    assert f"snapshot_download('{m['hf_repo']}', revision='{m['hf_model_revision']}', local_dir=" in text
    assert f"python {m['entry']} --policy-model {m['policy_model']}" in text
    assert '--policy-setup "${POLICY_SETUP}"' in text and '--env-name "${ENV_NAME}"' in text
    assert 'echo "action_type: ${ACTION_TYPE}"' in text and 'echo "env_name: ${TASK_NAME}"' in text
    for mod in m["import_check"]:
        assert f'"{mod}"' in text
    tasks = generate.simpler_tasks(task_sets.apply(_model(vla), task_set or None, vla), vla)
    blob = re.search(r"TASKS_JSON='(.*?)'\n", text, re.S).group(1)
    assert [t["name"] for t in json.loads(blob)] == [t["name"] for t in tasks]
    assert f"{sum(t['n_episodes'] for t in tasks)} episode(s) in total" in text
    if vla == "magma":
        assert f'checkout "{m["plugin_commit"]}"' in text and m["plugin_src"] in text
        assert "snapshot_download('laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg', revision=" in text
        assert "POLICY_ARGS=()" in text
    else:
        assert "PLUGIN_DIR" not in text
        assert 'POLICY_ARGS=("--action-ensemble-temp" "-0.8" )' in text


@pytest.mark.parametrize("vla", TARGETS)
def test_render_embeds_the_lock_file_byte_for_byte(vla, tmp_path):
    _render(vla, tmp_path)
    text = (tmp_path / f"{vla}.sh").read_text()
    blob = re.search(r"<< 'REQEOF'\n(.*?)REQEOF\n", text, re.S).group(1)
    assert base64.b64decode(blob) == (ROOT / _model(vla)["model"]["requirements"]).read_bytes()


def test_protocol_card_lists_checkpoint_prefetch_and_pins(tmp_path):
    _render("magma", tmp_path)
    text = (tmp_path / "magma.sh").read_text()
    m = _model("magma")["model"]
    assert f"{{repo: $(pc_q '{m['hf_repo']}'), revision: $(pc_q '{m['hf_model_revision']}')}}" in text
    assert f"$(pc_q '{m['prefetch'][0]['repo']}'), revision: $(pc_q '{m['prefetch'][0]['revision']}')" in text
    for key in ("harness_commit", "maniskill_commit", "plugin_commit"):
        assert f"echo \"  {key}: $(pc_q '{m[key]}')\"" in text


def test_generate_rejects_a_bad_model_block(tmp_path):
    cfg = copy.deepcopy(_model("spatialvla"))
    cfg["model"]["harness_commit"] = "main"
    with pytest.raises(SystemExit):
        generate.generate_simpler(cfg, False, "spatialvla")


# --- the task loop, run through bash + jq with a stub main_inference -------------------------------------------

FAKE_PY = r"""#!/bin/bash
# Stub for "python <entry> ...": writes SimplerEnv-style episode videos into --logging-dir.
# FAKE_PER_CALL videos per call; variant directories v<even> succeed, v<odd> fail.
LOGDIR=""; ENV=""
while [ $# -gt 0 ]; do
  case "$1" in --logging-dir) LOGDIR="$2"; shift ;; --env-name) ENV="$2"; shift ;; esac
  shift
done
echo "$*" >> "${FAKE_CALLS}"
VI="${LOGDIR##*/v}"
OUT="${LOGDIR}/dummy_drawer/arm_pd_ee_delta_pose/${ENV}/rob_0.6_0.1_rot_0.000_0.000_0.000_rgb_overlay_x"
mkdir -p "${OUT}"
for i in $(seq 1 "${FAKE_PER_CALL:-1}"); do
  if [ $((VI % 2)) -eq 0 ]; then tag=success; else tag=failure; fi
  : > "${OUT}/${tag}_obj_0.0_0.0_qpos_${i}.mp4"
done
: > "${OUT}/${tag}_obj_0.0_0.0.png"
echo "          Average success 0.5"
"""


def _loop_script(tmp_path, vla="spatialvla", task_set="actions"):
    _render(vla, tmp_path / "r", "--task-set", task_set)
    text = (tmp_path / "r" / f"{vla}.sh").read_text()
    block = text[text.index("# ── [6/7]"):text.index("# ── [7/7]")]
    block = block.replace("/tmp/", f"{tmp_path}/tmp/")
    fake = tmp_path / "fakepy"
    fake.write_text(FAKE_PY)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    awsdir = tmp_path / "bin"
    awsdir.mkdir()
    (awsdir / "aws").write_text("#!/bin/bash\nexit 0\n")
    (awsdir / "aws").chmod(0o755)
    prelude = f"""set -euo pipefail
export PATH="{awsdir}:$PATH"
mkdir -p "{tmp_path}/tmp"
log() {{ echo "[LOG] $*"; }}
die() {{ echo "[DIE] $*"; exit 7; }}
_trap_handler() {{ :; }}
trap '_trap_handler' ERR
PY="{fake}"; HARNESS_DIR="{tmp_path}"; CKPT_DIR="{tmp_path}/ckpt/spatialvla-4b-224-pt"
EVAL_BASE_DIR="{tmp_path}/eval"; VIDEO_BASE_DIR="{tmp_path}/videos"
S3_BUCKET=b; RUN_ID=R; REGION=us-east-1; INSTANCE_ID=i-0; INSTANCE_TYPE=g6.xlarge; STACK_NAME=S
STACK_VERSIONS="torch==2.5.1"; GPU_NAME="NVIDIA L4"; HARNESS_HEAD=h; MANISKILL_HEAD=m
"""
    script = tmp_path / "loop.sh"
    script.write_text(prelude + block)
    return script


def _run_loop(tmp_path, per_call, vla="spatialvla", task_set="actions"):
    script = _loop_script(tmp_path, vla, task_set)
    env = dict(os.environ, FAKE_PER_CALL=str(per_call), FAKE_CALLS=str(tmp_path / "calls.txt"))
    return subprocess.run([BASH, str(script)], capture_output=True, text=True, env=env, timeout=120)


@pytest.mark.skipif(BASH is None or JQ is None, reason="bash / jq not on PATH")
def test_task_loop_counts_outcomes_from_video_names(tmp_path):
    proc = _run_loop(tmp_path, per_call=1)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    assert len(calls) == 18                         # 2 tasks x 9 robot positions, one call each
    summary = (tmp_path / "tmp" / "spatialvla_task0_summary.txt").read_text()
    kv = dict(line.split(": ", 1) for line in summary.splitlines() if ": " in line)
    # variants v0, v2, v4, v6, v8 succeed -> 5 of 9
    assert (kv["n_success"], kv["n_episodes"], kv["success_rate"]) == ("5", "9", "0.556")
    assert kv["env_name"] == "google_robot_open_top_drawer" and kv["simpler_env"] == "OpenTopDrawerCustomInScene-v0"
    assert kv["action_type"] == "articulated" and kv["max_episode_steps"] == "113"
    assert kv["policy_setup"] == "google_robot" and kv["n_episodes_per_task"] == "9"
    vids = sorted(p.name for p in (tmp_path / "videos" / "task-0").iterdir())
    assert len(vids) == 9 and all(re.fullmatch(r"google_robot_open_top_drawer--(success|failure)-\d{3}\.mp4", v)
                                  for v in vids)
    assert sum(v.count("--success-") for v in vids) == 5
    # the policy args, the task args and the variant args reach main_inference in that order
    first = (tmp_path / "tmp" / "spatialvla_task0.log").read_text().splitlines()[0]
    assert "--policy-model spatialvla --ckpt-path " in first
    assert first.index("--action-ensemble-temp -0.8") < first.index("--robot google_robot_static") \
        < first.index("--robot-init-x 0.644")
    assert "--logging-dir " + str(tmp_path / "eval" / "task-0" / "v0") in first


@pytest.mark.skipif(BASH is None or JQ is None, reason="bash / jq not on PATH")
def test_task_loop_fails_when_the_episode_count_differs(tmp_path):
    proc = _run_loop(tmp_path, per_call=2)
    assert proc.returncode == 7
    assert "[DIE] Task 0 wrote 18 episode videos, expected 9" in proc.stdout


@pytest.mark.skipif(BASH is None or JQ is None, reason="bash / jq not on PATH")
def test_task_loop_magma_passes_no_policy_args(tmp_path):
    proc = _run_loop(tmp_path, per_call=9, vla="magma", task_set="actions")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    first = (tmp_path / "tmp" / "magma_task0.log").read_text().splitlines()[0]
    assert "python simpler_env/main_inference_magma.py --policy-model magma --ckpt-path" in first
    assert "--action-ensemble-temp" not in first
    kv = dict(line.split(": ", 1) for line in (tmp_path / "tmp" / "magma_task0_summary.txt").read_text().splitlines()
              if ": " in line)
    assert (kv["n_success"], kv["n_episodes"]) == ("18", "27")  # v0 and v2 succeed (9 each), v1 fails


# --- results.py reads the summary the template writes -------------------------------------------------------

def test_results_reads_task_name_and_action_type():
    s3 = FakeS3({BUCKET: {
        "R/task-0/summary.txt": "model: spatialvla\nsuite: simpler_google_vm\nenv_name: google_robot_open_top_drawer\n"
                                "simpler_env: OpenTopDrawerCustomInScene-v0\naction_type: articulated\n"
                                "success_rate: 0.556\nn_success: 5\nn_episodes: 9\n",
        "R/task-1/summary.txt": "model: magma\nsuite: simpler_bridge\nenv_name: widowx_carrot_on_plate\n"
                                "success_rate: 0.250\nn_success: 6\nn_episodes: 24\n",  # no tag: derived from the name
    }})
    r0 = results.task_row(s3, "us-east-1", BUCKET, "R", "R/task-0/")
    r1 = results.task_row(s3, "us-east-1", BUCKET, "R", "R/task-1/")
    assert (r0["task_name"], r0["action_type"], r0["k"], r0["n"]) == ("google_robot_open_top_drawer", "articulated", 5, 9)
    assert (r1["task_name"], r1["action_type"], r1["suite"]) == ("widowx_carrot_on_plate", "pick-place", "simpler_bridge")
