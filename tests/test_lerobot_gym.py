"""lerobot-gym (ACT on gym-aloha, Diffusion Policy on gym-pusht) and multitask-dit (LBM-style DiT on LIBERO).

Covers the task validation in generate.lerobot_gym_tasks, the rendered UserData (one checkpoint, migration and
lerobot-eval call per task, task sets), the video naming / outcome step run through bash + jq against an
eval_info.json in the format lerobot-eval writes, and the summary.txt fields results.py reads.
"""

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

yaml = pytest.importorskip("yaml")
pytest.importorskip("jinja2")

import generate  # noqa: E402
import results  # noqa: E402
import task_sets  # noqa: E402
from test_results import BUCKET, FakeS3  # noqa: E402

BASH = shutil.which("bash")
JQ = shutil.which("jq")


def _model(vla="lerobot-gym"):
    return yaml.safe_load((ROOT / "models" / f"{vla}.yaml").read_text())


def _render(vla, out_dir, *extra, ok=True):
    proc = subprocess.run([sys.executable, str(ROOT / "generate.py"), "--vla", vla, "--output-dir", str(out_dir),
                           *extra], cwd=ROOT, capture_output=True, text=True, timeout=120)
    if ok:
        assert proc.returncode == 0, f"{vla} {extra}:\n{proc.stdout}\n{proc.stderr}"
    return proc


# --- task validation --------------------------------------------------------------------------------------

def test_default_tasks_validate_with_tags_and_distinct_checkpoint_dirs():
    tasks = generate.lerobot_gym_tasks(_model())
    assert [t["task"] for t in tasks] == ["AlohaInsertion-v0", "AlohaTransferCube-v0", "PushT-v0"]
    assert [t["action_type"] for t in tasks] == ["bimanual", "bimanual", "push"]
    assert [t["env_type"] for t in tasks] == ["aloha", "aloha", "pusht"]
    assert len({t["ckpt_dir"] for t in tasks}) == 3
    assert all("/" not in t["ckpt_dir"] for t in tasks)


def test_yaml_tags_agree_with_the_name_rules():
    """Runs made without the yaml tag (or read back from old summaries) get the same action type."""
    for t in _model()["tasks"]:
        assert task_sets.classify(t["task"]) == t["action_type"]


@pytest.mark.parametrize("patch,msg", [
    ({"env_type": "libero"}, "env_type 'libero' is not one of"),
    ({"hf_model_revision": "main"}, "40-hex"),
    ({"hf_repo": "no-org"}, "is not <org>/<name>"),
    ({"task": "Aloha Insertion"}, "characters outside"),
    ({"policy_type": "act;rm"}, "characters outside"),
    ({"n_episodes": 0}, "positive integer"),
    ({"extra_args": ["--env.task=PushT-v0"]}, "is set by the template"),
    ({"policy_type": None}, "policy_type is required"),
    ({"action_type": "dance"}, "is not one of"),
])
def test_bad_task_entries_are_rejected(patch, msg):
    cfg = _model()
    cfg["tasks"][0].update(patch)
    with pytest.raises(ValueError, match=msg):
        generate.lerobot_gym_tasks(cfg)


def test_no_tasks_is_rejected():
    with pytest.raises(ValueError, match="no tasks"):
        generate.lerobot_gym_tasks({"tasks": []})


@pytest.mark.parametrize("drop", ["aloha", "pusht", "diffusion"])
def test_missing_uv_extra_for_an_env_or_policy_exits(drop, capsys):
    cfg = copy.deepcopy(_model())
    cfg["model"]["uv_extras"] = [e for e in cfg["model"]["uv_extras"] if e != drop]
    with pytest.raises(SystemExit):
        generate.generate_lerobot_gym(cfg, dry_run=True)
    assert f"lacks {drop}" in capsys.readouterr().err


# --- render ----------------------------------------------------------------------------------------------

@pytest.mark.skipif(BASH is None, reason="bash not on PATH")
@pytest.mark.parametrize("task_set,repos", [
    ("", ["act_aloha_sim_insertion_human", "act_aloha_sim_transfer_cube_human", "diffusion_pusht"]),
    ("aloha", ["act_aloha_sim_insertion_human", "act_aloha_sim_transfer_cube_human"]),
    ("pusht", ["diffusion_pusht"]),
])
def test_render_per_task_checkpoints_and_task_sets(task_set, repos, tmp_path):
    extra = ("--task-set", task_set) if task_set else ()
    _render("lerobot-gym", tmp_path, *extra)
    script = tmp_path / "lerobot-gym.sh"
    assert subprocess.run([BASH, "-n", str(script)]).returncode == 0
    text = script.read_text()
    tasks = generate.lerobot_gym_tasks(task_sets.apply(_model(), task_set or None, "lerobot-gym"))
    for t in tasks:
        # pinned download, migration into its own dir, and the task in TASKS_JSON
        assert f"snapshot_download('{t['hf_repo']}', revision='{t['hf_model_revision']}')" in text
        assert f"--pretrained-path {t['hf_repo']} --revision {t['hf_model_revision']}" in text
        assert f'"ckpt_dir": "{t["ckpt_dir"]}"' in text
    every = ["act_aloha_sim_insertion_human", "act_aloha_sim_transfer_cube_human", "diffusion_pusht"]
    for name in every:
        assert (f"lerobot/{name}', revision=" in text) == (name in repos), name
    assert text.count("_migrate.py \\") == len(repos)
    assert '--policy.path="${TASK_CKPT}"' in text and '--env.type="${ENV_TYPE}"' in text
    assert '--env.task="${GYM_TASK}"' in text
    assert 'echo "action_type: ${ACTION_TYPE}"' in text and 'echo "env_name: ${GYM_TASK}"' in text
    assert "uv sync --locked --python 3.12 --extra aloha --extra pusht --extra diffusion" in text
    assert "lerobot.policies.' + m" in text and "gym package OK" in text


def test_render_writes_the_checkpoints_into_the_protocol_card(tmp_path):
    _render("lerobot-gym", tmp_path)
    text = (tmp_path / "lerobot-gym.sh").read_text()
    for t in _model()["tasks"]:
        assert f"{t['hf_repo']}" in text and t["hf_model_revision"] in text
    assert "MIGEOF" in text and "mig.make_policy_config = _make_with_tuples" in text


# --- video naming + outcome (bash + jq, eval_info.json as lerobot-eval writes it) -------------------------

def _video_block(text):
    start = text.index('    VIDEO_DIR="${VIDEO_BASE_DIR}/task-${TASK_IDX}"')
    end = text.index("    VIDEO_COUNT=", start)
    return text[start:end]


@pytest.mark.skipif(BASH is None or JQ is None, reason="bash / jq not on PATH")
def test_videos_carry_the_gym_task_and_the_outcome(tmp_path):
    _render("lerobot-gym", tmp_path / "r")
    block = _video_block((tmp_path / "r" / "lerobot-gym.sh").read_text())
    out_dir = tmp_path / "eval" / "task-0"
    (out_dir / "videos" / "aloha_0").mkdir(parents=True)
    for i in range(3):
        (out_dir / "videos" / "aloha_0" / f"eval_episode_{i}.mp4").write_bytes(b"x")
    # per_task / overall layout of lerobot-eval at the pinned commit (checked with a local Push-T run)
    (out_dir / "eval_info.json").write_text(json.dumps({
        "per_task": [{"task_group": "aloha", "task_id": 0,
                      "metrics": {"sum_rewards": [1, 0, 3], "max_rewards": [4, 0, 4], "successes": [True, False, True],
                                  "video_paths": [], "predicted_video_paths": []}}],
        "per_group": {}, "overall": {"pc_success": 66.7, "n_success": 2, "n_episodes": 3}}))
    marker = tmp_path / "start.marker"
    marker.write_text("")
    import os
    import time
    old = time.time() - 60
    os.utime(marker, (old, old))
    script = (f'set -euo pipefail\nVIDEO_BASE_DIR="{tmp_path}/videos"\nTASK_IDX=0\nOUT_DIR="{out_dir}"\n'
              f'GYM_TASK=AlohaInsertion-v0\n' + block.replace("/tmp/lerobot-gym_task${TASK_IDX}_start.marker",
                                                               str(marker)))
    proc = subprocess.run([BASH, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    names = sorted(p.name for p in (tmp_path / "videos" / "task-0").iterdir())
    assert names == ["AlohaInsertion-v0--eval_episode_0-success.mp4", "AlohaInsertion-v0--eval_episode_1-failure.mp4",
                     "AlohaInsertion-v0--eval_episode_2-success.mp4"]


@pytest.mark.skipif(BASH is None or JQ is None, reason="bash / jq not on PATH")
def test_videos_without_eval_info_keep_a_plain_name(tmp_path):
    _render("lerobot-gym", tmp_path / "r")
    block = _video_block((tmp_path / "r" / "lerobot-gym.sh").read_text())
    out_dir = tmp_path / "eval"
    (out_dir / "videos" / "pusht_0").mkdir(parents=True)
    (out_dir / "videos" / "pusht_0" / "eval_episode_0.mp4").write_bytes(b"x")
    marker = tmp_path / "m"
    marker.write_text("")
    import os
    os.utime(marker, (1, 1))
    script = (f'set -euo pipefail\nVIDEO_BASE_DIR="{tmp_path}/videos"\nTASK_IDX=0\nOUT_DIR="{out_dir}"\n'
              f'GYM_TASK=PushT-v0\n' + block.replace("/tmp/lerobot-gym_task${TASK_IDX}_start.marker", str(marker)))
    proc = subprocess.run([BASH, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert [p.name for p in (tmp_path / "videos" / "task-0").iterdir()] == ["PushT-v0--eval_episode_0.mp4"]


# --- results.py reads the summary the template writes ------------------------------------------------------

def test_results_reads_gym_task_and_action_type():
    s3 = FakeS3({BUCKET: {
        "R/task-0/summary.txt": "model: lerobot-gym\npolicy_type: act\nsuite: aloha\nenv_name: AlohaInsertion-v0\n"
                                "action_type: bimanual\nsuccess_rate: 0.600\nn_success: 6\nn_episodes: 10\n",
        "R/task-1/summary.txt": "model: lerobot-gym\nsuite: pusht\nenv_name: PushT-v0\nsuccess_rate: 0.700\n"
                                "n_success: 7\nn_episodes: 10\n",  # no tag line: derived from the gym id
    }})
    r0 = results.task_row(s3, "us-east-1", BUCKET, "R", "R/task-0/")
    r1 = results.task_row(s3, "us-east-1", BUCKET, "R", "R/task-1/")
    assert (r0["task_name"], r0["action_type"], r0["k"], r0["n"]) == ("AlohaInsertion-v0", "bimanual", 6, 10)
    assert (r1["task_name"], r1["action_type"], r1["suite"]) == ("PushT-v0", "push", "pusht")


# --- multitask-dit (LBM-style DiT, shared LeRobot LIBERO template) -----------------------------------------

@pytest.mark.skipif(BASH is None, reason="bash not on PATH")
def test_multitask_dit_renders_on_the_libero_template(tmp_path):
    assert "multitask-dit" in generate.LEROBOT_LIBERO_TARGETS
    _render("multitask-dit", tmp_path)
    text = (tmp_path / "multitask-dit.sh").read_text()
    assert subprocess.run([BASH, "-n", str(tmp_path / "multitask-dit.sh")]).returncode == 0
    m = _model("multitask-dit")["model"]
    assert f"snapshot_download('{m['hf_repo']}', revision='{m['hf_model_revision']}')" in text
    assert "importlib.import_module('lerobot.policies.multi_task_dit')" in text
    assert "'--env.observation_height=256'" in text and "'--env.observation_width=256'" in text
    assert "--extra multi_task_dit --extra libero" in text
    assert "not TRI weights" in m["policy_desc"]


@pytest.mark.skipif(BASH is None, reason="bash not on PATH")
def test_import_check_reaches_python_with_quoted_module_names(tmp_path):
    # Regression: the check used to be python -c "..." with JSON lists inside; bash stripped the double quotes
    # and the instance failed with NameError: name 'act' is not defined.
    _render("lerobot-gym", tmp_path / "r")
    text = (tmp_path / "r" / "lerobot-gym.sh").read_text()
    start = text.index("uv run --active python - <<'PYCHECK'")
    end = text.index("\nPYCHECK\n", start) + len("\nPYCHECK\n")
    captured = tmp_path / "check.py"
    script = f'set -euo pipefail\nuv() {{ cat > "{captured}"; }}\ntee() {{ cat >/dev/null; }}\ndie() {{ exit 9; }}\n'
    proc = subprocess.run([BASH, "-c", script + text[start:end]], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    import ast
    loops = [n for n in ast.walk(ast.parse(captured.read_text())) if isinstance(n, ast.For)]
    names = [[e.value for e in loop.iter.elts] for loop in loops]
    policies = sorted({t["policy_type"] for t in _model()["tasks"]})
    assert names[0] == policies and all(isinstance(v, str) for v in names[0] + names[1]) and names[1]
