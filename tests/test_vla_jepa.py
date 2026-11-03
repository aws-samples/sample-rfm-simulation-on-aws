"""vla-jepa (VLA-JEPA LeRobot port on LIBERO, shared lerobot-libero template).

Covers the per-task entries (generate.lerobot_libero_named_tasks), the pinned backbone downloads
(generate.lerobot_extra_downloads), the rendered UserData for every task set, and two bash steps of the template
run with stubs: the backbone download block (local paths -> --policy.* args) and the instance-side task-name check.
"""

import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

yaml = pytest.importorskip("yaml")
pytest.importorskip("jinja2")

import generate  # noqa: E402
import task_sets  # noqa: E402

BASH = shutil.which("bash")
JQ = shutil.which("jq")
VLA = "vla-jepa"


def _model():
    return yaml.safe_load((ROOT / "models" / f"{VLA}.yaml").read_text())


def _render(out_dir, *extra, ok=True):
    proc = subprocess.run([sys.executable, str(ROOT / "generate.py"), "--vla", VLA, "--output-dir", str(out_dir),
                           *extra], cwd=ROOT, capture_output=True, text=True, timeout=120)
    if ok:
        assert proc.returncode == 0, f"{extra}:\n{proc.stdout}\n{proc.stderr}"
    return proc


# --- yaml --------------------------------------------------------------------------------------------------

def test_is_a_lerobot_libero_target_with_pinned_versions():
    assert VLA in generate.LEROBOT_LIBERO_TARGETS
    m = _model()["model"]
    assert m["hf_repo"] == "lerobot/VLA-JEPA-LIBERO"
    assert re.fullmatch(r"[0-9a-f]{40}", m["hf_model_revision"])
    assert re.fullmatch(r"[0-9a-f]{40}", m["lerobot_commit"])
    assert m["policy_module"] == "vla_jepa" and m["uv_extras"] == ["vla_jepa", "libero"]
    assert {d["policy_arg"] for d in m["extra_downloads"]} == {"qwen_model_name", "jepa_encoder_name"}
    assert all(re.fullmatch(r"[0-9a-f]{40}", d["revision"]) for d in m["extra_downloads"])


def test_default_and_actions_tasks_match_univla():
    cfg = _model()
    univla = yaml.safe_load((ROOT / "models" / "univla.yaml").read_text())

    def ids(tasks):
        return [(t["suite"], t["task_ids"][0], t["env_name"], t["n_episodes"]) for t in tasks]

    def ids_own(tasks):
        return [(t["suite"], t["task_id"], t["env_name"], t["n_episodes"]) for t in tasks]

    assert ids(cfg["tasks"]) == ids_own(univla["tasks"])
    assert ids(cfg["task_sets"]["actions"]["tasks"]) == ids_own(univla["task_sets"]["actions"]["tasks"])
    tags = task_sets.tags_of(task_sets.apply(cfg, "actions", VLA)["tasks"])
    assert tags == ["articulated", "push", "articulated"]
    probe = cfg["task_sets"]["probe"]["tasks"]
    assert len(probe) == 1 and probe[0]["n_episodes"] == 2


def test_every_task_names_its_libero_task_and_has_a_published_number():
    cfg = _model()
    every = cfg["tasks"] + [t for s in cfg["task_sets"].values() for t in s["tasks"]]
    out = generate.lerobot_libero_named_tasks(every, VLA)
    assert len(out) == len(every) and all(t["action_type"] in task_sets.ACTION_TYPES for t in out)
    assert all(t["published"] and t["seed"] == 7 for t in every)
    assert {t["suite"] for t in every} == set(generate.LIBERO_TASKS)


# --- task validation ---------------------------------------------------------------------------------------

def test_named_tasks_pass_entries_without_env_name_unchanged():
    t = {"suite": "libero_spatial", "task_ids": [0, 1], "n_episodes": 5}
    assert generate.lerobot_libero_named_tasks([t], "smolvla") == [t]


def test_named_tasks_tag_a_named_entry():
    out = generate.lerobot_libero_named_tasks(
        [{"suite": "libero_goal", "task_ids": [7], "env_name": "turn_on_the_stove", "n_episodes": 3}], VLA)
    assert out[0]["action_type"] == "articulated"


@pytest.mark.parametrize("entry,msg", [
    ({"suite": "libero_goal", "task_ids": [1], "env_name": "turn_on_the_stove"}, "is not task 1"),
    ({"suite": "libero_goal", "task_ids": [1, 6], "env_name": "put_the_bowl_on_the_stove"}, "one id"),
    ({"suite": "libero_goal", "task_ids": [10], "env_name": "x"}, "one id 0-9"),
    ({"suite": "libero_goal", "task_ids": [True], "env_name": "x"}, "one id"),
    ({"suite": "libero_90", "task_ids": [0], "env_name": "x"}, "suite must be one of"),
])
def test_named_tasks_reject(entry, msg):
    with pytest.raises(ValueError, match=msg):
        generate.lerobot_libero_named_tasks([entry], VLA)


# --- backbone downloads ------------------------------------------------------------------------------------

def test_extra_downloads_parse():
    out = generate.lerobot_extra_downloads(_model()["model"], VLA)
    assert [d["policy_arg"] for d in out] == ["qwen_model_name", "jepa_encoder_name"]
    assert out[0]["allow_patterns_py"] == ""
    assert ast.literal_eval(out[1]["allow_patterns_py"]) == ["config.json", "model.safetensors",
                                                             "video_preprocessor_config.json"]
    assert generate.lerobot_extra_downloads({}, "smolvla") == []


GOOD = {"repo": "Qwen/Qwen3-VL-2B-Instruct", "revision": "a" * 40, "policy_arg": "qwen_model_name"}


@pytest.mark.parametrize("bad,msg", [
    ({"revision": "main"}, "40-hex"),
    ({"repo": "no-org"}, "<org>/<name>"),
    ({"repo": "a/b'; rm -rf /"}, "<org>/<name>"),
    ({"policy_arg": "qwen-model"}, "policy config field"),
    ({"allow_patterns": ["a'b"]}, "plain glob"),
    ({"allow_patterns": "config.json"}, "plain glob"),
])
def test_extra_downloads_reject(bad, msg):
    with pytest.raises(ValueError, match=msg):
        generate.lerobot_extra_downloads({"extra_downloads": [dict(GOOD, **bad)]}, VLA)


def test_extra_downloads_reject_two_downloads_for_one_field():
    with pytest.raises(ValueError, match="one download per policy_arg"):
        generate.lerobot_extra_downloads({"extra_downloads": [GOOD, dict(GOOD, repo="x/y")]}, VLA)


# --- render ------------------------------------------------------------------------------------------------

@pytest.mark.skipif(BASH is None, reason="bash not on PATH")
@pytest.mark.parametrize("task_set", ["", "actions", "probe", "spatial", "object", "long"])
def test_renders_every_task_set(task_set, tmp_path):
    proc = _render(tmp_path, *(["--task-set", task_set] if task_set else []))
    text = (tmp_path / f"{VLA}.sh").read_text()
    assert subprocess.run([BASH, "-n", str(tmp_path / f"{VLA}.sh")]).returncode == 0
    cfg = task_sets.apply(_model(), task_set, VLA)
    for t in cfg["tasks"]:
        assert f'"env_name": "{t["env_name"]}"' in text
    if task_set:
        assert "Tasks:  " + task_set + ":" in proc.stdout


@pytest.mark.skipif(BASH is None, reason="bash not on PATH")
def test_render_pins_checkpoint_backbones_and_eval_settings(tmp_path):
    _render(tmp_path)
    text = (tmp_path / f"{VLA}.sh").read_text()
    m = _model()["model"]
    assert f"snapshot_download('{m['hf_repo']}', revision='{m['hf_model_revision']}')" in text
    assert "--extra vla_jepa --extra libero" in text
    assert "importlib.import_module('lerobot.policies.vla_jepa')" in text
    assert "'--env.observation_height=256'" in text and "'--env.observation_width=256'" in text
    for d in m["extra_downloads"]:
        assert f"snapshot_download('{d['repo']}', revision='{d['revision']}'" in text
        assert f'--policy.{d["policy_arg"]}="${{_EXTRA_PATH}}"' in text
        assert f'echo "stack_{d["policy_arg"]}: {d["repo"]}@{d["revision"]}"' in text
    assert '"${EXTRA_POLICY_ARGS[@]}"' in text
    assert 'echo "action_type: ${ACTION_TYPE}"' in text and 'echo "env_name: ${ENV_NAME}"' in text
    assert 'echo "published: ${PUBLISHED}"' in text


@pytest.mark.skipif(BASH is None, reason="bash not on PATH")
def test_other_lerobot_targets_get_no_backbone_block(tmp_path):
    proc = subprocess.run([sys.executable, str(ROOT / "generate.py"), "--vla", "smolvla", "--output-dir",
                           str(tmp_path)], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    text = (tmp_path / "smolvla.sh").read_text()
    assert "EXTRA_POLICY_ARGS" not in text
    assert subprocess.run([BASH, "-n", str(tmp_path / "smolvla.sh")]).returncode == 0


def test_generate_rejects_a_wrong_env_name():
    cfg = _model()
    cfg["tasks"][0]["env_name"] = "turn_on_the_stove"
    with pytest.raises(SystemExit):
        generate.generate_lerobot_libero(cfg, True, vla=VLA)


# --- bash steps with stubs ---------------------------------------------------------------------------------

def _block(text, start_marker, end_marker):
    start = text.index(start_marker)
    end = text.index(end_marker, start) + len(end_marker)
    return text[start:end]


@pytest.mark.skipif(BASH is None, reason="bash not on PATH")
def test_backbone_block_turns_snapshots_into_policy_args(tmp_path):
    _render(tmp_path / "r")
    text = (tmp_path / "r" / f"{VLA}.sh").read_text()
    block = _block(text, "EXTRA_POLICY_ARGS=()", 'log "  backbone args: ${EXTRA_POLICY_ARGS[*]}"')
    block = block.replace("/tmp/vla-jepa_extra", str(tmp_path / "extra"))
    codes = tmp_path / "codes"
    # uv stub: keep the python code, write a snapshot dir named after the repo into the .path file it names.
    stub = (
        'set -euo pipefail\n'
        'log() { :; }\ndie() { echo "DIE $*"; exit 9; }\n'
        'hf_dl_guard() { shift 2; [ "$1" = "--" ] && shift; "$@"; }\n'
        f'uv() {{ local code="${{@: -1}}"; printf "%s\\n#---\\n" "$code" >> "{codes}"; '
        'local f; f=$(printf "%s" "$code" | grep -o "open(.[^,]*" | sed "s/open(.//;s/.$//"); '
        'local repo; repo=$(printf "%s" "$code" | grep -o "snapshot_download(.[^,]*" | sed "s/snapshot_download(.//;s/.$//"); '
        f'mkdir -p "{tmp_path}/snap/$repo"; printf "%s" "{tmp_path}/snap/$repo" > "$f"; }}\n'
    )
    proc = subprocess.run([BASH, "-c", stub + block + '\nprintf "%s\\n" "${EXTRA_POLICY_ARGS[@]}"'],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.split() == [
        f"--policy.qwen_model_name={tmp_path}/snap/Qwen/Qwen3-VL-2B-Instruct",
        f"--policy.jepa_encoder_name={tmp_path}/snap/facebook/vjepa2-vitl-fpc64-256",
    ]
    for code in codes.read_text().split("#---\n")[:-1]:
        ast.parse(code)  # the python bash passes on is valid (quotes survived the double-quoted -c string)


@pytest.mark.skipif(BASH is None or JQ is None, reason="bash / jq not on PATH")
@pytest.mark.parametrize("libero_says,rc", [("put_the_bowl_on_the_stove", 0), ("turn_on_the_stove", 9)])
def test_instance_checks_the_libero_task_name(tmp_path, libero_says, rc):
    _render(tmp_path / "r")
    text = (tmp_path / "r" / f"{VLA}.sh").read_text()
    block = _block(text, '    if [ -n "${ENV_NAME}" ]; then', '        log "  env_name: ${ENV_NAME} (action_type ${ACTION_TYPE})"\n    fi\n')
    py = tmp_path / "lerobot" / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text(f"#!/bin/sh\necho 'LIBERO banner'\necho '{libero_says}'\n")
    os.chmod(py, 0o755)
    task = '{"suite": "libero_goal", "task_ids": [1], "env_name": "put_the_bowl_on_the_stove"}'
    head = (f'set -euo pipefail\nlog() {{ :; }}\ndie() {{ echo "DIE $*"; exit 9; }}\nLEROBOT_DIR="{tmp_path}/lerobot"\n'
            f"TASK='{task}'\nTASK_IDX=0\nSUITE=libero_goal\nTASK_IDS_ARR='[1]'\n"
            "ENV_NAME=put_the_bowl_on_the_stove\nACTION_TYPE=pick-place\n")
    proc = subprocess.run([BASH, "-c", head + block], capture_output=True, text=True, timeout=30)
    assert proc.returncode == rc, proc.stdout + proc.stderr
    if rc:
        assert "is 'turn_on_the_stove'" in proc.stdout
