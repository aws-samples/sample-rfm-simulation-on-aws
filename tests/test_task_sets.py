"""Task sets (`--task-set NAME`, models/<vla>.yaml task_sets) and action-type tags (task_sets.py).

Covers the selection logic, the yaml sets shipped in the repo, the render of every target that has one
(generate.py CLI + bash -n), the action_type line each touched template writes into summary.txt, the
task name in the GR00T video file names, and the results.py side (summary tag, derived tag for old runs,
--action-type filter, per-task table).
"""

import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import robotwin_plan  # noqa: E402
import task_sets  # noqa: E402
from targets import VLA_TARGETS  # noqa: E402

yaml = pytest.importorskip("yaml")
pytest.importorskip("jinja2")

BASH = shutil.which("bash")


def _model(vla):
    return yaml.safe_load((ROOT / "models" / f"{vla}.yaml").read_text())


SET_TARGETS = sorted(v for v in VLA_TARGETS if _model(v).get("task_sets"))


# --- classify -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,tag", [
    ("libero_sim/open_the_middle_drawer_of_the_cabinet", "articulated"),
    ("libero_sim/turn_on_the_stove", "articulated"),
    ("libero_sim/push_the_plate_to_the_front_of_the_stove", "push"),
    ("libero_sim/put_the_bowl_on_the_stove", "pick-place"),
    # a drawer named only as a place is not an articulated task
    ("libero_sim/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate", "pick-place"),
    # compound with a jointed sub-goal -> articulated
    ("libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it", "articulated"),
    ("libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it", "articulated"),
    ("gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env", "articulated"),
    ("robocasa_panda_omron/OpenSingleDoor_PandaOmron_Env", "articulated"),
    ("robocasa_panda_omron/TurnOnSinkFaucet_PandaOmron_Env", "articulated"),
    ("robocasa_panda_omron/CoffeePressButton_PandaOmron_Env", "press"),
    ("robocasa_panda_omron/TurnOnMicrowave_PandaOmron_Env", "press"),
    ("simpler_env_google/google_robot_close_drawer", "articulated"),
    ("simpler_env_google/google_robot_pick_coke_can", "pick-place"),
    ("simpler_env_google/google_robot_move_near", "pick-place"),
    ("simpler_env_widowx/widowx_carrot_on_plate", "pick-place"),
    ("handover_block", "bimanual"),
    ("lift_pot", "bimanual"),
    ("click_bell", "press"),
    ("turn_switch", "press"),
    ("beat_block_hammer", "tool-use"),
    ("stamp_seal", "tool-use"),
    ("LMPnPAppleToPlateDC_G1_gear_wbc", "locomotion"),
    ("", "other"),
    ("libero_spatial", "other"),
])
def test_classify(name, tag):
    assert task_sets.classify(name) == tag
    assert tag in task_sets.ACTION_TYPES


def test_known_table_only_uses_defined_tags():
    assert set(task_sets.KNOWN.values()) <= set(task_sets.ACTION_TYPES)
    assert all(k == k.lower() for k in task_sets.KNOWN)


def test_action_type_prefers_the_yaml_tag_and_rejects_unknown_tags():
    assert task_sets.action_type({"env_name": "libero_sim/put_the_bowl_on_the_stove", "action_type": "press"}) == "press"
    assert task_sets.action_type({"env_name": "libero_sim/put_the_bowl_on_the_stove"}) == "pick-place"
    with pytest.raises(ValueError, match="not one of"):
        task_sets.action_type({"env_name": "x", "action_type": "juggling"})


def test_annotate_lists_and_suite_dicts():
    out = task_sets.annotate([{"env_name": "libero_sim/turn_on_the_stove"}])
    assert out == [{"env_name": "libero_sim/turn_on_the_stove", "action_type": "articulated"}]
    by_suite = task_sets.annotate({"goal": [{"task": "click_bell"}]})
    assert by_suite["goal"][0]["action_type"] == "press"


# --- apply -----------------------------------------------------------------------------------------------

BASE = {
    "model": {"id": "t", "hf_subfolder": "libero_10", "hf_repo": "r"},
    "tasks": [{"env_name": "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"}],
    "task_sets": {"actions": {"description": "d", "model": {"hf_subfolder": "libero_goal"},
                              "tasks": [{"env_name": "libero_sim/turn_on_the_stove", "action_type": "articulated"}]}},
}


def test_apply_replaces_tasks_and_merges_model_overrides_without_touching_the_input():
    out = task_sets.apply(BASE, "actions", "t")
    assert out["tasks"] == BASE["task_sets"]["actions"]["tasks"]
    assert out["model"] == {"id": "t", "hf_subfolder": "libero_goal", "hf_repo": "r"}
    assert out["task_set"] == "actions"
    assert BASE["model"]["hf_subfolder"] == "libero_10" and len(BASE["tasks"]) == 1
    out["tasks"][0]["env_name"] = "changed"
    assert BASE["task_sets"]["actions"]["tasks"][0]["env_name"] == "libero_sim/turn_on_the_stove"


def test_apply_without_a_name_is_the_default_run():
    out = task_sets.apply(BASE, "", "t")
    assert out["tasks"] == BASE["tasks"] and out["task_set"] == ""
    assert task_sets.describe(out) == ""


@pytest.mark.parametrize("cfg,name,msg", [
    ({"tasks": []}, "actions", "defines no task_sets"),
    (BASE, "nope", "unknown set"),
    ({"task_sets": {"a": {"tasks": []}}}, "a", "non-empty list"),
    ({"task_sets": {"a": {"tasks": ["x"]}}}, "a", "non-empty list"),
    ({"task_sets": {"a": {"tasks": [{"task": "x", "action_type": "bad"}]}}}, "a", "not one of"),
    ({"task_sets": {"a": {"tasks": [{"task": "x"}], "model": {"k": {"nested": 1}}}}}, "a", "plain values"),
])
def test_apply_errors(cfg, name, msg):
    with pytest.raises(ValueError, match=msg):
        task_sets.apply(cfg, name, "t")


def test_describe_counts_robotwin_tasks_one_by_one():
    cfg = task_sets.apply({"task_sets": {"actions": {"tasks": [
        {"setting": "clean", "robotwin_tasks": ["click_bell", "lift_pot"], "n_episodes": 1}]}}}, "actions")
    assert task_sets.describe(cfg) == "actions: 2 task(s), action types bimanual, press"


# --- the sets shipped in models/*.yaml -------------------------------------------------------------------

def test_the_four_simulators_have_an_actions_set():
    assert {"gr00t", "rldx-kitchen", "rldx-simpler", "smolvla-robotwin", "lingbot-vla2"} <= set(SET_TARGETS)


@pytest.mark.parametrize("vla", SET_TARGETS)
def test_shipped_sets_apply_and_actions_has_no_pick_and_place(vla):
    cfg = _model(vla)
    for name in task_sets.names(cfg):
        out = task_sets.apply(cfg, name, vla)
        tags = task_sets.tags_of(out["tasks"])
        assert tags and set(tags) <= set(task_sets.ACTION_TYPES)
        if name == "actions":
            assert "pick-place" not in tags and "other" not in tags, (vla, tags)
        # the yaml tag (where written) agrees with the name rules, so old runs get the same tag in results
        for t in out["tasks"]:
            if t.get("action_type"):
                assert task_sets.classify(task_sets.task_name(t)) == t["action_type"], (vla, t)


@pytest.mark.parametrize("vla", [v for v in SET_TARGETS if v in ("smolvla-robotwin", "lingbot-vla2")])
def test_robotwin_sets_are_valid_plans(vla):
    out = task_sets.apply(_model(vla), "actions", vla)
    units = robotwin_plan.plan(out["tasks"])
    assert [u["task"] for u in units] and all(u["task"] in robotwin_plan.TASKS for u in units)


# --- render (generate.py CLI) ----------------------------------------------------------------------------

def _render(vla, out_dir, *extra, ok=True):
    proc = subprocess.run([sys.executable, str(ROOT / "generate.py"), "--vla", vla, "--output-dir", str(out_dir),
                           *extra], cwd=ROOT, capture_output=True, text=True, timeout=120)
    if ok:
        assert proc.returncode == 0, f"{vla} {extra}:\n{proc.stdout}\n{proc.stderr}"
    return proc


@pytest.mark.skipif(BASH is None, reason="bash not on PATH")
@pytest.mark.parametrize("vla", SET_TARGETS)
def test_task_set_renders_and_writes_action_type(vla, tmp_path):
    proc = _render(vla, tmp_path, "--task-set", "actions")
    assert "Tasks:  actions:" in proc.stdout
    text = (tmp_path / f"{vla}.sh").read_text()
    assert subprocess.run([BASH, "-n", str(tmp_path / f"{vla}.sh")]).returncode == 0
    assert 'echo "action_type: ${ACTION_TYPE}"' in text
    assert '"action_type": ' in text  # in TASKS_JSON
    cfg = task_sets.apply(_model(vla), "actions", vla)
    for t in cfg["tasks"]:
        for name in t.get("robotwin_tasks") or [t.get("env_name")]:
            assert name in text, (vla, name)


def test_gr00t_actions_uses_the_libero_goal_checkpoint(tmp_path):
    _render("gr00t", tmp_path, "--task-set", "actions")
    text = (tmp_path / "gr00t.sh").read_text()
    assert "NEEDED = 'libero_goal'" in text and 'HF_MODEL_PATH="${HF_REPO_PATH}/libero_goal"' in text
    _render("gr00t", tmp_path)
    text = (tmp_path / "gr00t.sh").read_text()
    assert "NEEDED = 'libero_10'" in text and 'HF_MODEL_PATH="${HF_REPO_PATH}/libero_10"' in text


def test_gr00t_videos_carry_the_task_name(tmp_path):
    _render("gr00t", tmp_path)
    text = (tmp_path / "gr00t.sh").read_text()
    assert 'TASK_SLUG=$(echo "${ENV_NAME##*/}"' in text
    assert '"${VIDEO_DIR}/${TASK_SLUG}--$(basename "${_mp4}")"' in text


@pytest.mark.skipif(BASH is None, reason="bash not on PATH")
def test_gr00t_task_slug_is_a_safe_file_name():
    snippet = ('ENV_NAME="libero_sim/turn_on_the_stove"; '
               "TASK_SLUG=$(echo \"${ENV_NAME##*/}\" | tr -c 'A-Za-z0-9_.\\n-' '_'); printf '%s' \"$TASK_SLUG\"")
    out = subprocess.run([BASH, "-c", snippet], capture_output=True, text=True)
    assert out.stdout == "turn_on_the_stove"
    out = subprocess.run([BASH, "-c", snippet.replace("turn_on_the_stove", "a b'c")], capture_output=True, text=True)
    assert out.stdout == "a_b_c"


@pytest.mark.parametrize("vla", ["gr00t", "rldx-kitchen", "smolvla-robotwin"])
def test_default_run_also_writes_action_type(vla, tmp_path):
    _render(vla, tmp_path)
    assert 'echo "action_type: ${ACTION_TYPE}"' in (tmp_path / f"{vla}.sh").read_text()


@pytest.mark.parametrize("extra,msg", [
    (("--task-set", "nope"), "unknown set"),
    (("--task-set", "actions", "--libero-variant", "plus"), "cannot be combined"),
])
def test_generate_rejects_bad_task_set_flags(extra, msg, tmp_path):
    proc = _render("gr00t", tmp_path, *extra, ok=False)
    assert proc.returncode == 2 and msg in proc.stderr


def test_generate_rejects_a_target_without_sets(tmp_path):
    proc = _render("pi", tmp_path, "--task-set", "actions", ok=False)
    assert proc.returncode == 2 and "defines no task_sets" in proc.stderr


def test_robotwin_task_set_cannot_replace_the_official_protocol(tmp_path):
    proc = _render("smolvla-robotwin", tmp_path, "--task-set", "actions", "--robotwin-protocol", "clean", ok=False)
    assert proc.returncode == 2 and "cannot be combined" in proc.stderr


def test_deploy_forwards_task_set_to_generate():
    src = (ROOT / "deploy.py").read_text()
    assert 'parser.add_argument("--task-set"' in src
    assert 'collect_args += ["--task-set", args.task_set]' in src
    assert "task_sets.apply(config, args.task_set, args.vla)" in src


# --- results.py ------------------------------------------------------------------------------------------

from test_results import BUCKET, FakeS3  # noqa: E402

import results  # noqa: E402


def test_task_row_reads_the_tag_and_derives_it_for_old_runs():
    s3 = FakeS3({BUCKET: {
        "R/task-0/summary.txt": "env_name: libero_sim/turn_on_the_stove\naction_type: articulated\n"
                                "success_rate: 1.0\nn_episodes: 5\nn_success: 5\n",
        "R/task-1/summary.txt": "suite: RoboTwin2/clean\nrobotwin_task: click_bell\nsuccess_rate: 0.6\n"
                                "n_episodes: 5\nn_success: 3\n",
        "R/task-2/summary.txt": "env_name: x\naction_type: not-a-tag\nsuccess_rate: 0\nn_episodes: 1\n",
    }})
    r0 = results.task_row(s3, "us-east-1", BUCKET, "R", "R/task-0/")
    r1 = results.task_row(s3, "us-east-1", BUCKET, "R", "R/task-1/")
    r2 = results.task_row(s3, "us-east-1", BUCKET, "R", "R/task-2/")
    assert (r0["action_type"], r0["task_name"]) == ("articulated", "libero_sim/turn_on_the_stove")
    assert (r1["action_type"], r1["task_name"]) == ("press", "click_bell")
    assert r2["action_type"] == "other"  # an unknown tag in summary.txt falls back to the name rules


def _rows():
    base = dict(bucket_stack="s", region="us-east-1", run="R", suite="RoboTwin2/clean", condition="subset",
                status="done", stack={})
    return [dict(base, task="task-0", task_name="click_bell", action_type="press", k=3, n=5, rate=0.6),
            dict(base, task="task-1", task_name="lift_pot", action_type="bimanual", k=2, n=5, rate=0.4),
            dict(base, task="task-2", task_name="stamp_seal", action_type="tool-use", status="running/failed")]


def test_per_task_table_sorts_by_action_type():
    table = results.per_task_table(_rows())
    assert [t["action_type"] for t in table] == ["bimanual", "press", "tool-use"]
    assert table[1]["k_n"] == "3/5" and table[2]["k_n"] == "-" and table[2]["rate"] == ""


def test_main_filters_by_action_type_and_writes_the_tags(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "boto3", types.ModuleType("boto3"))
    monkeypatch.setattr(results, "collect", lambda a, boto3: _rows())
    out_json = tmp_path / "r.json"
    assert results.main(["--action-type", "press", "--per-task", "--json", str(out_json), "--csv",
                         str(tmp_path / "r.csv")]) == 0
    text = capsys.readouterr().out
    assert "click_bell" in text and "lift_pot" not in text
    data = json.loads(out_json.read_text())
    assert [r["action_type"] for r in data["rows"]] == ["press"]
    assert data["table"][0]["action_types"] == "press"
    assert "action_types" in (tmp_path / "r.csv").read_text().splitlines()[0]
    assert results.main(["--action-type", "locomotion"]) == 0
    assert "No task with action type locomotion" in capsys.readouterr().out
