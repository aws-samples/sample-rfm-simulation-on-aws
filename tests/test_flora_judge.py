"""Offline tests for the Flora-Rewarder post-run judge (assets/flora-judge/flora_judge.py).

Covers what runs on the judge instance around NVIDIA's own scoring script: mapping a run's videos
to episodes (instruction + simulator success), parsing the terminal_reward output, the agreement
figures, the judge.json schema, the caption block for the gallery, and the prepare/report CLI.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("flora_judge", ROOT / "assets" / "flora-judge" / "flora_judge.py")
fj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fj)

# A small excerpt in the shape of LIBERO's libero_suite_task_map.py (first two tasks per suite).
TASK_MAP = '''libero_task_map = {
    "libero_spatial": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    ],
    "libero_10": [
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    ],
    "libero_90": ["KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet"],
}
'''


def _eval_info(groups):
    """LeRobot eval_info.json with per_task entries: groups = [(suite, task_id, [successes])]."""
    per_task = []
    for suite, tid, succ in groups:
        paths = [f"/tmp/x-eval/task-0/videos/{suite}_{tid}/eval_episode_{i}.mp4" for i in range(len(succ))]
        per_task.append({"task_group": suite, "task_id": tid,
                         "metrics": {"successes": succ, "video_paths": paths, "sum_rewards": [0] * len(succ)}})
    return {"per_task": per_task, "overall": {"pc_success": 50.0}}


# --- instructions -----------------------------------------------------------------------------------

def test_libero_languages_follow_libero_rule():
    langs = fj.libero_languages(TASK_MAP)
    assert langs["libero_spatial"][1] == "pick up the black bowl next to the ramekin and place it on the plate"
    assert langs["libero_10"][0] == "put both the alphabet soup and the tomato sauce in the basket"
    assert langs["libero_10"][1] == "turn on the stove and put the moka pot on it"
    assert langs["libero_90"][0] == "close the top drawer of the cabinet"  # SCENE10: one more digit


def test_libero_languages_never_executes_the_file():
    with pytest.raises(ValueError):
        fj.libero_languages("import os\nos.system('echo no')\n")


@pytest.mark.parametrize("text,expected", [
    ("Task:       open the drawer\n", "open the drawer"),
    ('Task:       Reach — "reach the red cube"\n', "reach the red cube"),
    ("Task:       gr1_unified/PnPCupToDrawerClose_GR1Arms\n", "PnPCupToDrawerClose GR1Arms"),
    ("Suite: libero_spatial\n", None),
    ("", None),
])
def test_sidecar_task(text, expected):
    assert fj.sidecar_task(text) == expected


# --- episode mapping --------------------------------------------------------------------------------

def test_openvla_style_name_gives_success_and_task():
    name = "2026_10_20-01_02_03--openvla_oft--episode=3--success=False--task=put_the_bowl_on_the_stove.mp4"
    info = fj.episode_info(name)
    assert info["sim_success"] is False and info["sim_success_source"] == "filename"
    assert info["instruction"] == "put the bowl on the stove" and info["instruction_source"] == "filename"


def test_cut_task_name_is_completed_from_the_task_map():
    langs = fj.libero_languages(TASK_MAP)
    name = "2026_10_17-03_06_03--episode=1--success=True--task=pick_up_the_black_bowl_between_the_plate.mp4"
    info = fj.episode_info(name, "", None, langs)
    assert info["instruction"] == "pick up the black bowl between the plate and the ramekin and place it on the plate"
    assert info["instruction_source"] == "filename+libero_task_map"
    # ambiguous prefix (both spatial tasks start with it) or no task map: keep the text as it is
    assert fj.complete_instruction("pick up the black bowl", langs) == "pick up the black bowl"
    assert fj.episode_info(name)["instruction"] == "pick up the black bowl between the plate"


def test_sidecar_task_and_result_win_over_nothing():
    info = fj.episode_info("ep0.mp4", "Task:       open the drawer\nresult:        True\n")
    assert (info["instruction"], info["sim_success"]) == ("open the drawer", True)
    assert (info["instruction_source"], info["sim_success_source"]) == ("sidecar", "sidecar")


def test_lerobot_prefixed_name_maps_to_eval_info_and_task_map():
    langs = fj.libero_languages(TASK_MAP)
    ei = _eval_info([("libero_spatial", 0, [True, False]), ("libero_spatial", 1, [False, True])])
    info = fj.episode_info("libero_spatial_1--eval_episode_1.mp4", "", ei, langs)
    assert info["sim_success"] is True and info["sim_success_source"] == "eval_info"
    assert info["instruction"] == langs["libero_spatial"][1] and info["instruction_source"] == "libero_task_map"
    info0 = fj.episode_info("libero_spatial_0--eval_episode_1.mp4", "", ei, langs)
    assert info0["sim_success"] is False and info0["instruction"] == langs["libero_spatial"][0]


def test_lerobot_flat_name_is_ambiguous_with_several_tasks():
    # Runs before the per-task prefix: eval_episode_0.mp4 of task 0 and task 1 overwrote each other.
    ei = _eval_info([("libero_spatial", 0, [True]), ("libero_spatial", 1, [False])])
    info = fj.episode_info("eval_episode_0.mp4", "", ei, fj.libero_languages(TASK_MAP))
    assert info["sim_success"] is None and info["instruction"] is None


def test_lerobot_flat_name_with_one_task_is_unique():
    ei = _eval_info([("libero_10", 1, [False, True])])
    info = fj.episode_info("eval_episode_1.mp4", "", ei, fj.libero_languages(TASK_MAP))
    assert info["sim_success"] is True and info["instruction"] == "turn on the stove and put the moka pot on it"


def test_split_lerobot_name():
    assert fj.split_lerobot_name("libero_10_3--eval_episode_2.mp4") == ("libero_10_3", "eval_episode_2.mp4")
    assert fj.split_lerobot_name("a--openvla_oft--episode=1.mp4") == (None, "a--openvla_oft--episode=1.mp4")


def _run_dir(tmp_path):
    run = tmp_path / "run"
    v0 = run / "task-0" / "videos"
    v0.mkdir(parents=True)
    (run / "task-0" / "eval_info.json").write_text(json.dumps(
        _eval_info([("libero_spatial", 0, [True, False]), ("libero_spatial", 1, [True])])))
    for name in ("libero_spatial_0--eval_episode_0.mp4", "libero_spatial_0--eval_episode_1.mp4",
                 "libero_spatial_1--eval_episode_0.mp4"):
        (v0 / name).write_bytes(b"")
    (v0 / "libero_spatial_0--eval_episode_0.txt").write_text("=== ctx ===\nSuite: libero_spatial\n")
    v1 = run / "task-1" / "videos"
    v1.mkdir(parents=True)
    (v1 / "x--episode=0--success=True.mp4").write_bytes(b"")  # no instruction anywhere -> not scored
    (run / "task-1" / "summary.txt").write_text("suite: libero_goal\n")
    # not rollouts: Cosmos Policy future-image montage, LeRobot predicted video
    (v1 / "x--with_future_img--episode=0--success=True--task=open_the_drawer.mp4").write_bytes(b"")
    (v0 / "libero_spatial_0--pred_episode_0.mp4").write_bytes(b"")
    return run


def test_build_episodes_and_jobs(tmp_path):
    run = _run_dir(tmp_path)
    eps = fj.build_episodes(run, fj.libero_languages(TASK_MAP))
    assert [e["id"] for e in eps] == ["ep0001", "ep0002", "ep0003", "ep0004"]
    assert [e["video_key"] for e in eps][0] == "task-0/videos/libero_spatial_0--eval_episode_0.mp4"
    assert [e["sim_success"] for e in eps] == [True, False, True, True]
    jobs = fj.jobs_for(eps)
    assert [j["id"] for j in jobs] == ["ep0001", "ep0002", "ep0003"]
    assert all(len(j["videos"]) == 1 and Path(j["videos"][0]).is_absolute() for j in jobs)
    assert jobs[2]["task"] == "pick up the black bowl next to the ramekin and place it on the plate"


# --- score parsing ----------------------------------------------------------------------------------

@pytest.mark.parametrize("record,expected", [
    ({"score": 4}, 4),
    ({"score": "2"}, 2),
    ({"score": 7, "output": "Rationale: x\n\nScore: 3"}, 3),
    ({"output": "Rationale: grasped and placed.\n\nScore: 5"}, 5),
    ({"output": "Score: 1\n"}, 1),
    ({"score": True}, None),
    ({"score": 0}, None),
    ({"output": "Score: 6"}, None),
    ({}, None),
])
def test_parse_score(record, expected):
    assert fj.parse_score(record) == expected


def test_read_scores_skips_garbage_lines():
    lines = ['{"id": "ep0001", "score": 5}', "", "not json", '{"no_id": 1}', '{"id": "ep0002", "score": 1}']
    assert set(fj.read_scores(lines)) == {"ep0001", "ep0002"}


# --- agreement --------------------------------------------------------------------------------------

def _ep(i, sim, score):
    return {"id": f"ep{i}", "sim_success": sim, "flora_score": score}


def test_summarize_agreement_confusion_and_means():
    eps = [_ep(1, True, 5), _ep(2, True, 4), _ep(3, True, 2), _ep(4, False, 1),
           _ep(5, False, 4), _ep(6, None, 3), _ep(7, False, None)]
    s = fj.summarize(eps)
    assert s["confusion"] == {"sim_success_flora_success": 2, "sim_success_flora_failure": 1,
                              "sim_failure_flora_success": 1, "sim_failure_flora_failure": 1}
    assert (s["n_episodes"], s["n_scored"], s["n_compared"], s["n_unscored"]) == (7, 6, 5, 1)
    assert s["n_agree"] == 3 and s["agreement"] == 0.6
    assert s["mean_score_sim_success"] == round(11 / 3, 3) and s["mean_score_sim_failure"] == 2.5
    assert s["score_histogram"] == {"1": 1, "2": 1, "3": 1, "4": 2, "5": 1}
    assert s["sim_success_rate"] == 0.6 and s["flora_success_rate"] == 0.5


def test_threshold_boundary():
    eps = [_ep(1, False, 3), _ep(2, True, 4)]
    assert fj.summarize(eps)["agreement"] == 1.0
    assert fj.summarize(eps, threshold=3)["agreement"] == 0.5
    assert fj.flora_success(4) is True and fj.flora_success(3) is False and fj.flora_success(None) is None


def test_summarize_empty():
    s = fj.summarize([])
    assert s["agreement"] is None and s["mean_score"] is None and s["n_compared"] == 0


# --- judge.json ---------------------------------------------------------------------------------------

def _manifest():
    return {"episodes": [
        {"id": "ep0001", "task_idx": 0, "video_key": "task-0/videos/a.mp4", "instruction": "open the drawer",
         "instruction_source": "sidecar", "sim_success": True, "sim_success_source": "filename"},
        {"id": "ep0002", "task_idx": 0, "video_key": "task-0/videos/b.mp4", "instruction": "open the drawer",
         "instruction_source": "sidecar", "sim_success": False, "sim_success_source": "filename"},
        {"id": "ep0003", "task_idx": 1, "video_key": "task-1/videos/c.mp4", "instruction": None,
         "instruction_source": "none", "sim_success": True, "sim_success_source": "filename"},
        {"id": "ep0004", "task_idx": 1, "video_key": "task-1/videos/d.mp4", "instruction": "x",
         "instruction_source": "sidecar", "sim_success": True, "sim_success_source": "filename"},
    ]}


def test_build_report_schema_and_reasons():
    scores = {"ep0001": {"id": "ep0001", "score": 5, "rationale": "Drawer fully open.", "expected_score": 4.9},
              "ep0002": {"id": "ep0002", "score": 4, "rationale": "Mostly open."}}
    meta = {"target": "smolvla", "run_id": "2026-10-20-0130", "model": "nvidia/Flora-Rewarder",
            "model_revision": "031299e46e452fc0cedb4e0975eb9fa95fc37783", "instance_type": "g7e.2xlarge",
            "wall_time_s": 1234}
    report = fj.build_report(_manifest(), scores, meta, dropped={"ep0004": "video unreadable: bad"})
    assert fj.validate_report(report) == []
    assert report["schema_version"] == 1 and report["success_threshold"] == 4
    assert report["instance_type"] == "g7e.2xlarge" and report["wall_time_s"] == 1234
    e1, e2, e3, e4 = report["episodes"]
    assert (e1["flora_score"], e1["flora_success"], e1["rationale"]) == (5, True, "Drawer fully open.")
    assert (e2["flora_score"], e2["flora_success"]) == (4, True)
    assert e3["skipped_reason"] == "no task instruction found" and e3["flora_success"] is None
    assert e4["skipped_reason"].startswith("video unreadable")
    s = report["summary"]
    assert (s["n_scored"], s["n_compared"], s["agreement"]) == (2, 2, 0.5)
    json.dumps(report)  # serialisable
    txt = fj.render_txt(report)
    assert "agreement with sim success: 50.0% (1/2)" in txt and "g7e.2xlarge" in txt
    assert "(not scored: no task instruction found)" in txt


def test_validate_report_catches_bad_scores():
    report = fj.build_report(_manifest(), {"ep0001": {"id": "ep0001", "score": 5}})
    report["episodes"][0]["flora_score"] = 9
    report["episodes"][1]["flora_score"] = 2
    report["episodes"][1]["flora_success"] = True
    problems = fj.validate_report(report)
    assert any("outside 1..5" in p for p in problems) and any("threshold" in p for p in problems)
    assert fj.validate_report({}) == [f"missing key {k}" for k in fj.REPORT_KEYS]


# --- gallery caption block ----------------------------------------------------------------------------

def test_annotate_caption_is_idempotent():
    report = {"success_threshold": 4, "model": "nvidia/Flora-Rewarder", "model_revision": "031299e46e45"}
    ep = {"flora_score": 2, "flora_success": False, "rationale": "Missed the bowl."}
    once = fj.annotate_caption("=== ctx ===\nSuite: s\n", ep, report)
    assert "Flora score: 2/5 (failure at >= 4)" in once and once.startswith("=== ctx ===")
    twice = fj.annotate_caption(once, dict(ep, flora_score=5, flora_success=True), report)
    assert twice.count(fj.CAPTION_MARK) == 1 and "Flora score: 5/5" in twice and "2/5" not in twice
    assert fj.annotate_caption("", ep, report).startswith(fj.CAPTION_MARK)


# --- CLI end to end -----------------------------------------------------------------------------------

def test_cli_prepare_report_captions(tmp_path):
    run = _run_dir(tmp_path)
    (tmp_path / "map.py").write_text(TASK_MAP)
    jobs, manifest = tmp_path / "jobs.jsonl", tmp_path / "manifest.json"
    assert fj.main(["prepare", "--run-dir", str(run), "--jobs", str(jobs), "--manifest", str(manifest),
                    "--libero-task-map", str(tmp_path / "map.py")]) == 0
    job_lines = [json.loads(x) for x in jobs.read_text().splitlines()]
    assert len(job_lines) == 3 and set(job_lines[0]) == {"id", "task", "videos"}
    # terminal_reward output: one line per job (official keys)
    scores = tmp_path / "terminal.jsonl"
    scores.write_text("".join(json.dumps({"id": j["id"], "task": j["task"], "score": s, "rationale": "r",
                                          "output": f"Rationale: r\n\nScore: {s}"}) + "\n"
                              for j, s in zip(job_lines, (5, 1, 4))))
    out_json, out_txt = tmp_path / "judge.json", tmp_path / "judge.txt"
    assert fj.main(["report", "--manifest", str(manifest), "--scores", str(scores), "--json", str(out_json),
                    "--txt", str(out_txt), "--meta", "instance_type=g7e.2xlarge", "wall_time_s=900",
                    "tensor_parallel=1"]) == 0
    report = json.loads(out_json.read_text())
    assert report["wall_time_s"] == 900 and report["tensor_parallel"] == 1
    assert report["summary"]["agreement"] == 1.0 and report["summary"]["n_compared"] == 3
    assert report["episodes"][3]["skipped_reason"] == "no task instruction found"
    caps = tmp_path / "caps"
    assert fj.main(["captions", "--manifest", str(manifest), "--report", str(out_json), "--out-dir", str(caps)]) == 0
    first = caps / "task-0" / "videos" / "libero_spatial_0--eval_episode_0.txt"
    assert first.read_text().startswith("=== ctx ===") and "Flora score: 5/5" in first.read_text()
    assert len(list(caps.rglob("*.txt"))) == 3


def test_cli_prepare_without_instructions_fails(tmp_path):
    run = tmp_path / "run" / "task-0" / "videos"
    run.mkdir(parents=True)
    (run / "eval_episode_0.mp4").write_bytes(b"")
    rc = fj.main(["prepare", "--run-dir", str(tmp_path / "run"), "--jobs", str(tmp_path / "j"),
                  "--manifest", str(tmp_path / "m")])
    assert rc == 3


def test_module_is_stdlib_only():
    src = (ROOT / "assets" / "flora-judge" / "flora_judge.py").read_text()
    imported = {line.split()[1].split(".")[0] for line in src.splitlines() if line.startswith(("import ", "from "))}
    assert imported <= set(sys.stdlib_module_names)
