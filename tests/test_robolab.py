"""RoboLab targets (robolab-cosmos3, robolab-pi05): variant rendering and the run report.

* every ``variants:`` entry of models/robolab-*.yaml renders through generate.generate_robolab, passes
  ``bash -n`` (and ``shellcheck -S error`` when it is installed) and carries the expected server / client
  launch lines, shard layout and checkpoint pins;
* the yaml guards (unknown variant, unsafe task names / args) stop the render;
* assets/robolab/robolab_report.py turns RoboLab's per-shard output (episode_results.jsonl + mp4s) into
  summary.txt / robolab_results.json / ``success=`` video names that results.py reads.
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

yaml = pytest.importorskip("yaml")
pytest.importorskip("jinja2")
import generate  # noqa: E402
import results  # noqa: E402

REPORT = ROOT / "assets" / "robolab" / "robolab_report.py"
BASH = shutil.which("bash")
SHELLCHECK = shutil.which("shellcheck")


def _config(vla, variant=None):
    cfg = generate._load_merged_config(ROOT / "simulator-config.yaml", vla)
    cfg = copy.deepcopy(cfg)
    if variant is not None:
        cfg["model"]["variant"] = variant
    return cfg


def _variants(vla):
    return list(yaml.safe_load((ROOT / "models" / f"{vla}.yaml").read_text())["variants"])


CASES = [(v, var) for v in generate.ROBOLAB_TARGETS for var in _variants(v)]


@pytest.mark.parametrize("vla,variant", CASES, ids=[f"{v}:{var}" for v, var in CASES])
def test_variant_renders(vla, variant, tmp_path):
    cfg = _config(vla, variant)
    text = generate.generate_robolab(cfg, False, vla=vla)
    script = tmp_path / f"{vla}-{variant}.sh"
    script.write_text(text)
    assert text.startswith("#!/bin/bash")
    assert "{%" not in text and "{#" not in text.replace("${#", "")
    if BASH:
        r = subprocess.run([BASH, "-n", str(script)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    if SHELLCHECK:
        r = subprocess.run([SHELLCHECK, "-S", "error", str(script)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout
    var = cfg["variants"][variant]
    proto = cfg["protocols"][var["protocol"]]
    assert f'--num-envs {proto["num_envs"]} --num-runs {proto["num_runs"]}' in text
    assert f'if [ "$SHARDS" -gt {var["shards"]} ]; then SHARDS={var["shards"]}; fi' in text
    assert f'SHARDS=$(( N_GPU / {var["gpus_per_shard"]} ))' in text
    assert cfg["model"]["robolab_commit"] in text and cfg["model"]["server_commit"] in text
    assert "--instruction-type default" in text
    assert f'--label "{proto["label"]}"' in text
    if proto["tasks"]:
        assert "TASKS=(" + " ".join(proto["tasks"]) + " )" in text
    else:
        assert "TASKS=()" in text and "task_metadata.json" in text
    if vla == "robolab-cosmos3":
        assert "policies/cosmos3/run.py --headless" in text
        assert f'--checkpoint-path "{var["hf_repo"]}" --hf-revision "{var["hf_model_revision"]}"' in text
        assert "--group=cu128-train --group=policy-server" in text
        assert " ".join(var["server_args"]) in text
    else:
        assert "policies/pi0_family/run.py --headless --policy pi05" in text
        assert f'--policy.config={var["policy_config"]} --policy.dir={var["checkpoint_dir"]}' in text
        assert "XLA_PYTHON_CLIENT_MEM_FRACTION=0.5" in text


def test_default_variants_are_the_small_subset():
    for vla in generate.ROBOLAB_TARGETS:
        cfg = _config(vla)
        name, var = generate.resolve_variant(cfg, vla)
        proto = cfg["protocols"][var["protocol"]]
        assert var["shards"] == 1 and len(proto["tasks"]) == 12, (vla, name)
        assert cfg["instance"]["preferred"] == var["instance"]["preferred"], vla


def test_unknown_variant_fails():
    with pytest.raises(SystemExit):
        generate.generate_robolab(_config("robolab-cosmos3", "nope"), False, vla="robolab-cosmos3")


@pytest.mark.parametrize("field,value", [
    ("task", "BananaInBowlTask; rm -rf /"),
    ("server_args", ["--port", "$(id)"]),
])
def test_unsafe_values_fail(field, value):
    cfg = _config("robolab-cosmos3")
    if field == "task":
        cfg["protocols"]["subset12"]["tasks"] = [value]
    else:
        cfg["variants"]["edge"]["server_args"] = value
    with pytest.raises(SystemExit):
        generate.generate_robolab(cfg, False, vla="robolab-cosmos3")


# ───────────────────────────── robolab_report.py ─────────────────────────────


def _episode(env, task, run, env_id, success, score=None):
    return {"env_name": env, "task_name": task, "run": run, "env_id": env_id, "episode": run * 2 + env_id,
            "success": success, "score": score, "instruction": f"do {task}", "policy": "cosmos3"}


def _fake_run(tmp_path):
    s0 = tmp_path / "shard-0" / "run"
    s1 = tmp_path / "shard-1" / "run"
    for d in (s0 / "BananaInBowlTask", s1 / "RubiksCubeTask"):
        d.mkdir(parents=True)
    eps0 = [_episode("BananaInBowlTask", "BananaInBowlTask", 0, 0, True, 1.0),
            _episode("BananaInBowlTask", "BananaInBowlTask", 0, 1, False, 0.5),
            # resumed run re-appends env 1: the later record wins
            _episode("BananaInBowlTask", "BananaInBowlTask", 0, 1, True, 1.0)]
    eps1 = [_episode("RubiksCubeTask", "RubiksCubeTask", 0, 0, False, 0.0)]
    (s0 / "episode_results.jsonl").write_text("\n".join(json.dumps(e) for e in eps0) + "\nnot json\n")
    (s1 / "episode_results.jsonl").write_text(json.dumps(eps1[0]) + "\n")
    for name in ("Put_the_banana_in_the_bowl_0_env0.mp4", "Put_the_banana_in_the_bowl_0_env0_viewport.mp4",
                 "Put_the_banana_in_the_bowl_0_env1.mp4"):
        (s0 / "BananaInBowlTask" / name).write_bytes(b"\x00mp4")
    (s1 / "RubiksCubeTask" / "Pick_the_cube_0_env0.mp4").write_bytes(b"\x00mp4")
    (s1 / "RubiksCubeTask" / "notes.txt").write_text("ignored")
    meta = tmp_path / "task_metadata.json"
    meta.write_text(json.dumps([{"task_name": "BananaInBowlTask", "difficulty_label": "simple", "attributes": "semantics"},
                                {"task_name": "RubiksCubeTask", "difficulty_label": "simple", "attributes": ""}]))
    return s0, s1, meta


def _run_report(tmp_path, *extra):
    s0, s1, meta = _fake_run(tmp_path)
    out = tmp_path / "upload"
    r = subprocess.run([sys.executable, str(REPORT), "--run-dir", str(s0), "--run-dir", str(s1),
                        "--run-dir", str(tmp_path / "missing-shard"), "--upload-dir", str(out),
                        "--label", "RoboLab/subset12", "--tasks", "BananaInBowlTask RubiksCubeTask PickDrillTask",
                        "--episodes-per-task", "2", "--task-metadata", str(meta),
                        "--meta", "model=robolab-cosmos3 (Cosmos3-Edge-Policy-DROID)", "--meta", "stack_robolab=9cce00ca",
                        *extra], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip().splitlines()[-1], out


def test_report_counts_and_files(tmp_path):
    kn, out = _run_report(tmp_path)
    assert kn == "2/3"
    rep = json.loads((out / "robolab_results.json").read_text())
    assert rep["n_episodes"] == 3 and rep["n_success"] == 2 and rep["n_expected"] == 6
    assert rep["tasks_complete"] == 1 and rep["n_tasks"] == 3
    rows = {r["task"]: r for r in rep["per_task"]}
    assert rows["BananaInBowlTask"]["k"] == 2 and rows["BananaInBowlTask"]["mean_score"] == 1.0
    assert rows["PickDrillTask"]["n"] == 0 and rows["PickDrillTask"]["success_rate"] is None
    assert rep["by_difficulty"]["simple"]["n"] == 3
    lo, hi = rep["ci95_wilson"]
    assert 0 < lo < 2 / 3 < hi <= 1
    vids = sorted(p.name for p in (out / "videos").glob("*.mp4"))
    assert vids == ["BananaInBowlTask_run0_env0_success=True.mp4", "BananaInBowlTask_run0_env0_success=True_viewport.mp4",
                    "BananaInBowlTask_run0_env1_success=True.mp4", "RubiksCubeTask_run0_env0_success=False.mp4"]
    cap = (out / "videos" / "RubiksCubeTask_run0_env0_success=False.txt").read_text()
    assert "Success:    False" in cap and "stack_robolab: 9cce00ca" in cap


def test_report_summary_is_read_by_results(tmp_path):
    _, out = _run_report(tmp_path)
    summ = (out / "summary.txt").read_text()
    kv = dict(results.KV_RE.findall(summ))
    assert kv["suite"] == "RoboLab/subset12"
    assert (kv["n_episodes"], kv["n_success"]) == ("3", "2")
    assert kv["model"].startswith("robolab-cosmos3")

    class FakeS3:
        def text(self, region, bucket, key):
            return summ if key.endswith("summary.txt") else None

        def keys(self, *a):
            return []

    row = results.task_row(FakeS3(), "us-east-1", "b", "2026-10-30-0000", "2026-10-30-0000/task-0/")
    assert row["suite"] == "RoboLab/subset12" and (row["k"], row["n"]) == (2, 3)
    assert row["stack"] == {"stack_robolab": "9cce00ca"}


def test_report_video_cap(tmp_path):
    _, out = _run_report(tmp_path, "--max-videos-per-task", "1")
    vids = sorted(p.name for p in (out / "videos").glob("*.mp4"))
    # one per (task, camera kind): Banana sensor + viewport, Rubiks sensor
    assert len(vids) == 3


def test_report_empty_run(tmp_path):
    out = tmp_path / "upload"
    r = subprocess.run([sys.executable, str(REPORT), "--run-dir", str(tmp_path / "none"), "--upload-dir", str(out),
                        "--label", "RoboLab/RoboLab-120", "--episodes-per-task", "10", "--n-tasks", "120"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip().endswith("0/0")
    rep = json.loads((out / "robolab_results.json").read_text())
    assert rep["n_expected"] == 1200 and rep["success_rate"] is None
