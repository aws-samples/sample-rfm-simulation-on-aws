"""Benchmark advisor (bench.py, bench/catalog.yaml) and the --tasks-file override it deploys with.

Covers the catalog against models/*.yaml and targets.py, the planner (several answer sets: filters, suite
coverage, cheapest-4 cut, --targets), the plan -> deploy-command dry run (tasks file per target in its own
format, generate.py render + bash -n, no AWS call), the --tasks-file override in task_sets / generate.py /
deploy.py, the pi task-id subset and RoboTwin condition label, and the HTML report from fixture summary.txt
trees (Wilson interval, tie rule, condition filter, self-contained page).
"""

import datetime as dt
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
yaml = pytest.importorskip("yaml")
pytest.importorskip("jinja2")
import bench  # noqa: E402
import robotwin_plan  # noqa: E402
import task_sets  # noqa: E402
from targets import VLA_TARGETS  # noqa: E402

BASH = shutil.which("bash")
CAT = bench.load_catalog()
REGION = "ap-northeast-2"
ACCT = "123456789012"


def _answers(**kw):
    a = dict(robot="single-arm", action="spatial", commercial="no", gpu="any", edge="no")
    a.update(kw)
    return a


def _names(sel):
    return [c["target"] for c in sel["candidates"]]


# --- catalog ------------------------------------------------------------------------------------------

def test_catalog_targets_exist_and_match_models():
    for name, e in CAT["targets"].items():
        assert name in VLA_TARGETS
        model = yaml.safe_load((ROOT / "models" / f"{name}.yaml").read_text())
        assert e["instance"] == model["instance"]["preferred"][0], name
        assert e["family"] in ("lerobot-libero", "pi", "robotwin")
        assert e["commercial_ok"] in (True, False, None)
        assert e["gpu_tier"] in bench.GPU_RANK
        assert set(e["suites_measured"]) <= set(e["suites"])
        assert e["default_run"]["episodes"] > 0 and len(e["default_run"]["minutes"]) == 2
    for name in CAT["not_benchable"]:
        assert name in VLA_TARGETS and name not in CAT["targets"]


def test_catalog_families_match_the_templates():
    import generate
    for name, e in CAT["targets"].items():
        if e["family"] == "lerobot-libero":
            assert name in generate.LEROBOT_LIBERO_TARGETS
        elif e["family"] == "robotwin":
            assert name in generate.LEROBOT_ROBOTWIN_TARGETS


# --- planner ------------------------------------------------------------------------------------------

def test_spatial_small_gpu_picks_four_cheapest_libero_spatial_targets():
    sel = bench.select(CAT, _answers(gpu="24gb"), REGION)
    assert sel["benchmark"] == "libero" and sel["suite"] == "libero_spatial"
    assert set(_names(sel)) == {"smolvla", "xvla", "vla-jepa", "pi"}
    why = {x["target"]: x["reason"] for x in sel["excluded"]}
    assert "48gb" in why["fastwam"] and "48gb" in why["lingbot-va"]
    assert "libero_10" in why["multitask-dit"]
    assert "not on a shared bench" in why["openvla-oft"]


def test_commercial_and_edge_filter_to_permissive_small_models():
    sel = bench.select(CAT, _answers(commercial="yes", edge="yes"), REGION)
    assert _names(sel) == ["smolvla", "xvla"]  # cheapest first
    why = {x["target"]: x["reason"] for x in sel["excluded"]}
    assert "commercial use not confirmed" in why["pi"] and "commercial use not confirmed" in why["fastwam"]
    assert "edge" in why["vla-jepa"]


def test_long_horizon_uses_libero_10():
    sel = bench.select(CAT, _answers(action="long-horizon"), REGION)
    assert sel["suite"] == "libero_10"
    assert len(sel["candidates"]) == bench.MAX_CANDIDATES
    acts = bench.actions_for(CAT, sel)
    assert [a["suite"] for a in acts] == ["libero_10"] * 10


def test_articulated_actions_are_valid_for_every_candidate():
    sel = bench.select(CAT, _answers(action="articulated"), REGION)
    acts = bench.actions_for(CAT, sel)
    assert acts and all(a["action_type"] == "articulated" for a in acts)
    for c in sel["candidates"]:
        assert {a["suite"] for a in acts} <= set(CAT["targets"][c["target"]]["suites"])
    assert sel["suite"] == "libero_10"  # the articulated suite most candidates cover (lingbot-va, multitask-dit too)
    assert "libero_10:3" in {a["id"] for a in acts}  # bowl in the bottom drawer and close it
    three = bench.select(CAT, _answers(action="articulated"), REGION, only=["smolvla", "xvla", "pi"])
    ids = {a["id"] for a in bench.actions_for(CAT, three)}
    assert {"libero_goal:0", "libero_goal:7", "libero_10:3"} <= ids  # middle drawer, stove knob: all-suite models


def test_bimanual_is_robotwin_and_commercial_excludes_everything():
    sel = bench.select(CAT, _answers(robot="bimanual", action="tool-use"), REGION)
    assert sel["benchmark"] == "robotwin" and set(_names(sel)) == {"smolvla-robotwin", "lingbot-vla2"}
    acts = bench.actions_for(CAT, sel)
    assert {a["task"] for a in acts} == {"beat_block_hammer", "stamp_seal"}
    none = bench.select(CAT, _answers(robot="bimanual", action="any", commercial="yes"), REGION)
    assert none["candidates"] == []
    assert any("CuRobo" in x["reason"] for x in none["excluded"])


def test_humanoid_has_no_bench_yet():
    sel = bench.select(CAT, _answers(robot="humanoid", action="any"), REGION)
    assert sel["candidates"] == [] and {x["target"] for x in sel["excluded"]} >= {"gr00t-gr1", "rldx-gr1"}


def test_targets_list_restricts_and_explains():
    sel = bench.select(CAT, _answers(gpu="24gb"), REGION, only=["smolvla", "xvla", "pi"])
    assert set(_names(sel)) == {"smolvla", "xvla", "pi"}
    assert {"target": "vla-jepa", "reason": "not in your --targets list"} in sel["excluded"]
    with pytest.raises(ValueError, match="fastwam.*48gb"):
        bench.select(CAT, _answers(gpu="24gb"), REGION, only=["smolvla", "fastwam"])


def test_answers_validation_accepts_yaml_booleans():
    assert bench.validate_answers(_answers(commercial=False, edge=True))["edge"] == "yes"
    with pytest.raises(ValueError, match="robot"):
        bench.validate_answers(_answers(robot="quadruped"))


def test_estimate_uses_seconds_per_episode_or_scales():
    smol = bench.estimate(CAT["targets"]["smolvla"], 20, "ap-northeast-2")
    assert smol["minutes"] == 35 and smol["usd_per_hour"] == 0.9896  # 30 + 10 x 30 s
    pi = bench.estimate(CAT["targets"]["pi"], 240, "ap-northeast-2")
    assert pi["minutes"] == 114 and "upper bound" in pi["basis"]
    assert bench.estimate(CAT["targets"]["smolvla"], 20, "sa-east-1")["usd"] is None


def test_interactive_plan_writes_a_plan(tmp_path):
    answers = iter(["1", "4", "2", "1", "2",  # single-arm, spatial, no commercial need, 24gb, no edge
                    "1,2", "", "y"])         # actions 1 and 2, default episodes, confirm
    out = tmp_path / "bench-plan.yaml"
    args = types.SimpleNamespace(catalog=str(bench.CATALOG_PATH), region=REGION, answers=None, episodes=None,
                                 actions=None, seed="7", out=str(out), yes=False, targets="smolvla,xvla,pi")
    log = tmp_path / "log.txt"
    with log.open("w") as fh:
        assert bench.cmd_plan(args, inp=lambda _="": next(answers), out=fh) == 0
    plan = bench.load_plan(out)
    assert [c["target"] for c in plan["candidates"]] == ["smolvla", "xvla", "pi"]
    assert [a["id"] for a in plan["actions"]] == ["libero_spatial:0", "libero_spatial:1"]
    assert plan["episodes"] == 10 and plan["seed"] == 7 and plan["condition"] == f"bench-{plan['id']}"
    assert plan["estimate"]["total_usd"] > 0 and "AWS does not certify" in plan["disclaimer"]


def test_parse_action_ids_limits():
    acts = bench.all_actions("libero")[:5]
    assert [a["id"] for a in bench.parse_action_ids("1, libero_spatial:2", acts)] == ["libero_spatial:0",
                                                                                      "libero_spatial:2"]
    with pytest.raises(ValueError):
        bench.parse_action_ids("1,2,3,4", acts)
    with pytest.raises(ValueError):
        bench.parse_action_ids("libero_goal:9", acts)


# --- run (dry run) ------------------------------------------------------------------------------------

def _plan(tmp_path, targets=("smolvla", "xvla", "pi"), actions=("libero_spatial:0", "libero_spatial:1"),
          answers=None, episodes=10, plan_id="a1b2c3"):
    sel = bench.select(CAT, answers or _answers(gpu="24gb"), REGION, only=list(targets))
    acts = bench.parse_action_ids(list(actions), bench.actions_for(CAT, sel))
    plan = bench.make_plan(CAT, sel, acts, episodes, 1234, REGION, plan_id=plan_id,
                           now=dt.datetime(2026, 11, 6, 1, 0, tzinfo=dt.timezone.utc))
    path = tmp_path / "bench-plan.yaml"
    bench.write_plan(plan, path)
    return bench.load_plan(path), path


def test_tasks_spec_per_family(tmp_path):
    plan, _ = _plan(tmp_path)
    smol = bench.tasks_spec(plan, "smolvla", CAT)
    assert smol["name"] == "bench-a1b2c3"
    assert smol["tasks"][1] == dict(suite="libero_spatial", task_ids=[1],
                                    env_name="pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
                                    n_episodes=10, seed=1234, condition="bench-a1b2c3",
                                    description=smol["tasks"][1]["description"])
    pi = bench.tasks_spec(plan, "pi", CAT)
    assert pi["tasks"][0]["num_episodes"] == 10 and pi["tasks"][0]["task_ids"] == [0]
    rplan, _ = _plan(tmp_path, targets=("smolvla-robotwin", "lingbot-vla2"), actions=("clean:stamp_seal",),
                     answers=_answers(robot="bimanual", action="tool-use"))
    rt = bench.tasks_spec(rplan, "lingbot-vla2", CAT)["tasks"][0]
    assert rt["robotwin_tasks"] == ["stamp_seal"] and rt["setting"] == "clean" and rt["condition"] == "bench-a1b2c3"
    units = robotwin_plan.plan(bench.tasks_spec(rplan, "lingbot-vla2", CAT)["tasks"])
    assert units[0]["condition"] == "bench-a1b2c3" and units[0]["seed"] == 1234


@pytest.mark.skipif(not BASH, reason="bash not on PATH")
def test_run_dry_run_prints_commands_and_renders(tmp_path):
    _, path = _plan(tmp_path)
    r = subprocess.run([sys.executable, str(ROOT / "bench.py"), "run", "--plan", str(path), "--dry-run"],
                       cwd=str(ROOT), capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "AWS_EC2_METADATA_DISABLED": "true"})
    assert r.returncode == 0, r.stdout + r.stderr
    d = tmp_path / "bench-plan.d"
    for t in ("smolvla", "xvla", "pi"):
        assert (f"python deploy.py --vla {t} --tasks-file {d / ('tasks-' + t + '.yaml')} --seed 1234 "
                f"--region {REGION}") in r.stdout
        assert f"render + bash -n {t}: ok" in r.stdout
    assert "nothing deployed" in r.stdout
    pi_sh = (d / "userdata" / "pi.sh").read_text()
    assert "/tmp/pi_task_ids_patch.py" in pi_sh and "for task_id in tqdm.tqdm(range(num_tasks_in_suite)):" in pi_sh
    assert '"condition": "bench-a1b2c3"' in pi_sh and 'echo "task_ids: ${K_TASK_IDS}"' in pi_sh
    smol = (d / "userdata" / "smolvla.sh").read_text()
    assert '"seed": 1234' in smol and '"condition": "bench-a1b2c3"' in smol and '"task_ids": [1]' in smol


def test_destroy_dry_run_keeps_data(tmp_path, capsys):
    _, path = _plan(tmp_path)
    assert bench.cmd_destroy(types.SimpleNamespace(plan=str(path), dry_run=True)) == 0
    out = capsys.readouterr().out
    assert out.count("--keep-data --yes") == 3 and "destroy.py --vla pi --region ap-northeast-2" in out


# --- --tasks-file override ----------------------------------------------------------------------------

def test_apply_file_replaces_tasks_and_records_the_name():
    cfg = yaml.safe_load((ROOT / "models" / "smolvla.yaml").read_text())
    spec = dict(name="bench-a1b2c3", description="d", tasks=[dict(suite="libero_goal", task_ids=[0], n_episodes=3)])
    out = task_sets.apply_file(cfg, spec, "f.yaml")
    assert out["tasks"] == spec["tasks"] and out["task_set"] == "bench-a1b2c3"
    assert "bench-a1b2c3: 1 task(s)" in task_sets.describe(out)
    assert cfg["tasks"][0]["suite"] == "libero_spatial"  # input untouched
    with pytest.raises(ValueError, match="non-empty"):
        task_sets.apply_file(cfg, dict(tasks=[]), "f.yaml")
    with pytest.raises(ValueError, match="name"):
        task_sets.apply_file(cfg, dict(name="a b", tasks=spec["tasks"]), "f.yaml")


def test_generate_rejects_tasks_file_with_task_set(tmp_path):
    f = tmp_path / "t.yaml"
    f.write_text(yaml.safe_dump(dict(tasks=[dict(suite="libero_spatial", task_ids=[0], n_episodes=1)])))
    r = subprocess.run([sys.executable, "generate.py", "--vla", "vla-jepa", "--tasks-file", str(f), "--task-set",
                        "probe", "--dry-run"], cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode != 0 and "--tasks-file and --task-set" in r.stderr


def test_pi_default_render_has_no_task_id_patch(tmp_path):
    r = subprocess.run([sys.executable, "generate.py", "--vla", "pi", "--output-dir", str(tmp_path)], cwd=str(ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    text = (tmp_path / "pi.sh").read_text()
    assert "pi_task_ids_patch" not in text and 'echo "n_tasks: 10"' in text


def test_pi_rejects_bad_task_ids(tmp_path):
    f = tmp_path / "t.yaml"
    f.write_text(yaml.safe_dump(dict(tasks=[dict(suite="libero_spatial", task_ids=[10], num_episodes=1)])))
    r = subprocess.run([sys.executable, "generate.py", "--vla", "pi", "--tasks-file", str(f), "--dry-run"],
                       cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode != 0 and "task ids 0-9" in r.stderr


def test_deploy_help_lists_tasks_file():
    r = subprocess.run([sys.executable, "deploy.py", "--help"], cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 0 and "--tasks-file" in r.stdout


# --- report -------------------------------------------------------------------------------------------

def _summary(**kv):
    return "".join(f"{k}: {v}\n" for k, v in kv.items())


def _fixture(root: Path, stack: str, run: str, tasks: list, condition="bench-a1b2c3", instance="g6.xlarge",
             wall=1800):
    """tasks = [(env_name, k, n)] -> <root>/<bucket>/<run>/task-N/{summary.txt, videos/*.mp4}."""
    b = root / f"vla-sim-results-{stack}-{REGION}-{ACCT}" / run
    for i, (name, k, n) in enumerate(tasks):
        d = b / f"task-{i}"
        (d / "videos").mkdir(parents=True)
        (d / "summary.txt").write_text(_summary(suite="libero_spatial", env_name=name, condition=condition,
                                                n_episodes=n, n_success=k, success_rate=k / n,
                                                instance_type=instance, run_wall_s=wall))
        for j in range(2):
            ok = j < k
            (d / "videos" / f"libero_spatial_{i}--eval_episode_{j}_success={ok}.mp4").write_bytes(b"\0")


def _report(tmp_path, plan_path, data):
    out = tmp_path / "report.html"
    args = types.SimpleNamespace(catalog=str(bench.CATALOG_PATH), plan=str(plan_path), from_dir=str(data),
                                 out=str(out), email=None, no_gallery=True)
    assert bench.cmd_report(args, out=open(tmp_path / "log.txt", "w")) == 0
    return out.read_text(), (tmp_path / "log.txt").read_text()


def test_report_from_fixture_summaries(tmp_path):
    _, path = _plan(tmp_path)
    data = tmp_path / "data"
    names = [bench.libero_tasks()["libero_spatial"][i] for i in (0, 1)]
    _fixture(data, "smolvla-demo", "2026-11-06-0110", [(names[0], 3, 10), (names[1], 4, 10)])
    _fixture(data, "xvla-demo", "2026-11-06-0111", [(names[0], 10, 10), (names[1], 9, 10)], instance="g6.2xlarge")
    _fixture(data, "pi-demo", "2026-11-06-0112", [(names[0], 9, 10), (names[1], 10, 10)], instance="g5.xlarge",
             wall=3600)
    # same bucket, another condition and an older run: both ignored
    _fixture(data, "smolvla-demo", "2026-11-06-0200", [(names[0], 10, 10)], condition="S0")
    _fixture(data, "smolvla-demo", "2026-11-01-0000", [(names[0], 10, 10)])
    page, log = _report(tmp_path, path, data)
    assert bench.DISCLAIMER in page and "<script" not in page and "<svg" in page
    assert "7/20 = 0.35" in page  # smolvla pooled, the S0 and older runs are not counted
    assert "19/20 = 0.95" in page and "3/10 = 0.30" in page
    assert "Tie: the 95% intervals of" in page and "xvla" in page.split("Recommendation")[1][:400]
    assert "$0.49" in page  # smolvla 30 min at the ap-northeast-2 g6.xlarge price
    assert "<video src=\"file://" in page
    assert "pi:" not in log


def test_rank_tie_rule_prefers_commercial_then_cost():
    def row(t, k, n, usd, com):
        lo, hi = bench.results.wilson(k, n)
        return dict(target=t, k=k, n=n, rate=k / n, lo=lo, hi=hi, usd=usd, est_usd=None, commercial_ok=com)
    r = bench.rank([row("pi", 19, 20, 1.0, None), row("xvla", 18, 20, 2.0, True), row("smolvla", 2, 20, 0.1, True)])
    assert r["pick"] == "xvla" and r["tie"] == ["xvla", "pi"] and "Tie" in r["text"]
    r = bench.rank([row("pi", 20, 20, 1.0, None), row("smolvla", 2, 20, 0.1, True)])
    assert r["pick"] == "pi" and r["tie"] == [] and "leads" in r["text"]
    assert "No finished" in bench.rank([dict(target="x", n=0)])["text"]


def test_report_without_results_says_so(tmp_path):
    _, path = _plan(tmp_path)
    (tmp_path / "empty").mkdir()
    page, _ = _report(tmp_path, path, tmp_path / "empty")
    assert "No finished bench task yet" in page and "no result yet" in page


def test_plan_file_is_plain_yaml_with_schema(tmp_path):
    plan, path = _plan(tmp_path)
    raw = yaml.safe_load(path.read_text())
    assert raw["schema"] == bench.PLAN_SCHEMA and json.loads(json.dumps(raw)) == raw
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema: other\n")
    with pytest.raises(ValueError):
        bench.load_plan(bad)


def test_child_env_gives_each_parallel_deploy_its_own_npm_cache(tmp_path):
    a = bench.child_env(tmp_path, "smolvla")
    b = bench.child_env(tmp_path, "xvla")
    assert a["npm_config_cache"] != b["npm_config_cache"]
    assert a["npm_config_cache"].startswith(str(tmp_path))


def test_rank_tie_keeps_rate_order_and_names_the_leader():
    def row(t, k, n, lo, hi, ok, usd):
        return dict(target=t, k=k, n=n, rate=k / n, lo=lo, hi=hi, commercial_ok=ok, usd=usd, est_usd=None)
    rows = [row("smolvla", 15, 20, 0.53, 0.89, True, 0.18), row("pi", 20, 20, 0.84, 1.0, None, 0.78)]
    r = bench.rank(rows)
    assert [x["target"] for x in r["order"]] == ["pi", "smolvla"]
    assert r["pick"] == "smolvla"
    assert r["text"].startswith("Highest observed: pi 20/20.")


def test_gallery_lookup_returns_the_bucket(monkeypatch):
    import gallery
    monkeypatch.setattr(gallery, "stack_outputs", lambda region, name: {
        "DistributionDomainName": "d.example", "GalleryBucketName": "b"})
    monkeypatch.setattr(gallery, "get_or_create_salt", lambda region, create=True: "salt")
    monkeypatch.setattr(gallery, "get_password", lambda region, gid: "pw")
    assert gallery.lookup("ap-northeast-2", "a@example.com")["bucket"] == "b"
