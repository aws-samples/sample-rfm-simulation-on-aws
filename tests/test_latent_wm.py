"""Latent world-model planners: dino-wm (gaoyuezhou/dino_wm) and jepa-wms (facebookresearch/jepa-wms), shared
templates/latent-wm-userdata.sh.j2 + assets/latent-wm/wm_plan_runner.py.

Covers task / download validation of generate.latent_wm_*, the rendered UserData (pins, downloads per task set,
embedded lock and runner), the runner's pieces (overrides, jepa-wms config patch, the PointMaze stubs), both adapters
end to end against stub upstream trees (no torch / hydra / simulators needed), the template's per-task loop run
through bash + jq with a stub runner, and the summary.txt fields results.py reads.
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
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "assets" / "latent-wm"))

yaml = pytest.importorskip("yaml")
pytest.importorskip("jinja2")

import wm_plan_runner as runner  # noqa: E402
from test_results import BUCKET, FakeS3  # noqa: E402

import generate  # noqa: E402
import results  # noqa: E402
import task_sets  # noqa: E402

BASH = shutil.which("bash")
JQ = shutil.which("jq")
TARGETS = ("dino-wm", "jepa-wms")


def _model(vla):
    return yaml.safe_load((ROOT / "models" / f"{vla}.yaml").read_text())


def _render(vla, out_dir, *extra, ok=True):
    proc = subprocess.run([sys.executable, str(ROOT / "generate.py"), "--vla", vla, "--output-dir", str(out_dir),
                           *extra], cwd=ROOT, capture_output=True, text=True, timeout=120)
    if ok:
        assert proc.returncode == 0, f"{vla} {extra}:\n{proc.stdout}\n{proc.stderr}"
    return proc


def _tasks(vla, task_set=None):
    cfg = task_sets.apply(_model(vla), task_set, vla)
    downloads = generate.latent_wm_downloads(cfg["model"], vla)
    return generate.latent_wm_tasks(cfg, vla, downloads)


# --- task / download validation ------------------------------------------------------------------------------

@pytest.mark.parametrize("vla,task_set,labels,tags", [
    ("dino-wm", None, ["pusht", "wall"], ["push", "navigation"]),
    ("dino-wm", "pusht", ["pusht"], ["push"]),
    ("dino-wm", "probe", ["pusht", "wall"], ["push", "navigation"]),
    ("jepa-wms", None, ["pusht--jepa-wm", "wall--jepa-wm"], ["push", "navigation"]),
    ("jepa-wms", "dino-wm-baseline", ["pusht--dino-wm", "wall--dino-wm"], ["push", "navigation"]),
    ("jepa-wms", "compare", ["pusht--jepa-wm", "pusht--dino-wm", "wall--jepa-wm", "wall--dino-wm"],
     ["push", "push", "navigation", "navigation"]),
])
def test_shipped_tasks(vla, task_set, labels, tags):
    tasks = _tasks(vla, task_set)
    assert [t["label"] for t in tasks] == labels
    assert [t["action_type"] for t in tasks] == tags
    for t in tasks:
        assert t["published"] and t["timeout_min"] >= 30 and t["path_args"][1].startswith("@ASSETS@/")
        full = 10 if vla == "jepa-wms" else 20  # jepa-wms: ~400 s per CEM episode on an L4 (probe)
        assert t["n_episodes"] == (2 if task_set == "probe" else full)


def test_navigation_is_an_action_type_and_wall_maps_to_it():
    assert "navigation" in task_sets.ACTION_TYPES
    assert task_sets.classify("wall") == "navigation" and task_sets.classify("pusht") == "push"


def test_published_numbers_are_the_papers():
    dino = {t["env"]: t["published"] for t in _tasks("dino-wm")}
    assert dino["pusht"].startswith("0.90") and dino["wall"].startswith("0.96")
    jepa = {t["label"]: t["published"] for t in _tasks("jepa-wms", "compare")}
    assert jepa["pusht--jepa-wm"].startswith("70.2") and jepa["wall--jepa-wm"].startswith("78.8")
    assert jepa["pusht--dino-wm"].startswith("66.0") and jepa["wall--dino-wm"].startswith("64.1")


def _cfg(vla, **task_over):
    cfg = copy.deepcopy(_model(vla))
    cfg["tasks"] = [dict(cfg["tasks"][0], **task_over)]
    return cfg


@pytest.mark.parametrize("vla,over,err", [
    ("dino-wm", dict(env="point_maze"), "env must be one of pusht, wall"),
    ("dino-wm", dict(n_episodes=0), "n_episodes must be an integer 1-100"),
    ("dino-wm", dict(n_episodes=True), "n_episodes must be an integer 1-100"),
    ("dino-wm", dict(seed=-1), "seed must be a non-negative integer"),
    ("dino-wm", dict(timeout_min=1000), "timeout_min must be an integer 5-600"),
    ("dino-wm", dict(set=["planner.max_iter=5; rm -rf /"]), "set entries must be key.path=value"),
    ("dino-wm", dict(ckpt="dino-wm-checkpoints"), "ckpt is a jepa-wms key"),
    ("dino-wm", dict(model="a b"), "model is required"),
    ("jepa-wms", dict(model="vj2ac"), "model must be one of jepa-wm, dino-wm"),
    ("jepa-wms", dict(ckpt="ckpt-nope"), "ckpt must name a model.downloads entry"),
    ("jepa-wms", dict(env_name="push t"), "has characters outside"),
])
def test_task_validation_errors(vla, over, err):
    cfg = _cfg(vla, **over)
    with pytest.raises(ValueError, match=re.escape(err)):
        generate.latent_wm_tasks(cfg, vla, generate.latent_wm_downloads(cfg["model"], vla))


def test_duplicate_task_is_rejected():
    cfg = copy.deepcopy(_model("dino-wm"))
    cfg["tasks"] = [cfg["tasks"][0], dict(cfg["tasks"][0])]
    with pytest.raises(ValueError, match="pusht is listed twice"):
        generate.latent_wm_tasks(cfg, "dino-wm", generate.latent_wm_downloads(cfg["model"], "dino-wm"))


@pytest.mark.parametrize("over,err", [
    (dict(url="http://osf.io/x"), "url must be https://"),
    (dict(sha256="abc"), "sha256 must be 64 hex"),
    (dict(file="../x.zip"), "file must be a relative path"),
    (dict(unzip_to=None, unzip_exclude=["a/*"]), "unzip_exclude needs unzip_to"),
    (dict(unzip_exclude=["a b"]), "plain glob patterns"),
    (dict(env="point_maze"), "is not a supported environment"),
])
def test_download_validation_errors(over, err):
    model = copy.deepcopy(_model("dino-wm")["model"])
    model["downloads"][1] = {k: v for k, v in dict(model["downloads"][1], **over).items() if v is not None}
    with pytest.raises(ValueError, match=re.escape(err)):
        generate.latent_wm_downloads(model, "dino-wm")


def test_download_names_are_unique():
    model = copy.deepcopy(_model("jepa-wms")["model"])
    model["downloads"][1]["name"] = model["downloads"][0]["name"]
    with pytest.raises(ValueError, match="is used twice"):
        generate.latent_wm_downloads(model, "jepa-wms")


def test_shipped_downloads_are_pinned():
    for vla in TARGETS:
        model = _model(vla)["model"]
        for d in model["downloads"]:
            assert re.fullmatch(r"[0-9a-f]{64}", d["sha256"]) and d["size"] > 1_000_000
        hub = [d for d in model["downloads"] if "huggingface.co" in d["url"]]
        assert all(f"/resolve/{model['checkpoint_revision']}/" in d["url"] for d in hub)


# --- render --------------------------------------------------------------------------------------------------

def _heredoc(text, tag):
    m = re.search(rf"<< '{tag}'\n(.*?){tag}\n", text, re.S)
    assert m, tag
    return base64.b64decode(m.group(1))


@pytest.mark.parametrize("vla", TARGETS)
def test_render_pins_lock_and_runner(vla, tmp_path):
    _render(vla, tmp_path)
    text = (tmp_path / f"{vla}.sh").read_text()
    model = _model(vla)["model"]
    assert f'git -C "${{CODE_DIR}}" checkout "{model["code_commit"]}"' in text
    assert f'git -C "${{DINOV2_DIR}}" checkout "{model["dinov2_commit"]}"' in text
    assert 'DINOV2_DIR="${TORCH_HOME}/hub/facebookresearch_dinov2_main"' in text
    assert _heredoc(text, "REQEOF") == (ROOT / model["requirements"]).read_bytes()
    assert _heredoc(text, "RUNNEREOF") == (ROOT / "assets" / "latent-wm" / "wm_plan_runner.py").read_bytes()
    assert "world-model planner, goal-image conditioned, not language" in text
    assert "sha256sum -c -" in text and 'cp -f "${ASSETS_DIR}/torch/dinov2_vits14_pretrain.pth"' in text
    assert "export WANDB_MODE=disabled" in text


def _download_files(text):
    return re.findall(r'^log "\[5/7\] (\S+) \(', text, re.M)


@pytest.mark.parametrize("vla,task_set,files", [
    ("dino-wm", None, ["dino_wm_ckpt.zip", "pusht_noise.zip", "wall_single.zip", "torch/dinov2_vits14_pretrain.pth"]),
    ("dino-wm", "pusht", ["dino_wm_ckpt.zip", "pusht_noise.zip", "torch/dinov2_vits14_pretrain.pth"]),
    ("jepa-wms", None, ["jepa-wms/jepa_wm_pusht.pth.tar", "jepa-wms/jepa_wm_wall.pth.tar", "pusht_noise.zip",
                        "wall_single.zip", "torch/dinov2_vits14_pretrain.pth"]),
    ("jepa-wms", "wall", ["jepa-wms/jepa_wm_wall.pth.tar", "wall_single.zip", "torch/dinov2_vits14_pretrain.pth"]),
    ("jepa-wms", "dino-wm-baseline", ["jepa-wms/dino_wm_pusht.pth.tar", "jepa-wms/dino_wm_wall.pth.tar",
                                      "pusht_noise.zip", "wall_single.zip", "torch/dinov2_vits14_pretrain.pth"]),
])
def test_render_downloads_only_what_the_tasks_need(vla, task_set, files, tmp_path):
    _render(vla, tmp_path, *(["--task-set", task_set] if task_set else []))
    assert _download_files((tmp_path / f"{vla}.sh").read_text()) == files


def test_render_pusht_unzip_skips_train_videos(tmp_path):
    _render("dino-wm", tmp_path)
    text = (tmp_path / "dino-wm.sh").read_text()
    line = next(x for x in text.splitlines() if "unzip -q" in x and "pusht_noise.zip" in x)
    assert "'pusht_noise/train/obses/*' 'pusht_noise/train/tokens.pth'" in line
    wall = next(x for x in text.splitlines() if "unzip -q" in x and "wall_single.zip" in x)
    assert "train" not in wall


@pytest.mark.parametrize("mutate,err", [
    (lambda m: m.pop("dinov2_commit"), "model.dinov2_commit is required"),
    (lambda m: m.update(code_commit="main"), "model.code_commit must be a 40-hex commit"),
    (lambda m: m.update(dinov2_weights="nope"), "must name a model.downloads entry"),
    (lambda m: m.pop("ckpt_root"), "model.ckpt_root"),
    (lambda m: m.update(requirements="assets/latent-wm/missing.txt"), "not found"),
    (lambda m: m.update(env={"A": "x y"}), "model.env"),
])
def test_render_rejects_bad_model_fields(mutate, err, tmp_path, capsys):
    cfg = copy.deepcopy(_model("dino-wm"))
    mutate(cfg["model"])
    with pytest.raises(SystemExit):
        generate.generate_latent_wm(cfg, True, "dino-wm")
    assert err in capsys.readouterr().err


def test_locks_pin_the_authors_versions():
    dino = (ROOT / "assets" / "latent-wm" / "dino-wm-requirements.txt").read_text()
    for pin in ("torch==2.3.0", "torchvision==0.18.0", "gym==0.23.1", "hydra-core==1.2.0", "pymunk==6.8.0",
                "decord==0.6.0", "accelerate==0.26.1", "numpy==1.26.4", "setuptools==70.0.0"):
        assert re.search(rf"^{re.escape(pin)}$", dino, re.M), pin
    jepa = (ROOT / "assets" / "latent-wm" / "jepa-wms-requirements.txt").read_text()
    for pin in ("torch==2.7.0", "torchvision==0.22.0", "gym==0.23.1", "pymunk==6.8.0"):
        assert re.search(rf"^{re.escape(pin)}$", jepa, re.M), pin
    for text in (dino, jepa):
        assert not re.search(r"^(d4rl|mujoco-py|metaworld)==", text, re.M)


# --- runner pieces -------------------------------------------------------------------------------------------

def test_parse_set_and_coerce():
    assert runner.parse_set(["planner.max_iter=5", "a.b=true"]) == [("planner.max_iter", "5"), ("a.b", "true")]
    with pytest.raises(ValueError):
        runner.parse_set(["x=$(id)"])
    assert [runner._coerce(v) for v in ("5", "0.5", "true", "null", "cem")] == [5, 0.5, True, None, "cem"]
    d = {"planner": {"iterations": 30}}
    runner.set_path(d, "planner.iterations", 2)
    runner.set_path(d, "meta.seed", 3)
    assert d == {"planner": {"iterations": 2}, "meta": {"seed": 3}}


def test_video_name_and_summarize():
    assert runner.video_name("pusht", 3, True) == "pusht--ep03-success.mp4"
    assert runner.video_name("wall--jepa-wm", 12, False) == "wall--jepa-wm--ep12-failure.mp4"
    res = runner.summarize("dino-wm", "pusht", "pusht", [dict(ep=1, success=False), dict(ep=0, success=True)])
    assert (res["n_success"], res["n_episodes"], res["success_rate"]) == (1, 2, 0.5)
    assert [e["ep"] for e in res["episodes"]] == [0, 1]


def _jepa_params():
    return {
        "folder": "${JEPAWM_LOGS}/x", "tag": "t", "eval_name": "simu_env_planning",
        "meta": {"eval_episodes": 96, "seed": 1, "quick_debug": True},
        "logging": {"optional_plots": True},
        "model_kwargs": {"checkpoint": "jepa-latest.pth.tar", "pretrain_kwargs": {"heads_cfg": {
            "architectures": {"image_head": {"kind": "vit"}, "state_head": {"kind": "vit"}},
            "pretrain_dec_path": {"state_head": "${JEPAWM_LOGS}/pt/x/jepa-latest.pth.tar",
                                  "image_head": "https://dl.fbaipublicfiles.com/jepa-wms/dec.pth.tar"}}}},
        "planner": {"iterations": 30, "decode_each_iteration": True},
    }


def _args(**kw):
    base = dict(adapter="jepa-wms", env="pusht", model="jepa-wm", n_episodes=4, seed=7, label="pusht--jepa-wm",
                image_head=None)
    return type("A", (), dict(base, **kw))()


def test_patch_jepa_wms_config_drops_unreleased_heads(tmp_path):
    notes = []
    p = runner.patch_jepa_wms_config(_jepa_params(), _args(), Path("/x/ck.pth.tar"), tmp_path,
                                     [("planner.iterations", "2")], notes)
    assert p["model_kwargs"]["checkpoint"] == "/x/ck.pth.tar" and p["folder"] == str(tmp_path / "logs")
    assert p["meta"] == {"eval_episodes": 4, "seed": 7, "quick_debug": False}
    assert p["model_kwargs"]["pretrain_kwargs"]["heads_cfg"] == {}   # a mapping: init_module calls .get on it
    assert p["planner"] == {"iterations": 2, "decode_each_iteration": False}
    assert p["logging"]["optional_plots"] is False
    assert any("state_head" in n for n in notes)


def test_patch_jepa_wms_config_keeps_a_local_image_head(tmp_path):
    p = runner.patch_jepa_wms_config(_jepa_params(), _args(image_head=str(tmp_path / "dec.pth.tar")),
                                     Path("/x/ck"), tmp_path, [], [])
    heads = p["model_kwargs"]["pretrain_kwargs"]["heads_cfg"]
    assert list(heads["architectures"]) == ["image_head"]
    assert heads["pretrain_dec_path"] == {"image_head": str(tmp_path / "dec.pth.tar")}
    assert p["planner"]["decode_each_iteration"] is True


def test_jepa_wms_config_needs_exactly_one_released_file(tmp_path):
    d = tmp_path / "configs/evals/simu_env_planning/pt/jepa-wm"
    d.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="expected one released eval config"):
        runner.jepa_wms_config(tmp_path, "pusht", "jepa-wm")
    (d / "pt_x.yaml").write_text("a: 1\n")
    assert runner.jepa_wms_config(tmp_path, "pusht", "jepa-wm").name == "pt_x.yaml"


def test_pointmaze_stubs_let_the_env_package_import(tmp_path):
    code = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {str(ROOT / 'assets' / 'latent-wm')!r})
        import wm_plan_runner as r
        notes = []
        r.stub_pointmaze_deps(notes)
        from d4rl import offline_env
        import mujoco_py
        try:
            offline_env.OfflineEnv()
        except RuntimeError as e:
            print("raised:", e)
        print(len(notes))
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "raised: d4rl is stubbed" in out.stdout and out.stdout.strip().endswith("2")


def test_parse_args_errors(capsys):
    base = ["--code-dir", "c", "--env", "pusht", "--model", "m", "--data-root", "d", "--n-episodes", "1",
            "--seed", "0", "--out", "o", "--label", "l"]
    with pytest.raises(SystemExit, match="--ckpt-root is required"):
        runner.main(["--adapter", "dino-wm", *base])
    with pytest.raises(SystemExit, match="--ckpt is required"):
        runner.main(["--adapter", "jepa-wms", *base])
    with pytest.raises(SystemExit, match="--model must be one of"):
        runner.main(["--adapter", "jepa-wms", "--ckpt", "x", *base])
    with pytest.raises(SystemExit):
        runner.main(["--adapter", "dino-wm", *base[:2], "--env", "point_maze_x", *base[4:]])


# --- adapters end to end against stub upstream trees -----------------------------------------------------------

DINO_STUB_PLAN = '''
import os
from planning.evaluator import PlanEvaluator
def planning_main(cfg_dict):
    n = cfg_dict["n_evals"]
    ok = [i % 2 == 0 for i in range(n)]
    ev = PlanEvaluator()
    ev.ok = ok
    ev.eval_actions(None, filename="plan0")
    ev.eval_actions(None, filename="output_final", save_video=True)
    with open("seen.txt", "w") as f:
        f.write(os.getcwd() + "\\n" + repr(sorted(cfg_dict)))
    return {"final_eval/success_rate": sum(ok) / n}
'''
DINO_STUB_EVAL = '''
class PlanEvaluator:
    ok = ()

    def eval_actions(self, actions, action_len=None, filename="output", save_video=False):
        ok = list(self.ok) if filename == "output_final" else [False] * len(self.ok)
        if save_video:
            for i, s in enumerate(ok):
                open(f"{filename}_{i}_{'success' if s else 'failure'}.mp4", "wb").write(b"mp4")
        return {}, ok, None, None
'''


def test_dino_wm_adapter_end_to_end(tmp_path, monkeypatch):
    code = tmp_path / "dino_wm"
    (code / "planning").mkdir(parents=True)
    (code / "plan.py").write_text(DINO_STUB_PLAN)
    (code / "planning" / "__init__.py").write_text("")
    (code / "planning" / "evaluator.py").write_text(DINO_STUB_EVAL)
    seen = {}

    def compose(c, name, overrides):
        seen.update(code=c, name=name, overrides=overrides)
        return {"n_evals": int(next(o for o in overrides if o.startswith("n_evals=")).split("=")[1]),
                "planner": {"name": "mpc_cem"}, "goal_source": "dset", "goal_H": 5, "objective": {}}

    monkeypatch.setattr(runner, "compose_plan_cfg", compose)
    monkeypatch.setattr(runner, "shim_decord", lambda notes: None)
    monkeypatch.setattr(runner, "stub_pointmaze_deps", lambda notes: None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    for k in ("DATASET_DIR", "WANDB_MODE"):
        monkeypatch.setenv(k, "unset")
    for m in ("plan", "planning", "planning.evaluator"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    rc = runner.main(["--adapter", "dino-wm", "--code-dir", str(code), "--env", "pusht", "--model", "pusht",
                      "--ckpt-root", str(tmp_path / "ck"), "--data-root", str(tmp_path / "data"), "--n-episodes", "3",
                      "--seed", "99", "--out", str(out), "--label", "pusht", "--set", "planner.max_iter=5"])
    assert rc == 0
    assert seen["name"] == "plan_pusht"
    assert "model_name=pusht" in seen["overrides"] and "planner.max_iter=5" in seen["overrides"]
    assert "n_plot_samples=3" in seen["overrides"] and "seed=99" in seen["overrides"]
    res = json.loads((out / "results.json").read_text())
    assert (res["n_success"], res["n_episodes"]) == (2, 3)
    assert [e["video"] for e in res["episodes"]] == ["pusht--ep00-success.mp4", "pusht--ep01-failure.mp4",
                                                     "pusht--ep02-success.mp4"]
    assert [e["seed"] for e in res["episodes"]] == [1, 100, 199]
    assert sorted(p.name for p in (out / "videos").iterdir()) == [e["video"] for e in res["episodes"]]
    assert (out / "seen.txt").read_text().startswith(str(out))   # planning ran in the output dir
    assert res["upstream"]["final_eval/success_rate"] == pytest.approx(2 / 3)
    assert os.environ["DATASET_DIR"] == str((tmp_path / "data").resolve())


JEPA_STUB_FILES = {
    "evals/__init__.py": "",
    "evals/scaffold.py": '''
import types
from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
def main(eval_name, args_eval, resume_preempt=False):
    assert eval_name == "simu_env_planning"
    from pathlib import Path
    work = Path(args_eval["folder"]) / "simu_env_planning" / args_eval["tag"]
    cfg = types.SimpleNamespace(tasks=["pusht-base"], work_dir=work)
    ev = PlanEvaluator()
    n = args_eval["meta"]["eval_episodes"]
    for ep in list(range(n)) + [0]:            # the authors pad with a duplicate dummy episode
        ev.eval(cfg, None, None, task_idx=0, ep=ep)
    (work).mkdir(parents=True, exist_ok=True)
    (work / "eval.csv").write_text("episode_success+pusht-base,total_time\\n0.5,1\\n")
''',
    "evals/simu_env_planning/__init__.py": "",
    "evals/simu_env_planning/planning/__init__.py": "",
    "evals/simu_env_planning/planning/plan_evaluator.py": '''
class PlanEvaluator:
    def eval(self, cfg, agent, env, task_idx=-1, ep=0):
        success = ep % 2 == 1
        vis = cfg.work_dir / cfg.tasks[task_idx] / f"ep_{ep}" / "visualisation"
        vis.mkdir(parents=True, exist_ok=True)
        (vis / f"video_agent_goal_{'succ' if success else 'fail'}.mp4").write_bytes(b"mp4")
        return (1, success, 0.0, 0.0, 0.0, -1, -1, -1, 0.0, -1, -1)
''',
    "src/__init__.py": "",
    "src/utils/__init__.py": "",
    "src/utils/distributed.py": "def init_distributed(rank_and_world_size=(None, None)):\n    return 1, 0\n",
    "configs/evals/simu_env_planning/pt/jepa-wm/pt_L2_cem.yaml": yaml.safe_dump({
        "folder": "${JEPAWM_LOGS}/pt", "tag": "x", "eval_name": "simu_env_planning",
        "meta": {"eval_episodes": 96, "seed": 1}, "logging": {"optional_plots": True},
        "model_kwargs": {"checkpoint": "jepa-latest.pth.tar", "pretrain_kwargs": {"heads_cfg": {
            "architectures": {"state_head": {}}, "pretrain_dec_path": {"state_head": "/nope"}}}},
        "planner": {"iterations": 30, "decode_each_iteration": True}}),
}


def test_jepa_wms_adapter_end_to_end(tmp_path, monkeypatch):
    code = tmp_path / "jepa-wms"
    for rel, body in JEPA_STUB_FILES.items():
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        (code / rel).write_text(body)
    monkeypatch.setattr(runner, "shim_decord", lambda notes: None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    for k in ("JEPAWM_DSET", "JEPAWM_LOGS", "JEPAWM_HOME", "WANDB_MODE"):
        monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv(k, "")
        monkeypatch.delenv(k)
    for m in [m for m in sys.modules if m == "evals" or m.startswith(("evals.", "src."))] + ["src"]:
        monkeypatch.delitem(sys.modules, m, raising=False)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    rc = runner.main(["--adapter", "jepa-wms", "--code-dir", str(code), "--env", "pusht", "--model", "jepa-wm",
                      "--ckpt", str(tmp_path / "ck.pth.tar"), "--data-root", str(tmp_path / "data"),
                      "--n-episodes", "3", "--seed", "5", "--out", str(out), "--label", "pusht--jepa-wm",
                      "--set", "planner.iterations=2"])
    assert rc == 0
    res = json.loads((out / "results.json").read_text())
    assert (res["n_success"], res["n_episodes"]) == (1, 3)   # the padded duplicate is not counted
    assert [e["video"] for e in res["episodes"]] == ["pusht--jepa-wm--ep00-failure.mp4",
                                                     "pusht--jepa-wm--ep01-success.mp4",
                                                     "pusht--jepa-wm--ep02-failure.mp4"]
    assert res["upstream"]["episode_success+pusht-base"] == "0.5"
    assert res["upstream"]["config"] == "configs/evals/simu_env_planning/pt/jepa-wm/pt_L2_cem.yaml"
    patched = yaml.safe_load((out / "eval_config.yaml").read_text())
    assert patched["planner"]["iterations"] == 2 and patched["meta"]["eval_episodes"] == 3
    assert patched["model_kwargs"]["checkpoint"] == str((tmp_path / "ck.pth.tar").resolve())
    assert os.environ["JEPAWM_DSET"] == str((tmp_path / "data").resolve())


# --- the template's per-task loop through bash + jq -------------------------------------------------------------

FAKE_RUNNER_PY = r"""#!/usr/bin/env python3
import json, os, sys
argv = sys.argv[2:]
open(os.environ["FAKE_CALLS"], "a").write(" ".join(argv) + "\n")
if os.environ.get("FAKE_FAIL"):
    sys.exit(3)
a = dict(zip(argv[::2], argv[1::2]))
out, label, n = a["--out"], a["--label"], int(a["--n-episodes"])
n_done = n + int(os.environ.get("FAKE_EXTRA", "0"))
os.makedirs(out + "/videos", exist_ok=True)
eps = []
for i in range(n_done):
    ok = i % 4 == 0
    v = f"{label}--ep{i:02d}-{'success' if ok else 'failure'}.mp4"
    open(f"{out}/videos/{v}", "wb").write(b"mp4")
    eps.append(dict(ep=i, success=ok, video=v))
json.dump(dict(n_success=sum(e["success"] for e in eps), n_episodes=n_done, episodes=eps,
               upstream=dict(wall_seconds=12.5), notes=["stub"]), open(out + "/results.json", "w"))
"""


def _run_loop(tmp_path, vla, task_set=None, **env):
    _render(vla, tmp_path / "r", *(["--task-set", task_set] if task_set else []))
    text = (tmp_path / "r" / f"{vla}.sh").read_text()
    block = text[text.index("# ── [6/7]"):text.index("# ── [7/7]")].replace("/tmp/", f"{tmp_path}/tmp/")
    fake = tmp_path / "fakepy"
    fake.write_text(FAKE_RUNNER_PY)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "aws").write_text("#!/bin/bash\nexit 0\n")
    (tmp_path / "bin" / "aws").chmod(0o755)
    prelude = f"""set -euo pipefail
export PATH="{tmp_path}/bin:$PATH"
mkdir -p "{tmp_path}/tmp" "{tmp_path}/code"
log() {{ echo "[LOG] $*"; }}
die() {{ echo "[DIE] $*"; exit 7; }}
_trap_handler() {{ :; }}
trap '_trap_handler' ERR
PY="{fake}"; RUNNER="{tmp_path}/wm_plan_runner.py"; CODE_DIR="{tmp_path}/code"
ASSETS_DIR="{tmp_path}/assets"; DATA_DIR="{tmp_path}/assets/data"
EVAL_BASE_DIR="{tmp_path}/eval"; VIDEO_BASE_DIR="{tmp_path}/videos"
S3_BUCKET=b; RUN_ID=R; REGION=us-east-1; INSTANCE_ID=i-0; INSTANCE_TYPE=g6e.xlarge; STACK_NAME=S
STACK_VERSIONS="torch==2.3.0"; GPU_NAME="NVIDIA L40S"; CODE_HEAD=c; DINOV2_HEAD=d
"""
    (tmp_path / "loop.sh").write_text(prelude + block)
    e = dict(os.environ, FAKE_CALLS=str(tmp_path / "calls.txt"), **env)
    return subprocess.run([BASH, str(tmp_path / "loop.sh")], capture_output=True, text=True, env=e, timeout=120)


def _kv(path):
    return dict(line.split(": ", 1) for line in path.read_text().splitlines() if ": " in line)


@pytest.mark.skipif(BASH is None or JQ is None, reason="bash / jq not on PATH")
def test_task_loop_dino_wm_writes_one_summary_per_task(tmp_path):
    proc = _run_loop(tmp_path, "dino-wm")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    assert len(calls) == 2
    assert "--adapter dino-wm" in calls[0] and "--env pusht --model pusht" in calls[0]
    assert f"--ckpt-root {tmp_path}/assets/dino_wm_ckpt" in calls[0] and "--set planner.max_iter=5" in calls[0]
    assert "--env wall --model wall_single" in calls[1]
    for i, (env, tag) in enumerate([("pusht", "push"), ("wall", "navigation")]):
        kv = _kv(tmp_path / "tmp" / f"dino-wm_task{i}_summary.txt")
        assert (kv["env_name"], kv["suite"], kv["action_type"]) == (env, env, tag)
        assert (kv["n_success"], kv["n_episodes"], kv["success_rate"]) == ("5", "20", "0.250")
        assert kv["model_kind"] == "world-model planner, goal-image conditioned, not language"
        assert kv["instruction"] == "none (goal image)" and kv["runner_notes"] == "stub"
        assert kv["published"].startswith("0.9") and kv["wall_seconds"] == "12.5"
        assert kv["stack_eval_args"].startswith("python wm_plan_runner.py --adapter dino-wm")
        vids = sorted(p.name for p in (tmp_path / "videos" / f"task-{i}").iterdir())
        assert len(vids) == 20 and vids[0] == f"{env}--ep00-success.mp4"


@pytest.mark.skipif(BASH is None or JQ is None, reason="bash / jq not on PATH")
def test_task_loop_jepa_wms_compare_labels_and_checkpoints(tmp_path):
    proc = _run_loop(tmp_path, "jepa-wms", "compare")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    assert [re.search(r"--label (\S+)", c).group(1) for c in calls] == \
        ["pusht--jepa-wm", "pusht--dino-wm", "wall--jepa-wm", "wall--dino-wm"]
    assert f"--ckpt {tmp_path}/assets/jepa-wms/dino_wm_pusht.pth.tar" in calls[1]
    kv = _kv(tmp_path / "tmp" / "jepa-wms_task1_summary.txt")
    assert kv["condition"] == "wm=dino-wm" and kv["wm_model"] == "dino-wm" and kv["env_name"] == "pusht"
    assert kv["checkpoint"] == "facebook/jepa-wms@9b9c41ef/dino_wm_pusht.pth.tar"


@pytest.mark.skipif(BASH is None or JQ is None, reason="bash / jq not on PATH")
def test_task_loop_fails_when_the_episode_count_differs(tmp_path):
    proc = _run_loop(tmp_path, "dino-wm", "pusht", FAKE_EXTRA="1")
    assert proc.returncode == 7
    assert "[DIE] Task 0 ran 21 episodes, expected 20" in proc.stdout


@pytest.mark.skipif(BASH is None or JQ is None, reason="bash / jq not on PATH")
def test_task_loop_fails_when_the_runner_fails(tmp_path):
    proc = _run_loop(tmp_path, "jepa-wms", "wall", FAKE_FAIL="1")
    assert proc.returncode == 7
    assert "[DIE] task 0 (wall--jepa-wm) failed (exit=3)" in proc.stdout


# --- results.py reads the summary the template writes ------------------------------------------------------------

def test_results_reads_env_name_condition_and_action_type():
    s3 = FakeS3({BUCKET: {
        "R/task-0/summary.txt": "model: jepa-wms\nsuite: wall\nenv_name: wall\naction_type: navigation\n"
                                "condition: wm=jepa-wm\nsuccess_rate: 0.750\nn_success: 15\nn_episodes: 20\n",
    }})
    r0 = results.task_row(s3, "us-east-1", BUCKET, "R", "R/task-0/")
    assert (r0["task_name"], r0["action_type"], r0["suite"], r0["k"], r0["n"]) == ("wall", "navigation", "wall", 15, 20)
    assert r0["condition"] == "wm=jepa-wm"
