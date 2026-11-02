"""LIBERO targets that run the authors' own LIBERO loop: univla (baaivision/UniVLA) and ge-act (AgibotTech/
Genie-Envisioner), shared templates/libero-own-userdata.sh.j2 + assets/libero/own_eval_runner.py.

Covers the task validation and run grouping of generate.libero_own_*, the rendered UserData (pins, downloads, embedded
lock and runner, per-suite arguments), the runner's pieces (exactly-once source patches, task subset, recorder,
outcome pairing), both adapters end to end against stub upstream trees (no torch / LIBERO needed), the template's
per-task loop run through bash + jq with a stub runner, and the summary.txt fields results.py reads.
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
sys.path.insert(0, str(ROOT / "assets" / "libero"))

yaml = pytest.importorskip("yaml")
pytest.importorskip("jinja2")

import own_eval_runner as runner  # noqa: E402
from test_results import BUCKET, FakeS3  # noqa: E402

import generate  # noqa: E402
import results  # noqa: E402

BASH = shutil.which("bash")
JQ = shutil.which("jq")
TARGETS = ("univla", "ge-act")
DEFAULT_IDS = [1, 6, 8]
ACTION_IDS = [0, 5, 7]


def _model(vla):
    return yaml.safe_load((ROOT / "models" / f"{vla}.yaml").read_text())


def _render(vla, out_dir, *extra, ok=True):
    proc = subprocess.run([sys.executable, str(ROOT / "generate.py"), "--vla", vla, "--output-dir", str(out_dir),
                           *extra], cwd=ROOT, capture_output=True, text=True, timeout=120)
    if ok:
        assert proc.returncode == 0, f"{vla} {extra}:\n{proc.stdout}\n{proc.stderr}"
    return proc


def _cfg(vla, tasks):
    cfg = copy.deepcopy(_model(vla))
    cfg["tasks"] = tasks
    return cfg


# --- task validation and run grouping ------------------------------------------------------------------------

@pytest.mark.parametrize("vla", TARGETS)
@pytest.mark.parametrize("task_set,ids,tags", [
    (None, DEFAULT_IDS, ["pick-place"] * 3),
    ("actions", ACTION_IDS, ["articulated", "push", "articulated"]),
])
def test_shipped_tasks(vla, task_set, ids, tags):
    import task_sets
    cfg = task_sets.apply(_model(vla), task_set, vla)
    tasks = generate.libero_own_tasks(cfg, vla)
    assert [t["task_id"] for t in tasks] == ids
    assert [t["env_name"] for t in tasks] == [generate.LIBERO_TASKS["libero_goal"][i] for i in ids]
    assert [t["action_type"] for t in tasks] == tags
    assert all(t["n_episodes"] == 10 and t["suite"] == "libero_goal" for t in tasks)
    runs = generate.libero_own_runs(tasks)
    assert len(runs) == 1 and runs[0]["task_ids"] == ",".join(map(str, ids)) and runs[0]["tasks"] == [0, 1, 2]


def test_libero_task_table_shape():
    assert set(generate.LIBERO_TASKS) == {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
    for names in generate.LIBERO_TASKS.values():
        assert len(names) == 10 and len(set(names)) == 10
        assert all(re.fullmatch(r"[A-Za-z0-9_]+", n) for n in names)
    assert generate.LIBERO_TASKS["libero_object"][3] == "pick_up_the_bbq_sauce_and_place_it_in_the_basket"


@pytest.mark.parametrize("task,err", [
    ({"suite": "libero_90", "task_id": 0, "n_episodes": 1}, "suite must be one of"),
    ({"suite": "libero_goal", "task_id": 10, "n_episodes": 1}, "task_id must be an integer 0-9"),
    ({"suite": "libero_goal", "task_id": "1", "n_episodes": 1}, "task_id must be an integer"),
    ({"suite": "libero_goal", "task_id": 1, "n_episodes": 0}, "n_episodes must be an integer 1-50"),
    ({"suite": "libero_goal", "task_id": 1, "n_episodes": 51}, "n_episodes must be an integer 1-50"),
    ({"suite": "libero_goal", "task_id": 1, "n_episodes": 2, "env_name": "turn_on_the_stove"}, "is not task 1"),
    ({"suite": "libero_goal", "task_id": 1, "n_episodes": 2, "action_type": "flying"}, "action_type"),
])
def test_task_validation_errors(task, err):
    with pytest.raises(ValueError, match=err):
        generate.libero_own_tasks(_cfg("univla", [task]), "univla")


def test_duplicate_task_is_rejected():
    t = {"suite": "libero_goal", "task_id": 1, "n_episodes": 2}
    with pytest.raises(ValueError, match="listed twice"):
        generate.libero_own_tasks(_cfg("univla", [t, dict(t)]), "univla")


def test_runs_group_consecutive_tasks_of_one_suite_and_episode_count():
    tasks = generate.libero_own_tasks(_cfg("univla", [
        {"suite": "libero_goal", "task_id": 1, "n_episodes": 5},
        {"suite": "libero_goal", "task_id": 2, "n_episodes": 5},
        {"suite": "libero_goal", "task_id": 3, "n_episodes": 3},   # other count -> new run
        {"suite": "libero_10", "task_id": 0, "n_episodes": 3},     # other suite -> new run
        {"suite": "libero_goal", "task_id": 4, "n_episodes": 3},   # not consecutive with run 2 -> new run
    ]), "univla")
    runs = generate.libero_own_runs(tasks)
    assert [(r["suite"], r["task_ids"], r["n_episodes"], r["tasks"]) for r in runs] == [
        ("libero_goal", "1,2", 5, [0, 1]), ("libero_goal", "3", 3, [2]), ("libero_10", "0", 3, [3]),
        ("libero_goal", "4", 3, [4])]
    assert runs[1]["expect_names"] == "open_the_top_drawer_and_put_the_bowl_inside"


@pytest.mark.parametrize("vla,pins", [
    ("univla", {"torch": "2.4.0", "transformers": "4.44.0", "tiktoken": "0.6.0", "robosuite": "1.4.0",
                "mujoco": "3.3.1", "numpy": "1.26.4", "moviepy": "1.0.3"}),
    ("ge-act", {"torch": "2.7.1", "diffusers": "0.32.0", "transformers": "4.51.3", "accelerate": "1.0.0",
                "robosuite": "1.4.0", "mujoco": "3.3.1", "numpy": "1.26.4"}),
])
def test_lock_pins(vla, pins):
    lock = (ROOT / _model(vla)["model"]["requirements"]).read_text()
    got = dict(re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)", lock, re.M))
    for pkg, ver in pins.items():
        assert got.get(pkg) == ver, (vla, pkg, got.get(pkg))
    assert "termcolor" in got           # robosuite 1.4.0 imports it without declaring it
    if vla == "univla":
        assert "flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310" in lock


# --- rendered UserData ---------------------------------------------------------------------------------------

def _heredoc(text, tag):
    m = re.search(rf"<< '{tag}'\n(.*?){tag}\n", text, re.S)
    assert m, tag
    return base64.b64decode(m.group(1))


@pytest.mark.parametrize("vla", TARGETS)
@pytest.mark.parametrize("task_set", [None, "actions"])
def test_render_pins_lock_and_runner(vla, task_set, tmp_path):
    _render(vla, tmp_path, *(["--task-set", task_set] if task_set else []))
    text = (tmp_path / f"{vla}.sh").read_text()
    m = _model(vla)["model"]
    assert subprocess.run(["bash", "-n", str(tmp_path / f"{vla}.sh")]).returncode == 0
    for pin in (m["code_commit"], m["libero_commit"], m["revision"][:8]):
        assert pin in text
    assert f'checkout "{m["code_commit"]}"' in text and f'checkout "{m["libero_commit"]}"' in text
    assert _heredoc(text, "REQEOF") == (ROOT / m["requirements"]).read_bytes()
    assert _heredoc(text, "RUNNEREOF") == (ROOT / "assets" / "libero" / "own_eval_runner.py").read_bytes()
    assert f'--adapter "{m["adapter"]}"' in text
    assert "{suite}" not in text and "{assets}" not in text
    assert "--no-deps -e \"${LIBERO_DIR}\" --config-settings editable_mode=compat" in text
    assert "protocol.yaml" in text                         # protocol card macro rendered
    ids = ACTION_IDS if task_set else DEFAULT_IDS
    assert f'--task-ids "{",".join(map(str, ids))}"' in text
    for d in m["downloads"]:
        if "repo" in d:
            assert f"snapshot_download('{d['repo']}', revision='{d['revision']}'" in text


def test_render_ge_act_downloads_only_the_suites_it_runs(tmp_path):
    _render("ge-act", tmp_path)
    text = (tmp_path / "ge-act.sh").read_text()
    shas = {d["suite"]: d["sha256"] for d in _model("ge-act")["model"]["downloads"] if "suite" in d}
    assert shas["libero_goal"] in text
    assert all(shas[s] not in text for s in ("libero_object", "libero_spatial", "libero_10"))
    assert "export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1" in text
    assert '"--ckpt" "/home/ubuntu/ge-act/assets/ge-act/ge_act_libero_goal.safetensors" "--exec-step" "8" ' \
           '"--threshold" "20"' in text
    assert "allow_patterns=['model_index.json', 'tokenizer/*', 'text_encoder/*', 'vae/*']" in text


def test_render_univla_args_and_pythonpath(tmp_path):
    _render("univla", tmp_path)
    text = (tmp_path / "univla.sh").read_text()
    assert '"--emu-hub" "/home/ubuntu/univla/assets/UniVLA/UNIVLA_LIBERO_IMG_BS192_8K"' in text
    assert 'export PYTHONPATH="${CODE_DIR}/reference/Emu3:${CODE_DIR}/reference/RoboVLMs/eval/libero:${CODE_DIR}"' \
        in text
    assert "allow_patterns=['UNIVLA_LIBERO_IMG_BS192_8K/*']" in text
    assert "allow_patterns=None" in text                  # the vision tokenizer is fetched whole


@pytest.mark.parametrize("mutate,err", [
    (lambda m: m.update(code_commit="main"), "code_commit must be a 40-hex commit"),
    (lambda m: m.update(adapter="openvla"), "model.adapter must be one of"),
    (lambda m: m["downloads"][1].update(sha256="abc"), "need an https url"),
    (lambda m: m["downloads"][1].update(url="http://example.com/x"), "need an https url"),
    (lambda m: m["downloads"][0].update(revision="main"), "40-hex revision"),
    (lambda m: m.update(runner_args=["--ckpt", "$(reboot)"]), "unsafe token"),
    (lambda m: m.update(env={"lower": "1"}), "model.env"),
    (lambda m: m["downloads"][1].update(file="../etc/x"), "relative `dir`"),
])
def test_render_rejects_bad_model_fields(mutate, err, tmp_path, capsys):
    cfg = copy.deepcopy(_model("ge-act"))
    mutate(cfg["model"])
    with pytest.raises(SystemExit):
        generate.generate_libero_own(cfg, False, "ge-act")
    assert err in capsys.readouterr().err


# --- runner pieces -------------------------------------------------------------------------------------------

def test_load_patched_requires_each_patch_exactly_once(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("A = 'x'\nB = 'y'\nB2 = 'y'\n")
    mod = runner.load_patched(f, "patched_mod_test", [("A = 'x'", "A = 'z'")])
    assert mod.A == "z" and sys.modules["patched_mod_test"] is mod
    with pytest.raises(RuntimeError, match="occurs 2 times"):
        runner.load_patched(f, "patched_mod_test2", [("'y'", "'q'")])
    with pytest.raises(RuntimeError, match="occurs 0 times"):
        runner.load_patched(f, "patched_mod_test3", [("nope", "")])
    sys.modules.pop("patched_mod_test", None)


class _Task:
    def __init__(self, name):
        self.name, self.language = name, name.replace("_", " ")


class _Suite:
    def __init__(self, names):
        self.tasks = [_Task(n) for n in names]
        self.n_tasks = len(names)

    def get_task(self, i):
        return self.tasks[i]

    def get_task_init_states(self, i):
        return [f"init-{i}-{k}" for k in range(50)]


def test_suite_subset_maps_local_to_real_ids():
    sub = runner.SuiteSubset(_Suite(generate.LIBERO_TASKS["libero_goal"]), [7, 0])
    assert sub.n_tasks == 2 and sub.get_num_tasks() == 2
    assert sub.get_task(0).name == "turn_on_the_stove" and sub.get_task_init_states(1)[3] == "init-0-3"
    assert sub.get_task_names() == ["turn_on_the_stove", "open_the_middle_drawer_of_the_cabinet"]
    with pytest.raises(ValueError, match="out of range"):
        runner.SuiteSubset(_Suite(["a"]), [1])


def test_recorder_splits_episodes_on_reset_and_flips_frames():
    np = pytest.importorskip("numpy")

    class Env:
        def reset(self):
            return {}

        def step(self, a):
            return {"agentview_image": np.arange(4).reshape(2, 2)}, 0.0, False, {}

        def set_init_state(self, s):
            return {"agentview_image": np.zeros((2, 2))}

    rec = runner.Recorder()
    get_env = runner.recording_get_libero_env(lambda task, **kw: (Env(), task.language), rec)
    env, desc = get_env(_Task("put_the_bowl_on_the_plate"), resolution=256)
    assert desc == "put the bowl on the plate"
    env.set_init_state("s")              # proxied, not recorded
    env.step(None)                       # before any reset: ignored
    env.reset()
    env.step(None)
    env.step(None)
    env.reset()                          # second episode, no frames yet
    assert [len(e["frames"]) for e in rec.episodes] == [2, 0]
    assert rec.episodes[0]["frames"][0].tolist() == [[3, 2], [1, 0]]   # rotated 180 degrees


def test_read_outcomes_and_collect(tmp_path):
    log = tmp_path / "l.txt"
    log.write_text("Task: x\nSuccess: True\n# successes: 1\nSuccess: False\nnot Success: True\nSuccess: True\n")
    assert runner.read_outcomes([log]) == [True, False, True]
    rec = runner.Recorder()
    for name in ("a", "a", "b"):
        rec.begin(_Task(name))
        rec.episodes[-1]["frames"] = [0]
    with pytest.raises(ValueError, match="expected 2 tasks x 1"):
        runner.collect(rec, [True, False, True], [3, 4], 1, ["a", "b"])
    with pytest.raises(ValueError, match="recorded episodes but 2"):
        runner.collect(rec, [True, False], [3, 4], 1, ["a", "b"])
    rec.episodes.append(dict(task_name="b", language="b", frames=[]))
    tasks = runner.collect(rec, [True, False, True, False], [3, 4], 2, ["a", "b"])
    assert [(t["task_id"], [e["success"] for e in t["episodes"]]) for t in tasks] == [(3, [True, False]),
                                                                                      (4, [True, False])]
    assert tasks[1]["episodes"][1]["video"] == "b--episode_1-failure.mp4"
    with pytest.raises(ValueError, match=r"episodes of \[.b.\], expected c"):
        runner.collect(rec, [True] * 4, [3, 4], 2, ["a", "c"])


def test_write_results_skips_frameless_episodes(tmp_path):
    written = []
    tasks = [dict(task_id=1, task_name="t", language="l", episodes=[
        dict(episode=0, success=True, frames=2, video="t--episode_0-success.mp4", _frames=[1, 2]),
        dict(episode=1, success=False, frames=0, video="t--episode_1-failure.mp4", _frames=[])])]
    out = runner.write_results(tmp_path, "univla", "libero_goal", tasks,
                               writer=lambda p, f, fps: written.append((Path(p).name, len(f), fps)))
    assert written == [("t--episode_0-success.mp4", 2, 20)]
    data = json.loads((tmp_path / "results.json").read_text())
    assert data["tasks"][0]["n_success"] == 1 and data["tasks"][0]["n_episodes"] == 2
    assert [e["video"] for e in data["tasks"][0]["episodes"]] == ["t--episode_0-success.mp4", ""]
    assert out[0]["episodes"][0].get("_frames") is None


def test_unloaded_tensors():
    assert runner.unloaded_tensors({"a": [2, 3], "b": [4], "c": [1]}, {"a": (2, 3), "b": (5,)}) == ["b", "c"]
    assert runner.unloaded_tensors({"a": [2]}, {"a": (2,), "extra": (1,)}) == []


@pytest.mark.parametrize("argv,err", [
    (["--task-ids", "1,2", "--expect-names", "a"], "same, non-zero number"),
    (["--n-episodes", "0"], "--n-episodes must be >= 1"),
    (["--expect-names", "a b"], "must match"),
])
def test_parse_args_errors(argv, err, capsys):
    base = {"--adapter": "univla", "--code-dir": "c", "--suite": "libero_goal", "--task-ids": "1",
            "--expect-names": "a", "--n-episodes": "1", "--out": "o", "--emu-hub": "e", "--vq-hub": "q",
            "--vision-hub": "v"}
    for k, v in zip(argv[::2], argv[1::2]):
        base[k] = v
    with pytest.raises(SystemExit):
        runner.parse_args([x for kv in base.items() for x in kv])
    assert err in capsys.readouterr().err


def test_parse_args_needs_the_adapter_paths(capsys):
    with pytest.raises(SystemExit):
        runner.parse_args(["--adapter", "ge-act", "--code-dir", "c", "--suite", "libero_goal", "--task-ids", "1",
                           "--expect-names", "a", "--n-episodes", "1", "--out", "o", "--ltx-dir", "x"])
    assert "--adapter ge-act needs --ckpt" in capsys.readouterr().err


# --- both adapters end to end against stub upstream trees -----------------------------------------------------

STUB_LIBERO = """
import os
NAMES = {names!r}
class Task:
    def __init__(self, name):
        self.name = name
        self.language = name.replace('_', ' ')
class Suite:
    def __init__(self, suite):
        self.tasks = [Task(n) for n in NAMES[suite]]
        self.n_tasks = len(self.tasks)
    def get_task(self, i):
        return self.tasks[i]
    def get_task_init_states(self, i):
        return list(range(50))
class benchmark:
    @staticmethod
    def get_benchmark_dict():
        return {{s: (lambda s=s: Suite(s)) for s in NAMES}}
"""

# Each episode succeeds when OUTCOMES[global episode index] == '1' (env var STUB_OUTCOMES).
STUB_ENV = """
import os
import numpy as np
COUNT = [0]
class Inner:
    done = False
class Env:
    def __init__(self, task):
        self.task, self.env, self.t = task, Inner(), 0
    def reset(self):
        self.t = 0
        self.ep = COUNT[0]
        COUNT[0] += 1
        return {}
    def set_init_state(self, s):
        return {'agentview_image': np.zeros((4, 4, 3), dtype=np.uint8)}
    def step(self, a):
        self.t += 1
        win = os.environ['STUB_OUTCOMES'][self.ep] == '1'
        done = win and self.t >= 3
        self.env.done = done
        return {'agentview_image': np.full((4, 4, 3), self.t, dtype=np.uint8)}, 0.0, done, {}
def get_libero_env(task, resolution=256, **kw):
    return Env(task), task.language
"""

# Upstream-shaped loop (UniVLA evaluate_libero_emu.evaluate): the three removed imports are at the top.
STUB_UNIVLA_EVAL = """
from robovlms.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
import os, time
from pytorch_lightning import seed_everything
from robovlms.utils.config_utils import load_config
from model_wrapper_emu import EmuVLAModel
from libero_utils import get_libero_env
from libero.libero import benchmark
def evaluate(model, model_name=None, debug=False, resize_size=256, num_trials_per_task=50, num_steps_wait=10,
             local_log_dir=None, task_suite_name="libero_object"):
    log_file = open(os.path.join(local_log_dir, task_suite_name + '-' + time.strftime('%H%M') + '.txt'), 'w')
    suite = benchmark.get_benchmark_dict()[task_suite_name]()
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        init = suite.get_task_init_states(task_id)
        env, desc = get_libero_env(task, resolution=256)
        log_file.write(f"\\nTask: {desc}\\n")
        for ep in range(num_trials_per_task):
            env.reset()
            model.reset()
            obs = env.set_init_state(init[ep])
            done = False
            for t in range(6):
                obs, r, done, info = env.step(model.step(obs, desc))
                if done:
                    break
            log_file.write(f"Success: {done}\\n")
    log_file.close()
"""

STUB_UNIVLA_WRAPPER = """
import os
class EmuVLAModel:
    def __init__(self, emu_hub, vq_hub, vision_hub, device):
        fast_path = "/share/project/yuqi.wang/UniVLA/pretrain/fast"
        assert os.path.isdir(fast_path), fast_path
        self.hubs = (emu_hub, vq_hub, vision_hub, device)
    def reset(self):
        pass
    def step(self, obs, goal):
        return [0] * 7
"""

STUB_GE_EVAL = """
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from yaml import load, Loader
from utils.libero_sim_utils import get_libero_env
from libero.libero import benchmark
class Tensor:
    def __init__(self, shape):
        self.shape = shape
class Model:
    def state_dict(self):
        return {k: Tensor(tuple(v)) for k, v in json.load(open(os.environ['STUB_MODEL_SHAPES'])).items()}
class InferenceLibero:
    def __init__(self, config_file, output_dir=None, task_suite_name='libero_goal', model_path=None, exec_step=8,
                 threshold=20, device='cuda:0', **kw):
        cd = load(open(config_file), Loader=Loader)
        assert cd['diffusion_model']['model_path'] == model_path
        assert os.path.isfile(cd['data']['val']['stat_file'])   # relative to the checkout (runner chdir)
        self.args = type('A', (), {'data': cd['data']})()
        self.task_suite_name = task_suite_name
        self.task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
        os.makedirs(output_dir, exist_ok=True)
        self.log_file = open(os.path.join(output_dir, 'inference_x.txt'), 'w')
        self.excution_step, self.threshold = exec_step, threshold
    def prepare_models(self):
        self.diffusion_model = Model()
    def infer(self, num_trails_per_task, image_shape=(256, 256)):
        for task_id in range(self.task_suite.n_tasks):
            task = self.task_suite.get_task(task_id)
            init = self.task_suite.get_task_init_states(task_id)
            env, desc = get_libero_env(task, image_height=image_shape[0], image_width=image_shape[1])
            for ep in range(num_trails_per_task):
                if env.env.done:
                    env, desc = get_libero_env(task, image_height=image_shape[0], image_width=image_shape[1])
                    env.set_init_state(init[ep])
                env.reset()
                env.set_init_state(init[ep])
                done = False
                for t in range(6):
                    obs, r, done, info = env.step([0] * 7)
                    if done or env.env.done:
                        break
                self.log_file.write(f"Success: {done}\\n")
"""

STUB_SAFETENSORS = """
import json
class _Slice:
    def __init__(self, s):
        self.s = s
    def get_shape(self):
        return self.s
class safe_open:
    def __init__(self, path, framework=None):
        self.d = json.load(open(path))
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def keys(self):
        return list(self.d)
    def get_slice(self, k):
        return _Slice(self.d[k])
"""

STUB_TORCH = """
def manual_seed(s):
    pass
class cuda:
    @staticmethod
    def manual_seed_all(s):
        pass
def device(x):
    return x
"""

DRIVER = """
import sys
sys.path.insert(0, {runner_dir!r})
import own_eval_runner as r
r.VIDEO_WRITER = lambda path, frames, fps: open(path, 'w').write(str(len(frames)))
sys.exit(r.main(sys.argv[1:]))
"""


def _stub_tree(tmp_path):
    np = pytest.importorskip("numpy")  # noqa: F841 - the stub env builds numpy frames
    stubs = tmp_path / "stubs"
    (stubs / "libero" / "libero").mkdir(parents=True)
    (stubs / "libero" / "__init__.py").write_text("")
    (stubs / "libero" / "libero" / "__init__.py").write_text(STUB_LIBERO.format(names=generate.LIBERO_TASKS))
    (stubs / "torch").mkdir()
    (stubs / "torch" / "__init__.py").write_text(STUB_TORCH)
    (stubs / "safetensors").mkdir()
    (stubs / "safetensors" / "__init__.py").write_text(STUB_SAFETENSORS)
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER.format(runner_dir=str(ROOT / "assets" / "libero")))
    return stubs, driver


def _run_driver(tmp_path, stubs, driver, args, outcomes, extra_env=None):
    env = dict(os.environ, PYTHONPATH=str(stubs), STUB_OUTCOMES=outcomes, **(extra_env or {}))
    return subprocess.run([sys.executable, str(driver), *args], capture_output=True, text=True, env=env,
                          cwd=tmp_path, timeout=120)


def test_univla_adapter_end_to_end(tmp_path):
    stubs, driver = _stub_tree(tmp_path)
    code = tmp_path / "UniVLA"
    ev = code / "reference" / "RoboVLMs" / "eval" / "libero"
    ev.mkdir(parents=True)
    (code / "reference" / "Emu3").mkdir()
    (code / "pretrain" / "fast").mkdir(parents=True)
    (ev / "evaluate_libero_emu.py").write_text(STUB_UNIVLA_EVAL)
    (ev / "model_wrapper_emu.py").write_text(STUB_UNIVLA_WRAPPER)
    (ev / "libero_utils.py").write_text(STUB_ENV)
    names = generate.LIBERO_TASKS["libero_goal"]
    out = tmp_path / "out"
    args = ["--adapter", "univla", "--code-dir", str(code), "--suite", "libero_goal", "--task-ids", "7,0",
            "--expect-names", f"{names[7]},{names[0]}", "--n-episodes", "3", "--out", str(out),
            "--emu-hub", "e", "--vq-hub", "q", "--vision-hub", "v"]
    proc = _run_driver(tmp_path, stubs, driver, args, "101011")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads((out / "results.json").read_text())
    assert [(t["task_id"], t["task_name"], t["n_success"]) for t in data["tasks"]] == [(7, names[7], 2),
                                                                                       (0, names[0], 2)]
    assert [e["success"] for e in data["tasks"][1]["episodes"]] == [False, True, True]
    vids = sorted(p.name for p in (out / "videos").iterdir())
    assert vids == sorted([f"{names[7]}--episode_0-success.mp4", f"{names[7]}--episode_1-failure.mp4",
                           f"{names[7]}--episode_2-success.mp4", f"{names[0]}--episode_0-failure.mp4",
                           f"{names[0]}--episode_1-success.mp4", f"{names[0]}--episode_2-success.mp4"])
    # a failure runs all 6 steps, a success stops at step 3
    assert (out / "videos" / f"{names[7]}--episode_1-failure.mp4").read_text() == "6"
    assert (out / "videos" / f"{names[7]}--episode_0-success.mp4").read_text() == "3"


def test_univla_adapter_fails_when_a_patch_target_moved(tmp_path):
    stubs, driver = _stub_tree(tmp_path)
    code = tmp_path / "UniVLA"
    ev = code / "reference" / "RoboVLMs" / "eval" / "libero"
    ev.mkdir(parents=True)
    (ev / "evaluate_libero_emu.py").write_text(STUB_UNIVLA_EVAL)
    (ev / "model_wrapper_emu.py").write_text(STUB_UNIVLA_WRAPPER.replace("/share/project", "/elsewhere"))
    (ev / "libero_utils.py").write_text(STUB_ENV)
    args = ["--adapter", "univla", "--code-dir", str(code), "--suite", "libero_goal", "--task-ids", "7",
            "--expect-names", "turn_on_the_stove", "--n-episodes", "1", "--out", str(tmp_path / "o"),
            "--emu-hub", "e", "--vq-hub", "q", "--vision-hub", "v"]
    proc = _run_driver(tmp_path, stubs, driver, args, "1")
    assert proc.returncode != 0 and "occurs 0 times" in proc.stderr   # the FAST path literal moved upstream


def test_univla_adapter_refuses_wrong_task_names(tmp_path):
    stubs, driver = _stub_tree(tmp_path)
    code = tmp_path / "UniVLA"
    ev = code / "reference" / "RoboVLMs" / "eval" / "libero"
    ev.mkdir(parents=True)
    (code / "pretrain" / "fast").mkdir(parents=True)
    (ev / "evaluate_libero_emu.py").write_text(STUB_UNIVLA_EVAL)
    (ev / "model_wrapper_emu.py").write_text(STUB_UNIVLA_WRAPPER)
    (ev / "libero_utils.py").write_text(STUB_ENV)
    args = ["--adapter", "univla", "--code-dir", str(code), "--suite", "libero_goal", "--task-ids", "7",
            "--expect-names", "open_the_middle_drawer_of_the_cabinet", "--n-episodes", "1", "--out", str(tmp_path / "o"),
            "--emu-hub", "e", "--vq-hub", "q", "--vision-hub", "v"]
    proc = _run_driver(tmp_path, stubs, driver, args, "1")
    assert proc.returncode != 0 and "suite task names ['turn_on_the_stove']" in proc.stderr


def _ge_tree(tmp_path):
    code = tmp_path / "Genie-Envisioner"
    (code / "experiments").mkdir(parents=True)
    (code / "utils").mkdir()
    (code / "utils" / "__init__.py").write_text("")
    (code / "utils" / "libero_sim_utils.py").write_text(STUB_ENV)
    (code / "experiments" / "eval_libero.py").write_text(STUB_GE_EVAL)
    cfgdir = code / "configs" / "ltx_model" / "libero"
    cfgdir.mkdir(parents=True)
    (cfgdir / "libero_all.json").write_text("{}")
    (cfgdir / "action_model_libero.yaml").write_text(textwrap.dedent("""\
        pretrained_model_name_or_path: PATH_TO_PRETRAINED_WEIGHTS_OF_VAE_AND_TOKENIZER
        output_dir: PATH_TO_SAVE_LOGS_AND_CHECKPOINTS
        diffusion_model:
          model_path: PATH/TO/GE_ACT/TRAINED_ON_LIBERO_DATASET.SAFETENSOR
        data:
          train: {sample_size: [256, 256]}
          val: {stat_file: configs/ltx_model/libero/libero_all.json}
        """))
    ckpt = tmp_path / "ge_act_libero_goal.safetensors"
    ckpt.write_text(json.dumps({"a": [2, 3], "b": [4]}))
    return code, ckpt


@pytest.mark.parametrize("model_shapes,ok", [({"a": [2, 3], "b": [4], "c": [1]}, True),
                                             ({"a": [2, 3], "b": [5]}, False)])
def test_ge_act_adapter_end_to_end(tmp_path, model_shapes, ok):
    stubs, driver = _stub_tree(tmp_path)
    code, ckpt = _ge_tree(tmp_path)
    shapes = tmp_path / "shapes.json"
    shapes.write_text(json.dumps(model_shapes))
    names = generate.LIBERO_TASKS["libero_goal"]
    out = tmp_path / "out"
    args = ["--adapter", "ge-act", "--code-dir", str(code), "--suite", "libero_goal", "--task-ids", "1,6",
            "--expect-names", f"{names[1]},{names[6]}", "--n-episodes", "2", "--out", str(out),
            "--ltx-dir", str(tmp_path / "ltx"), "--ckpt", str(ckpt), "--threshold", "20"]
    proc = _run_driver(tmp_path, stubs, driver, args, "1101", {"STUB_MODEL_SHAPES": str(shapes)})
    if not ok:
        assert proc.returncode != 0 and "1 checkpoint tensors did not load" in proc.stderr
        return
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads((out / "results.json").read_text())
    assert [[e["success"] for e in t["episodes"]] for t in data["tasks"]] == [[True, True], [False, True]]
    cfg = yaml.safe_load((out / "log" / "action_model_libero.yaml").read_text())
    assert cfg["pretrained_model_name_or_path"] == str(tmp_path / "ltx")
    assert cfg["diffusion_model"]["model_path"] == str(ckpt)
    # a success sets env.env.done, so the loop rebuilds the env for the next episode; still one video per episode
    assert len(list((out / "videos").iterdir())) == 4


# --- template task loop (bash + jq) with a stub runner ---------------------------------------------------------

FAKE_RUNNER_PY = r"""#!/usr/bin/env python3
import json, os, sys
argv = sys.argv[1:]
if argv and argv[0].endswith('own_eval_runner.py'):
    argv = argv[1:]
a = dict(zip(argv[::2], argv[1::2]))
out = a['--out']
os.makedirs(out + '/videos', exist_ok=True)
ids = [int(x) for x in a['--task-ids'].split(',')]
names = a['--expect-names'].split(',')
n = int(a['--n-episodes']) + int(os.environ.get('FAKE_EXTRA', '0'))
open(os.environ['FAKE_CALLS'], 'a').write(' '.join(argv) + '\n')
if os.environ.get('FAKE_FAIL'):
    sys.exit(3)
tasks = []
for tid, name in zip(ids, names):
    eps = []
    for k in range(n):
        s = (k % 2 == 0)
        v = f"{name}--episode_{k}-{'success' if s else 'failure'}.mp4"
        open(f"{out}/videos/{v}", 'w').write('x')
        eps.append(dict(episode=k, success=s, frames=3, video=v))
    tasks.append(dict(task_id=tid, task_name=name, language=name.replace('_', ' '), episodes=eps,
                      n_success=sum(e['success'] for e in eps), n_episodes=n))
json.dump(dict(tasks=tasks), open(out + '/results.json', 'w'))
"""


def _loop_script(tmp_path, vla="univla", task_set="actions"):
    _render(vla, tmp_path / "r", *(["--task-set", task_set] if task_set else []))
    text = (tmp_path / "r" / f"{vla}.sh").read_text()
    runs = re.search(r"^RUNS_JSON='.*'$", text, re.M).group(0)
    block = text[text.index("# ── [6/7]"):text.index("# ── [7/7]")]
    block = block.replace("/tmp/", f"{tmp_path}/tmp/")
    fake = tmp_path / "fakepy"
    fake.write_text(FAKE_RUNNER_PY)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    awsdir = tmp_path / "bin"
    awsdir.mkdir()
    (awsdir / "aws").write_text("#!/bin/bash\nexit 0\n")
    (awsdir / "aws").chmod(0o755)
    prelude = f"""set -euo pipefail
export PATH="{awsdir}:$PATH"
mkdir -p "{tmp_path}/tmp" "{tmp_path}/code"
log() {{ echo "[LOG] $*"; }}
die() {{ echo "[DIE] $*"; exit 7; }}
_trap_handler() {{ :; }}
trap '_trap_handler' ERR
PY="{fake}"; RUNNER="{tmp_path}/own_eval_runner.py"; CODE_DIR="{tmp_path}/code"
EVAL_BASE_DIR="{tmp_path}/eval"; VIDEO_BASE_DIR="{tmp_path}/videos"
S3_BUCKET=b; RUN_ID=R; REGION=us-east-1; INSTANCE_ID=i-0; INSTANCE_TYPE=g6.2xlarge; STACK_NAME=S
STACK_VERSIONS="torch==2.4.0"; GPU_NAME="NVIDIA L4"; CODE_HEAD=c; LIBERO_HEAD=l
{runs}
"""
    script = tmp_path / "loop.sh"
    script.write_text(prelude + block)
    return script


def _run_loop(tmp_path, vla="univla", task_set="actions", **env):
    script = _loop_script(tmp_path, vla, task_set)
    e = dict(os.environ, FAKE_CALLS=str(tmp_path / "calls.txt"), **env)
    return subprocess.run([BASH, str(script)], capture_output=True, text=True, env=e, timeout=120)


@pytest.mark.skipif(BASH is None or JQ is None, reason="bash / jq not on PATH")
def test_task_loop_writes_one_summary_per_libero_task(tmp_path):
    proc = _run_loop(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    assert len(calls) == 1 and "--task-ids 0,5,7" in calls[0] and "--adapter univla" in calls[0]
    for i, (tid, tag) in enumerate(zip(ACTION_IDS, ["articulated", "push", "articulated"])):
        summary = (tmp_path / "tmp" / f"univla_task{i}_summary.txt").read_text()
        kv = dict(line.split(": ", 1) for line in summary.splitlines() if ": " in line)
        assert kv["env_name"] == generate.LIBERO_TASKS["libero_goal"][tid] and kv["task_id"] == str(tid)
        assert (kv["n_success"], kv["n_episodes"], kv["success_rate"]) == ("5", "10", "0.500")
        assert kv["action_type"] == tag and kv["suite"] == "libero_goal"
        assert kv["instruction"] == kv["env_name"].replace("_", " ")
        assert kv["stack_eval_args"].startswith("python own_eval_runner.py --adapter univla")
        vids = sorted(p.name for p in (tmp_path / "videos" / f"task-{i}").iterdir())
        assert len(vids) == 10 and all(v.startswith(kv["env_name"] + "--episode_") for v in vids)


@pytest.mark.skipif(BASH is None or JQ is None, reason="bash / jq not on PATH")
def test_task_loop_fails_when_the_episode_count_differs(tmp_path):
    proc = _run_loop(tmp_path, vla="ge-act", task_set=None, FAKE_EXTRA="1")
    assert proc.returncode == 7
    assert "[DIE] Task 0 ran 11 episodes, expected 10" in proc.stdout


@pytest.mark.skipif(BASH is None or JQ is None, reason="bash / jq not on PATH")
def test_task_loop_fails_when_the_runner_fails(tmp_path):
    proc = _run_loop(tmp_path, vla="ge-act", task_set=None, FAKE_FAIL="1")
    assert proc.returncode == 7
    assert "[DIE] run 0 (libero_goal) failed (exit=3)" in proc.stdout


# --- results.py reads the summary the template writes -------------------------------------------------------

def test_results_reads_task_name_and_action_type():
    s3 = FakeS3({BUCKET: {
        "R/task-0/summary.txt": "model: univla\nsuite: libero_goal\ntask_id: 5\n"
                                "env_name: push_the_plate_to_the_front_of_the_stove\naction_type: push\n"
                                "success_rate: 0.500\nn_success: 5\nn_episodes: 10\n",
    }})
    r0 = results.task_row(s3, "us-east-1", BUCKET, "R", "R/task-0/")
    assert (r0["task_name"], r0["action_type"], r0["suite"], r0["k"], r0["n"]) == \
        ("push_the_plate_to_the_front_of_the_stove", "push", "libero_goal", 5, 10)
