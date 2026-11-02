"""Run LIBERO episodes through a model author's own evaluation loop and keep one video per episode.

Used by templates/libero-own-userdata.sh.j2 (targets `univla`, `ge-act`). The authors' scripts evaluate every task
of a suite and save no per-episode video (UniVLA: a GIF only with --debug; GE-Act: an MP4 only when a step raises),
so this runner calls their loop unchanged and adds three things around it:

* a task subset: the suite object the loop reads (`n_tasks`, `get_task`, `get_task_init_states`) is wrapped so it
  sees only the requested task ids, in the given order;
* a recorder: the `get_libero_env` the loop calls is wrapped so every `env.step` frame (agent view, rotated 180
  degrees as both loops do for the policy) is kept, and every `env.reset` starts a new episode;
* outcomes: both loops write one ``Success: True|False`` line per episode to their log file; the runner reads those
  lines in order and pairs them with the recorded episodes (a count mismatch is an error).

Output (``--out DIR``)::

    DIR/results.json                                  per task: task id, name, language, per-episode outcome
    DIR/videos/<task name>--episode_<k>-<success|failure>.mp4

Adapters (``--adapter``):

* ``univla``: baaivision/UniVLA reference/RoboVLMs/eval/libero/evaluate_libero_emu.py ``evaluate()`` with
  ``EmuVLAModel`` from model_wrapper_emu.py. Two source patches at import, each matched exactly once: the FAST
  tokenizer path hard-coded to the authors' cluster (``/share/project/...``) points at the checkout's
  ``pretrain/fast``, and the three imports the evaluation never calls (``robovlms`` is not part of the repository,
  ``pytorch_lightning`` only seeds) are removed; the runner seeds python / numpy / torch with 0 as
  ``seed_everything(0)`` in the script's ``parser_args`` does.
* ``ge-act``: AgibotTech/Genie-Envisioner experiments/eval_libero.py ``InferenceLibero.infer()`` with
  configs/ltx_model/libero/action_model_libero.yaml, unpatched; the runner writes a copy of that config with the
  three paths filled in (LTX-Video tokenizer / text encoder / VAE, GE-Act checkpoint, output dir) and checks that
  every checkpoint tensor reached the model (the upstream loader is ``strict=False``).

Stdlib-only at import time (tests import it without torch / LIBERO); the adapters import their stacks lazily.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import types
from pathlib import Path

SUCCESS_LINE = re.compile(r"^Success: (True|False)\s*$")
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


# ── generic pieces (unit-tested) ──────────────────────────────────────────────

def load_patched(path, name, replacements):
    """Import the file at `path` as module `name` after replacing each (old, new) pair. Every `old` must occur
    exactly once, so an upstream change fails loudly instead of running unpatched. Registers the module in
    sys.modules first (later `from <name> import ...` statements get this copy)."""
    path = str(path)
    src = Path(path).read_text()
    for old, new in replacements:
        n = src.count(old)
        if n != 1:
            raise RuntimeError(f"patch for {path}: {old!r} occurs {n} times (expected exactly 1)")
        src = src.replace(old, new)
    mod = types.ModuleType(name)
    mod.__file__ = path
    sys.modules[name] = mod
    exec(compile(src, path, "exec"), mod.__dict__)  # nosec B102 - the pinned upstream file, patched above
    return mod


class SuiteSubset:
    """The task-suite interface the authors' loops read, restricted to `task_ids` (in that order)."""

    def __init__(self, suite, task_ids):
        n = suite.get_num_tasks() if hasattr(suite, "get_num_tasks") else suite.n_tasks
        bad = [i for i in task_ids if not 0 <= int(i) < n]
        if bad:
            raise ValueError(f"task ids {bad} out of range for a suite of {n} tasks")
        self._suite = suite
        self.task_ids = [int(i) for i in task_ids]
        self.n_tasks = len(self.task_ids)

    def get_num_tasks(self):
        return self.n_tasks

    def get_task(self, j):
        return self._suite.get_task(self.task_ids[j])

    def get_task_init_states(self, j):
        return self._suite.get_task_init_states(self.task_ids[j])

    def get_task_names(self):
        return [self._suite.get_task(i).name for i in self.task_ids]

    def __getattr__(self, item):
        return getattr(self._suite, item)


class Recorder:
    """Episodes in the order the loop runs them: one per env.reset(), frames from env.step()."""

    def __init__(self, camera="agentview_image"):
        self.camera = camera
        self.episodes = []

    def begin(self, task):
        self.episodes.append(dict(task_name=str(task.name), language=str(task.language), frames=[]))

    def add(self, obs):
        if not self.episodes or not isinstance(obs, dict) or self.camera not in obs:
            return
        self.episodes[-1]["frames"].append(obs[self.camera][::-1, ::-1])


class RecordingEnv:
    """LIBERO env proxy: reset() opens an episode in the recorder, step() records the returned observation."""

    def __init__(self, env, recorder, task):
        self._env = env
        self._recorder = recorder
        self._task = task

    def reset(self, *a, **kw):
        out = self._env.reset(*a, **kw)
        self._recorder.begin(self._task)
        return out

    def step(self, *a, **kw):
        out = self._env.step(*a, **kw)
        self._recorder.add(out[0] if isinstance(out, tuple) else out)
        return out

    def __getattr__(self, item):
        return getattr(self._env, item)


def recording_get_libero_env(orig, recorder):
    """Wrap a `get_libero_env(task, ...)` -> (env, description) function of the authors' utilities."""

    def wrapped(task, *a, **kw):
        env, description = orig(task, *a, **kw)
        return RecordingEnv(env, recorder, task), description

    return wrapped


def read_outcomes(log_paths):
    """`Success: True|False` lines of the authors' log files, in order."""
    out = []
    for p in log_paths:
        for line in Path(p).read_text(errors="replace").splitlines():
            m = SUCCESS_LINE.match(line.strip())
            if m:
                out.append(m.group(1) == "True")
    return out


def video_name(task_name, k, success):
    return f"{task_name}--episode_{k}-{'success' if success else 'failure'}.mp4"


def collect(recorder, outcomes, task_ids, n_episodes, expect_names):
    """Pair recorded episodes with outcomes; group them per task. Raises ValueError on any count / name mismatch."""
    eps = recorder.episodes
    want = len(task_ids) * n_episodes
    if len(eps) != len(outcomes):
        raise ValueError(f"{len(eps)} recorded episodes but {len(outcomes)} 'Success:' lines in the log")
    if len(eps) != want:
        raise ValueError(f"{len(eps)} episodes ran, expected {len(task_ids)} tasks x {n_episodes} = {want}")
    tasks = []
    for j, tid in enumerate(task_ids):
        chunk = list(zip(eps[j * n_episodes:(j + 1) * n_episodes], outcomes[j * n_episodes:(j + 1) * n_episodes]))
        names = {e["task_name"] for e, _ in chunk}
        if names != {expect_names[j]}:
            raise ValueError(f"task {tid}: episodes of {sorted(names)}, expected {expect_names[j]}")
        tasks.append(dict(task_id=tid, task_name=expect_names[j], language=chunk[0][0]["language"],
                          episodes=[dict(episode=k, success=bool(s), frames=len(e["frames"]),
                                         video=video_name(expect_names[j], k, s), _frames=e["frames"])
                                    for k, (e, s) in enumerate(chunk)]))
    return tasks


def imageio_writer(path, frames, fps):
    import imageio
    w = imageio.get_writer(str(path), fps=fps)
    for f in frames:
        w.append_data(f)
    w.close()


VIDEO_WRITER = imageio_writer  # tests swap in a stub


def write_results(out_dir, adapter, suite, tasks, fps=20, writer=None):
    """results.json + one mp4 per episode with frames (`writer(path, frames, fps)`, default VIDEO_WRITER)."""
    out_dir = Path(out_dir)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    writer = writer or VIDEO_WRITER
    clean = []
    for t in tasks:
        eps = []
        for e in t["episodes"]:
            frames = e.pop("_frames", [])
            if frames:
                writer(out_dir / "videos" / e["video"], frames, fps)
            else:
                e["video"] = ""
            eps.append(e)
        clean.append(dict(t, episodes=eps, n_success=sum(e["success"] for e in eps), n_episodes=len(eps)))
    (out_dir / "results.json").write_text(json.dumps(dict(adapter=adapter, suite=suite, tasks=clean), indent=2))
    return clean


def unloaded_tensors(ckpt_shapes, model_shapes):
    """Checkpoint tensor names that a strict=False load would skip: absent from the model or of another shape."""
    return sorted(k for k, shp in ckpt_shapes.items() if k not in model_shapes or list(model_shapes[k]) != list(shp))


def check_names(names, expect):
    if list(names) != list(expect):
        raise ValueError(f"suite task names {list(names)} != expected {list(expect)}")


def seed_all(seed):
    import random

    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── adapters ──────────────────────────────────────────────────────────────────

UNIVLA_FAST_PATH = 'fast_path = "/share/project/yuqi.wang/UniVLA/pretrain/fast"'
UNIVLA_UNUSED_IMPORTS = (
    "from robovlms.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict\n",
    "from robovlms.utils.config_utils import load_config\n",
    "from pytorch_lightning import seed_everything\n",
)


def run_univla(a, recorder, log_dir):
    code = Path(a.code_dir).resolve()
    ev_dir = code / "reference" / "RoboVLMs" / "eval" / "libero"
    for p in (ev_dir, code / "reference" / "Emu3", code):  # libero_utils / emu3.mllm / models.policy_head
        sys.path.insert(0, str(p))
    mw = load_patched(ev_dir / "model_wrapper_emu.py", "model_wrapper_emu",
                      [(UNIVLA_FAST_PATH, f"fast_path = {str(code / 'pretrain' / 'fast')!r}")])
    ev = load_patched(ev_dir / "evaluate_libero_emu.py", "evaluate_libero_emu",
                      [(imp, "") for imp in UNIVLA_UNUSED_IMPORTS])
    real = ev.benchmark.get_benchmark_dict()[a.suite]()
    subset = SuiteSubset(real, a.task_ids)
    check_names(subset.get_task_names(), a.expect_names)
    ev.benchmark = types.SimpleNamespace(get_benchmark_dict=lambda: {a.suite: lambda: subset})
    ev.get_libero_env = recording_get_libero_env(ev.get_libero_env, recorder)
    seed_all(0)
    import torch
    model = mw.EmuVLAModel(emu_hub=a.emu_hub, vq_hub=a.vq_hub, vision_hub=a.vision_hub, device=torch.device("cuda"))
    ev.evaluate(model, task_suite_name=a.suite, local_log_dir=str(log_dir), debug=False,
                num_trials_per_task=a.n_episodes)
    return sorted(Path(log_dir).glob(f"{a.suite}-*.txt"))


def ge_act_config(src_yaml, dst_yaml, ltx_dir, ckpt, output_dir):
    """Copy of the authors' LIBERO action-model config with the three PATH placeholders filled in."""
    import yaml
    cd = yaml.load(Path(src_yaml).read_text(), Loader=yaml.Loader)  # nosec B506 - the pinned upstream config
    cd["pretrained_model_name_or_path"] = str(ltx_dir)
    cd["diffusion_model"]["model_path"] = str(ckpt)
    cd["output_dir"] = str(output_dir)
    Path(dst_yaml).write_text(yaml.dump(cd, sort_keys=False))
    return cd


def run_ge_act(a, recorder, log_dir):
    code = Path(a.code_dir).resolve()
    os.chdir(code)  # the config's class / stat-file paths are relative to the checkout
    sys.path.insert(0, str(code))
    import importlib.util
    spec = importlib.util.spec_from_file_location("ge_eval_libero", code / "experiments" / "eval_libero.py")
    el = importlib.util.module_from_spec(spec)
    sys.modules["ge_eval_libero"] = el
    spec.loader.exec_module(el)
    el.get_libero_env = recording_get_libero_env(el.get_libero_env, recorder)
    cfg = Path(log_dir) / "action_model_libero.yaml"
    ge_act_config(code / a.config, cfg, a.ltx_dir, a.ckpt, log_dir)
    inf = el.InferenceLibero(config_file=str(cfg), output_dir=str(log_dir), task_suite_name=a.suite,
                             model_path=str(a.ckpt), exec_step=a.exec_step, threshold=a.threshold,
                             device="cuda:0")
    inf.task_suite = SuiteSubset(inf.task_suite, a.task_ids)
    check_names(inf.task_suite.get_task_names(), a.expect_names)
    inf.prepare_models()
    from safetensors import safe_open
    with safe_open(str(a.ckpt), framework="pt") as f:
        ckpt_shapes = {k: f.get_slice(k).get_shape() for k in f.keys()}
    missing = unloaded_tensors(ckpt_shapes, {k: v.shape for k, v in inf.diffusion_model.state_dict().items()})
    if missing:
        raise RuntimeError(f"{len(missing)} checkpoint tensors did not load into the model, e.g. {missing[:5]}")
    print(f"[runner] checkpoint check OK: every tensor of {Path(a.ckpt).name} matches the model")
    inf.infer(num_trails_per_task=a.n_episodes, image_shape=inf.args.data["train"]["sample_size"])
    inf.log_file.close()
    return [inf.log_file.name]


ADAPTERS = {"univla": run_univla, "ge-act": run_ge_act}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--adapter", required=True, choices=sorted(ADAPTERS))
    p.add_argument("--code-dir", required=True, help="the authors' repository checkout")
    p.add_argument("--suite", required=True, choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    p.add_argument("--task-ids", required=True, help="comma-separated LIBERO task ids, in run order")
    p.add_argument("--expect-names", required=True, help="comma-separated task names (bddl) for those ids")
    p.add_argument("--n-episodes", required=True, type=int)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=int, default=20)
    # univla
    p.add_argument("--emu-hub")
    p.add_argument("--vq-hub")
    p.add_argument("--vision-hub")
    # ge-act
    p.add_argument("--config", default="configs/ltx_model/libero/action_model_libero.yaml")
    p.add_argument("--ltx-dir")
    p.add_argument("--ckpt")
    p.add_argument("--exec-step", type=int, default=8)
    p.add_argument("--threshold", type=int, default=20)
    a = p.parse_args(argv)
    a.task_ids = [int(x) for x in a.task_ids.split(",") if x != ""]
    a.expect_names = [x for x in a.expect_names.split(",") if x != ""]
    if not a.task_ids or len(a.task_ids) != len(a.expect_names):
        p.error("--task-ids and --expect-names need the same, non-zero number of entries")
    if any(not _NAME_RE.match(n) for n in a.expect_names):
        p.error("--expect-names entries must match [A-Za-z0-9_.-]+")
    if a.n_episodes < 1:
        p.error("--n-episodes must be >= 1")
    need = {"univla": ("emu_hub", "vq_hub", "vision_hub"), "ge-act": ("ltx_dir", "ckpt")}[a.adapter]
    miss = [f"--{k.replace('_', '-')}" for k in need if not getattr(a, k)]
    if miss:
        p.error(f"--adapter {a.adapter} needs {' '.join(miss)}")
    return a


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out).resolve()
    log_dir = out / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    recorder = Recorder()
    logs = ADAPTERS[a.adapter](a, recorder, log_dir)
    if not logs:
        raise RuntimeError(f"no log file of the {a.adapter} evaluation under {log_dir}")
    tasks = collect(recorder, read_outcomes(logs), a.task_ids, a.n_episodes, a.expect_names)
    for t in write_results(out, a.adapter, a.suite, tasks, fps=a.fps):
        print(f"[runner] {t['task_name']}: {t['n_success']}/{t['n_episodes']} success")
    return 0


if __name__ == "__main__":
    sys.exit(main())
