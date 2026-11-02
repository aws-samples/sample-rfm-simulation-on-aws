"""Run the authors' goal-image planning evaluation of a latent world model and write one results.json.

These models are not policies: an action-conditioned world model predicts future latent states, and a CEM / MPC
planner searches action sequences whose predicted final latent matches the encoding of a goal image. The planning
loop, the environments, the success test and the episode sampling are the authors' code; this runner only fixes
the inputs (checkpoint path, dataset root, episode count, seed, an MPC iteration cap), records the per-episode
outcome the authors' evaluator computes, and renames the rollout videos it writes.

Adapters:
  dino-wm   gaoyuezhou/dino_wm plan.py: planning_main() with conf/plan_<env>.yaml (MPC + CEM). The final rollout
            video of each episode (output_final_<i>_<success|failure>.mp4) is the authors' side-by-side: simulator
            frames on top, the world model's decoded prediction below, goal image on the right.
  jepa-wms  facebookresearch/jepa-wms evals.main (simu_env_planning) with the released eval config of the model
            (configs/evals/simu_env_planning/<env>/<model>/*.yaml). Videos = the simulator frames of the executed
            plan (visualisation/video_agent_goal_<succ|fail>.mp4).

Writes <out>/results.json:
  {"adapter", "env", "model", "n_episodes", "n_success", "success_rate", "episodes": [{"ep", "success", "video"}],
   "upstream": {...the authors' own aggregate where they print one...}, "notes": [...]}
and <out>/videos/<label>--ep<k>-<success|failure>.mp4.

Usage (on the instance, from the userdata):
  python wm_plan_runner.py --adapter dino-wm --code-dir /home/ubuntu/dino-wm/code --env pusht --model pusht \
      --ckpt-root /home/ubuntu/dino-wm/assets/dino_wm_ckpt --data-root /home/ubuntu/dino-wm/assets/data \
      --n-episodes 20 --seed 99 --out /tmp/dino-wm-eval/task-0 --label pusht--dino-wm [--set planner.max_iter=6]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
import types
from pathlib import Path

ADAPTERS = ("dino-wm", "jepa-wms")
# dino_wm conf/plan_<env>.yaml and the model folder under <ckpt_root>/outputs/ for each environment.
DINO_WM_ENVS = {"pusht": "plan_pusht", "wall": "plan_wall", "point_maze": "plan_point_maze"}
# jepa-wms configs/evals/simu_env_planning/<dir>/ for each environment.
JEPA_WMS_ENVS = {"pusht": "pt", "wall": "wall", "point_maze": "mz"}
JEPA_WMS_MODELS = ("jepa-wm", "dino-wm")
_SET_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*=[A-Za-z0-9_.+-]*$")


def parse_set(items) -> list:
    """`key.sub=value` overrides (plain tokens only); raises ValueError."""
    out = []
    for s in items or []:
        if not _SET_RE.match(s):
            raise ValueError(f"--set {s!r}: expected key.path=value with plain tokens")
        k, v = s.split("=", 1)
        out.append((k, v))
    return out


def _coerce(v: str):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() in ("null", "none"):
        return None
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def set_path(d: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        if not isinstance(d.get(k), dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def video_name(label: str, ep: int, success) -> str:
    tag = "success" if success is True else "failure" if success is False else "unknown"
    return f"{label}--ep{ep:02d}-{tag}.mp4"


def summarize(adapter, env, model, episodes, upstream=None, notes=None) -> dict:
    """results.json body from per-episode outcomes (success True / False)."""
    eps = sorted(episodes, key=lambda e: e["ep"])
    k = sum(1 for e in eps if e["success"] is True)
    n = len(eps)
    return dict(adapter=adapter, env=env, model=model, n_episodes=n, n_success=k,
                success_rate=round(k / n, 4) if n else None, episodes=eps, upstream=upstream or {},
                notes=list(notes or []))


# ── shims ────────────────────────────────────────────────────────────────────────────────────────
def stub_pointmaze_deps(notes: list) -> None:
    """dino_wm's env/__init__.py imports env.pointmaze, which imports d4rl and gym.envs.mujoco (-> mujoco_py) at
    module level, so every environment needs them installed. Push-T and Wall never touch them: register empty
    modules under those names when they are not installed (PointMaze then fails loudly at env creation)."""
    for name in ("mujoco_py", "d4rl"):
        try:
            __import__(name)
        except Exception:
            mod = types.ModuleType(name)
            mod.__path__ = []  # package, so `from d4rl import offline_env` resolves the submodule below
            sys.modules[name] = mod
            notes.append(f"{name} not installed: empty stub module (only PointMaze uses it)")
    if "d4rl" in sys.modules and not hasattr(sys.modules["d4rl"], "offline_env"):
        off = types.ModuleType("d4rl.offline_env")

        class OfflineEnv:  # base class of the PointMaze env; never instantiated for Push-T / Wall
            def __init__(self, *a, **k):
                raise RuntimeError("d4rl is stubbed: PointMaze needs d4rl + mujoco-py installed")

        off.OfflineEnv = OfflineEnv
        sys.modules["d4rl.offline_env"] = off
        sys.modules["d4rl"].offline_env = off


def shim_decord(notes: list) -> None:
    """decord has no wheel for some platforms (e.g. aarch64 CPU hosts used for a local smoke). The dino_wm Push-T
    dataset only calls VideoReader(path).get_batch(frames) -> THWC uint8 tensor; provide that through imageio.
    Never used when decord imports (the instance lock installs it)."""
    try:
        import decord  # noqa: F401
        return
    except Exception:
        pass
    import imageio.v2 as iio
    import numpy as np
    import torch

    class VideoReader:
        def __init__(self, path, num_threads=1, **_):
            self._frames = np.stack([f for f in iio.mimread(path, memtest=False)])

        def __len__(self):
            return len(self._frames)

        def get_batch(self, idx):
            return torch.from_numpy(self._frames[list(idx)])

    mod = types.ModuleType("decord")
    mod.VideoReader = VideoReader
    mod.cpu = mod.gpu = lambda *_a, **_k: None  # decord.cpu(0) context argument (jepa-wms imports it)
    mod.bridge = types.SimpleNamespace(set_bridge=lambda *_a, **_k: None)
    sys.modules["decord"] = mod
    notes.append("decord not installed: imageio-based VideoReader shim (local smoke only)")


# ── dino-wm ──────────────────────────────────────────────────────────────────────────────────────
def compose_plan_cfg(code: Path, config_name: str, overrides: list) -> dict:
    """conf/<config_name>.yaml + Hydra overrides -> the dict plan.py's main() hands to planning_main()
    (utils.cfg_to_dict of the composed config), without Hydra's run-dir / launcher handling."""
    from hydra import compose, initialize_config_dir
    from utils import cfg_to_dict  # dino_wm/utils.py

    with initialize_config_dir(config_dir=str(code / "conf"), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict.pop("hydra", None)
    return cfg_dict


def run_dino_wm(a, sets, notes) -> dict:
    code = Path(a.code_dir).resolve()
    out = Path(a.out).resolve()
    os.environ["DATASET_DIR"] = str(Path(a.data_root).resolve())
    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.path.insert(0, str(code))
    os.chdir(code)
    stub_pointmaze_deps(notes)
    shim_decord(notes)
    import plan as dplan  # the authors' plan.py (imports env, planning, datasets)
    from planning.evaluator import PlanEvaluator

    final = {}
    orig_eval = PlanEvaluator.eval_actions

    def eval_actions(self, actions, action_len=None, filename="output", save_video=False):
        res = orig_eval(self, actions, action_len, filename=filename, save_video=save_video)
        if filename == "output_final":
            final["successes"] = [bool(x) for x in res[1]]
        return res

    PlanEvaluator.eval_actions = eval_actions
    overrides = [f"model_name={a.model}", f"ckpt_base_path={Path(a.ckpt_root).resolve()}",
                 f"n_evals={a.n_episodes}", f"seed={a.seed}", f"n_plot_samples={a.n_episodes}"]
    overrides += [f"{k}={v}" for k, v in sets]
    cfg_dict = compose_plan_cfg(code, DINO_WM_ENVS[a.env], overrides)
    cfg_dict["saved_folder"] = str(out)
    cfg_dict["wandb_logging"] = False
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)  # the authors' code writes plan_targets.pkl, logs.json and the videos into the cwd
    json.dump({"overrides": overrides, "planner": cfg_dict.get("planner"), "objective": cfg_dict.get("objective"),
               "goal_source": cfg_dict.get("goal_source"), "goal_H": cfg_dict.get("goal_H")},
              open(out / "plan_config.json", "w"), indent=1, default=str)
    t0 = time.time()
    logs = dplan.planning_main(cfg_dict=cfg_dict)
    if "successes" not in final:
        raise RuntimeError("the final evaluation did not run (no output_final call)")
    vdir = out / "videos"
    vdir.mkdir(exist_ok=True)
    episodes = []
    for i, ok in enumerate(final["successes"]):
        src = out / f"output_final_{i}_{'success' if ok else 'failure'}.mp4"
        name = video_name(a.label, i, ok)
        if src.is_file():
            shutil.move(str(src), str(vdir / name))
        episodes.append(dict(ep=i, seed=a.seed * i + 1, success=ok, video=name if (vdir / name).is_file() else ""))
    up = {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in (logs or {}).items()}
    up["wall_seconds"] = round(time.time() - t0, 1)
    return summarize("dino-wm", a.env, a.model, episodes, upstream=up, notes=notes)


# ── jepa-wms ─────────────────────────────────────────────────────────────────────────────────────
def jepa_wms_config(code: Path, env: str, model: str) -> Path:
    files = sorted(glob.glob(str(code / "configs/evals/simu_env_planning" / JEPA_WMS_ENVS[env] / model / "*.yaml")))
    if len(files) != 1:
        raise RuntimeError(f"expected one released eval config for {env}/{model}, found {files}")
    return Path(files[0])


def patch_jepa_wms_config(params: dict, a, ckpt: Path, out: Path, sets, notes) -> dict:
    """Released eval config -> single-GPU run of this target (paths, episode count, seed, heads)."""
    params["folder"] = str(out / "logs")
    params["tag"] = f"{a.label}"
    params["model_kwargs"]["checkpoint"] = str(ckpt)
    params["meta"]["eval_episodes"] = a.n_episodes
    params["meta"]["seed"] = a.seed
    params["meta"]["quick_debug"] = False
    params.setdefault("logging", {})["optional_plots"] = False
    heads = params["model_kwargs"]["pretrain_kwargs"].get("heads_cfg") or {}
    arch = heads.get("architectures") or {}
    dec = heads.get("pretrain_dec_path") or {}
    # The state-readout head the configs name lives under the authors' own log folder (${JEPAWM_LOGS}/...) and is not
    # released; it only feeds optional plots. Drop it. The image head (VM2M decoder) only decodes the planner's
    # imagined frames (plan_vis); kept when --image-head points at the downloaded file.
    if "state_head" in arch:
        arch.pop("state_head")
        dec.pop("state_head", None)
        notes.append("state_head dropped (not released; optional plots only)")
    if a.image_head:
        dec["image_head"] = str(Path(a.image_head).resolve())
    else:
        arch.pop("image_head", None)
        dec.pop("image_head", None)
        params["planner"]["decode_each_iteration"] = False
        notes.append("image_head off (no decoded imagination frames; videos are simulator frames)")
    if not arch:  # init_module reads heads_cfg.get(...): an empty mapping, not null, means "no heads"
        params["model_kwargs"]["pretrain_kwargs"]["heads_cfg"] = {}
    for k, v in sets:
        set_path(params, k, _coerce(v))
    return params


def run_jepa_wms(a, sets, notes) -> dict:
    import yaml

    code = Path(a.code_dir).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["JEPAWM_DSET"] = str(Path(a.data_root).resolve())
    os.environ.setdefault("JEPAWM_LOGS", str(out / "logs"))
    os.environ.setdefault("JEPAWM_HOME", str(code.parent))
    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.path.insert(0, str(code))
    os.chdir(code)
    shim_decord(notes)
    src = jepa_wms_config(code, a.env, a.model)
    params = yaml.safe_load(src.read_text())
    ckpt = Path(a.ckpt).resolve()
    params = patch_jepa_wms_config(params, a, ckpt, out, sets, notes)
    fname = out / "eval_config.yaml"
    fname.write_text(yaml.safe_dump(params, sort_keys=False))

    from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator

    outcomes = []
    orig_eval = PlanEvaluator.eval

    def eval_(self, cfg, agent, env, task_idx=-1, ep=0):
        res = orig_eval(self, cfg, agent, env, task_idx=task_idx, ep=ep)
        outcomes.append(dict(ep=int(ep), task=str(cfg.tasks[task_idx]), success=bool(res[1]),
                             work_dir=str(Path(cfg.work_dir) / cfg.tasks[task_idx] / f"ep_{ep}")))
        return res

    PlanEvaluator.eval = eval_
    from evals.scaffold import main as eval_main
    from src.utils.distributed import init_distributed

    try:
        import torch
        import torch.distributed as dist
    except ImportError:  # stub trees in the unit tests
        torch = dist = None
    if torch is not None and not torch.cuda.is_available() and not dist.is_initialized():
        # CPU-only hosts (local smoke): the authors' init asks for "cpu:gloo,cuda:nccl", and CPU builds of torch
        # have no NCCL. A one-process gloo group is what that call would give on CPU; init_distributed() then
        # returns early (already initialised).
        os.environ.update(MASTER_ADDR="localhost", MASTER_PORT=os.environ.get("MASTER_PORT", "29533"),
                          WORLD_SIZE="1", RANK="0", LOCAL_RANK="0")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
        notes.append("no CUDA: single-process gloo group (local smoke)")
    init_distributed(rank_and_world_size=(0, 1))
    t0 = time.time()
    eval_main(params["eval_name"], args_eval=params)
    vdir = out / "videos"
    vdir.mkdir(exist_ok=True)
    episodes, seen = [], set()
    for o in outcomes:
        if o["ep"] in seen:  # the authors' loop pads with duplicate dummy episodes when world_size > episodes
            continue
        seen.add(o["ep"])
        tag = "succ" if o["success"] else "fail"
        src_v = Path(o["work_dir"]) / "visualisation" / f"video_agent_goal_{tag}.mp4"
        name = video_name(a.label, o["ep"], o["success"])
        if src_v.is_file():
            shutil.copy(str(src_v), str(vdir / name))
        episodes.append(dict(ep=o["ep"], success=o["success"], video=name if (vdir / name).is_file() else ""))
    up = {"config": str(src.relative_to(code)), "wall_seconds": round(time.time() - t0, 1)}
    csv = sorted(Path(params["folder"]).glob("**/eval.csv"))
    if csv:
        up["eval_csv"] = str(csv[-1])
        lines = csv[-1].read_text().splitlines()
        if len(lines) >= 2:
            head, row = lines[0].split(","), lines[-1].split(",")
            up.update({h: r for h, r in zip(head, row) if h.startswith("episode_success")})
    return summarize("jepa-wms", a.env, a.model, episodes, upstream=up, notes=notes)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--adapter", choices=ADAPTERS, required=True)
    p.add_argument("--code-dir", required=True, help="checkout of the authors' repository")
    p.add_argument("--env", required=True, choices=sorted(DINO_WM_ENVS))
    p.add_argument("--model", required=True,
                   help="dino-wm: model folder under <ckpt-root>/outputs; jepa-wms: jepa-wm | dino-wm (config dir)")
    p.add_argument("--ckpt-root", help="dino-wm: folder that holds outputs/<model>/")
    p.add_argument("--ckpt", help="jepa-wms: the downloaded .pth.tar")
    p.add_argument("--image-head", help="jepa-wms: VM2M decoder .pth.tar (optional)")
    p.add_argument("--data-root", required=True, help="folder with pusht_noise/ and / or wall_single/")
    p.add_argument("--n-episodes", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--label", required=True, help="video file name prefix (task name)")
    p.add_argument("--set", action="append", default=[], help="config override key.path=value (repeatable)")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    sets = parse_set(a.set)
    if a.n_episodes < 1:
        raise SystemExit("--n-episodes must be >= 1")
    if a.adapter == "dino-wm" and not a.ckpt_root:
        raise SystemExit("--ckpt-root is required for dino-wm")
    if a.adapter == "jepa-wms":
        if not a.ckpt:
            raise SystemExit("--ckpt is required for jepa-wms")
        if a.model not in JEPA_WMS_MODELS:
            raise SystemExit(f"--model must be one of {JEPA_WMS_MODELS} for jepa-wms")
    notes: list = []
    out = Path(a.out).resolve()
    res = run_dino_wm(a, sets, notes) if a.adapter == "dino-wm" else run_jepa_wms(a, sets, notes)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(f"[wm_plan_runner] {a.adapter} {a.env}/{a.model}: {res['n_success']}/{res['n_episodes']} success")
    return 0


if __name__ == "__main__":
    sys.exit(main())
