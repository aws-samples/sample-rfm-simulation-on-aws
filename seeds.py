"""Shared seed for a comparison: draw one seed when a test starts, give the same seed to every model.

Two numbers can only be compared if the randomness behind them is known. This module turns that into
three rules (docs/results.md "Shared seed"):

1. ``--seed auto`` draws one seed with ``secrets.randbits(31)`` when the test starts; ``--seed <n>`` reuses
   a published seed; ``--seed model`` (the default) keeps the per-task seeds of ``models/<target>.yaml``
   (the settings behind the published numbers).
2. The same value goes to every target of the comparison. Where the target's evaluation has a seed knob,
   the tool writes it into every task; where it has none, the protocol card says so and why
   (``seed_control``), instead of pretending the run is seeded.
3. The value, where it came from (drawn / given / model) and when it was drawn are written into
   ``protocol.yaml`` next to every result, so anyone can rerun the comparison with ``--seed <n>``.

Seeds have two jobs that targets wire differently: the *env* seed decides start states, object placement
and resets; the *policy* seed decides sampling noise (diffusion / flow noise, token sampling). The table
below records, per target, which of the two the tool controls today.
"""
from __future__ import annotations

import copy
import datetime as _dt
import re
import secrets

SEED_BITS = 31  # fits every int32 seed argument (numpy, torch, gymnasium, LeRobot, openpi)
SEED_MODES = ("model", "auto")

# How each target applies a seed. `where` = which config key the tool writes:
#   "tasks"      -> every task entry's `seed` (the template reads `.seed` per task)
#   "eval"       -> `eval.seed` (RoboLab: one seed for the run)
#   None         -> the tool cannot set it; `reason` says why.
# env / policy = what the seed reaches at the pinned versions ("yes", "no", "fixed" = start states fixed
# by the benchmark file, so there is nothing random to seed).
_LIBERO_FIXED = "fixed"  # LIBERO / LIBERO-Pro start states come from the benchmark's init-state files
SEED_CONTROL: dict[str, dict] = {
    # LeRobot lerobot-eval --seed: set_seed (Python / NumPy / torch) + env.reset(seed=...)
    **{t: dict(where="tasks", env=_LIBERO_FIXED, policy="yes", via="lerobot-eval --seed")
       for t in ("smolvla", "xvla", "lingbot-va", "fastwam")},
    **{t: dict(where="tasks", env="yes", policy="yes", via="lerobot-eval --seed (RoboTwin setup_demo seed)")
       for t in ("smolvla-robotwin", "lingbot-vla2")},
    "molmoact2": dict(where="tasks", env=_LIBERO_FIXED, policy="yes", via="lerobot-eval --seed"),
    "openvla-oft": dict(where="tasks", env=_LIBERO_FIXED, policy="yes", via="run_libero_eval.py --seed"),
    "pi": dict(where="tasks", env=_LIBERO_FIXED, policy="yes", via="openpi examples/libero main.py --args.seed"),
    "cosmos-policy": dict(where="tasks", env=_LIBERO_FIXED, policy="yes",
                          via="run_libero_eval --seed (with --randomize_seed False)"),
    "openarm-isaac": dict(where="tasks", env="yes", policy="yes", via="run_eval.py --seed (Isaac Lab env + policy)"),
    "gr00t-bridge": dict(where="tasks", env="no", policy="yes",
                         via="rollout_policy.py --seed (seeds Python / NumPy / torch)",
                         reason="the WidowX env wrapper at this pin ignores reset seeds"),
    "robolab-cosmos3": dict(where="eval", env="yes", policy="yes", via="RoboLab eval --seed (one seed per run)"),
    "robolab-pi05": dict(where="eval", env="yes", policy="yes", via="RoboLab eval --seed (one seed per run)"),
    "aspire": dict(where=None, env=_LIBERO_FIXED, policy="no",
                   reason="scored on LIBERO-Pro saved init states 0..4 (eval_seeds index them); "
                          "the frozen program has no sampling"),
    "gr00t": dict(where=None, env="no", policy="no",
                  reason="rollout_policy.py at Isaac-GR00T 4b1dca9 has no seed and calls env.reset() without "
                         "one (random resets); upstream 65cc4a1 adds --seed"),
    "gr00t-gr1": dict(where=None, env="no", policy="no",
                      reason="the template passes no seed to the RoboCasa GR-1 rollout (random resets)"),
    "gr00t-g1": dict(where=None, env="no", policy="no",
                     reason="rollout_policy.py at Isaac-GR00T 7786639 has a seedless env.reset()"),
    "lap": dict(where=None, env="no", policy="no", reason="the template passes no seed to the LAP evaluation client"),
    "vla-adapter": dict(where=None, env=_LIBERO_FIXED, policy="no",
                        reason="the template passes no seed to the upstream eval script"),
    "real2sim": dict(where=None, env="no", policy="no",
                     reason="reconstruction pipeline, not a policy evaluation (segment_args fix its own seed)"),
    "openarm-lift-act": dict(where=None, env="no", policy="no",
                             reason="the template passes no seed to the ACT play script; it writes no summary.txt"),
    "rlinf-oft-libero": dict(where=None, env="no", policy="no",
                             reason="the template passes no seed to RLinf evaluations/run_eval.sh (the eval config's "
                                    "own seed applies)"),
    **{t: dict(where=None, env="no", policy="no", reason="the template passes no seed to the RLDX evaluation entry point")
       for t in ("rldx", "rldx-simpler", "rldx-bridge", "rldx-gr1", "rldx-kitchen")},
}


def control(target: str) -> dict:
    """Seed control of a target (unknown targets: not controllable)."""
    return SEED_CONTROL.get(target, dict(where=None, env="no", policy="no", reason="target not in seeds.SEED_CONTROL"))


def draw() -> int:
    """One fresh seed in [0, 2**31)."""
    return secrets.randbits(SEED_BITS)


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(value: str | int | None) -> str | int:
    """`model` / `auto` / a non-negative int below 2**31 (as int). Raises ValueError otherwise."""
    if value is None:
        return "model"
    if isinstance(value, int):
        n = value
    else:
        s = str(value).strip().lower()
        if s in SEED_MODES:
            return s
        if not re.fullmatch(r"\d{1,10}", s):
            raise ValueError(f"--seed must be model, auto or an integer 0..{2 ** SEED_BITS - 1}, got {value!r}")
        n = int(s)
    if not 0 <= n < 2 ** SEED_BITS:
        raise ValueError(f"--seed must be in 0..{2 ** SEED_BITS - 1}, got {n}")
    return n


def resolve(value: str | int | None, drawn_utc: str | None = None) -> dict:
    """Seed plan for one test: {mode, value, drawn_utc}. `auto` draws now (once per call)."""
    mode = parse(value)
    if mode == "model":
        return dict(mode="model", value=None, drawn_utc=None)
    if mode == "auto":
        return dict(mode="drawn", value=draw(), drawn_utc=drawn_utc or _utc_now())
    # A value with a draw time was drawn by the caller (deploy.py draws once and hands the number on).
    return dict(mode="drawn" if drawn_utc else "given", value=mode, drawn_utc=drawn_utc)


def _task_lists(tasks):
    if isinstance(tasks, list):
        yield tasks
    elif isinstance(tasks, dict):  # openvla-oft: tasks by suite
        for v in tasks.values():
            yield from _task_lists(v)


def apply(config: dict, target: str, plan: dict) -> dict:
    """Merged config with the plan's seed written where the target reads it. Returns a new dict and adds
    `seed_plan` (what was applied, for the protocol card). `model` mode changes nothing but the record."""
    out = copy.deepcopy(config)
    ctl = control(target)
    applied = False
    if plan.get("value") is not None and ctl.get("where") == "tasks":
        for lst in _task_lists(out.get("tasks")):
            for t in lst:
                if isinstance(t, dict):
                    t["seed"] = int(plan["value"])
                    applied = True
    elif plan.get("value") is not None and ctl.get("where") == "eval":
        out.setdefault("eval", {})["seed"] = int(plan["value"])
        applied = True
    if plan.get("value") is None:
        status = "model"  # per-task seeds of models/<target>.yaml (or the template default)
    elif applied:
        status = "applied"
    else:
        status = "not-controllable"
    out["seed_plan"] = dict(
        mode=plan.get("mode", "model"), value=plan.get("value"), drawn_utc=plan.get("drawn_utc"), status=status,
        env=ctl.get("env", "no"), policy=ctl.get("policy", "no"), via=ctl.get("via", ""),
        reason=ctl.get("reason", "") if status != "applied" or ctl.get("env") == "no" else "",
    )
    return out
