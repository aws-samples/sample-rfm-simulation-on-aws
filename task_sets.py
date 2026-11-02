"""Named task sets and action-type tags.

A target's default run is the ``tasks:`` list of ``models/<target>.yaml``. A *task set* is another list in the
same file, under ``task_sets.<name>``, that ``deploy.py`` / ``generate.py --task-set <name>`` puts in place of
``tasks:`` (same stack, same results bucket, same template). The ``actions`` set of a target holds tasks the
pinned checkpoint was trained or evaluated on upstream that are *not* pick-and-place (doors, drawers, knobs,
buttons, bimanual handover, tool use), so the same checkpoint can be shown on other kinds of motion:

    python deploy.py --vla rldx-kitchen --task-set actions

Shape of one set (every key except ``tasks`` is optional)::

    task_sets:
      actions:
        description: "one line shown by deploy.py"
        model:                      # merged over `model:` for this run only (the libero_plus precedent)
          hf_subfolder: libero_goal
        tasks:                      # same entries as `tasks:` of that target (+ optional action_type)
          - env_name: ...
            action_type: articulated

Every task also gets an ``action_type`` (``ACTION_TYPES``): the one written in the yaml, else the tag
``classify`` derives from the task name. The templates write it into ``summary.txt``; ``vlasim results``
carries it into its CSV / JSON rows (and derives it from the task name for runs made before the tag existed).

Stdlib-only and side-effect free on import (vlasim doctor imports the CLIs on hosts without deps).
"""
from __future__ import annotations

import copy
import re

# Tag -> one-line meaning (docs/showcase.md groups the clips by these).
ACTION_TYPES = {
    "pick-place": "grasp an object and put it somewhere (the default of most benchmarks)",
    "articulated": "open / close / turn a jointed fixture: door, drawer, knob, faucet, lid",
    "press": "press or tap a button, bell or switch without grasping",
    "push": "move an object by pushing it, without grasping",
    "bimanual": "two arms on one object (handover, two-handed lift)",
    "tool-use": "grasp a tool and act on another object with it (hammer, stamp)",
    "locomotion": "the robot walks or drives as part of the task",
    "other": "none of the above / not recognised",
}

# Exact task names (after the namespace prefix is stripped) whose tag the keyword rules would get wrong or that
# deserve a precise tag. Keys are lower case.
KNOWN = {
    # RoboTwin 2.0 (Aloha-AgileX dual arm) — envs/<task>.py @0aeea2d6
    "beat_block_hammer": "tool-use",
    "stamp_seal": "tool-use",
    "click_bell": "press",
    "click_alarmclock": "press",
    "press_stapler": "press",
    "turn_switch": "press",
    "handover_block": "bimanual",
    "handover_mic": "bimanual",
    "lift_pot": "bimanual",
    "open_laptop": "articulated",
    "open_microwave": "articulated",
    # RoboCasa Kitchen (PandaOmron) — robocasa/environments/kitchen/single_stage/*
    "turnonmicrowave": "press",
    "turnoffmicrowave": "press",
    "coffeepressbutton": "press",
    # LIBERO
    "push_the_plate_to_the_front_of_the_stove": "push",
    # Unitree G1 loco-manipulation (gr00t-g1)
    "lmpnpappletoplatedc_g1_gear_wbc": "locomotion",
}

# First matching rule wins; applied to the bare lower-case task name. A compound task with a jointed sub-goal
# ("put the bowl in the drawer and close it", PnPCupToDrawerClose) is tagged articulated: that sub-goal is
# what sets it apart from plain pick-and-place. A fixture named only as a place ("the bowl in the top drawer")
# does not count — the rule keys on the verbs open / close / turn.
_RULES = (
    ("locomotion", re.compile(r"walk|loco|navigate")),
    ("tool-use", re.compile(r"hammer|stamp|wipe|scoop|sweep")),
    ("bimanual", re.compile(r"handover|hand_over|bimanual|dual_arm")),
    ("press", re.compile(r"press|button|bell|click")),
    ("push", re.compile(r"(^|_)push(_|$)")),
    ("articulated", re.compile(r"(^|_)(open|close|turn)(_|$)|^(open|close|turn)|close$")),
    ("pick-place", re.compile(r"pick|place|put|pnp|stack|move|lift|carry|insert|_(on|in|into|onto)_")),
)
_NAMESPACE = re.compile(r"^.*/")
_ROBOCASA_SUFFIX = re.compile(r"_(pandaomron|panda|gr1[a-z0-9]*)_env$")


def task_key(name: str) -> str:
    """Bare lower-case task name: namespace (``libero_sim/``, ``robocasa_panda_omron/``) and the RoboCasa
    ``_<Robot>_Env`` suffix removed, so the same task gets the same tag in every target."""
    n = _NAMESPACE.sub("", str(name or "").strip()).lower()
    n = _ROBOCASA_SUFFIX.sub("", n)
    return n


def classify(name: str) -> str:
    """Action type of a task name (``ACTION_TYPES`` key); ``other`` when nothing matches."""
    key = task_key(name)
    if not key:
        return "other"
    if key in KNOWN:
        return KNOWN[key]
    for tag, rule in _RULES:
        if rule.search(key):
            return tag
    return "other"


def task_name(task: dict) -> str:
    """The field that names the task in a `tasks:` entry of any target (first one present)."""
    for k in ("env_name", "task", "robotwin_task", "task_name", "instruction", "suite"):
        if task.get(k):
            return str(task[k])
    return ""


def action_type(task: dict) -> str:
    """The yaml `action_type` when set (validated), else classify(task_name(task))."""
    tag = task.get("action_type")
    if tag:
        if tag not in ACTION_TYPES:
            raise ValueError(f"action_type {tag!r} is not one of {', '.join(ACTION_TYPES)}")
        return str(tag)
    return classify(task_name(task))


def annotate(tasks):
    """Copy of a `tasks:` value with `action_type` on every dict entry (lists and suite dicts of lists)."""
    if isinstance(tasks, list):
        return [dict(t, action_type=action_type(t)) if isinstance(t, dict) else t for t in tasks]
    if isinstance(tasks, dict):  # openvla-oft: tasks by suite
        return {k: annotate(v) for k, v in tasks.items()}
    return tasks


def tags_of(tasks) -> list:
    """action_type of every task of a `tasks:` list; a RoboTwin entry (`robotwin_tasks: [..]`) counts per task."""
    out = []
    for t in tasks or []:
        if isinstance(t, dict) and isinstance(t.get("robotwin_tasks"), list):
            out += [classify(x) for x in t["robotwin_tasks"]]
        elif isinstance(t, dict):
            out.append(action_type(t))
    return out


def names(config: dict) -> list:
    return sorted((config.get("task_sets") or {}).keys())


def apply(config: dict, name: str | None, target: str = "") -> dict:
    """Merged config with `tasks:` replaced by `task_sets.<name>.tasks` and `model` overrides merged.

    Returns a new dict and records the choice under `task_set` (None/"" = the default `tasks:`, which is
    returned unchanged apart from that record). Raises ValueError for an unknown set or a malformed one.
    """
    out = copy.deepcopy(config)
    if not name:
        out["task_set"] = ""
        return out
    sets = config.get("task_sets") or {}
    where = f"models/{target}.yaml" if target else "the model yaml"
    if not sets:
        raise ValueError(f"--task-set {name}: {where} defines no task_sets")
    if name not in sets:
        raise ValueError(f"--task-set {name}: unknown set (available in {where}: {', '.join(sorted(sets))})")
    spec = sets[name] or {}
    tasks = spec.get("tasks")
    if not isinstance(tasks, list) or not tasks or not all(isinstance(t, dict) for t in tasks):
        raise ValueError(f"task_sets.{name}.tasks in {where} must be a non-empty list of task entries")
    for t in tasks:
        action_type(t)  # validates an explicit tag
    overrides = spec.get("model") or {}
    if not isinstance(overrides, dict) or any(isinstance(v, (dict, list)) for v in overrides.values()):
        raise ValueError(f"task_sets.{name}.model in {where} must map model keys to plain values")
    out["tasks"] = copy.deepcopy(tasks)
    if overrides:
        out["model"] = {**(out.get("model") or {}), **overrides}
    out["task_set"] = name
    return out


def describe(config: dict) -> str:
    """One line for the deploy banner: set name, task count and the action types it covers."""
    name = config.get("task_set") or ""
    tasks = config.get("tasks") or []
    if not name or not isinstance(tasks, list):
        return ""
    tags = tags_of(tasks)
    desc = ((config.get("task_sets") or {}).get(name) or {}).get("description", "")
    return f"{name}: {len(tags)} task(s), action types {', '.join(sorted(set(tags)))}" + (f" ({desc})" if desc else "")
