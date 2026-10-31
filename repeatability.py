"""Re-run null per simulator: does running the same stack again give the same episodes?

A verdict between two policies is only as good as the noise floor under it. On some simulators an identical
re-run (same stack, same seed, same start states) gives the same episodes back; on GPU physics it does not,
and one run per policy is then not enough to call a winner (docs/results.md "Re-run null per simulator").

This table records, per simulator, what has been measured with this tool:

- ``yes``         an identical re-run gave the same episodes (or the same k within a point or two);
                  ``results --compare`` keeps its verdict from one run per side.
- ``no``          an identical re-run changed episodes and could change a verdict; ``results --compare``
                  prints "needs re-run" unless every such side has two or more finished runs, and with two
                  or more runs it prints the re-run spread and checks the verdict on every pair of runs.
- ``unmeasured``  nobody has measured it yet; the verdict is printed as before, with a note.

Stdlib-only and side-effect free on import. ``tests/test_repeatability.py`` fails if a target is missing.
"""
from __future__ import annotations

REPEATABLE = ("yes", "no", "unmeasured")

SIMULATORS: dict[str, dict] = {
    "libero": dict(
        label="LIBERO (MuJoCo, CPU physics)", repeatable="yes",
        evidence="same seed twice: 50 of 50 episodes identical for SmolVLA and X-VLA on LIBERO-Spatial, also on "
                 "another machine and region (2026-10-31); exact re-runs of 4 policies: 10 of 12 cells identical "
                 "k, at most 2 points apart"),
    "robolab": dict(
        label="RoboLab (Isaac Lab, GPU physics)", repeatable="no",
        evidence="identical re-run, 12 tasks x 10 episodes: 25 of 120 pi0.5 episodes and 19 of 120 Cosmos3 "
                 "episodes changed, and one pair verdict changed (tie, p = 0.69 -> p = 0.023) (2026-10-31)"),
    "isaaclab": dict(
        label="Isaac Lab (GPU physics)", repeatable="no",
        evidence="Isaac Lab Arena, same seed, same machine: 9 to 46 of 100 episodes changed between two runs "
                 "(2026-10-31); RoboLab (Isaac Lab) above"),
    "simpler": dict(label="SimplerEnv (SAPIEN / ManiSkill)", repeatable="unmeasured", evidence=""),
    "robocasa": dict(label="RoboCasa (MuJoCo)", repeatable="unmeasured", evidence=""),
    "robotwin": dict(label="RoboTwin 2.0 (SAPIEN)", repeatable="unmeasured", evidence=""),
    "mujoco": dict(label="MuJoCo (other scenes)", repeatable="unmeasured", evidence=""),
}

TARGET_SIMULATOR: dict[str, str] = {
    **{t: "libero" for t in ("aspire", "cosmos-policy", "fastwam", "gr00t", "lap", "lingbot-va", "molmoact2",
                             "openvla-oft", "pi", "rldx", "smolvla", "vla-adapter", "xvla")},
    **{t: "robolab" for t in ("robolab-cosmos3", "robolab-pi05")},
    **{t: "isaaclab" for t in ("openarm-isaac", "openarm-lift-act")},
    **{t: "simpler" for t in ("gr00t-bridge", "rldx-bridge", "rldx-simpler")},
    **{t: "robocasa" for t in ("gr00t-gr1", "rldx-gr1", "rldx-kitchen")},
    **{t: "robotwin" for t in ("smolvla-robotwin", "lingbot-vla2")},
    **{t: "mujoco" for t in ("gr00t-g1", "real2sim")},
}

_ORDER = {"no": 0, "unmeasured": 1, "yes": 2}


def for_target(name: str | None) -> dict:
    """Simulator entry for a target or stack name (`robolab-pi05`, `robolab-pi05-demo`, `pi`); unknown -> unmeasured.
    The longest target name that the stack name starts with wins (so `pi` does not swallow `pi05` stacks)."""
    name = (name or "").lower()
    best = max((t for t in TARGET_SIMULATOR if name == t or name.startswith(t + "-")), key=len, default=None)
    if best is None:
        return dict(simulator="unknown", label="unknown simulator", repeatable="unmeasured", evidence="")
    sim = TARGET_SIMULATOR[best]
    return dict(simulator=sim, **SIMULATORS[sim])


def for_simulator(sim: str) -> dict:
    sim = (sim or "").lower()
    if sim in SIMULATORS:
        return dict(simulator=sim, **SIMULATORS[sim])
    return dict(simulator=sim or "unknown", label=sim or "unknown simulator", repeatable="unmeasured", evidence="")


def worst(*classes: str) -> str:
    """The less repeatable of several classes (`no` < `unmeasured` < `yes`)."""
    return min(classes, key=lambda c: _ORDER.get(c, 1))
