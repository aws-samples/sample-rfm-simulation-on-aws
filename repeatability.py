"""Re-run null per simulator: does running the same stack again give the same episodes?

A verdict between two policies is only as good as the noise floor under it. On some simulators an identical
re-run (same stack, same seed, same start states) gives the same episodes back; on Isaac Lab it does not
(the physics repeats, the rendered camera images and the GPU policy inference do not), and one run per policy is then not enough to call a winner (docs/results.md "Re-run null per simulator").

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
        label="RoboLab (Isaac Lab, rendering + inference)", repeatable="no",
        evidence="identical re-run, 12 tasks x 10 episodes: 25 of 120 pi0.5 episodes and 19 of 120 Cosmos3 "
                 "episodes changed, and one pair verdict changed (tie, p = 0.69 -> p = 0.023) (2026-10-31); replaying the "
                 "same actions repeats the physics bit for bit (0 of 40 outcomes change) but the camera images differ on "
                 "every frame (0 of 331 identical), and the policy server is not bit-exact without deterministic XLA "
                 "ops (2026-10-31)"),
    "isaaclab": dict(
        label="Isaac Lab (rendering + inference)", repeatable="no",
        evidence="Isaac Lab Arena, same seed, same machine: 9 to 46 of 100 episodes changed between two runs "
                 "(2026-10-31); RoboLab (Isaac Lab) above"),
    "simpler": dict(label="SimplerEnv (SAPIEN / ManiSkill)", repeatable="unmeasured", evidence=""),
    "robocasa": dict(label="RoboCasa (MuJoCo)", repeatable="unmeasured", evidence=""),
    "robotwin": dict(label="RoboTwin 2.0 (SAPIEN)", repeatable="unmeasured", evidence=""),
    "mujoco": dict(label="MuJoCo (other scenes)", repeatable="unmeasured", evidence=""),
}

TARGET_SIMULATOR: dict[str, str] = {
    **{t: "libero" for t in ("aspire", "cosmos-policy", "fastwam", "gr00t", "lap", "lingbot-va", "molmoact2",
                             "openvla-oft", "pi", "rldx", "rlinf-oft-libero", "smolvla", "vla-adapter",
                             "xvla")},
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


# ── Re-runs per deploy (deploy.py / generate.py --repeats) ─────────────────────────────────────────────
# Targets whose UserData can run the whole evaluation more than once in one deploy (a new RUN_ID per repeat,
# same stack, seed and start states), so results --compare finds two runs of the same fingerprint.
REPEATS_SUPPORTED = ("robolab-cosmos3", "robolab-pi05")
MAX_REPEATS = 3

# Measured instance hours of one run of the RoboLab subset (one L40S, 1 shard), from boot to summary upload,
# with the number of episodes of that run (2026-10-31, 5 runs per policy, g6e.2xlarge / g6e.4xlarge).
MEASURED_RUN_HOURS = {
    "robolab-pi05": dict(episodes=120, hours_min=2.08, hours_max=2.25),
    "robolab-cosmos3": dict(episodes=120, hours_min=2.83, hours_max=2.94),
}


def default_repeats(target: str | None) -> int:
    """Runs per deploy when --repeats is not given: 2 where an identical re-run is known not to repeat
    (repeatable = no) and the template supports it, else 1."""
    if for_target(target)["repeatable"] == "no" and (target or "") in REPEATS_SUPPORTED:
        return 2
    return 1


def resolve_repeats(target: str, value: str | int | None) -> tuple[int, str]:
    """(runs, reason) for `--repeats auto|N`. Raises ValueError for a value the target cannot take."""
    rep = for_target(target)
    if value in (None, "", "auto"):
        n = default_repeats(target)
        if n > 1:
            return n, f"default for {rep['label']}: an identical re-run does not repeat, so one run cannot call a winner"
        if rep["repeatable"] == "no":
            return 1, (f"{rep['label']} does not repeat, but this target runs once per deploy; "
                       "deploy it again for the re-run")
        return 1, f"{rep['label']}: repeatable = {rep['repeatable']}"
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"--repeats must be auto or an integer 1..{MAX_REPEATS}, got {value!r}") from None
    if not 1 <= n <= MAX_REPEATS:
        raise ValueError(f"--repeats must be 1..{MAX_REPEATS}, got {n}")
    if n > 1 and target not in REPEATS_SUPPORTED:
        raise ValueError(f"--repeats {n}: {target} runs once per deploy (supported: {', '.join(REPEATS_SUPPORTED)}); "
                         "deploy it again for a re-run")
    if n == 1 and rep["repeatable"] == "no":
        return 1, f"--repeats 1 on {rep['label']}: results --compare will say 'needs re-run' for this side"
    return n, f"--repeats {n}"


def plan_line(target: str, repeats: int, episodes_per_run: int | None) -> str | None:
    """One line before deploy: runs x episodes and the instance hours expected from measured runs (None when
    the episode count is unknown)."""
    if not episodes_per_run:
        return None
    total = repeats * episodes_per_run
    line = f"{repeats} run(s) x {episodes_per_run} episodes = {total} episodes"
    m = MEASURED_RUN_HOURS.get(target)
    if m:
        lo, hi = (round(total / m["episodes"] * m[k], 1) for k in ("hours_min", "hours_max"))
        line += (f", ~{lo}-{hi} instance-h (scaled from measured {m['episodes']}-episode runs of "
                 f"{m['hours_min']}-{m['hours_max']} h; set-up is shared, so this is an upper bound)")
    return line


def check_fit(target: str, repeats: int, plan: dict) -> str | None:
    """Error text when `repeats` runs are not expected to fit one deploy (instance.creationpolicy_timeout of the
    variant), else None. Only the 1-shard RoboLab subset has measured run times; other variants take 1 run."""
    if repeats <= 1 or not plan:
        return None
    if plan.get("shards", 1) > 1:
        return (f"--repeats {repeats}: run time is measured only for the 1-shard subset, this variant runs "
                f"{plan['shards']} shards; use --repeats 1 and deploy again for the re-run.")
    m = MEASURED_RUN_HOURS.get(target)
    if not (m and plan.get("episodes") and plan.get("deploy_timeout_min")):
        return None
    need_min = repeats * plan["episodes"] / m["episodes"] * m["hours_max"] * 60
    if need_min > plan["deploy_timeout_min"]:
        return (f"--repeats {repeats}: ~{need_min / 60:.1f} h expected, more than this variant's deploy limit "
                f"({plan['deploy_timeout_min'] / 60:.1f} h, instance.creationpolicy_timeout). "
                "Use --repeats 1 and deploy again for the re-run.")
    return None
