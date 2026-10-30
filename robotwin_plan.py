"""RoboTwin 2.0 run plan for the LeRobot x RoboTwin targets (smolvla-robotwin, lingbot-vla2).

generate.py turns the plan into the flat task list of the userdata (one ``lerobot-eval`` call per
RoboTwin task and setting = one ``task-N/`` prefix in the results bucket); deploy.py prints the
episode count and a runtime estimate before it deploys.

Protocols (``--robotwin-protocol``):

``subset``      the ``tasks:`` entries of ``models/<vla>.yaml`` (default: 5 tasks x 5 episodes, clean).
``clean``       official Easy setting: every task x 100 episodes, ``demo_clean``.
``randomized``  official Hard setting: every task x 100 episodes, ``demo_randomized``.
``full``        both settings (the leaderboard protocol): 2 x 49 x 100 = 9,800 episodes.

The non-subset protocols do not fit one run (the CloudFormation WaitCondition is capped at 12 h), so
``--robotwin-shard K/N`` keeps every N-th (setting, task) unit starting at K (1-based); run the shards
one after the other and read them together with ``vlasim results --all-runs``.

Stdlib-only and side-effect free on import (deploy.py / tests import it without the AWS deps).
"""

# The 50 tasks of RoboTwin 2.0, mirrored from ROBOTWIN_TASKS in lerobot/envs/robotwin.py (pinned
# LeRobot commit) = the envs/ directory of RoboTwin @0aeea2d6.
TASKS = (
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size", "click_alarmclock",
    "click_bell", "dump_bin_bigbin", "grab_roller", "handover_block", "handover_mic", "hanging_mug", "lift_pot",
    "move_can_pot", "move_pillbottle_pad", "move_playingcard_away", "move_stapler_pad", "open_laptop",
    "open_microwave", "pick_diverse_bottles", "pick_dual_bottles", "place_a2b_left", "place_a2b_right",
    "place_bread_basket", "place_bread_skillet", "place_burger_fries", "place_can_basket",
    "place_cans_plasticbox", "place_container_plate", "place_dual_shoes", "place_empty_cup", "place_fan",
    "place_mouse_pad", "place_object_basket", "place_object_scale", "place_object_stand", "place_phone_stand",
    "place_shoe", "press_stapler", "put_bottles_dustbin", "put_object_cabinet", "rotate_qrcode", "scan_object",
    "shake_bottle", "shake_bottle_horizontally", "stack_blocks_three", "stack_blocks_two", "stack_bowls_three",
    "stack_bowls_two", "stamp_seal", "turn_switch",
)
# LeRobot docs (robotwin.mdx): open_laptop's check_success() reads self.arm_tag, which is only set by the
# scripted expert, so a policy eval crashes on it. Left out of "all" (it can still be named explicitly).
BROKEN = ("open_laptop",)
ALL_TASKS = tuple(t for t in TASKS if t not in BROKEN)

# Official per-task step limits (RoboTwin task_config/_eval_step_limit.yml @0aeea2d6). The userdata reads
# the same file on the instance and passes it as --env.episode_length; this copy is only for estimates.
STEP_LIMIT = {
    "adjust_bottle": 400, "beat_block_hammer": 400, "blocks_ranking_rgb": 1200, "blocks_ranking_size": 1200,
    "click_alarmclock": 400, "click_bell": 400, "dump_bin_bigbin": 600, "grab_roller": 400, "handover_block": 800,
    "handover_mic": 600, "hanging_mug": 900, "lift_pot": 400, "move_can_pot": 400, "move_pillbottle_pad": 400,
    "move_playingcard_away": 400, "move_stapler_pad": 400, "open_laptop": 700, "open_microwave": 1500,
    "pick_diverse_bottles": 400, "pick_dual_bottles": 400, "place_a2b_left": 400, "place_a2b_right": 400,
    "place_bread_basket": 700, "place_bread_skillet": 500, "place_burger_fries": 500, "place_can_basket": 700,
    "place_cans_plasticbox": 800, "place_container_plate": 400, "place_dual_shoes": 600, "place_empty_cup": 500,
    "place_fan": 400, "place_mouse_pad": 400, "place_object_basket": 700, "place_object_scale": 400,
    "place_object_stand": 400, "place_phone_stand": 400, "place_shoe": 500, "press_stapler": 400,
    "put_bottles_dustbin": 1700, "put_object_cabinet": 700, "rotate_qrcode": 400, "scan_object": 500,
    "shake_bottle": 700, "shake_bottle_horizontally": 700, "stack_blocks_three": 1200, "stack_blocks_two": 800,
    "stack_bowls_three": 1200, "stack_bowls_two": 900, "stamp_seal": 400, "turn_switch": 400,
}

# Simulator stack pins shared by every RoboTwin target (models/<vla>.yaml `robotwin:` overrides a key).
# Source of truth for the pip set: LeRobot docker/Dockerfile.benchmark.robotwin @552777b7 (minus pytorch3d,
# which RoboTwin only imports for point clouds, and with mplib / sapien installed --no-deps, see the template).
PINS = {
    "repo": "https://github.com/RoboTwin-Platform/RoboTwin.git",
    "commit": "0aeea2d669c0f8516f4d5785f0aa33ba812c14b4",  # = LeRobot's ROBOTWIN_SHA (2026-04-19)
    "assets_repo": "TianxingChen/RoboTwin2.0",
    # Last dataset revision before the 2026-09-22 embodiment (curobo.yml) refresh; embodiments.zip and
    # objects.zip here are byte-identical to the 2025-12 files the pinned code shipped with.
    "assets_revision": "981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042",
    "curobo_repo": "https://github.com/NVlabs/curobo.git",
    "curobo_ref": "v0.7.8",  # RoboTwin script/_install.sh; still exposes the curobo.types.math API it imports
    "curobo_commit": "d64c4b005459db10c5dd867d8b30a87d5bda9bdb",
    "sapien_version": "3.0.0b1",
    "mplib_version": "0.2.1",
    # Installed against the frozen LeRobot lock (constraints), so none of them can move a locked package.
    "pip": ["toppra==0.6.10", "transforms3d==0.4.2", "trimesh==4.4.3", "open3d==0.19.0", "h5py",
            "pyyaml", "termcolor", "requests", "lxml", "networkx", "pyperclip"],
    # Wall-clock budget per simulated step for the per-task timeout (not a speed estimate).
    "sec_per_step": 1.0,
    # Run lerobot-eval with HF_HUB_OFFLINE=1 against the pinned snapshots (see the template, step 6).
    "hf_offline": True,
}

# RoboTwin setting name (results label RoboTwin2/<setting>) -> task_config file the env loads.
SETTINGS = {"clean": "demo_clean", "randomized": "demo_randomized"}
PROTOCOLS = ("subset", "clean", "randomized", "full")
OFFICIAL_EPISODES = 100
# RoboTwin's script/eval_policy.py starts at seed 100000 * (1 + seed) with seed 0.
OFFICIAL_START_SEED = 100000


def label(setting: str) -> str:
    return f"RoboTwin2/{setting}"


def _task_list(value) -> list:
    if value in ("all", None):
        return list(ALL_TASKS)
    tasks = [t.strip() for t in (value.split(",") if isinstance(value, str) else value) if str(t).strip()]
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        raise ValueError(f"unknown RoboTwin task(s): {', '.join(unknown)}")
    if len(set(tasks)) != len(tasks):
        raise ValueError("duplicate RoboTwin task in the list")
    return tasks


def parse_shard(shard: str):
    """'K/N' (1-based) -> (K, N); '' -> None."""
    if not shard:
        return None
    try:
        k, n = (int(x) for x in shard.split("/"))
    except ValueError:
        raise ValueError(f"--robotwin-shard must look like K/N, got {shard!r}") from None
    if n < 1 or not 1 <= k <= n:
        raise ValueError(f"--robotwin-shard {shard}: need 1 <= K <= N")
    return k, n


def plan(yaml_tasks: list, protocol: str = "subset", shard: str = "", seed: int = OFFICIAL_START_SEED) -> list:
    """Flat list of {setting, task, n_episodes, seed, description}: one lerobot-eval call each.

    ``yaml_tasks`` = the ``tasks:`` entries of models/<vla>.yaml, each
    ``{setting, robotwin_tasks: [..] | all, n_episodes, seed?, description?}``; used for ``subset`` only.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown RoboTwin protocol {protocol!r} (choose from {', '.join(PROTOCOLS)})")
    units = []
    if protocol == "subset":
        for entry in yaml_tasks:
            setting = entry.get("setting", "clean")
            if setting not in SETTINGS:
                raise ValueError(f"tasks[].setting must be one of {', '.join(SETTINGS)}, got {setting!r}")
            n = int(entry.get("n_episodes", 0))
            if n < 1:
                raise ValueError("tasks[].n_episodes must be >= 1")
            for task in _task_list(entry.get("robotwin_tasks")):
                units.append(dict(setting=setting, task=task, n_episodes=n, seed=int(entry.get("seed", seed)),
                                  description=entry.get("description", "")))
    else:
        settings = list(SETTINGS) if protocol == "full" else [protocol]
        for setting in settings:
            for task in ALL_TASKS:
                units.append(dict(setting=setting, task=task, n_episodes=OFFICIAL_EPISODES, seed=seed,
                                  description=f"RoboTwin 2.0 official protocol ({setting}, {OFFICIAL_EPISODES} ep)"))
    if not units:
        raise ValueError("the RoboTwin plan is empty")
    sh = parse_shard(shard)
    if sh:
        k, n = sh
        units = units[k - 1::n]
        if not units:
            raise ValueError(f"--robotwin-shard {shard} selects no task")
    return units


def episodes(units: list) -> int:
    return sum(u["n_episodes"] for u in units)


def worst_case_hours(units: list, sec_per_step: float) -> float:
    """Upper bound of the episode loop: every episode runs to its step limit (successes stop earlier)."""
    return sum(u["n_episodes"] * STEP_LIMIT[u["task"]] for u in units) * sec_per_step / 3600.0
