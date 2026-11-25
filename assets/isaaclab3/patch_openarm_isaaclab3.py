#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Make enactic/openarm_isaac_lab @ bad82e2 (written for Isaac Lab 2.3) run on Isaac Lab 3.0.

Why this exists
---------------
The openarm-isaac (bimanual Reach) and openarm-lift-act (unimanual Lift-Cube) targets moved to
Isaac Lab 3.0.0 for CVE-2026-65141 (eval injection in ``isaaclab.utils.string.string_to_callable``,
fixed in 3.0). Upstream openarm_isaac_lab has no 3.0 release, so the pinned tree is patched in place
after the clone. Only the Isaac Lab 3.0 breaking changes that the two task chains hit are touched
(docs/source/isaaclab/migrating.rst of Isaac Lab v3.0.0):

  * ``.data.*`` of assets and sensors returns a ``ProxyArray``. Its deprecation bridge covers torch
    functions and indexing, but NOT ``@torch.jit.script`` functions: ``combine_frame_transforms``,
    ``quat_mul`` and ``quat_error_magnitude`` reject a ProxyArray argument, and the Reach rewards and
    the Lift rewards / observations / ``object_reached_goal`` termination (the demo recorder's success
    signal) pass ``robot.data.root_pos_w`` straight into them. Every ``<x>.data.<attr>`` read in those
    mdp modules gets ``.torch``.
  * Quaternions changed from (w, x, y, z) to (x, y, z, w): the Lift table (90 deg about z) and the
    cube (identity) initial rotations are rewritten. The bimanual Reach chain has no literal
    quaternion.
  * The 2.x noise aliases ``AdditiveUniformNoiseCfg`` / ``AdditiveGaussianNoiseCfg`` were removed: the
    bimanual Reach env imports the first one, it becomes ``UniformNoiseCfg`` (the alias target in 2.3).
  * ``SimulationCfg.physx`` is gone (``sim.physics`` takes ``isaaclab_physx.physics.PhysxCfg``): the
    Lift env's four ``self.sim.physx.<field> = v`` lines become one ``PhysxCfg(...)`` with the same
    values (the other PhysX fields keep their defaults, as before).

Deprecated-but-working 2.x names are left alone (``effort_limit_sim`` / ``velocity_limit_sim``,
``RigidBodyPropertiesCfg``, ``ArticulationRootPropertiesCfg``, ``ViewerCfg``, ``.copy()``/``.replace()``).

Design: the same rules as the repo's other source patchers. Exact full-line anchors, each required
exactly once; idempotent (an already patched tree is a no-op); fail loud when an anchor is neither in
its unpatched nor its patched form, so a pin bump that moves these lines stops the run here instead of
in the simulator. The ``.torch`` rewrite checks the number of reads per file against the pinned tree.

Usage:  python patch_openarm_isaaclab3.py /path/to/openarm_isaac_lab
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PKG = "source/openarm/openarm/tasks/manager_based/openarm_manipulation"

# (relative path, anchor lines, replacement lines): the anchor block (consecutive exact lines) is
# replaced by the replacement block.
LINE_EDITS: list[tuple[str, list[str], list[str]]] = [
    (
        f"{PKG}/unimanual/lift/lift_env_cfg.py",
        ["            pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]"],
        ["            pos=[0.5, 0, 0], rot=[0, 0, 0.707, 0.707]  # PATCHED(isaaclab3): (x, y, z, w)"],
    ),
    (
        f"{PKG}/unimanual/lift/lift_env_cfg.py",
        [
            "        self.sim.physx.bounce_threshold_velocity = 0.01",
            "        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4",
            "        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024",
            "        self.sim.physx.friction_correlation_distance = 0.00625",
        ],
        [
            "        # PATCHED(isaaclab3): SimulationCfg.physx -> sim.physics = PhysxCfg(...), same values",
            "        from isaaclab_physx.physics import PhysxCfg",
            "",
            "        self.sim.physics = PhysxCfg(",
            "            bounce_threshold_velocity=0.01,",
            "            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,",
            "            gpu_total_aggregate_pairs_capacity=16 * 1024,",
            "            friction_correlation_distance=0.00625,",
            "        )",
        ],
    ),
    (
        # the 2.x aliases AdditiveUniformNoiseCfg / AdditiveGaussianNoiseCfg (= UniformNoiseCfg / GaussianNoiseCfg,
        # isaaclab 2.3 utils/noise/__init__.py) are gone in 3.0: ImportError on the bimanual Reach chain (GPU job j4)
        f"{PKG}/bimanual/reach/reach_env_cfg.py",
        ["from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise"],
        ["from isaaclab.utils.noise import UniformNoiseCfg as Unoise  # PATCHED(isaaclab3): 2.x alias removed"],
    ),
    (
        f"{PKG}/unimanual/reach/reach_env_cfg.py",   # not on either target's chain today; same alias, kept importable
        ["from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise"],
        ["from isaaclab.utils.noise import UniformNoiseCfg as Unoise  # PATCHED(isaaclab3): 2.x alias removed"],
    ),
    (
        f"{PKG}/unimanual/lift/config/joint_pos_env_cfg.py",
        ["                pos=[0.4, 0, 0.055], rot=[1, 0, 0, 0]"],
        ["                pos=[0.4, 0, 0.055], rot=[0, 0, 0, 1]  # PATCHED(isaaclab3): (x, y, z, w)"],
    ),
]

# ProxyArray reads that feed torch.jit.script math: file -> number of `<x>.data.<attr>` reads at the pin.
TORCH_READS: dict[str, int] = {
    f"{PKG}/bimanual/reach/mdp/rewards.py": 8,
    f"{PKG}/unimanual/lift/mdp/terminations.py": 3,
    f"{PKG}/unimanual/lift/mdp/rewards.py": 7,
    f"{PKG}/unimanual/lift/mdp/observations.py": 3,
}
_READ = re.compile(r"\b(\w+\.data\.[A-Za-z_]\w*)\b(?!\.torch)")
_READ_DONE = re.compile(r"\b\w+\.data\.[A-Za-z_]\w*\.torch\b")


def _find_block(lines: list[str], block: list[str]) -> list[int]:
    n = len(block)
    return [i for i in range(len(lines) - n + 1) if lines[i : i + n] == block]


def apply_line_edits(root: Path) -> int:
    applied = 0
    for rel, anchor, replacement in LINE_EDITS:
        fp = root / rel
        if not fp.exists():
            raise SystemExit(f"[patch-openarm] FATAL: expected file missing: {rel}")
        lines = fp.read_text().split("\n")
        hits = _find_block(lines, anchor)
        if _find_block(lines, replacement) and not hits:
            print(f"[patch-openarm] skip (already applied): {rel}  ({anchor[0].strip()[:50]})")
            continue
        if len(hits) != 1:
            raise SystemExit(
                f"[patch-openarm] FATAL: anchor found {len(hits)}x in {rel} (expected 1): {anchor[0]!r}\n"
                "               Source layout changed (pin moved?) - review the patcher."
            )
        i = hits[0]
        lines[i : i + len(anchor)] = replacement
        fp.write_text("\n".join(lines))
        print(f"[patch-openarm] applied: {rel}  ({anchor[0].strip()[:50]})")
        applied += 1
    return applied


def apply_torch_reads(root: Path) -> int:
    applied = 0
    for rel, expected in TORCH_READS.items():
        fp = root / rel
        if not fp.exists():
            raise SystemExit(f"[patch-openarm] FATAL: expected file missing: {rel}")
        text = fp.read_text()
        todo = len(_READ.findall(text))
        done = len(_READ_DONE.findall(text))
        if todo == 0 and done == expected:
            print(f"[patch-openarm] skip (already applied): {rel}  ({done} .torch reads)")
            continue
        if todo != expected or done != 0:
            raise SystemExit(
                f"[patch-openarm] FATAL: {rel} has {todo} unpatched + {done} patched .data reads, "
                f"expected {expected} unpatched. Source changed (pin moved?) - review the patcher."
            )
        fp.write_text(_READ.sub(r"\1.torch", text))
        print(f"[patch-openarm] applied: {rel}  ({todo} .data reads -> .torch)")
        applied += 1
    return applied


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_openarm_isaaclab3.py <openarm_isaac_lab checkout>")
    root = Path(sys.argv[1]).resolve()
    if not (root / PKG).is_dir():
        raise SystemExit(f"[patch-openarm] FATAL: {root} does not look like an openarm_isaac_lab checkout")
    n = apply_line_edits(root) + apply_torch_reads(root)
    print(f"[patch-openarm] OK - {n} edit(s) applied for Isaac Lab 3.0.")


if __name__ == "__main__":
    main()
