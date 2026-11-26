#!/usr/bin/env python3
"""Make a pinned LeRobot source tree installable into the Isaac Lab 3.0 environment.

Why this exists
---------------
openarm-isaac installs LeRobot (the pi05 inference stack) into the same Python environment as
Isaac Lab, because the policy runs in-process on the sim GPU. On Isaac Lab 3.0 that environment is
the image's uv venv (Python 3.12, torch 2.11 cu130, numpy 2.5, packaging 26). LeRobot @ d1b1c5c
(v0.5.2) needs Python >= 3.12, so the Python 3.11 rewrites the Isaac Lab 2.3 image needed
(PEP-695 syntax, the requires-python gate) are gone. What is left is the version overlap:
LeRobot caps ``numpy<2.3.0`` (for opencv-python-headless), ``packaging<26.0`` and
``setuptools<81.0.0``, while Isaac Lab 3.0 ships numpy 2.5.1, packaging 26.0 and setuptools 81.

A plain ``pip install lerobot[pi]`` would downgrade them under Isaac Sim (the 2.3 lesson: a numpy
change under Isaac Sim's torch broke ``import torch``). So the Isaac Lab environment wins for the
packages Isaac Sim itself runs on:

  * for each PROTECTED package that is installed, every LeRobot requirement on it whose specifier
    excludes the installed version loses its version bounds (name, extras and marker kept), and
  * a pip constraints file pins every installed PROTECTED package to its installed version, so a
    transitive demand for another version is a loud resolver error instead of a silent swap.

LeRobot uses numpy through plain array calls (np.digitize / np.linspace in pi05) and packaging
through version.parse, both stable across these versions. Other LeRobot pins (transformers, av, ...)
are left alone: pip installs LeRobot's versions, Isaac Lab imports them lazily if at all.

Idempotent (a relaxed requirement no longer conflicts), fail-loud on a missing pyproject.

Usage:  python patch_lerobot_isaaclab.py <lerobot-src-root> --constraints-out <file>
        [--installed name=version ...]   (override the detected versions; for tests)
"""

from __future__ import annotations

import argparse
import sys
from importlib import metadata
from pathlib import Path

import tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# Packages Isaac Sim / Isaac Lab run on; the installed version must not move.
PROTECTED = ("numpy", "packaging", "setuptools", "torch", "torchvision")


def _installed(overrides: dict[str, str]) -> dict[str, str]:
    found = {}
    for name in PROTECTED:
        if name in overrides:
            found[name] = overrides[name]
            continue
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return found


def _requirement_strings(pyproject: dict) -> list[str]:
    project = pyproject.get("project", {})
    reqs = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        reqs.extend(group)
    return reqs


def _unbounded(req: Requirement) -> str:
    out = req.name
    if req.extras:
        out += "[" + ",".join(sorted(req.extras)) + "]"
    if req.marker is not None:
        out += f"; {req.marker}"
    return out


def relax(root: Path, installed: dict[str, str]) -> list[tuple[str, str]]:
    """Rewrite conflicting requirement strings in pyproject.toml; return (old, new) pairs."""
    fp = root / "pyproject.toml"
    if not fp.exists():
        raise SystemExit(f"[patch-lerobot] FATAL: {root} does not look like a lerobot checkout (no pyproject.toml)")
    text = fp.read_text()
    pyproject = tomllib.loads(text)
    changes = []
    for raw in _requirement_strings(pyproject):
        try:
            req = Requirement(raw)
        except Exception:  # noqa: BLE001 - self-references like "lerobot[x]" parse fine; anything odd is left as is
            continue
        name = canonicalize_name(req.name)
        version = installed.get(name)
        if version is None or not req.specifier or req.specifier.contains(version, prereleases=True):
            continue
        new = _unbounded(req)
        old_quoted, new_quoted = f'"{raw}"', f'"{new}"'
        if old_quoted not in text:
            raise SystemExit(f"[patch-lerobot] FATAL: requirement {raw!r} not found verbatim in pyproject.toml")
        text = text.replace(old_quoted, new_quoted)
        changes.append((raw, new))
    if changes:
        fp.write_text(text)
    return changes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("root", type=Path)
    p.add_argument("--constraints-out", type=Path, required=True)
    p.add_argument("--installed", action="append", default=[], metavar="NAME=VERSION")
    args = p.parse_args()
    overrides = dict(item.split("=", 1) for item in args.installed)
    installed = _installed(overrides)
    for name, version in sorted(installed.items()):
        print(f"[patch-lerobot] Isaac Lab env: {name}=={version}")
    missing = sorted(set(PROTECTED) - set(installed))
    if missing:
        print(f"[patch-lerobot] [warn] not installed (no constraint): {', '.join(missing)}")
    for old, new in relax(args.root.resolve(), installed):
        print(f"[patch-lerobot] relaxed: {old!r} -> {new!r}")
    args.constraints_out.write_text("".join(f"{n}=={v}\n" for n, v in sorted(installed.items())))
    print(f"[patch-lerobot] OK - constraints written to {args.constraints_out}")


if __name__ == "__main__":
    sys.exit(main())
