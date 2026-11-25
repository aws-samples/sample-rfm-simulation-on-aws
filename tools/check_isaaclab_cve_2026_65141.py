#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Check whether an installed Isaac Lab is affected by CVE-2026-65141.

CVE-2026-65141 (NVIDIA security bulletin, October 2026, CVSS 7.8, CWE-95): Isaac Lab 0.0 - 2.3.2
``isaaclab.utils.string.string_to_callable()`` passes any string that parses as a lambda to
``eval()``. Manager term configs and ``update_dict`` / ``from_dict`` resolve config strings through it,
so a crafted config (file, override, serialized cfg) runs code when it is loaded. Fixed in Isaac Lab
3.0.0-beta2 and later (the lambda is AST-checked and evaluated without builtins).

Run it with the Python that has isaaclab installed, e.g. inside an NGC image:

    /workspace/isaaclab/isaaclab.sh -p tools/check_isaaclab_cve_2026_65141.py        # 3.0
    /workspace/isaaclab/_isaac_sim/python.sh tools/check_isaaclab_cve_2026_65141.py  # 2.x

or point it at a ``string.py`` source file (no Isaac Lab install needed):

    python tools/check_isaaclab_cve_2026_65141.py --source IsaacLab/source/isaaclab/isaaclab/utils/string.py

Exit status: 0 = fixed (every crafted string rejected, the benign lambda still works),
1 = vulnerable, 2 = could not test (string_to_callable not importable).

Safety: no shell command and no network access is ever run. The crafted strings only create empty marker
files in a fresh temporary directory, which is deleted afterwards; the bulletin-style ``os.system``
string is resolved but never called (on a vulnerable version, resolving it returns the lambda; the
payload would only run when the lambda is called, which this check does not do).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path


def _load(source: str | None):
    """Return (string_to_callable, description of where it came from)."""
    if source:
        path = Path(source).resolve()
        spec = importlib.util.spec_from_file_location("_isaaclab_utils_string", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.string_to_callable, str(path)
    try:
        from isaaclab.utils.string import string_to_callable

        return string_to_callable, "isaaclab.utils.string"
    except Exception as e:  # noqa: BLE001 - isaaclab.utils pulls optional deps; fall back to the file itself
        spec = importlib.util.find_spec("isaaclab")  # locates the package without importing it
        if spec is None or not spec.submodule_search_locations:
            raise ImportError(f"isaaclab is not installed in this Python ({sys.executable}): {e!r}") from e
        path = Path(next(iter(spec.submodule_search_locations))) / "utils" / "string.py"
        return _load(str(path))[0], f"{path} (isaaclab.utils import failed: {e!r})"


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("isaaclab")
    except Exception:  # noqa: BLE001
        return "unknown"


def run(string_to_callable, workdir: Path) -> dict:
    m_import = workdir / "marker-import-default"
    m_open = workdir / "marker-open-default"
    m_call = workdir / "marker-call"
    crafted = {
        # Runs at resolution time: a lambda default argument is evaluated by eval() itself.
        "import_in_default": f"lambda x=__import__('pathlib').Path({str(m_import)!r}).touch(): x",
        # Same without a dunder name: needs only the eval() builtins.
        "builtin_in_default": f"lambda x=open({str(m_open)!r}, 'w').close(): x",
        # Runs when the resolved term is called (as a manager calls a term func every step).
        "import_in_body": f"lambda x: __import__('pathlib').Path({str(m_call)!r}).touch()",
        # The bulletin-style payload: resolved only, never called.
        "os_system_never_called": "lambda x: __import__('os').system('touch /tmp/pwned')",
    }
    results = {}
    for key, text in crafted.items():
        try:
            fn = string_to_callable(text)
            accepted = callable(fn)
            if accepted and key == "import_in_body":
                fn(0)  # harmless: touches the marker in the temp dir
            results[key] = "accepted"
        except Exception as e:  # noqa: BLE001 - any rejection counts
            results[key] = f"rejected ({type(e).__name__})"
    markers = {p.name: p.exists() for p in (m_import, m_open, m_call)}
    try:
        benign = string_to_callable("lambda x: x + 1")(1) == 2
    except Exception as e:  # noqa: BLE001
        benign = f"error ({type(e).__name__}: {e})"
    try:
        module_attr = string_to_callable("math:sqrt")(9.0) == 3.0
    except Exception as e:  # noqa: BLE001
        module_attr = f"error ({type(e).__name__}: {e})"
    fixed = (all(v.startswith("rejected") for v in results.values()) and not any(markers.values())
             and benign is True and module_attr is True)
    return {"crafted": results, "markers_created": markers, "benign_lambda_ok": benign,
            "module_attr_ok": module_attr, "verdict": "FIXED" if fixed else "VULNERABLE"}


def main() -> int:
    p = argparse.ArgumentParser(description="Check an Isaac Lab install for CVE-2026-65141.")
    p.add_argument("--source", help="test this isaaclab/utils/string.py instead of the installed isaaclab")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    args = p.parse_args()
    try:
        string_to_callable, origin = _load(args.source)
    except Exception as e:  # noqa: BLE001
        print(f"[cve-2026-65141] CANNOT TEST: {e}", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="cve-2026-65141-") as tmp:
        report = run(string_to_callable, Path(tmp))
    report.update({"isaaclab_version": _version() if not args.source else "n/a (--source)", "origin": origin,
                   "python": sys.version.split()[0]})
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"[cve-2026-65141] isaaclab {report['isaaclab_version']} ({origin}), python {report['python']}")
        for key, value in report["crafted"].items():
            print(f"[cve-2026-65141]   {key}: {value}")
        print(f"[cve-2026-65141]   markers created: {report['markers_created']}")
        print(f"[cve-2026-65141]   benign 'lambda x: x + 1': {report['benign_lambda_ok']}, "
              f"'math:sqrt': {report['module_attr_ok']}")
        print(f"[cve-2026-65141] {report['verdict']}")
    return 0 if report["verdict"] == "FIXED" else 1


if __name__ == "__main__":
    sys.exit(main())
