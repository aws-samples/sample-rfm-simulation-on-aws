"""LIBERO real2sim swap: runtime patches and task selection (runs on the instance; stdlib only).

  python3 r2s_select.py patch-hope <hope_objects.py>
      One-time, idempotent: hope_objects.py reads the HOPE asset directory from a module global
      (R2S_HOPE_DIR, default "stable_hope_objects" = stock LIBERO) instead of the literal path.
  python3 r2s_select.py use <hope_objects.py> <asset subdir>
      Point that global at one asset directory (stable_hope_objects | stable_hope_objects_r2s_<mode>)
      before an evaluation process starts. Works for any process that imports this LIBERO checkout
      (OpenVLA-OFT's conda env, openpi's Docker-mounted third_party/libero), no env var plumbing.
  python3 r2s_select.py tasks <libero package dir> <suite> <object,...> [--any] [--manifest out.json]
      Print the 0-based task ids (comma-separated) of <suite> whose BDDL lists one of the objects in
      (:obj_of_interest ...) (default) or anywhere in (:objects ...) (--any).
  python3 r2s_select.py loop <eval script> <id,id,...>
      Restrict the eval script's suite loop (`for task_id in tqdm.tqdm(range(...)):`, OpenVLA-OFT
      run_libero_eval.py and openpi examples/libero/main.py) to these ids. Run after stackpatch.py,
      which restores the upstream script first.
"""
import ast
import json
import os
import re
import sys

GLOBAL_LINE = 'R2S_HOPE_DIR = "stable_hope_objects"  # real2sim swap: set by r2s_select.py use\n'
STOCK_PATH = 'f"assets/stable_hope_objects/{obj_name}/{obj_name}.xml"'
PATCHED_PATH = 'f"assets/{R2S_HOPE_DIR}/{obj_name}/{obj_name}.xml"'
GLOBAL_RE = re.compile(r'^R2S_HOPE_DIR = "([A-Za-z0-9_]+)"', re.M)
LOOP_RE = re.compile(r"for task_id in tqdm\.tqdm\(range\([A-Za-z_][A-Za-z0-9_.]*\)\):")


def patch_hope(path):
    s = open(path).read()
    if GLOBAL_RE.search(s) and PATCHED_PATH in s:
        return "already patched"
    if s.count(STOCK_PATH) != 1:
        raise SystemExit(f"r2s patch-hope: expected 1 stock asset path in {path}, found {s.count(STOCK_PATH)}")
    s = s.replace(STOCK_PATH, PATCHED_PATH)
    anchor = "from libero.libero.envs.base_object import register_object\n"
    if anchor not in s:
        raise SystemExit(f"r2s patch-hope: import anchor not found in {path}")
    s = s.replace(anchor, anchor + "\n" + GLOBAL_LINE, 1)
    open(path, "w").write(s)
    return "patched"


def use(path, subdir):
    s = open(path).read()
    if not GLOBAL_RE.search(s):
        raise SystemExit(f"r2s use: {path} is not patched (run patch-hope first)")
    assets = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), "..", "..", "assets"))
    if not os.path.isdir(os.path.join(assets, subdir)):
        raise SystemExit(f"r2s use: no asset dir {subdir!r} in {assets}")
    open(path, "w").write(GLOBAL_RE.sub(f'R2S_HOPE_DIR = "{subdir}"', s, count=1))
    return subdir


def suite_tasks(pkg_dir, suite):
    """Task names of a suite in benchmark order, from libero/libero/benchmark/libero_suite_task_map.py."""
    src = open(os.path.join(pkg_dir, "benchmark", "libero_suite_task_map.py")).read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "libero_task_map" for t in node.targets):
            m = ast.literal_eval(node.value)
            if suite not in m:
                raise SystemExit(f"r2s tasks: unknown suite {suite!r} (have {sorted(m)})")
            return m[suite]
    raise SystemExit("r2s tasks: libero_task_map not found")


def bddl_section(text, key):
    m = re.search(r"\(:" + key + r"\b(.*?)\)", text, re.S)
    return m.group(1).split() if m else []


def bddl_objects(text):
    """Object instance -> category from (:objects ...) and (:fixtures ...) (`name_1 name_2 - category`)."""
    out = {}
    for key in ("objects", "fixtures"):
        toks = bddl_section(text, key)
        pending = []
        i = 0
        while i < len(toks):
            if toks[i] == "-" and i + 1 < len(toks):
                for n in pending:
                    out[n] = toks[i + 1]
                pending = []
                i += 2
                continue
            pending.append(toks[i])
            i += 1
    return out


def select_tasks(pkg_dir, suite, objects, any_role=False):
    names = suite_tasks(pkg_dir, suite)
    out = []
    for i, name in enumerate(names):
        path = os.path.join(pkg_dir, "bddl_files", suite, name + ".bddl")
        text = open(path).read()
        cats = bddl_objects(text)
        pool = list(cats) if any_role else bddl_section(text, "obj_of_interest")
        hit = sorted({cats.get(n, "") for n in pool} & set(objects))
        if hit:
            lang = re.search(r"\(:language\s+(.*?)\)", text, re.S)
            out.append(dict(task_id=i, name=name, objects=hit,
                            language=" ".join(lang.group(1).split()) if lang else name.replace("_", " ")))
    return out


def loop(path, ids):
    if not re.fullmatch(r"\d+(,\d+)*", ids):
        raise SystemExit(f"r2s loop: ids must be comma-separated integers, got {ids!r}")
    s = open(path).read()
    s2, k = LOOP_RE.subn(f"for task_id in tqdm.tqdm([{ids}]):", s)
    if k != 1:
        raise SystemExit(f"r2s loop: expected 1 suite loop in {path}, found {k}")
    open(path, "w").write(s2)


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    cmd = argv[0]
    if cmd == "patch-hope":
        print("r2s patch-hope:", patch_hope(argv[1]))
    elif cmd == "use":
        print("r2s use:", use(argv[1], argv[2]))
    elif cmd == "tasks":
        manifest = None
        if "--manifest" in argv:
            j = argv.index("--manifest")
            manifest = argv[j + 1]
            argv = argv[:j] + argv[j + 2:]
        any_role = "--any" in argv
        argv = [a for a in argv if a != "--any"]
        objects = [x for x in argv[3].split(",") if x]
        rows = select_tasks(argv[1], argv[2], objects, any_role)
        if not rows:
            raise SystemExit(f"r2s tasks: no {argv[2]} task has {'any' if any_role else 'an object of interest'} "
                             f"among {objects}")
        if manifest:
            with open(manifest, "w") as f:
                json.dump(rows, f, indent=1)
        print(",".join(str(r["task_id"]) for r in rows))
    elif cmd == "loop":
        loop(argv[1], argv[2])
        print(f"r2s loop: task ids {argv[2]}")
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
