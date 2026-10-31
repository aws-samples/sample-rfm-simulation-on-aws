"""Restrict openpi's examples/libero/main.py task loop to the LIBERO-Plus variants picked for this task.

usage: python3 openpi_task_ids_patch.py <main.py> <comma-separated 0-based task ids>

Upstream iterates every task of the suite (`for task_id in tqdm.tqdm(range(num_tasks_in_suite)):`); a
LIBERO-Plus suite has ~2,400 variants. The LIBERO client runs in a container whose environment is fixed
by examples/libero/compose.yml, so the ids are written into the script as a literal list instead of
being read from an environment variable. The replay video name is also prefixed with the variant id:
upstream names it after the instruction only, so the variants of one dimension that share an
instruction (camera, light, layout, ...) would overwrite each other's video.

Run after stackpatch.py (which restores the upstream file first); fails loudly if a site is missing.
"""
import re
import sys

p, ids_csv = sys.argv[1], sys.argv[2]
ids = [int(x) for x in ids_csv.split(",") if x.strip()]
if not ids:
    sys.exit("LIBERO-PLUS PATCH FAIL: no task ids")
s = open(p).read()
s, k = re.subn(r"for task_id in tqdm\.tqdm\(range\(num_tasks_in_suite\)\):",
               f"for task_id in tqdm.tqdm({ids!r}):", s)
if k != 1:
    sys.exit(f"LIBERO-PLUS PATCH FAIL: expected 1 task loop in {p}, found {k}")
s, k = re.subn(r'f"rollout_\{task_segment\}_\{suffix\}\.mp4"', 'f"rollout_{task_id:05d}_{task_segment}_{suffix}.mp4"', s)
if k != 1:
    sys.exit(f"LIBERO-PLUS PATCH FAIL: expected 1 replay video name in {p}, found {k}")
open(p, "w").write(s)
print(f"LIBERO-PLUS PATCH: task loop runs {len(ids)} variant(s), videos prefixed with the variant id")
