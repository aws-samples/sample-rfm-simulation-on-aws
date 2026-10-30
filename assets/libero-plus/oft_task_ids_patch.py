"""Restrict OpenVLA-OFT's run_libero_eval.py task loop to the LIBERO-Plus variants picked for this task.

usage: python3 oft_task_ids_patch.py <run_libero_eval.py>

Upstream iterates every task of the suite (`for task_id in tqdm.tqdm(range(num_tasks)):`); a LIBERO-Plus
suite has ~2,400 variants, so the loop is rewritten to read the comma-separated 0-based ids from the
LIBERO_PLUS_TASK_IDS environment variable at run time. Run after stackpatch.py (which restores the
upstream file first); fails loudly if the loop is not found.
"""
import re
import sys

p = sys.argv[1]
s = open(p).read()
s2, k = re.subn(r"for task_id in tqdm\.tqdm\(range\(num_tasks\)\):",
                "for task_id in tqdm.tqdm([int(x) for x in __import__('os').environ['LIBERO_PLUS_TASK_IDS'].split(',')]):",
                s)
if k != 1:
    sys.exit(f"LIBERO-PLUS PATCH FAIL: expected 1 task loop in {p}, found {k}")
open(p, "w").write(s2)
print("LIBERO-PLUS PATCH: task loop reads LIBERO_PLUS_TASK_IDS")
