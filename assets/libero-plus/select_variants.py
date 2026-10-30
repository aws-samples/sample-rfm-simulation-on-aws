"""Pick the LIBERO-Plus variants of one perturbation dimension to evaluate (runs on the instance).

usage: python3 select_variants.py <task_classification.json> <suite> <category> <n> <seed>
                                  [--format json|csv] [--manifest out.json]

task_classification.json ships with the LIBERO-Plus fork (libero/libero/benchmark/): per suite, a list
of {id (1-based, = position in the suite + 1), name, category, difficulty_level}. The fork's benchmark
keeps the suite order at task_order_index 0, so suite task index = id - 1 (what lerobot-eval
--env.task_ids and the OpenVLA-OFT task loop take).

n = 0 selects every variant of the category; otherwise n variants are drawn without replacement with
random.Random("<seed>:<suite>:<category>"), so the same seed always evaluates the same variants.
Prints the 0-based ids (sorted) as a JSON list (default) or comma-separated. Stdlib only.
"""
import argparse
import json
import random
import sys


def select(classification, suite, category, n, seed):
    if suite not in classification:
        raise SystemExit(f"select_variants: suite {suite!r} not in classification ({sorted(classification)})")
    pool = sorted((t for t in classification[suite] if t.get("category") == category), key=lambda t: t["id"])
    if not pool:
        cats = sorted({t.get("category") for t in classification[suite]})
        raise SystemExit(f"select_variants: no {category!r} variants in {suite} (categories: {cats})")
    if n and n < len(pool):
        pool = sorted(random.Random(f"{seed}:{suite}:{category}").sample(pool, n), key=lambda t: t["id"])
    return pool


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("classification")
    p.add_argument("suite")
    p.add_argument("category")
    p.add_argument("n", type=int)
    p.add_argument("seed", type=int)
    p.add_argument("--format", choices=("json", "csv"), default="json")
    p.add_argument("--manifest", help="also write the selected entries (id, name, difficulty) as JSON")
    a = p.parse_args(argv)
    with open(a.classification) as f:
        classification = json.load(f)
    picked = select(classification, a.suite, a.category, a.n, a.seed)
    ids = [t["id"] - 1 for t in picked]
    if a.manifest:
        with open(a.manifest, "w") as f:
            json.dump([dict(task_id=t["id"] - 1, name=t["name"], category=t["category"],
                            difficulty_level=t.get("difficulty_level")) for t in picked], f, indent=1)
    print(json.dumps(ids, separators=(",", ":")) if a.format == "json" else ",".join(map(str, ids)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
