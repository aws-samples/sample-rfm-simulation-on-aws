"""LIBERO-Plus robustness sweep: shared constants and the sweep plan (stdlib only).

LIBERO-Plus (Fei et al., arXiv:2510.13626, github.com/sylvestf/LIBERO-plus) is a fork of LIBERO that
installs under the same ``libero`` package and replaces each suite's 10 tasks with ~2,400-2,600
perturbed variants, each tagged with one of seven perturbation dimensions in
``libero/libero/benchmark/task_classification.json``. A target that already runs LIBERO opts in with
``--libero-variant plus``: generate.py replaces its ``tasks`` with one entry per perturbation dimension
(``condition: LIBERO-Plus/<dim>``), the UserData installs the fork + its asset pack at the pins below,
and on the instance ``assets/libero-plus/select_variants.py`` picks N variants of that dimension from
the classification file (seeded, so a re-run evaluates the same variants). One episode per variant,
as in the paper. Off by default: without the flag nothing in the rendered UserData changes.

The fork's code repository declares no license (no LICENSE file, GitHub reports none) and the asset
pack (Hugging Face dataset Sylvest/LIBERO-plus) is tagged MIT on its dataset card. Treat the
benchmark as research / evaluation use: both are downloaded on the instance at run time and never
vendored here.
"""

# Short CLI / summary names → the category strings used in task_classification.json.
# Order = the column order of the LIBERO-Plus leaderboard (README of the fork).
CATEGORIES = {
    "camera": "Camera Viewpoints",
    "robot": "Robot Initial States",
    "language": "Language Instructions",
    "light": "Light Conditions",
    "background": "Background Textures",
    "noise": "Sensor Noise",
    "layout": "Objects Layout",
}

# Suites the fork ships with a classification entry (libero_90 has no perturbed variants).
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

# Pins shared by every target (a models/<id>.yaml libero_plus block may override any of them).
DEFAULTS = {
    "repo_url": "https://github.com/sylvestf/LIBERO-plus.git",
    # git ls-remote https://github.com/sylvestf/LIBERO-plus.git HEAD (main, pushed 2026-01-21);
    # the same SHA LeRobot pins in docker/Dockerfile.benchmark.libero_plus.
    "repo_commit": "4976dc30028e805ff8094b55501d532c48fec182",
    "assets_repo": "Sylvest/LIBERO-plus",  # Hugging Face dataset, MIT on the dataset card
    "assets_revision": "dd2bd61b7d9a6fef1abc52d606e983b41886a149",
    "assets_file": "assets.zip",
    "assets_sha256": "96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf",
    "assets_bytes": 6395849578,  # 6.4 GB zip
    "suite": "libero_spatial",
    "perturb": list(CATEGORIES),
    "variants_per_category": 10,
    "episodes_per_variant": 1,
    "seed": 7,
}

VARIANTS = ("libero", "plus")


def parse_perturb(value):
    """'camera,light' / ['camera', 'light'] / 'all' → ordered list of category keys (ValueError on junk)."""
    if value is None:
        return None
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    items = [str(x).strip().lower() for x in items if str(x).strip()]
    if items == ["all"]:
        return list(CATEGORIES)
    bad = [x for x in items if x not in CATEGORIES]
    if bad or not items:
        raise ValueError(f"unknown perturbation(s) {bad or items}; choose from {', '.join(CATEGORIES)} or all")
    return [k for k in CATEGORIES if k in items]  # canonical order, de-duplicated


def plan(block, perturb=None, n=None, suite=None):
    """Merge DEFAULTS + a models/<id>.yaml ``libero_plus`` block + CLI overrides into a sweep plan.

    Returns (cfg, tasks): cfg = the merged settings, tasks = one task entry per perturbation dimension
    in the ``tasks`` schema the templates already loop over. Raises ValueError on a bad setting.
    """
    cfg = {**DEFAULTS, **(block or {})}
    if suite:
        cfg["suite"] = suite
    if cfg["suite"] not in SUITES:
        raise ValueError(f"libero_plus.suite must be one of {', '.join(SUITES)}, got {cfg['suite']!r}")
    cats = parse_perturb(perturb if perturb is not None else cfg["perturb"])
    n = int(cfg["variants_per_category"] if n is None else n)
    if n < 0:
        raise ValueError("variants per category must be >= 0 (0 = every variant of the category)")
    eps = int(cfg["episodes_per_variant"])
    if eps < 1:
        raise ValueError("libero_plus.episodes_per_variant must be >= 1")
    seed = int(cfg["seed"])
    cfg.update(perturb=cats, variants_per_category=n, episodes_per_variant=eps, seed=seed)
    count = "all" if n == 0 else str(n)
    tasks = [{
        "suite": cfg["suite"],
        "perturbation": c,
        "perturbation_category": CATEGORIES[c],
        "variants": n,
        "variant_seed": seed,
        "condition": f"LIBERO-Plus/{c}",
        "description": f"LIBERO-Plus {cfg['suite']} / {CATEGORIES[c]} ({count} variants x {eps} ep)",
    } for c in cats]
    return cfg, tasks


def episodes(cfg):
    """Episode count of a plan (None when variants_per_category = 0, i.e. the full category)."""
    n = cfg["variants_per_category"]
    return None if n == 0 else n * len(cfg["perturb"]) * cfg["episodes_per_variant"]
