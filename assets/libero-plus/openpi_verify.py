"""Check LIBERO-Plus inside openpi's LIBERO client image before a sweep (runs in the container).

usage: python examples/libero/lp_verify.py <suite> <comma-separated 0-based variant ids>

Goes through the whole chain the client uses, including the imports the fork defers to env
construction: `libero` resolves to the fork (suite has the perturbed variants), wand (ImageMagick) and
scikit-image import, numpy / robosuite are the image's pins, and for each id (one per perturbation
dimension) the variant's env is built, reset to its init state and stepped once. Finally main.py is
loaded (not run) so the client's own imports (openpi_client, tyro) are checked too.
"""
import os
import runpy
import sys

import numpy
import robosuite
import skimage.filters  # noqa: F401  (fork: blur / noise perturbations)
import wand.image  # noqa: F401  (fork: ImageMagick motion blur; needs libmagickwand in the image)
import libero
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

suite_name, ids = sys.argv[1], [int(x) for x in sys.argv[2].split(",") if x.strip()]
print("libero:", list(libero.__path__), "/ numpy", numpy.__version__, "/ robosuite", robosuite.__version__)
assert int(numpy.__version__.split(".")[0]) < 2, "numpy " + numpy.__version__ + " (the client pins numpy 1.22.4)"
suite = benchmark.get_benchmark_dict()[suite_name]()
assert suite.n_tasks > 100, "libero still resolves to vanilla LIBERO (10 tasks per suite)"
print(suite_name, "variants:", suite.n_tasks)
for tid in ids:
    task = suite.get_task(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(7)
    env.reset()
    init_states = suite.get_task_init_states(tid)
    obs = env.set_init_state(init_states[0])
    obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
    assert obs["agentview_image"].shape == (256, 256, 3), obs["agentview_image"].shape
    env.close()
    print("variant", tid, task.name, "OK")
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py"), run_name="lp_verify")
print("LIBERO-PLUS VERIFY OK")
