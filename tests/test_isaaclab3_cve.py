"""CVE-2026-65141: every Isaac Lab target runs a fixed Isaac Lab (3.0.0-beta2 or later), never 2.x.

Isaac Lab 0.0 - 2.3.2 eval()s any lambda-looking string in isaaclab.utils.string.string_to_callable (no 2.x
fix). The guard below fails when a model yaml, a template or a generate.py default names an Isaac Lab 2.x
image again. The rest checks the 3.0 wiring the GPU runs depend on (driver floor, uid 1000 mounts, the
isaaclab python probe, the openarm_isaac_lab patch) and the standalone checker tools/check_isaaclab_cve_2026_65141.py.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
yaml = pytest.importorskip("yaml")
pytest.importorskip("jinja2")
import generate  # noqa: E402

ISAACLAB3_TARGETS = ("openarm-isaac", "openarm-lift-act", "robolab-cosmos3", "robolab-pi05")
GA_DIGEST = "sha256:f70acc9ed3bc3fc67099a0c5715379c432cc9965623cabbaf54f74599997e245"
BETA2_DIGEST = "sha256:970621075d00059c309f847fa835709de3fa634537407a7009c586c97893d218"
# An Isaac Lab 2.x (or 1.x / 0.x) image or tag: isaac-lab:2.3.0, isaac_lab_tag: "2.2.0@...", --isaaclab-tag=2.
_OLD_IMAGE = re.compile(r"isaac-lab:[0-2]\.\d|isaac_lab_tag:\s*[\"']?[0-2]\.\d|isaaclab-tag=[0-2]\.\d")


def _sources():
    yield from sorted((ROOT / "models").glob("*.yaml"))
    yield from sorted((ROOT / "templates").glob("*.j2"))
    yield ROOT / "generate.py"


@pytest.mark.parametrize("path", list(_sources()), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_isaac_lab_2x_image_anywhere(path):
    hits = [f"{i}: {line.strip()}" for i, line in enumerate(path.read_text().splitlines(), 1) if _OLD_IMAGE.search(line)]
    assert not hits, f"Isaac Lab 2.x image (CVE-2026-65141) in {path.relative_to(ROOT)}:\n" + "\n".join(hits)


def test_guard_regex_catches_the_old_pins():
    for old in ('isaac_lab_image: "nvcr.io/nvidia/isaac-lab:2.3.0"', 'isaac_lab_tag: "2.2.0@sha256:b4d8"',
                'model.get("isaac_lab_image", "nvcr.io/nvidia/isaac-lab:2.3.0")', "--isaaclab-tag=2.3.0"):
        assert _OLD_IMAGE.search(old), old
    for new in (f"nvcr.io/nvidia/isaac-lab:3.0.0@{GA_DIGEST}", f'isaac_lab_tag: "3.0.0-beta2@{BETA2_DIGEST}"'):
        assert not _OLD_IMAGE.search(new), new


def _model(vla):
    return yaml.safe_load((ROOT / "models" / f"{vla}.yaml").read_text())["model"]


@pytest.mark.parametrize("vla", ["openarm-isaac", "openarm-lift-act"])
def test_openarm_targets_pin_the_3_0_0_ga_image_by_digest(vla):
    assert _model(vla)["isaac_lab_image"] == f"nvcr.io/nvidia/isaac-lab:3.0.0@{GA_DIGEST}"
    assert generate.ISAAC_LAB_3_IMAGE == f"nvcr.io/nvidia/isaac-lab:3.0.0@{GA_DIGEST}"


@pytest.mark.parametrize("vla", ["robolab-cosmos3", "robolab-pi05"])
def test_robolab_targets_pin_3_0_0_beta2_and_driver_580(vla):
    m = _model(vla)
    assert m["isaac_lab_tag"] == f"3.0.0-beta2@{BETA2_DIGEST}"
    assert int(m["min_driver_major"]) >= 580


def _render(vla):
    cfg = generate._load_merged_config(ROOT / "simulator-config.yaml", vla)
    if vla == "openarm-isaac":
        return generate.generate_openarm_isaac(cfg, dry_run=True)
    if vla == "openarm-lift-act":
        return generate.generate_openarm_lift_act(cfg, dry_run=True)
    return generate.generate_robolab(cfg, True, vla)


@pytest.mark.parametrize("vla", ["openarm-isaac", "openarm-lift-act"])
def test_openarm_userdata_is_wired_for_isaac_lab_3(vla):
    text = _render(vla)
    # no hard-coded 2.x Kit python: isaaclab lives in the 3.0 image's venv, reached via the probe
    assert "/workspace/isaaclab/_isaac_sim/python.sh" not in text
    assert "--entrypoint /workspace/assets/isaaclab_python.sh" in text
    assert 'CONTAINER_PY="$ASSETS/isaaclab_python.sh"' in text
    assert '"$CONTAINER_PY" "$ASSETS/patch_openarm_isaaclab3.py" "$WORK"' in text
    # uid 1000 image: host mounts handed to 1000:1000, clones off the root-owned /workspace
    assert "CONTAINER_UID=1000" in text and 'chown -R "${CONTAINER_UID}:${CONTAINER_UID}"' in text
    assert 'WORK="${SCRATCH}/openarm_isaac_lab"' in text
    # driver floor before the pull; Ubuntu 24.04 AMI has no awscli apt package
    assert '[ "${DRIVER_MAJOR:-0}" -ge 580 ]' in text
    assert "ca-certificates awscli" not in text and "command -v aws" in text
    assert "Isaac Lab 2.3" not in text and "Isaac Lab 3.0 / Isaac Sim 6.2" in text


def test_openarm_isaac_userdata_patches_lerobot_for_the_isaac_lab_env():
    text = _render("openarm-isaac")
    assert "patch_lerobot_py311" not in text
    assert '"$ASSETS/patch_lerobot_isaaclab.py" "$LEROBOT_SRC" --constraints-out "$CONSTRAINTS"' in text


@pytest.mark.parametrize("vla", ["robolab-cosmos3", "robolab-pi05"])
def test_robolab_builds_the_isaac60_stack_as_uid_1000(vla):
    text = _render(vla)
    # --isaac60 sets runtime user 1000:1000; --isaaclab-tag must come after it (it only swaps the base tag)
    assert f'build_docker.sh --isaac60 "--isaaclab-tag=3.0.0-beta2@{BETA2_DIGEST}"' in text
    assert 'chown -R 1000:1000 "${OUT_DIR}/shard-${_s}" "${CACHE_DIR}/shard-${_s}"' in text
    assert "ca-certificates awscli" not in text and "command -v aws" in text


def test_cdk_boots_the_driver_580_ami_for_every_isaac_lab_3_target():
    ts = (ROOT / "cdk" / "lib" / "vla-simulator-stack.ts").read_text()
    listed = re.search(r"ISAACLAB3_TARGETS = \[([^\]]*)\]", ts).group(1)
    assert sorted(re.findall(r"'([^']+)'", listed)) == sorted(ISAACLAB3_TARGETS)


# ── openarm_isaac_lab patch (assets/isaaclab3/patch_openarm_isaaclab3.py) on a synthetic tree ──────────

def _patcher():
    import importlib.util

    spec = importlib.util.spec_from_file_location("patch_openarm", ROOT / "assets" / "isaaclab3" / "patch_openarm_isaaclab3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_openarm(tmp_path, mod):
    for rel, anchor, _ in mod.LINE_EDITS:
        fp = tmp_path / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text((fp.read_text() if fp.exists() else "x = 1\n") + "\n".join(anchor) + "\n")
    for rel, n in mod.TORCH_READS.items():
        fp = tmp_path / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text("".join(f"v{i} = robot.data.root_pos_w[:, {i}]\n" for i in range(n)))
    return tmp_path


def _run_patcher(tree):
    return subprocess.run([sys.executable, str(ROOT / "assets" / "isaaclab3" / "patch_openarm_isaaclab3.py"), str(tree)],
                          capture_output=True, text=True)


def test_openarm_patch_applies_once_and_is_idempotent(tmp_path):
    mod = _patcher()
    tree = _fake_openarm(tmp_path, mod)
    first = _run_patcher(tree)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "9 edit(s) applied" in first.stdout
    reach = (tree / mod.PKG / "bimanual/reach/reach_env_cfg.py").read_text()
    assert "import UniformNoiseCfg as Unoise" in reach and "AdditiveUniformNoiseCfg" not in reach
    lift = (tree / mod.PKG / "unimanual/lift/lift_env_cfg.py").read_text()
    assert "rot=[0, 0, 0.707, 0.707]" in lift and "self.sim.physx." not in lift and "PhysxCfg(" in lift
    rewards = (tree / mod.PKG / "bimanual/reach/mdp/rewards.py").read_text()
    assert rewards.count("robot.data.root_pos_w.torch[") == 8
    second = _run_patcher(tree)
    assert second.returncode == 0 and "0 edit(s) applied" in second.stdout


def test_openarm_patch_fails_on_a_removed_2x_noise_alias(tmp_path):
    mod = _patcher()
    tree = _fake_openarm(tmp_path, mod)
    fp = tree / mod.PKG / "bimanual/reach/mdp/rewards.py"
    fp.write_text(fp.read_text() + "from isaaclab.utils.noise import AdditiveGaussianNoiseCfg\n")
    r = _run_patcher(tree)
    assert r.returncode != 0 and "removed in 3.0 still used" in (r.stdout + r.stderr)


def test_openarm_patch_fails_loudly_when_the_pinned_sources_moved(tmp_path):
    mod = _patcher()
    tree = _fake_openarm(tmp_path, mod)
    fp = tree / mod.PKG / "unimanual/lift/mdp/rewards.py"
    fp.write_text(fp.read_text() + "extra = object.data.root_pos_w\n")
    r = _run_patcher(tree)
    assert r.returncode != 0 and "pin moved" in (r.stdout + r.stderr)


# ── LeRobot bound relaxer (assets/openarm-isaac/patch_lerobot_isaaclab.py, Python >= 3.11) ─────────────

def test_lerobot_patch_relaxes_only_conflicting_bounds(tmp_path):
    pytest.importorskip("tomllib")
    pytest.importorskip("packaging")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "lerobot"\nrequires-python = ">=3.12"\ndependencies = [\n'
        '    "torch>=2.7,<2.12.0",\n    "numpy>=2.0.0,<2.3.0", # comment kept\n    "packaging>=24.2,<26.0",\n'
        '    "einops>=0.8.0,<0.9.0",\n]\n[project.optional-dependencies]\nav-dep = ["av>=15.0.0,<16.0.0"]\n'
    )
    out = tmp_path / "c.txt"
    r = subprocess.run([sys.executable, str(ROOT / "assets" / "openarm-isaac" / "patch_lerobot_isaaclab.py"), str(tmp_path),
                        "--constraints-out", str(out), "--installed", "numpy=2.5.1", "--installed", "packaging=26.0",
                        "--installed", "torch=2.11.0+cu130", "--installed", "setuptools=81.0.0",
                        "--installed", "torchvision=0.26.0+cu130"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    text = (tmp_path / "pyproject.toml").read_text()
    assert '"numpy", # comment kept' in text and '"packaging",' in text
    assert '"torch>=2.7,<2.12.0"' in text and '"einops>=0.8.0,<0.9.0"' in text and '"av>=15.0.0,<16.0.0"' in text
    assert "numpy==2.5.1\n" in out.read_text() and "torch==2.11.0+cu130\n" in out.read_text()


# ── tools/check_isaaclab_cve_2026_65141.py on a vulnerable and a fixed string_to_callable ─────────────

_VULNERABLE = '''
import ast, importlib
def is_lambda_expression(name):
    try:
        return isinstance(ast.parse(name, mode="eval").body, ast.Lambda)
    except SyntaxError:
        return False
def string_to_callable(name):
    if is_lambda_expression(name):
        return eval(name)
    mod_name, attr_name = name.split(":")
    return getattr(importlib.import_module(mod_name), attr_name)
'''

_FIXED = _VULNERABLE.replace("        return eval(name)\n", '''        for node in ast.walk(ast.parse(name, mode="eval")):
            if isinstance(node, (ast.Attribute, ast.Call)) or (isinstance(node, ast.Name) and node.id.startswith("__")):
                raise ValueError("unsafe lambda")
        return eval(name, {"__builtins__": {}}, {})
''')


@pytest.mark.parametrize("source, rc, verdict", [(_VULNERABLE, 1, "VULNERABLE"), (_FIXED, 0, "FIXED")])
def test_cve_checker_tells_vulnerable_from_fixed(tmp_path, source, rc, verdict):
    src = tmp_path / "string.py"
    src.write_text(source)
    pwned = Path("/tmp/pwned")
    before = pwned.exists()
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "check_isaaclab_cve_2026_65141.py"), "--source", str(src)],
                       capture_output=True, text=True)
    assert r.returncode == rc, r.stdout + r.stderr
    assert r.stdout.strip().endswith(verdict)
    assert pwned.exists() == before  # the os.system payload is resolved, never called


def test_cve_checker_cannot_test_without_isaaclab(tmp_path):
    r = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "check_isaaclab_cve_2026_65141.py")],
                       capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 2 and "CANNOT TEST" in r.stderr
