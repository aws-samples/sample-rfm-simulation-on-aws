"""Every place that lists the ``--vla`` targets must agree with ``targets.VLA_TARGETS``.

The CLIs are read with ``ast`` (not imported) so this test needs only the stdlib + pytest:
importing deploy.py/generate.py would pull in boto3 and jinja2.
"""

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from targets import JUDGE_TARGETS, VLA_TARGETS  # noqa: E402

EXPECTED = set(VLA_TARGETS)
JUDGES = set(JUDGE_TARGETS)
APP_TS = (ROOT / "cdk" / "bin" / "app.ts").read_text()


def _module_constants(tree: ast.Module) -> dict:
    """Module-level ``NAME = <literal>`` / ``NAME = tuple(<literal or name>)`` assignments."""
    consts = {"VLA_TARGETS": list(VLA_TARGETS)}  # imported from targets.py
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = _resolve(node.value, consts)
            except ValueError:
                pass
    return consts


def _resolve(node: ast.AST, consts: dict) -> list:
    """Turn a choices= expression into a list of strings (literals, *NAME splats, tuple()/list())."""
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out = []
        for elt in node.elts:
            if isinstance(elt, ast.Starred):
                out.extend(_resolve(elt.value, consts))
            elif isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.append(elt.value)
            else:
                raise ValueError(ast.dump(elt))
        return out
    if isinstance(node, ast.Dict):
        return [k.value for k in node.keys if isinstance(k, ast.Constant)]
    if isinstance(node, ast.Name) and node.id in consts:
        return list(consts[node.id])
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"tuple", "list"} and len(node.args) == 1:
        return _resolve(node.args[0], consts)
    raise ValueError(ast.dump(node))


# --vla arguments that intentionally have no choices= (best-effort parsers, not user-facing):
#   vlasim._peek_core_args only peeks at forwarded tokens for the preflight doctor; the real
#   validation happens in the deploy.py/destroy.py parser it forwards to.
NO_CHOICES_OK = {("vlasim.py", "_peek_core_args")}


def _vla_choices(filename: str) -> list:
    tree = ast.parse((ROOT / filename).read_text())
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            for child in ast.walk(fn):
                child._enclosing_fn = fn.name
    consts = _module_constants(tree)
    found = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--vla"):
            kw = {k.arg: k.value for k in node.keywords}
            if (filename, getattr(node, "_enclosing_fn", None)) in NO_CHOICES_OK and "choices" not in kw:
                continue
            assert "choices" in kw, f"{filename}: --vla has no choices="
            found.append(_resolve(kw["choices"], consts))
    assert found, f"{filename}: no add_argument('--vla', ...) found"
    return found


def _ts_object_keys(name: str) -> set:
    m = re.search(r"const " + name + r"\s*:\s*Record<string, string>\s*=\s*\{(.*?)\n\};", APP_TS, re.S)
    assert m, f"cdk/bin/app.ts: {name} not found"
    return set(re.findall(r"^\s*'([a-z0-9-]+)'\s*:", m.group(1), re.M))


def _literal_assign(filename: str, name: str):
    tree = ast.parse((ROOT / filename).read_text())
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name:
            return ast.literal_eval(n.value)
    raise AssertionError(f"{filename}: {name} not found")


def test_no_duplicates():
    assert len(VLA_TARGETS) == len(EXPECTED)


@pytest.mark.parametrize("filename", ["deploy.py", "destroy.py", "generate.py", "vlasim.py"])
def test_cli_vla_choices(filename):
    for choices in _vla_choices(filename):
        assert len(choices) == len(set(choices)), f"{filename}: duplicate --vla choice"
        assert set(choices) == EXPECTED, (
            f"{filename}: missing {sorted(EXPECTED - set(choices))}, extra {sorted(set(choices) - EXPECTED)}")


def test_models_yaml():
    stems = {p.stem for p in (ROOT / "models").glob("*.yaml")}
    want = EXPECTED | JUDGES
    assert stems == want, f"models/: missing {sorted(want - stems)}, extra {sorted(stems - want)}"


def test_judges_are_not_vla_targets():
    assert len(JUDGE_TARGETS) == len(JUDGES) and not JUDGES & EXPECTED


@pytest.mark.parametrize("judge", JUDGE_TARGETS)
def test_judge_template_and_cdk(judge):
    assert (ROOT / "templates" / f"{judge}-userdata.sh.j2").is_file()
    ts = (ROOT / "cdk" / "lib" / "judge-stack.ts").read_text()
    m = re.search(r"export const JUDGE_TARGETS = \[(.*?)\] as const;", ts)
    assert m and set(re.findall(r"'([a-z0-9-]+)'", m.group(1))) == JUDGES


def test_judge_cli_choices():
    """deploy.py / destroy.py --judge and generate.py --judge-target take their choices from JUDGE_TARGETS."""
    for filename, flag in (("deploy.py", "--judge"), ("destroy.py", "--judge"), ("generate.py", "--judge-target")):
        tree = ast.parse((ROOT / filename).read_text())
        consts = {"JUDGE_TARGETS": list(JUDGE_TARGETS)}
        found = [_resolve({k.arg: k.value for k in n.keywords}["choices"], consts) for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"
                 and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == flag]
        assert found and all(set(c) == JUDGES for c in found), filename


def test_app_ts_allowlist():
    m = re.search(r"if \(!vla \|\| !\[(.*?)\]\.includes\(vla\)\)", APP_TS, re.S)
    assert m, "cdk/bin/app.ts: vla allowlist not found"
    assert set(re.findall(r"'([a-z0-9-]+)'", m.group(1))) == EXPECTED


@pytest.mark.parametrize("name", ["stackNameMap", "descriptionMap"])
def test_app_ts_maps(name):
    keys = _ts_object_keys(name)
    assert keys == EXPECTED, f"app.ts {name}: missing {sorted(EXPECTED - keys)}, extra {sorted(keys - EXPECTED)}"


def test_stack_names_match_cdk():
    """destroy.STACK_NAMES / deploy's LeRobot+local maps must name the same stacks as app.ts stackNameMap."""
    m = re.search(r"const stackNameMap\s*:\s*Record<string, string>\s*=\s*\{(.*?)\n\};", APP_TS, re.S)
    ts = dict(re.findall(r"^\s*'([a-z0-9-]+)'\s*:\s*'([^']+)'", m.group(1), re.M))
    # openvla-oft is the only computed entry (oftStackName(liberoSuite)); its default is OpenVLA-OFT-Demo.
    ts["openvla-oft"] = "OpenVLA-OFT-Demo"
    assert set(ts) == EXPECTED

    assert _literal_assign("destroy.py", "STACK_NAMES") == ts
    for const in ("LEROBOT_LIBERO_STACKS", "OTHER_LOCAL_STACKS", "LEROBOT_ROBOTWIN_STACKS"):
        mapping = _literal_assign("deploy.py", const)
        assert {k: ts[k] for k in mapping} == mapping, f"deploy.{const} disagrees with app.ts"


def test_generate_lerobot_family_matches_deploy():
    gen = _module_constants(ast.parse((ROOT / "generate.py").read_text()))
    dep = _module_constants(ast.parse((ROOT / "deploy.py").read_text()))
    assert set(gen["LEROBOT_LIBERO_TARGETS"]) == set(dep["LEROBOT_LIBERO_TARGETS"]) <= EXPECTED


def test_generate_robotwin_family_matches_deploy():
    gen = _module_constants(ast.parse((ROOT / "generate.py").read_text()))
    dep = _module_constants(ast.parse((ROOT / "deploy.py").read_text()))
    assert set(gen["LEROBOT_ROBOTWIN_TARGETS"]) == set(dep["LEROBOT_ROBOTWIN_TARGETS"]) <= EXPECTED
    assert not set(gen["LEROBOT_ROBOTWIN_TARGETS"]) & set(gen["LEROBOT_LIBERO_TARGETS"])
