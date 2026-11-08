"""Isaac Lab rerender_on_reset: openarm-isaac re-renders its cameras on reset by default (eval.rerender_on_reset),
and the value is recorded next to num_envs in summary.txt, protocol.yaml and eval_info.json.

Isaac Lab's ManagerBasedEnvCfg.rerender_on_reset defaults to False, which leaves the first camera frame of every
episode stale (the previous episode's last frame). The Isaac Sim parts cannot run here, so env.py / run_eval.py are
checked on their source (AST), and the rendered userdata on its text.
"""

import ast
import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
pytest.importorskip("yaml")
pytest.importorskip("jinja2")
import generate  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
from test_protocol_card import _run_card  # noqa: E402

ASSETS = ROOT / "assets" / "openarm-isaac"


def _config():
    return generate._load_merged_config(ROOT / "simulator-config.yaml", "openarm-isaac")


def _render(config):
    return generate.generate_openarm_isaac(config, dry_run=True)


def test_yaml_default_is_true():
    assert _config()["eval"]["rerender_on_reset"] is True


def test_default_render_passes_true_to_the_container_and_the_summary():
    text = _render(_config())
    assert '-e "RERENDER_ON_RESET=1"' in text
    assert '--rerender-on-reset "${RERENDER_ON_RESET}"' in text
    assert 'echo "rerender_on_reset: true"' in text and 'echo "num_envs: 1"' in text


def test_false_is_rendered_as_false():
    cfg = copy.deepcopy(_config())
    cfg["eval"]["rerender_on_reset"] = False
    text = _render(cfg)
    assert '-e "RERENDER_ON_RESET=0"' in text and 'echo "rerender_on_reset: false"' in text


def test_missing_eval_block_defaults_to_true():
    cfg = copy.deepcopy(_config())
    cfg.pop("eval", None)
    assert '-e "RERENDER_ON_RESET=1"' in _render(cfg)


@pytest.mark.parametrize("bad", ["yes", 1, None])
def test_non_bool_is_rejected(bad):
    cfg = copy.deepcopy(_config())
    cfg["eval"]["rerender_on_reset"] = bad
    with pytest.raises(SystemExit):
        _render(cfg)


def _make_env_node():
    tree = ast.parse((ASSETS / "env.py").read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_env")


def test_env_py_sets_the_flag_before_gym_make():
    fn = _make_env_node()
    names = [a.arg for a in fn.args.args]
    assert "rerender_on_reset" in names
    default = fn.args.defaults[names.index("rerender_on_reset") - (len(names) - len(fn.args.defaults))]
    assert isinstance(default, ast.Constant) and default.value is True
    set_line = make_line = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "rerender_on_reset"
                                                and isinstance(t.value, ast.Name) and t.value.id == "env_cfg"
                                                for t in node.targets):
            set_line = node.lineno
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "make" \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "gym":
            make_line = node.lineno
    assert set_line is not None and make_line is not None and set_line < make_line


def test_run_eval_passes_the_flag_and_records_it():
    src = (ASSETS / "run_eval.py").read_text()
    assert '"--rerender-on-reset", type=int, choices=(0, 1), default=1' in src
    assert "rerender_on_reset=rerender_on_reset" in src
    assert 'info["env_settings"] = {"num_envs": int(vec_env.num_envs), "rerender_on_reset": rerender_on_reset}' in src


def test_card_records_num_envs_and_rerender(tmp_path):
    card, _ = _run_card(tmp_path, "n_episodes: 1\nn_success: 0\nnum_envs: 1\nrerender_on_reset: true\n")
    assert (card["num_envs"], card["rerender_on_reset"]) == ("1", "true")


def test_card_says_unknown_when_not_recorded(tmp_path):
    card, _ = _run_card(tmp_path, "n_episodes: 1\nn_success: 0\n")
    assert (card["num_envs"], card["rerender_on_reset"]) == ("unknown", "unknown")


def test_summary_has_one_key_per_line_for_results_and_the_card():
    # results.py and the protocol card read `n_episodes:` / `n_success:` at the start of a line; the old combined
    # "episode_length: N, n_episodes: M" line left k and n unknown.
    text = _render(_config())
    assert 'echo "episode_length: ${EPISODE_LENGTH}"' in text
    assert 'echo "n_episodes: ${N_EPISODES}"' in text and 'echo "n_success: ${NSUCC}"' in text
    assert ", n_episodes:" not in text
