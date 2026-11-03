"""tools/intro-video/scenes.json: every clip it names exists under docs/showcase/, every scene type has a renderer."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "intro-video"


def _scenes():
    return json.loads((TOOL / "scenes.json").read_text())


def _clips(cfg):
    for sc in cfg["scenes"] + [cfg["preview_scene"]]:
        for c in sc.get("clips", []):
            yield c["src"]
        for nd in sc.get("nodes", []):
            if nd.get("clip"):
                yield nd["clip"]
        if sc.get("clip"):
            yield sc["clip"]


def test_clips_exist():
    missing = [s for s in _clips(_scenes()) if not (ROOT / "docs" / "showcase" / s).is_file()]
    assert not missing, missing


def test_scene_types_have_renderers():
    src = (TOOL / "render.py").read_text()
    known = set(re.findall(r'"(\w+)": scene_\w+', src))
    cfg = _scenes()
    types = {sc["type"] for sc in cfg["scenes"]} | {cfg["preview_scene"]["type"]}
    assert types <= known, types - known


def test_every_scene_is_narrated():
    for sc in _scenes()["scenes"]:
        assert sc["narration"].strip(), sc["id"]
