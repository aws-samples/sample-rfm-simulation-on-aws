"""Every ``--vla`` target renders through generate.py and the result passes ``bash -n``.

Mirrors the CI "render userdata" step: generate.py is run as a CLI (it reads models/<target>.yaml
and simulator-config.yaml, writes <output-dir>/<target>.sh and runs ``bash -n`` itself). The test
writes to a pytest tmp dir so it never touches assets/userdata/, and re-runs ``bash -n`` so a
regression in generate.py's own check cannot hide a broken script.

Every completion e-mail also carries the "Video history" link only when the bootstrap exported
GalleryUrl (deploy.py with the gallery on); the block is run through bash with and without it.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from targets import VLA_TARGETS  # noqa: E402

pytest.importorskip("jinja2")
pytest.importorskip("yaml")

BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not on PATH")

# openvla-oft is one stack per LIBERO suite; render each one, not just the default ("long" is an
# alias of "10" and renders the same stack, so it is not repeated).
OFT_SUITES = ("spatial", "object", "goal", "10")
CASES = [(v, []) for v in VLA_TARGETS if v != "openvla-oft"]
CASES += [("openvla-oft", ["--libero-suite", s]) for s in OFT_SUITES]


def _render(vla, extra, out_dir):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "generate.py"), "--vla", vla, "--output-dir", str(out_dir), *extra],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"generate.py --vla {vla} {' '.join(extra)} failed:\n{proc.stdout}\n{proc.stderr}"
    return out_dir / f"{vla}.sh"


@pytest.mark.parametrize("vla,extra", CASES, ids=[" ".join([v, *e]).replace("--libero-suite ", "") for v, e in CASES])
def test_render_and_bash_n(vla, extra, tmp_path):
    script = _render(vla, extra, tmp_path)
    text = script.read_text()
    assert text.startswith("#!"), f"{script.name} has no shebang"
    # ``{{`` is legitimate in rendered scripts (docker --format '{{.Id}}'); block tags are not.
    assert "{%" not in text, f"{script.name} has an unrendered Jinja block tag"
    check = subprocess.run([BASH, "-n", str(script)], capture_output=True, text=True)
    assert check.returncode == 0, f"bash -n {script.name}:\n{check.stderr}"
    _check_gallery_block(text, script.name)


GALLERY_URL = "https://d111.cloudfront.net/g/0123456789abcdef0123456789abcdef/"


def _check_gallery_block(text, name):
    """The SNS message gets the gallery link only when GalleryUrl is set; s3 sync lines stay."""
    assert text.count('GALLERY_BLOCK=""') == 1, f"{name}: gallery block missing or repeated"
    msg = text[text.index('SNS_MESSAGE="'):]
    assert "${GALLERY_BLOCK}Download results:" in msg, f"{name}: SNS message does not use GALLERY_BLOCK"
    assert re.search(r"aws s3 sync s3://\$\{S3_BUCKET\}/\$\{RUN_ID\}/", msg), f"{name}: s3 sync line lost"
    start = text.index('GALLERY_BLOCK=""')
    block = text[start:text.index("\nfi\n", start) + 4]
    snippet = block + 'printf "%s" "${GALLERY_BLOCK}Download results:"\n'
    for env, expect_link in (({"GalleryUrl": GALLERY_URL}, True), ({"GalleryUrl": ""}, False), ({}, False)):
        out = subprocess.run([BASH, "-u", "-c", snippet], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", **env})
        assert out.returncode == 0, f"{name}: gallery block failed: {out.stderr}"
        if expect_link:
            assert out.stdout == f"Video history (all your runs, by model; password: python vlasim.py gallery):\n  {GALLERY_URL}\n\nDownload results:"
        else:
            assert out.stdout == "Download results:", f"{name}: block shown without GalleryUrl"


def test_every_target_is_covered():
    assert {v for v, _ in CASES} == set(VLA_TARGETS)
