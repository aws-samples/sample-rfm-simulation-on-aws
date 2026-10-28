"""Single source of truth for the deployable ``--vla`` targets.

deploy.py, destroy.py, generate.py and vlasim.py all take their ``--vla`` choices from here.
Each target also needs ``models/<target>.yaml`` and an entry in the ``cdk/bin/app.ts`` allowlist,
stack-name map and description map; ``tests/test_targets_parity.py`` fails if any of those drift.

``JUDGE_TARGETS`` are post-run judges, not ``--vla`` choices: they score the videos of a finished
run of a ``--vla`` target (``python deploy.py --judge --vla <target>``). Each one has
``models/<judge>.yaml`` + ``templates/<judge>-userdata.sh.j2`` and deploys through
``cdk/bin/judge.ts``; the parity test checks those too.

Stdlib-only and side-effect free on import (vlasim doctor imports it on hosts without deps).
"""

VLA_TARGETS = (
    "gr00t", "gr00t-gr1", "gr00t-g1", "pi", "openvla-oft",
    "lap", "rldx", "rldx-simpler", "rldx-gr1", "rldx-kitchen", "openarm-isaac", "openarm-lift-act",
    "molmoact2", "xvla", "smolvla", "lingbot-va", "vla-adapter", "fastwam", "cosmos-policy", "aspire",
)

JUDGE_TARGETS = ("flora-judge",)

if __name__ == "__main__":
    import sys

    print(" ".join(JUDGE_TARGETS if sys.argv[1:] == ["--judge"] else VLA_TARGETS))
