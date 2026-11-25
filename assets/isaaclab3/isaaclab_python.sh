#!/bin/bash
# isaaclab_python.sh — run the Python interpreter that can import isaaclab inside an NGC isaac-lab
# image, with the given arguments (drop-in for `python ARGS...`).
#
# Why a probe and not one fixed path:
#   Isaac Lab 3.0 (Isaac Sim 6.x, nvcr.io/nvidia/isaac-lab:3.0.0) installs isaaclab and its
#     dependencies into a uv venv (ENV VIRTUAL_ENV=/opt/isaaclab-venv, docker/Dockerfile.base).
#     `isaaclab.sh -p` runs that venv's interpreter with Isaac Sim's environment sourced
#     (setup_python_env.sh). `_isaac_sim/python.sh` is Kit's own interpreter and no longer sees
#     isaaclab there.
#   Isaac Lab 2.x installs isaaclab into Kit's python, where `_isaac_sim/python.sh` is the
#     interpreter (`isaaclab.sh -p` resolves to it as well).
# The first candidate that imports isaaclab wins. The `python` alias in ~/.bashrc is not used: it
# is not loaded in a non-interactive `docker run ... bash script.sh`.
#
#   isaaclab_python.sh --resolve   print the chosen command and exit (export it as
#                                  ISAACLAB_PYTHON_CMD so later calls skip the probe)
#   ISAACLAB_PYTHON_CMD="<cmd>"    use <cmd> without probing
# Exits 127 with the list of tried candidates when none imports isaaclab.
set -u
_root="${ISAACLAB_PATH:-/workspace/isaaclab}"
_tried=()
if [ -z "${ISAACLAB_PYTHON_CMD:-}" ]; then
    for _cand in "${_root}/isaaclab.sh -p" "${_root}/_isaac_sim/python.sh" "/isaac-sim/python.sh"; do
        _tried+=("$_cand")
        [ -x "${_cand%% *}" ] || continue
        # shellcheck disable=SC2086  # word splitting of "<script> -p" is intended
        if $_cand -c 'import isaaclab' > /dev/null 2>&1 < /dev/null; then
            ISAACLAB_PYTHON_CMD="$_cand"
            break
        fi
    done
fi
if [ -z "${ISAACLAB_PYTHON_CMD:-}" ]; then
    echo "[isaaclab_python] FATAL: no interpreter in this image imports isaaclab (tried: ${_tried[*]})" >&2
    exit 127
fi
if [ "${1:-}" = "--resolve" ]; then
    echo "$ISAACLAB_PYTHON_CMD"
    exit 0
fi
# shellcheck disable=SC2086
exec $ISAACLAB_PYTHON_CMD "$@"
