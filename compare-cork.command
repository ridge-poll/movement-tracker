#!/bin/zsh
set -e
cd -- "$(dirname -- "$0")"
if [[ -x .venv/bin/python ]]; then
    experiment_python=.venv/bin/python
elif [[ -x ../../work/venv/bin/python ]]; then
    experiment_python=../../work/venv/bin/python
else
    print "Create the Python environment using README.md, then run this launcher again."
    exit 1
fi
exec "$experiment_python" -m movement_tracker.compare_rotations \
    '/Users/ridgepoll/Documents/Codex/2026-10-17/videos/spidey_back.MOV' \
    --start-frame 1 --end-frame 50
