#!/bin/bash
# Background-friendly crun alloc.
# Allocates an A100-80G x8 node, runs `tail -f /dev/null` so the
# container entrypoint never exits — Claude can `docker exec` into it
# from any session for the alloc lifetime.
#
# Usage:
#   ssh computelab-sc 'setsid nohup bash /path/to/_crun_alloc.sh > /tmp/crun_alloc.log 2>&1 < /dev/null &'
#   poll /tmp/crun_alloc.log for the running container hostname.

# `script` provides a pseudo-TTY so crun's `-i` (interactive-attach)
# doesn't bail out when stdin is /dev/null (the background-launch
# case). Output is appended to /tmp/crun_typescript so script's own
# logging doesn't clutter the alloc log.
exec script -qfe -c "crun --wait -i \
    --gpus 8 \
    -q 'gpu.product_slug=*a100*80* and cpu.arch=x86_64 ' \
    -t 9:0:0 \
    -img gitlab-master.nvidia.com:5005/devtech-compute/distributed-recommender:devel_latest \
    -a '--shm-size 8G' \
    --exclusive \
    tail -f /dev/null" /dev/null
