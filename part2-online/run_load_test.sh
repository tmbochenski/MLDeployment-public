#!/usr/bin/env bash
set -e

USERS=30
SPAWN_RATE=5
WAIT_MIN=0.1
WAIT_MAX=0.5
RUN_TIME=10s
HOST="https://oip-server-2tlztupk5a-ey.a.run.app"

export WAIT_MIN
export WAIT_MAX

locust -f locustfile.py --host "$HOST" -u "$USERS" -r "$SPAWN_RATE" -t "$RUN_TIME"