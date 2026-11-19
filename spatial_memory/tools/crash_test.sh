#!/bin/bash
# kill -9 crash-recovery test.
# Usage: ./crash_test.sh [db_path] [runs]
set -u
DB=${1:-/tmp/sm_crash.db}
RUNS=${2:-5}
BIN=${BIN:-./crash_test}
PASS=0
FAIL=0

for run in $(seq 1 "$RUNS"); do
    rm -f "$DB" "$DB-wal" "$DB-shm"
    "$BIN" write "$DB" >/dev/null 2>/tmp/sm_writer.log &
    PID=$!
    sleep 3
    kill -9 "$PID" 2>/dev/null
    wait "$PID" 2>/dev/null

    WRITTEN=$(grep -o 'written=[0-9]*' /tmp/sm_writer.log | tail -1 | cut -d= -f2)
    OUT=$("$BIN" verify "$DB")
    OBS=$(echo "$OUT" | grep -o 'obs=[0-9]*' | cut -d= -f2)
    VERDICT=$(echo "$OUT" | tail -1)

    # committed data should mostly survive: recovered >= 80% of last reported batch
    NEED=$(( ${WRITTEN:-0} * 8 / 10 ))
    if [ "$VERDICT" = "VERIFY_OK" ] && [ "${OBS:-0}" -ge "$NEED" ]; then
        echo "run $run: OK   written=$WRITTEN recovered_obs=$OBS"
        PASS=$((PASS+1))
    else
        echo "run $run: FAIL written=$WRITTEN recovered_obs=$OBS"
        echo "$OUT"
        FAIL=$((FAIL+1))
    fi
done

rm -f "$DB" "$DB-wal" "$DB-shm"
echo "----"
echo "crash test: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
