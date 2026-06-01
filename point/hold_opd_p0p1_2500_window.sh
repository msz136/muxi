#!/usr/bin/env bash
cd /data/msz/point || exit 1
./run_opd_p0p1_studentrollout_full2500_save500_zero3_mb16_accum1_from_coldstart100.sh
status=$?
echo "[window] training exited status=${status}"
exec bash
