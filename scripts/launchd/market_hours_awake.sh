#!/bin/bash
# Hold a no-idle-sleep assertion from now until 15:40 IST on weekdays, so the V83 agent isn't
# suspended mid-session (2026-10-08 lost 11:00-15:30 to sleep). Launched at 08:55 Mon-Fri by
# com.algoagent.market_hours_awake (launchd also fires a missed run on wake).
# -i idle sleep, -m disk sleep, -s system sleep (AC power only). It CANNOT stop a closed lid on
# battery — that still needs power (+ external display for lid-closed use).
dow=$(TZ=Asia/Kolkata date +%u)
[ "$dow" -gt 5 ] && exit 0
now=$(TZ=Asia/Kolkata date +%s)
start=$(TZ=Asia/Kolkata date -j -f "%H:%M:%S" "08:50:00" +%s)
end=$(TZ=Asia/Kolkata date -j -f "%H:%M:%S" "15:40:00" +%s)
[ "$now" -lt "$start" ] && exit 0   # RunAtLoad / a wake before the session must not hold all night
secs=$(( end - now ))
[ "$secs" -le 0 ] && exit 0
echo "$(date '+%F %T') holding awake for ${secs}s (until 15:40 IST)"
exec /usr/bin/caffeinate -ims -t "$secs"
