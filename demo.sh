#!/bin/bash
# Start or stop the whole demo on this Mac.
#   ./demo.sh start   sim (8818) + data farm (8819) + viewer (8820) + the show (robots, training)
#   ./demo.sh stop    stops all of it (the show spends OpenRouter tokens while it runs)
# Viewer: http://127.0.0.1:8820 ; logs in /tmp/winners-*.log
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"

stop() {
  pkill -f winners.show 2>/dev/null
  pkill -f "node server/server.js" 2>/dev/null
  pkill -f "vite" 2>/dev/null
  sleep 1
  echo "stopped"
}

start() {
  stop >/dev/null
  # The key is read from ~/.hermes/.env and only lives in this process.
  eval "$(grep -E '^(export )?OPENROUTER_API_KEY=' ~/.hermes/.env)"
  export OPENROUTER_API_KEY
  rm -rf /tmp/farm-data "$ROOT/arena/data" "$ROOT/input_layer/runs/g1-show2"
  cd "$ROOT/arena" || exit 1
  nohup node server/server.js > /tmp/winners-sim.log 2>&1 &
  SIM_PORT=8819 ARENA_DATA_DIR=/tmp/farm-data nohup node server/server.js > /tmp/winners-farm.log 2>&1 &
  nohup npm run viewer > /tmp/winners-viewer.log 2>&1 &
  sleep 4
  cd "$ROOT/input_layer" || exit 1
  source .venv/bin/activate
  nohup python -u -m winners.show --config configs/g1_arena.yaml --out runs/g1-show2 > /tmp/winners-show.log 2>&1 &
  echo "started: open http://127.0.0.1:8820 (give the robots ~30 s to appear)"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  *) echo "usage: $0 start|stop" ;;
esac
