#!/bin/bash
# Uruchom trening w tle z logami

echo "Rozpoczynam trening w tle..."
echo "Logi będą zapisywane do training.log"

nohup ./train.sh allegro/herbert-base-cased 1 1 > training.log 2>&1 &

PID=$!
echo "Proces PID: $PID"
echo ""
echo "Aby sprawdzić postęp:"
echo "  tail -f training.log"
echo ""
echo "Aby zatrzymać:"
echo "  kill $PID"
echo ""
echo "Aby sprawdzić czy działa:"
echo "  ps aux | grep $PID"
