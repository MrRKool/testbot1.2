#!/bin/bash

# Check if optimization is running
if [ -f optimization.pid ]; then
    PID=$(cat optimization.pid)
    if ps -p $PID > /dev/null; then
        echo "Stopping optimization process (PID: $PID)..."
        kill $PID
        rm optimization.pid
        echo "Optimization stopped"
    else
        echo "Optimization process not found. PID file exists but process is dead."
        rm optimization.pid
    fi
else
    echo "No optimization process found (no PID file)"
fi 