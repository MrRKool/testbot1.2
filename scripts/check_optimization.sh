#!/bin/bash

# Check if optimization is running
if [ -f optimization.pid ]; then
    PID=$(cat optimization.pid)
    if ps -p $PID > /dev/null; then
        echo "Optimization is running with PID: $PID"
        echo "Last 10 lines of log:"
        tail -n 10 logs/optimization.log
    else
        echo "Optimization process not found. PID file exists but process is dead."
        rm optimization.pid
    fi
else
    echo "No optimization process found (no PID file)"
fi 