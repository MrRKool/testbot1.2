#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Create log directory if it doesn't exist
mkdir -p logs

# Function to start the optimization
start_optimization() {
    nohup python bot/backtesting/run_optimization.py > logs/optimization.log 2>&1 &
    echo $! > optimization.pid
    echo "Optimization started with PID: $(cat optimization.pid)"
}

# Function to check if process is running
check_process() {
    if [ -f optimization.pid ]; then
        PID=$(cat optimization.pid)
        if ps -p $PID > /dev/null; then
            return 0
        fi
    fi
    return 1
}

# Start the optimization
start_optimization

# Monitor and restart if needed
while true; do
    sleep 60  # Check every minute
    
    if ! check_process; then
        echo "Process not running. Restarting..."
        start_optimization
    fi
done 