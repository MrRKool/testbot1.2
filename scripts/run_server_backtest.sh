#!/bin/bash

# Set error handling
set -e
set -o pipefail

# Configuration
LOG_DIR="logs"
RESULTS_DIR="results"
PLOTS_DIR="results/plots"
CONFIG_FILE="config/trading_parameters.yaml"
MAX_PARALLEL_JOBS=4  # Adjust based on server CPU cores

# Create necessary directories
mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$PLOTS_DIR"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/server_backtest.log"
}

# Function to check if a command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        log "ERROR: $1 is required but not installed"
        exit 1
    fi
}

# Function to run backtest for a symbol
run_backtest() {
    local symbol=$1
    local log_file="$LOG_DIR/${symbol}_backtest.log"
    
    log "Starting backtest for $symbol"
    python scripts/run_backtest.py --symbol "$symbol" --config "$CONFIG_FILE" > "$log_file" 2>&1
    
    if [ $? -eq 0 ]; then
        log "Backtest completed successfully for $symbol"
    else
        log "ERROR: Backtest failed for $symbol"
        return 1
    fi
}

# Function to process results
process_results() {
    local symbol=$1
    log "Processing results for $symbol"
    
    # Generate summary
    python scripts/summarize_trade_log.py --symbol "$symbol" >> "$LOG_DIR/server_backtest.log" 2>&1
    
    # Check data integrity
    python scripts/check_data_integrity.py --symbol "$symbol" >> "$LOG_DIR/server_backtest.log" 2>&1
}

# Main execution
main() {
    log "Starting server backtest"
    
    # Check required commands
    check_command python
    check_command pip
    
    # Check Python dependencies
    log "Checking Python dependencies"
    pip install -r requirements.txt
    
    # Get symbols from config
    if [ ! -f "$CONFIG_FILE" ]; then
        log "ERROR: Config file not found: $CONFIG_FILE"
        exit 1
    fi
    
    # Extract symbols from config (assuming YAML format)
    symbols=($(grep -A 10 "symbols:" "$CONFIG_FILE" | grep "-" | sed 's/- //'))
    
    if [ ${#symbols[@]} -eq 0 ]; then
        log "ERROR: No symbols found in config"
        exit 1
    fi
    
    log "Found ${#symbols[@]} symbols to process"
    
    # Run backtests in parallel
    for symbol in "${symbols[@]}"; do
        # Check if we've reached the maximum parallel jobs
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL_JOBS ]; do
            sleep 1
        done
        
        # Start backtest in background
        run_backtest "$symbol" &
    done
    
    # Wait for all backtests to complete
    wait
    
    # Process results
    for symbol in "${symbols[@]}"; do
        process_results "$symbol"
    done
    
    # Generate final summary
    log "Generating final summary"
    python scripts/summarize_trade_log.py --all >> "$LOG_DIR/server_backtest.log" 2>&1
    
    log "Server backtest completed"
}

# Run main function
main 