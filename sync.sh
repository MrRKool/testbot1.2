#!/bin/bash

# Configuration
VPS_IP="147.93.59.80"
VPS_USER="root"
VPS_PATH="/root/trading-bot"
LOCAL_PATH="."

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting synchronization...${NC}"

# Function to handle errors
handle_error() {
    echo -e "${RED}Error: $1${NC}"
    exit 1
}

# Stop the bot on VPS if it's running
echo "Stopping bot on VPS..."
ssh $VPS_USER@$VPS_IP "pkill -f 'python main.py'" || echo "No bot process found"

# Sync files to VPS
echo "Syncing files to VPS..."
rsync -avz --exclude 'venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'logs' \
    --exclude 'data' \
    $LOCAL_PATH/ $VPS_USER@$VPS_IP:$VPS_PATH/ || handle_error "Failed to sync files"

# Create necessary directories on VPS if they don't exist
echo "Creating necessary directories..."
ssh $VPS_USER@$VPS_IP "mkdir -p $VPS_PATH/logs $VPS_PATH/data"

# Set up environment on VPS
echo "Setting up environment on VPS..."
ssh $VPS_USER@$VPS_IP "cd $VPS_PATH && \
    if [ ! -d 'venv' ]; then \
        python3 -m venv venv && \
        source venv/bin/activate && \
        pip install -r requirements.txt; \
    fi"

# Start the bot on VPS
echo "Starting bot on VPS..."
ssh $VPS_USER@$VPS_IP "cd $VPS_PATH && \
    source .env && \
    source venv/bin/activate && \
    nohup python main.py > logs/trading_bot.log 2>&1 &"

echo -e "${GREEN}Synchronization completed successfully!${NC}"
echo "Bot has been restarted on VPS"
echo "You can check the logs with: ssh $VPS_USER@$VPS_IP 'tail -f $VPS_PATH/logs/trading_bot.log'" 