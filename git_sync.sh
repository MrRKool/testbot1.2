#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
VPS_IP="147.93.59.80"
VPS_USER="root"
VPS_PATH="/root/trading-bot"
BRANCH="main"

echo -e "${GREEN}Starting git synchronization...${NC}"

# Function to handle errors
handle_error() {
    echo -e "${RED}Error: $1${NC}"
    exit 1
}

# Check if there are any uncommitted changes
if [[ -n $(git status -s) ]]; then
    echo -e "${YELLOW}Warning: You have uncommitted changes. Please commit or stash them first.${NC}"
    git status
    exit 1
fi

# Pull latest changes from remote
echo "Pulling latest changes from remote..."
git pull origin $BRANCH || handle_error "Failed to pull from remote"

# Push changes to remote
echo "Pushing changes to remote..."
git push origin $BRANCH || handle_error "Failed to push to remote"

# Stop the bot on VPS if it's running
echo "Stopping bot on VPS..."
ssh $VPS_USER@$VPS_IP "pkill -f 'python main.py'" || echo "No bot process found"

# Update and restart bot on VPS
echo "Updating and restarting bot on VPS..."
ssh $VPS_USER@$VPS_IP "cd $VPS_PATH && \
    git fetch origin && \
    git reset --hard origin/$BRANCH && \
    source .env && \
    source venv/bin/activate && \
    pip install -r requirements.txt && \
    nohup python main.py > logs/trading_bot.log 2>&1 &"

echo -e "${GREEN}Git synchronization completed successfully!${NC}"
echo "Bot has been restarted on VPS"
echo "You can check the logs with: ssh $VPS_USER@$VPS_IP 'tail -f $VPS_PATH/logs/trading_bot.log'" 