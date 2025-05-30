#!/bin/bash

# Configuration
VPS="root@147.93.59.80"
REMOTE_DIR="/root/trading-bot"
LOCAL_DIR="."

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${GREEN}[*] $1${NC}"
}

print_error() {
    echo -e "${RED}[!] $1${NC}"
}

# Check if SSH key exists
if [ ! -f ~/.ssh/id_rsa ]; then
    print_error "SSH key not found. Please generate one with: ssh-keygen -t rsa"
    exit 1
fi

# Ensure SSH key is added to VPS
print_status "Checking SSH connection..."
ssh -o BatchMode=yes -o ConnectTimeout=5 $VPS echo "SSH connection successful" || {
    print_error "SSH connection failed. Please ensure your SSH key is added to the VPS."
    print_error "You can add it with: ssh-copy-id $VPS"
    exit 1
}

# Create remote directory if it doesn't exist
print_status "Creating remote directory if it doesn't exist..."
ssh $VPS "mkdir -p $REMOTE_DIR"

# Sync everything except venv, __pycache__, .git, .DS_Store, db files, and logs
print_status "Syncing files to VPS..."
rsync -avz --exclude 'venv' \
          --exclude '__pycache__' \
          --exclude '*.pyc' \
          --exclude '.git' \
          --exclude '.DS_Store' \
          --exclude 'trading_bot.db*' \
          --exclude 'logs/*' \
          --exclude '.env' \
          ./ $VPS:$REMOTE_DIR/

# Setup new virtual environment and install requirements
print_status "Setting up virtual environment and installing requirements..."
ssh $VPS "cd $REMOTE_DIR && \
    python3.11 -m venv venv && \
    source venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt"

# Copy .env file separately (if it exists)
if [ -f .env ]; then
    print_status "Copying .env file..."
    scp .env $VPS:$REMOTE_DIR/.env
else
    print_error "Warning: .env file not found locally"
fi

# Test the installation
print_status "Testing the installation..."
ssh $VPS "cd $REMOTE_DIR && source venv/bin/activate && python main.py --test"

print_status "Sync completed successfully!" 