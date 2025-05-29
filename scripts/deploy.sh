#!/bin/bash
# Synchroniseer bot naar VPS en installeer requirements

VPS=root@147.93.59.80
REMOTE_DIR=/root/trading-bot

# Installeer Python 3.11 op VPS als het nog niet bestaat
ssh $VPS "if ! command -v python3.11 &> /dev/null; then \
    apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev; \
fi"

# Maak backup van huidige versie
ssh $VPS "cd /root && mv trading-bot trading-bot-backup-\$(date +%Y%m%d-%H%M%S)"

# Sync alles behalve venv, __pycache__, .git, .DS_Store, db files
rsync -avz --exclude 'venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' --exclude '.DS_Store' --exclude 'trading_bot.db*' ./ $VPS:$REMOTE_DIR/

# Setup nieuwe virtual environment en installeer requirements
ssh $VPS "cd $REMOTE_DIR && \
    python3.11 -m venv venv && \
    source venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt"

# Start de bot in test modus
ssh $VPS "cd $REMOTE_DIR && source venv/bin/activate && python main.py --test" 