#!/bin/bash

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and required packages
sudo apt-get install -y python3 python3-pip python3-venv

# Create project directory
mkdir -p ~/trading-bot
cd ~/trading-bot

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required Python packages
pip install --upgrade pip
pip install numpy>=1.24.0 numba pandas aiohttp psutil

# Copy service file
sudo cp trading-bot.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start the service
sudo systemctl enable trading-bot.service
sudo systemctl start trading-bot.service

# Create logs directory
mkdir -p logs

echo "Installation completed. Check status with: sudo systemctl status trading-bot.service" 