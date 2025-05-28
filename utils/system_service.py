import os
import sys
import platform
import subprocess
import logging
from pathlib import Path
import time
import signal
import psutil

class ServiceManager:
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.is_macos = platform.system() == 'Darwin'
        self.pid_file = Path('bot.pid')
        self.logger = logging.getLogger(__name__)
        
        if self.is_macos:
            self.launch_agents_dir = Path.home() / 'Library/LaunchAgents'
            self.plist_file = self.launch_agents_dir / f'com.{service_name}.plist'
        else:
            self.service_file = Path('/etc/systemd/system') / f'{service_name}.service'
            
    def create_service(self):
        """Create service file based on OS."""
        try:
            if self.is_macos:
                return self._create_launch_agent()
            else:
                return self._create_systemd_service()
        except Exception as e:
            self.logger.error(f"Failed to create service: {str(e)}")
            return False
            
    def start_service(self):
        """Start the service with improved error handling."""
        try:
            if self.is_macos:
                return self._start_launch_agent()
            else:
                return self._start_systemd_service()
        except Exception as e:
            self.logger.error(f"Failed to start service: {str(e)}")
            return False
            
    def stop_service(self):
        """Stop the service with improved error handling."""
        try:
            if self.is_macos:
                return self._stop_launch_agent()
            else:
                return self._stop_systemd_service()
        except Exception as e:
            self.logger.error(f"Failed to stop service: {str(e)}")
            return False
            
    def _create_launch_agent(self):
        """Create LaunchAgent plist file for macOS with improved configuration."""
        try:
            self.launch_agents_dir.mkdir(parents=True, exist_ok=True)
            
            # Get absolute paths
            python_path = sys.executable
            main_script = os.path.abspath("main.py")
            log_dir = os.path.abspath("logs")
            
            # Create log directory if it doesn't exist
            os.makedirs(log_dir, exist_ok=True)
            
            plist_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.{self.service_name}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>{main_script}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardErrorPath</key>
    <string>{log_dir}/error.log</string>
    <key>StandardOutPath</key>
    <string>{log_dir}/output.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
        <key>PATH</key>
        <string>{os.environ.get('PATH', '')}</string>
    </dict>
    <key>WorkingDirectory</key>
    <string>{os.getcwd()}</string>
    <key>ProcessType</key>
    <string>Interactive</string>
    <key>ThrottleInterval</key>
    <integer>5</integer>
    <key>ExitTimeOut</key>
    <integer>10</integer>
    <key>AbandonProcessGroup</key>
    <true/>
</dict>
</plist>'''
            
            with open(self.plist_file, 'w') as f:
                f.write(plist_content)
                
            self.logger.info(f"Created LaunchAgent at {self.plist_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create LaunchAgent: {str(e)}")
            return False
            
    def _start_launch_agent(self):
        """Start the LaunchAgent service with improved error handling."""
        try:
            # Check if service is already running
            if self._is_service_running():
                self.logger.info("Service is already running")
                return True
                
            # Ensure plist file exists
            if not self.plist_file.exists():
                if not self._create_launch_agent():
                    return False
                
            # Load the service
            result = subprocess.run(
                ['launchctl', 'load', str(self.plist_file)],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                self.logger.error(f"Failed to load service: {result.stderr}")
                return False
                
            # Wait for service to start
            time.sleep(2)
            
            # Verify service is running
            if not self._is_service_running():
                self.logger.error("Service failed to start")
                return False
                
            self.logger.info("Service started successfully")
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.error("Timeout while starting service")
            return False
        except Exception as e:
            self.logger.error(f"Failed to start service: {str(e)}")
            return False
            
    def _stop_launch_agent(self):
        """Stop the LaunchAgent service with improved error handling."""
        try:
            # Check if service is running
            if not self._is_service_running():
                self.logger.info("Service is not running")
                return True
                
            # Unload the service
            result = subprocess.run(
                ['launchctl', 'unload', str(self.plist_file)],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                self.logger.error(f"Failed to unload service: {result.stderr}")
                return False
                
            # Wait for service to stop
            time.sleep(2)
            
            # Verify service is stopped
            if self._is_service_running():
                self.logger.error("Service failed to stop")
                return False
                
            self.logger.info("Service stopped successfully")
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.error("Timeout while stopping service")
            return False
        except Exception as e:
            self.logger.error(f"Failed to stop service: {str(e)}")
            return False
            
    def _is_service_running(self):
        """Check if the service is running with improved error handling."""
        try:
            # Check PID file first
            if self.pid_file.exists():
                try:
                    pid = int(self.pid_file.read_text().strip())
                    process = psutil.Process(pid)
                    if process.is_running() and process.name().lower().startswith('python'):
                        return True
                except (ValueError, psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Check launchctl list
            result = subprocess.run(
                ['launchctl', 'list', f'com.{self.service_name}'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and '0' in result.stdout:
                return True
                
            return False
            
        except subprocess.TimeoutExpired:
            self.logger.error("Timeout checking service status")
            return False
        except Exception as e:
            self.logger.error(f"Failed to check service status: {str(e)}")
            return False
            
    def _create_systemd_service(self):
        """Create systemd service file for Linux."""
        try:
            service_content = f'''[Unit]
Description={self.service_name}
After=network.target

[Service]
Type=simple
User={os.getenv('USER')}
WorkingDirectory={os.getcwd()}
ExecStart={sys.executable} {os.path.abspath("main.py")}
Restart=always
RestartSec=5
StandardOutput=append:{os.path.abspath("logs/output.log")}
StandardError=append:{os.path.abspath("logs/error.log")}

[Install]
WantedBy=multi-user.target'''
            
            with open(self.service_file, 'w') as f:
                f.write(service_content)
                
            self.logger.info(f"Created systemd service at {self.service_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create systemd service: {str(e)}")
            return False
            
    def _start_systemd_service(self):
        """Start the systemd service."""
        try:
            result = subprocess.run(['systemctl', 'start', self.service_name], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Failed to start service: {result.stderr}")
                return False
                
            self.logger.info("Service started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start service: {str(e)}")
            return False
            
    def _stop_systemd_service(self):
        """Stop the systemd service."""
        try:
            result = subprocess.run(['systemctl', 'stop', self.service_name], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Failed to stop service: {result.stderr}")
                return False
                
            self.logger.info("Service stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop service: {str(e)}")
            return False 