from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
import uvicorn
import os
from typing import Dict, List
import json
from datetime import datetime

from src.monitoring.monitoring_manager import MonitoringManager
from src.monitoring.event_manager import EventManager
from src.monitoring.error_manager import ErrorManager
from src.api.chat import ChatAPI

class WebServer:
    """Web server voor de trading bot interface."""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize FastAPI app
        self.app = FastAPI(title="AI Trading Bot Web Interface")
        
        # Initialize managers
        self.monitoring_manager = MonitoringManager(config)
        self.event_manager = EventManager(config)
        self.error_manager = ErrorManager(config)
        self.chat_api = ChatAPI(config)
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory="src/web/static"), name="static")
        
        # Setup templates
        self.templates = Jinja2Templates(directory="src/web/templates")
        
        # Setup routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_index(request: Request):
            return self.templates.TemplateResponse("index.html", {"request": request})
            
        @self.app.get("/api/metrics/summary")
        async def get_metrics_summary():
            return self.monitoring_manager.get_metrics_summary()
            
        @self.app.get("/api/events/summary")
        async def get_events_summary():
            return self.event_manager.get_events_summary()
            
        @self.app.get("/api/errors/summary")
        async def get_errors_summary():
            return self.error_manager.get_errors_summary()
            
        # Mount chat API
        self.app.mount("/ws", self.chat_api.get_app())
        
    async def start(self):
        """Start de web server."""
        try:
            # Start managers
            await self.monitoring_manager.start()
            await self.event_manager.start()
            await self.error_manager.start()
            
            # Start server
            config = uvicorn.Config(
                self.app,
                host=self.config.get('web_host', '0.0.0.0'),
                port=self.config.get('web_port', 8000),
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            print(f"Error starting web server: {str(e)}")
            raise
            
    async def stop(self):
        """Stop de web server."""
        try:
            # Stop managers
            await self.monitoring_manager.stop()
            await self.event_manager.stop()
            await self.error_manager.stop()
            
        except Exception as e:
            print(f"Error stopping web server: {str(e)}")
            raise 