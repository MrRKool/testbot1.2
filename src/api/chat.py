from fastapi import FastAPI, WebSocket, HTTPException
from typing import Dict, List, Optional
import json
import asyncio
from datetime import datetime
import logging
from src.ai.models.chat_model import ChatModel

class ChatAPI:
    """API voor chat interactie met de AI."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize FastAPI app
        self.app = FastAPI(title="AI Trading Bot Chat API")
        
        # Initialize chat model
        self.chat_model = ChatModel(config)
        
        # Store active connections
        self.active_connections: List[WebSocket] = []
        
        # Setup routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.websocket("/ws/chat")
        async def websocket_endpoint(websocket: WebSocket):
            await self._handle_websocket(websocket)
            
        @self.app.get("/api/chat/history")
        async def get_chat_history():
            return await self.chat_model.get_chat_history()
            
        @self.app.post("/api/chat/clear")
        async def clear_chat_history():
            return await self.chat_model.clear_chat_history()
            
    async def _handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection."""
        try:
            await websocket.accept()
            self.active_connections.append(websocket)
            
            # Send welcome message
            await websocket.send_json({
                "type": "system",
                "message": "Welkom bij de AI Trading Bot chat! Hoe kan ik je helpen?"
            })
            
            while True:
                try:
                    # Receive message
                    data = await websocket.receive_json()
                    
                    # Process message
                    response = await self._process_message(data)
                    
                    # Send response
                    await websocket.send_json(response)
                    
                except Exception as e:
                    self.logger.error(f"Error processing message: {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Er is een fout opgetreden bij het verwerken van je bericht."
                    })
                    
        except Exception as e:
            self.logger.error(f"WebSocket error: {str(e)}")
            
        finally:
            self.active_connections.remove(websocket)
            
    async def _process_message(self, data: Dict) -> Dict:
        """Process incoming message and generate response."""
        try:
            message = data.get("message", "")
            message_type = data.get("type", "user")
            
            # Get AI response
            response = await self.chat_model.generate_response(message)
            
            # Store in chat history
            await self.chat_model.add_to_history({
                "timestamp": datetime.now().isoformat(),
                "type": message_type,
                "message": message,
                "response": response
            })
            
            return {
                "type": "ai",
                "message": response
            }
            
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
    async def broadcast_message(self, message: Dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                self.logger.error(f"Error broadcasting message: {str(e)}")
                
    def get_app(self) -> FastAPI:
        """Get FastAPI app instance."""
        return self.app 