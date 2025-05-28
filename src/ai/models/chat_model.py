from typing import Dict, List, Optional
import json
import asyncio
from datetime import datetime
import logging
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ChatModel:
    """Model voor chat interactie met de AI."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Chat settings
        self.model_name = config.get('chat_model', 'gpt2')
        self.max_history = config.get('max_chat_history', 100)
        self.chat_history_path = config.get('chat_history_path', 'chat_history')
        
        # Initialize model and tokenizer
        self._init_model()
        
        # Load chat history
        self.chat_history = self._load_chat_history()
        
    def _init_model(self):
        """Initialize chat model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                
        except Exception as e:
            self.logger.error(f"Error initializing chat model: {str(e)}")
            raise
            
    def _load_chat_history(self) -> List[Dict]:
        """Load chat history from file."""
        try:
            # Create history directory if it doesn't exist
            os.makedirs(self.chat_history_path, exist_ok=True)
            
            history_file = os.path.join(self.chat_history_path, 'chat_history.json')
            
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    return json.load(f)
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error loading chat history: {str(e)}")
            return []
            
    async def _save_chat_history(self):
        """Save chat history to file."""
        try:
            history_file = os.path.join(self.chat_history_path, 'chat_history.json')
            
            with open(history_file, 'w') as f:
                json.dump(self.chat_history, f)
                
        except Exception as e:
            self.logger.error(f"Error saving chat history: {str(e)}")
            
    async def generate_response(self, message: str) -> str:
        """Generate AI response to message."""
        try:
            # Prepare input
            input_text = self._prepare_input(message)
            
            # Generate response
            inputs = self.tokenizer(input_text, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                
            outputs = self.model.generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                temperature=0.7
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response
            response = self._clean_response(response, input_text)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "Sorry, ik kon geen antwoord genereren. Probeer het later opnieuw."
            
    def _prepare_input(self, message: str) -> str:
        """Prepare input text for model."""
        try:
            # Get recent history
            recent_history = self.chat_history[-5:] if self.chat_history else []
            
            # Format history
            history_text = ""
            for entry in recent_history:
                history_text += f"User: {entry['message']}\n"
                history_text += f"AI: {entry['response']}\n"
                
            # Add current message
            input_text = f"{history_text}User: {message}\nAI:"
            
            return input_text
            
        except Exception as e:
            self.logger.error(f"Error preparing input: {str(e)}")
            return f"User: {message}\nAI:"
            
    def _clean_response(self, response: str, input_text: str) -> str:
        """Clean up model response."""
        try:
            # Remove input text from response
            response = response[len(input_text):].strip()
            
            # Remove any remaining "User:" or "AI:" prefixes
            response = response.replace("User:", "").replace("AI:", "").strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error cleaning response: {str(e)}")
            return response
            
    async def add_to_history(self, entry: Dict):
        """Add entry to chat history."""
        try:
            self.chat_history.append(entry)
            
            # Trim history if needed
            if len(self.chat_history) > self.max_history:
                self.chat_history = self.chat_history[-self.max_history:]
                
            # Save history
            await self._save_chat_history()
            
        except Exception as e:
            self.logger.error(f"Error adding to chat history: {str(e)}")
            
    async def get_chat_history(self) -> List[Dict]:
        """Get chat history."""
        return self.chat_history
        
    async def clear_chat_history(self) -> bool:
        """Clear chat history."""
        try:
            self.chat_history = []
            await self._save_chat_history()
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing chat history: {str(e)}")
            return False 