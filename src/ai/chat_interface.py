import logging
import json
from openai import OpenAI
from datetime import datetime

class AIChatInterface:
    def __init__(self, config):
        """Initialize the AI chat interface."""
        self.logger = logging.getLogger('chat_interface')
        self.client = OpenAI(api_key=config['openai']['api_key'])
        self.model = config['openai']['model']
        self.temperature = config['openai']['temperature']
        self.max_tokens = config['openai']['max_tokens']
        self.chat_history = []
        
        # Log de gebruikte API key en modelnaam
        self.logger.info(f"Initializing chat interface with model: {self.model}")
        self.logger.info(f"API key (first 10 chars): {config['openai']['api_key'][:10]}...")
        
        # System message definiëren
        self.system_message = {
            "role": "system",
            "content": """Je bent een geavanceerde trading AI assistent. Je kunt:
            1. Marktanalyses uitvoeren
            2. Trading strategieën uitleggen
            3. Risico's inschatten
            4. Performance metrics analyseren
            5. Technische indicatoren uitleggen
            
            Je antwoordt altijd in het Nederlands en gebruikt duidelijke, professionele taal."""
        }
        
        self.logger.info("AI Chat Interface initialized")
    
    def chat(self, message):
        """Process a user message and return AI response."""
        try:
            # Voeg gebruikersbericht toe aan geschiedenis
            self.chat_history.append({"role": "user", "content": message})
            
            # Bereid messages voor
            messages = [self.system_message] + self.chat_history
            
            # API call maken
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Antwoord ophalen en toevoegen aan geschiedenis
            ai_response = response.choices[0].message.content
            self.chat_history.append({"role": "assistant", "content": ai_response})
            
            return ai_response
            
        except Exception as e:
            self.logger.error(f"Error in chat: {str(e)}")
            return f"Er is een fout opgetreden: {str(e)}"
    
    def get_market_analysis(self):
        """Get current market analysis."""
        try:
            message = "Geef een gedetailleerde analyse van de huidige marktsituatie."
            return self.chat(message)
        except Exception as e:
            self.logger.error(f"Error in market analysis: {str(e)}")
            return f"Er is een fout opgetreden bij de marktanalyse: {str(e)}"
    
    def get_strategy_explanation(self):
        """Get explanation of current trading strategy."""
        try:
            message = "Leg uit welke trading strategieën momenteel actief zijn en hoe ze werken."
            return self.chat(message)
        except Exception as e:
            self.logger.error(f"Error in strategy explanation: {str(e)}")
            return f"Er is een fout opgetreden bij de strategie-uitleg: {str(e)}"
    
    def get_performance_metrics(self):
        """Get current performance metrics."""
        try:
            message = "Geef een overzicht van de huidige performance metrics en hun interpretatie."
            return self.chat(message)
        except Exception as e:
            self.logger.error(f"Error in performance metrics: {str(e)}")
            return f"Er is een fout opgetreden bij het ophalen van metrics: {str(e)}"
    
    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []
        self.logger.info("Chat history cleared") 