import speech_recognition as sr
import pyttsx3
import threading
import queue
import logging
from typing import Optional, Callable, Dict, Any
import json
import os
from datetime import datetime

class VoiceInterface:
    """Voice interface for controlling the trading bot."""
    
    def __init__(self, config_path: str = 'config/voice_config.json'):
        self.logger = logging.getLogger(__name__)
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.command_queue = queue.Queue()
        self.is_listening = False
        self.callbacks: Dict[str, Callable] = {}
        
        # Load voice configuration
        self._load_config(config_path)
        
        # Initialize voice settings
        self._setup_voice()
        
    def _load_config(self, config_path: str):
        """Load voice configuration."""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {
                'voice_id': 'dutch',
                'rate': 150,
                'volume': 1.0,
                'commands': {
                    'start_trading': ['start trading', 'begin trading', 'start bot'],
                    'stop_trading': ['stop trading', 'stop bot', 'halt trading'],
                    'check_status': ['status', 'how are you', 'what\'s your status'],
                    'show_metrics': ['show metrics', 'show performance', 'show results'],
                    'adjust_risk': ['adjust risk', 'change risk', 'modify risk'],
                    'change_strategy': ['change strategy', 'switch strategy', 'new strategy']
                }
            }
            os.makedirs('config', exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
                
    def _setup_voice(self):
        """Setup voice settings."""
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if self.config['voice_id'] in voice.id.lower():
                self.engine.setProperty('voice', voice.id)
                break
                
        self.engine.setProperty('rate', self.config['rate'])
        self.engine.setProperty('volume', self.config['volume'])
        
    def register_callback(self, command: str, callback: Callable):
        """Register callback for voice command."""
        self.callbacks[command] = callback
        
    def speak(self, text: str):
        """Convert text to speech."""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            self.logger.error(f"Error in text-to-speech: {str(e)}")
            
    def _process_command(self, text: str) -> Optional[str]:
        """Process voice command and return corresponding action."""
        text = text.lower()
        
        for action, commands in self.config['commands'].items():
            if any(cmd in text for cmd in commands):
                return action
                
        return None
        
    def _listen_loop(self):
        """Main listening loop."""
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            
            while self.is_listening:
                try:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    text = self.recognizer.recognize_google(audio, language='nl-NL')
                    
                    self.logger.info(f"Recognized: {text}")
                    
                    action = self._process_command(text)
                    if action and action in self.callbacks:
                        self.command_queue.put((action, text))
                        
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    self.speak("Ik heb je niet goed verstaan. Kun je dat herhalen?")
                except sr.RequestError as e:
                    self.logger.error(f"Error with speech recognition service: {str(e)}")
                    self.speak("Er is een probleem met de spraakherkenning. Probeer het later opnieuw.")
                except Exception as e:
                    self.logger.error(f"Unexpected error in voice recognition: {str(e)}")
                    
    def _process_commands(self):
        """Process queued commands."""
        while self.is_listening:
            try:
                action, text = self.command_queue.get(timeout=1)
                if action in self.callbacks:
                    self.callbacks[action](text)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing command: {str(e)}")
                
    def start(self):
        """Start voice interface."""
        if not self.is_listening:
            self.is_listening = True
            
            # Start listening thread
            self.listen_thread = threading.Thread(target=self._listen_loop)
            self.listen_thread.daemon = True
            self.listen_thread.start()
            
            # Start command processing thread
            self.process_thread = threading.Thread(target=self._process_commands)
            self.process_thread.daemon = True
            self.process_thread.start()
            
            self.speak("Spraakinterface is geactiveerd. Hoe kan ik je helpen?")
            
    def stop(self):
        """Stop voice interface."""
        self.is_listening = False
        if hasattr(self, 'listen_thread'):
            self.listen_thread.join()
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
            
    def update_config(self, key: str, value: Any):
        """Update voice configuration."""
        if key in self.config:
            self.config[key] = value
            if key in ['voice_id', 'rate', 'volume']:
                self._setup_voice()
                
            # Save updated config
            with open('config/voice_config.json', 'w') as f:
                json.dump(self.config, f, indent=4) 