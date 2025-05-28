import logging
import yaml
import os
from chat_interface import AIChatInterface

def load_config():
    """Load configuration from file."""
    try:
        with open('config/config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        return {}

def main():
    """Main function to run the chat interface."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/chat.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Load configuration
    config = load_config()
    
    # Initialize chat interface
    chat_interface = AIChatInterface(config)
    
    print("\nWelkom bij de Trading AI Chat Interface!")
    print("Type 'exit' om te stoppen")
    print("Type 'clear' om de chat geschiedenis te wissen")
    print("Type 'analysis' voor een marktanalyse")
    print("Type 'strategy' voor uitleg over de huidige strategie")
    print("Type 'metrics' voor performance metrics")
    print("\nStel je vraag:")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() == 'exit':
                print("Tot ziens!")
                break
            elif user_input.lower() == 'clear':
                chat_interface.clear_history()
                print("Chat geschiedenis gewist")
            elif user_input.lower() == 'analysis':
                result = chat_interface.get_market_analysis()
                print("\nMarktanalyse:")
                print(result.get('analysis', 'Geen analyse beschikbaar'))
            elif user_input.lower() == 'strategy':
                result = chat_interface.get_strategy_explanation()
                print("\nStrategie uitleg:")
                print(result.get('explanation', 'Geen strategie-uitleg beschikbaar'))
            elif user_input.lower() == 'metrics':
                result = chat_interface.get_performance_metrics()
                print("\nPerformance metrics:")
                print(result.get('metrics', 'Geen metrics beschikbaar'))
            else:
                response = chat_interface.chat(user_input)
                print(f"\nAI: {response}")
                
        except KeyboardInterrupt:
            print("\nProgramma gestopt door gebruiker")
            break
        except Exception as e:
            print(f"\nEr is een fout opgetreden: {str(e)}")
            continue

if __name__ == "__main__":
    main() 