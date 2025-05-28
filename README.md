# Geavanceerde Cryptocurrency Trading Bot

Een professionele cryptocurrency trading bot met spraakbesturing, machine learning en geavanceerde monitoring.

## Features

- 🤖 Reinforcement Learning voor slimme trading beslissingen
- 🎤 Spraakbesturing in het Nederlands
- 📊 Real-time monitoring en metrics
- 🔄 Automatische herstelmechanismen
- 📈 Geavanceerde technische analyse
- 🔐 Veilige API integratie
- 📱 Telegram notificaties
- 💾 Efficiënt geheugengebruik

## Vereisten

- Python 3.8 of hoger
- pip (Python package manager)
- Virtual environment (wordt automatisch aangemaakt)
- API keys voor de gewenste exchanges
- Telegram bot token (optioneel)

## Installatie

1. Clone de repository:
```bash
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot
```

2. Run het setup script:
```bash
python setup.py
```

3. Configureer de bot:
- Vul je API keys in in het `.env` bestand
- Pas de trading parameters aan in `config/bot_config.json`
- Configureer de spraakbesturing in `config/voice_config.json`

## Gebruik

1. Activeer de virtual environment:
```bash
# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

2. Start de bot:
```bash
python main.py
```

3. Start de monitor:
```bash
python monitor_bot.py
```

## Spraakcommando's

De bot ondersteunt de volgende spraakcommando's:

- "Start trading" - Start de trading bot
- "Stop trading" - Stop de trading bot
- "Wat is je status?" - Toont de huidige status
- "Toon metrics" - Toont de trading prestaties
- "Pas risico aan naar [laag/gemiddeld/hoog]" - Past het risiconiveau aan
- "Verander strategie naar [conservatief/gematigd/agressief]" - Wijzigt de trading strategie

## Configuratie

### Bot Configuratie (`config/bot_config.json`)
```json
{
    "trading": {
        "exchange": "binance",
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "timeframe": "1h",
        "max_positions": 3,
        "risk_per_trade": 0.02
    },
    "monitoring": {
        "check_interval": 300,
        "alert_thresholds": {
            "cpu_percent": 80,
            "memory_percent": 80,
            "disk_percent": 80,
            "api_latency_ms": 1000
        }
    }
}
```

### Spraak Configuratie (`config/voice_config.json`)
```json
{
    "voice_id": "dutch",
    "rate": 150,
    "volume": 1.0,
    "commands": {
        "start_trading": ["start trading", "begin trading", "start bot"],
        "stop_trading": ["stop trading", "stop bot", "halt trading"],
        "check_status": ["status", "how are you", "what's your status"],
        "show_metrics": ["show metrics", "show performance", "show results"],
        "adjust_risk": ["adjust risk", "change risk", "modify risk"],
        "change_strategy": ["change strategy", "switch strategy", "new strategy"]
    }
}
```

## Monitoring

De bot monitort:
- CPU en geheugengebruik
- Netwerk latency
- Trading prestaties
- API gezondheid
- Systeem resources

## Veiligheid

- API keys worden veilig opgeslagen in `.env`
- Alle gevoelige data is versleuteld
- Rate limiting voor API calls
- Automatische error recovery
- Backup systeem voor configuratie

## Troubleshooting

1. **Bot start niet**
   - Controleer of de virtual environment actief is
   - Controleer of alle dependencies geïnstalleerd zijn
   - Controleer de API keys in `.env`

2. **Spraakherkenning werkt niet**
   - Controleer of de microfoon correct is aangesloten
   - Controleer of de juiste taal is ingesteld
   - Controleer de spraakconfiguratie

3. **Trading errors**
   - Controleer de API connectie
   - Controleer de trading parameters
   - Controleer de exchange status

## Contributing

1. Fork de repository
2. Maak een feature branch
3. Commit je changes
4. Push naar de branch
5. Maak een Pull Request

## License

Dit project is gelicenseerd onder de MIT License - zie het [LICENSE](LICENSE) bestand voor details.

## Contact

Voor vragen of ondersteuning, open een issue of neem contact op via:
- Email: your.email@example.com
- Telegram: @yourusername 