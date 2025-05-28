Configuration
============

The trading bot uses a YAML configuration file for all settings. The main configuration file is located at ``config/config.yaml``.

Configuration Structure
---------------------

Exchange Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   exchange:
     name: bybit
     testnet: true
     api:
       key: your_api_key
       secret: your_api_secret
     rate_limit:
       requests_per_second: 5
       max_retries: 3

Trading Parameters
~~~~~~~~~~~~~~~~

.. code-block:: yaml

   trading:
     symbols:
       - BTCUSDT
       - ETHUSDT
     timeframes:
       - 1m
       - 5m
       - 15m
       - 1h
     min_volume: 1000000
     max_spread: 0.002

Strategy Parameters
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   strategy:
     indicators:
       rsi:
         period: 14
         overbought: 70
         oversold: 30
       macd:
         fast_period: 12
         slow_period: 26
         signal_period: 9
     market_regime:
       trend:
         enabled: true
         lookback_period: 20
       volatility:
         enabled: true
         threshold: 0.02

Risk Management
~~~~~~~~~~~~~

.. code-block:: yaml

   risk:
     max_position_size: 1000
     max_leverage: 5
     stop_loss:
       enabled: true
       percentage: 0.02
     take_profit:
       enabled: true
       percentage: 0.04

Performance Tracking
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   performance:
     enabled: true
     metrics:
       - win_rate
       - profit_factor
       - sharpe_ratio
     save_trades: true

Logging
~~~~~~~

.. code-block:: yaml

   logging:
     level: INFO
     format: json
     file: logs/trading_bot.log
     max_size: 10485760
     backup_count: 5

Database
~~~~~~~~

.. code-block:: yaml

   database:
     url: postgresql://user:password@localhost:5432/trading_bot
     pool_size: 5
     max_overflow: 10

Monitoring
~~~~~~~~~

.. code-block:: yaml

   monitoring:
     enabled: true
     prometheus:
       port: 9090
     grafana:
       port: 3000

Backup
~~~~~~

.. code-block:: yaml

   backup:
     enabled: true
     directory: backups
     interval: 86400
     keep_last: 7

Environment Variables
-------------------

The following environment variables can be used to override configuration settings:

- ``BYBIT_API_KEY``: Bybit API key
- ``BYBIT_API_SECRET``: Bybit API secret
- ``DATABASE_URL``: Database connection URL
- ``LOG_LEVEL``: Logging level
- ``TELEGRAM_BOT_TOKEN``: Telegram bot token
- ``TELEGRAM_CHAT_ID``: Telegram chat ID

Configuration Validation
----------------------

The configuration can be validated using the config validator:

.. code-block:: bash

   python utils/config_validator.py

This will check for:
- Required fields
- Valid values
- Type checking
- Dependencies between settings

Common Issues
------------

1. Invalid API Keys
~~~~~~~~~~~~~~~~~

If you get API key errors:
- Check if the keys are correct
- Verify if testnet is enabled/disabled as needed
- Ensure the keys have the correct permissions

2. Database Connection
~~~~~~~~~~~~~~~~~~~~

If you have database connection issues:
- Check the database URL format
- Verify the database is running
- Check user permissions

3. Logging Issues
~~~~~~~~~~~~~~~

If logs are not being written:
- Check if the log directory exists
- Verify write permissions
- Check the log level setting 