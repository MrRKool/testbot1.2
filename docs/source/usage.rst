Usage
=====

Starting the Bot
--------------

To start the trading bot:

.. code-block:: bash

   python main.py

Or using the Makefile:

.. code-block:: bash

   make run

Command Line Arguments
--------------------

The bot supports the following command line arguments:

.. code-block:: bash

   python main.py --help

   usage: main.py [-h] [--config CONFIG] [--testnet] [--debug] [--backtest]

   optional arguments:
     -h, --help           show this help message and exit
     --config CONFIG      path to config file (default: config/config.yaml)
     --testnet           use testnet (default: False)
     --debug             enable debug mode (default: False)
     --backtest          run in backtest mode (default: False)

Running Backtests
---------------

To run a backtest:

.. code-block:: bash

   python backtest.py --config config/backtest_config.yaml

Or using the Makefile:

.. code-block:: bash

   make backtest

Backtest Configuration
~~~~~~~~~~~~~~~~~~~~

The backtest configuration file should include:

.. code-block:: yaml

   backtest:
     start_date: "2024-01-01"
     end_date: "2024-03-01"
     initial_balance: 10000
     symbols:
       - BTCUSDT
     timeframes:
       - 1h
     strategy:
       name: "custom_strategy"
       parameters:
         rsi_period: 14
         macd_fast: 12
         macd_slow: 26

Monitoring
---------

The bot provides several monitoring options:

1. Logs
~~~~~~~

View the logs:

.. code-block:: bash

   tail -f logs/trading_bot.log

2. Performance Metrics
~~~~~~~~~~~~~~~~~~~~~

View performance metrics:

.. code-block:: bash

   python utils/performance_analyzer.py

3. Prometheus/Grafana
~~~~~~~~~~~~~~~~~~~~

Access the Prometheus metrics at:
- http://localhost:9090

Access the Grafana dashboard at:
- http://localhost:3000

Telegram Notifications
--------------------

The bot can send notifications to Telegram. Configure in ``config.yaml``:

.. code-block:: yaml

   telegram:
     enabled: true
     bot_token: your_bot_token
     chat_id: your_chat_id
     notifications:
       trades: true
       errors: true
       performance: true

Common Operations
---------------

1. Starting a New Trade
~~~~~~~~~~~~~~~~~~~~~~

The bot will automatically start trades based on the configured strategy. You can monitor active trades in the logs or through Telegram notifications.

2. Stopping a Trade
~~~~~~~~~~~~~~~~~~

To stop a trade:

.. code-block:: bash

   python utils/trade_manager.py --action stop --trade_id TRADE_ID

3. Viewing Active Trades
~~~~~~~~~~~~~~~~~~~~~~~

To view active trades:

.. code-block:: bash

   python utils/trade_manager.py --action list

4. Updating Configuration
~~~~~~~~~~~~~~~~~~~~~~~

After updating the configuration:

.. code-block:: bash

   python utils/config_validator.py
   python main.py --config config/config.yaml

Troubleshooting
-------------

1. Bot Not Starting
~~~~~~~~~~~~~~~~~~

Check:
- Configuration file exists and is valid
- API keys are correct
- Database is running
- Log directory exists and is writable

2. No Trades Being Executed
~~~~~~~~~~~~~~~~~~~~~~~~~

Check:
- Strategy parameters
- Market conditions
- Risk management settings
- Logs for any errors

3. Performance Issues
~~~~~~~~~~~~~~~~~~~

Check:
- System resources (CPU, memory)
- Database performance
- Network connectivity
- Rate limiting settings 