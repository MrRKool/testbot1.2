API Reference
============

Core Components
-------------

PriceFetcher
~~~~~~~~~~~

.. automodule:: utils.price_fetcher
   :members:
   :undoc-members:
   :show-inheritance:

ConfigValidator
~~~~~~~~~~~~~

.. automodule:: utils.config_validator
   :members:
   :undoc-members:
   :show-inheritance:

PerformanceAnalyzer
~~~~~~~~~~~~~~~~~

.. automodule:: utils.performance_analyzer
   :members:
   :undoc-members:
   :show-inheritance:

Exchange API
-----------

Bybit API
~~~~~~~~

The bot uses the Bybit API v5. Key endpoints:

Market Data
^^^^^^^^^^

- GET /v5/market/tickers
  - Get ticker information
  - Parameters:
    - category: spot/linear/inverse
    - symbol: optional

- GET /v5/market/kline
  - Get candlestick data
  - Parameters:
    - category: spot/linear/inverse
    - symbol: required
    - interval: 1m/3m/5m/15m/30m/1h/2h/4h/6h/12h/1d/1w/1M
    - limit: 1-1000

Trading
^^^^^^^

- POST /v5/order/create
  - Create order
  - Parameters:
    - category: spot/linear/inverse
    - symbol: required
    - side: Buy/Sell
    - orderType: Market/Limit
    - qty: required
    - price: required for Limit orders

- POST /v5/order/cancel
  - Cancel order
  - Parameters:
    - category: spot/linear/inverse
    - symbol: required
    - orderId: required

Account
^^^^^^^

- GET /v5/account/wallet-balance
  - Get wallet balance
  - Parameters:
    - accountType: UNIFIED/CONTRACT

Rate Limits
----------

The API has the following rate limits:

- Market Data: 120 requests per second
- Trading: 50 requests per second
- Account: 20 requests per second

Error Codes
----------

Common error codes:

- 10001: Invalid API key
- 10002: Invalid signature
- 10003: Invalid timestamp
- 10004: Invalid request
- 10005: Invalid permission
- 10006: IP address not allowed
- 10007: API key expired
- 10008: Invalid request size
- 10009: Invalid request parameter
- 10010: Invalid request format

WebSocket API
-----------

The bot also supports WebSocket connections for real-time data:

Market Data
~~~~~~~~~~

- Topic: tickers
  - Subscribe: ``{"op": "subscribe", "args": ["tickers.BTCUSDT"]}``
  - Unsubscribe: ``{"op": "unsubscribe", "args": ["tickers.BTCUSDT"]}``

- Topic: kline
  - Subscribe: ``{"op": "subscribe", "args": ["kline.1.BTCUSDT"]}``
  - Unsubscribe: ``{"op": "unsubscribe", "args": ["kline.1.BTCUSDT"]}``

Trading
~~~~~~~

- Topic: orders
  - Subscribe: ``{"op": "subscribe", "args": ["orders"]}``
  - Unsubscribe: ``{"op": "unsubscribe", "args": ["orders"]}``

- Topic: positions
  - Subscribe: ``{"op": "subscribe", "args": ["positions"]}``
  - Unsubscribe: ``{"op": "unsubscribe", "args": ["positions"]}``

WebSocket Rate Limits
-------------------

- Connection limit: 5 connections per IP
- Subscription limit: 50 subscriptions per connection
- Message size: 1MB per message
- Heartbeat: 20 seconds

Authentication
------------

API requests must be authenticated using:

1. API Key in header: ``X-BAPI-API-KEY``
2. Timestamp in header: ``X-BAPI-TIMESTAMP``
3. Signature in header: ``X-BAPI-SIGN``

Signature generation:

.. code-block:: python

   def generate_signature(api_secret, timestamp, recv_window, params):
       param_str = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
       sign_str = f"{timestamp}{api_key}{recv_window}{param_str}"
       return hmac.new(
           api_secret.encode('utf-8'),
           sign_str.encode('utf-8'),
           hashlib.sha256
       ).hexdigest() 