import pandas as pd
import numpy as np
from typing import Tuple, Optional

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices: pd.Series, 
                  fast_period: int = 12, 
                  slow_period: int = 26, 
                  signal_period: int = 9) -> Tuple[pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(prices: pd.Series, 
                            period: int = 20, 
                            std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    middle_band = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    return upper_band, middle_band, lower_band

def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average (EMA)."""
    return prices.ewm(span=period, adjust=False).mean()

def calculate_atr(high: pd.Series, 
                 low: pd.Series, 
                 close: pd.Series, 
                 period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()

def calculate_adx(high: pd.Series, 
                 low: pd.Series, 
                 close: pd.Series, 
                 period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Average Directional Index (ADX)."""
    high_diff = high.diff()
    low_diff = low.diff()
    
    plus_dm = high_diff.where((high_diff > 0) & (high_diff > -low_diff), 0)
    minus_dm = -low_diff.where((low_diff > 0) & (low_diff > high_diff), 0)
    
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift(1)),
        'lc': abs(low - close.shift(1))
    }).max(axis=1)
    
    smoothed_plus_dm = plus_dm.rolling(window=period).sum()
    smoothed_minus_dm = minus_dm.rolling(window=period).sum()
    smoothed_tr = tr.rolling(window=period).sum()
    
    plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
    minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx, plus_di, minus_di

def calculate_stochastic_rsi(prices: pd.Series, 
                           rsi_period: int = 14, 
                           stoch_period: int = 14, 
                           k_period: int = 3, 
                           d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Calculate Stochastic RSI."""
    rsi = calculate_rsi(prices, rsi_period)
    stoch_rsi = (rsi - rsi.rolling(window=stoch_period).min()) / \
                (rsi.rolling(window=stoch_period).max() - rsi.rolling(window=stoch_period).min())
    k = stoch_rsi.rolling(window=k_period).mean() * 100
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_obv(close: pd.Series, volume: pd.Series, smoothing: int = 5) -> pd.Series:
    """Calculate On-Balance Volume (OBV)."""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv.rolling(window=smoothing).mean() 