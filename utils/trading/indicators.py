import numpy as np
import pandas as pd
import talib

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    try:
        return pd.Series(talib.RSI(data['close'].values, timeperiod=period), index=data.index)
    except Exception as e:
        raise Exception(f"Error calculating RSI: {str(e)}")

def calculate_macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
    """Calculate MACD indicator."""
    try:
        macd, signal, hist = talib.MACD(
            data['close'].values,
            fastperiod=fast_period,
            slowperiod=slow_period,
            signalperiod=signal_period
        )
        return (
            pd.Series(macd, index=data.index),
            pd.Series(signal, index=data.index),
            pd.Series(hist, index=data.index)
        )
    except Exception as e:
        raise Exception(f"Error calculating MACD: {str(e)}")

def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> tuple:
    """Calculate Bollinger Bands."""
    try:
        upper, middle, lower = talib.BBANDS(
            data['close'].values,
            timeperiod=period,
            nbdevup=std_dev,
            nbdevdn=std_dev
        )
        return (
            pd.Series(upper, index=data.index),
            pd.Series(middle, index=data.index),
            pd.Series(lower, index=data.index)
        )
    except Exception as e:
        raise Exception(f"Error calculating Bollinger Bands: {str(e)}")

def calculate_ema(data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate EMA indicator."""
    try:
        return pd.Series(talib.EMA(data['close'].values, timeperiod=period), index=data.index)
    except Exception as e:
        raise Exception(f"Error calculating EMA: {str(e)}")

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR indicator."""
    try:
        return pd.Series(talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=period), index=data.index)
    except Exception as e:
        raise Exception(f"Error calculating ATR: {str(e)}")

def calculate_adx(data: pd.DataFrame, period: int = 14) -> tuple:
    """Calculate ADX indicator."""
    try:
        adx = talib.ADX(data['high'].values, data['low'].values, data['close'].values, timeperiod=period)
        plus_di = talib.PLUS_DI(data['high'].values, data['low'].values, data['close'].values, timeperiod=period)
        minus_di = talib.MINUS_DI(data['high'].values, data['low'].values, data['close'].values, timeperiod=period)
        return (
            pd.Series(adx, index=data.index),
            pd.Series(plus_di, index=data.index),
            pd.Series(minus_di, index=data.index)
        )
    except Exception as e:
        raise Exception(f"Error calculating ADX: {str(e)}")

def calculate_stochastic_rsi(data: pd.DataFrame, rsi_period: int = 14, stoch_period: int = 14, k_period: int = 3, d_period: int = 3) -> tuple:
    """Calculate Stochastic RSI."""
    try:
        rsi = calculate_rsi(data, rsi_period)
        stoch_k, stoch_d = talib.STOCH(
            rsi.values, rsi.values, rsi.values,
            fastk_period=k_period,
            slowk_period=stoch_period,
            slowk_matype=0,
            slowd_period=d_period,
            slowd_matype=0
        )
        return (
            pd.Series(stoch_k, index=data.index),
            pd.Series(stoch_d, index=data.index)
        )
    except Exception as e:
        raise Exception(f"Error calculating Stochastic RSI: {str(e)}")

def calculate_obv(data: pd.DataFrame, smoothing: int = 5) -> pd.Series:
    """Calculate On Balance Volume with smoothing."""
    try:
        obv = talib.OBV(data['close'].values, data['volume'].values)
        if smoothing > 1:
            obv = talib.SMA(obv, timeperiod=smoothing)
        return pd.Series(obv, index=data.index)
    except Exception as e:
        raise Exception(f"Error calculating OBV: {str(e)}")

def calculate_klinger_oscillator(data: pd.DataFrame, short_period: int = 34, long_period: int = 55) -> pd.Series:
    """Calculate Klinger Volume Oscillator (KVO).
    Combines price and volume to detect trend reversals.
    Usage: Buy/sell on signal line crossovers."""
    try:
        # Calculate trend direction
        trend = np.where(data['close'] > data['close'].shift(1), 1, -1)
        
        # Calculate volume force
        volume_force = data['volume'] * trend
        
        # Calculate short and long EMAs of volume force
        short_ema = talib.EMA(volume_force, timeperiod=short_period)
        long_ema = talib.EMA(volume_force, timeperiod=long_period)
        
        # Calculate KVO
        kvo = short_ema - long_ema
        
        # Calculate signal line (EMA of KVO)
        signal = talib.EMA(kvo, timeperiod=13)
        
        return pd.Series(kvo, index=data.index), pd.Series(signal, index=data.index)
    except Exception as e:
        raise Exception(f"Error calculating Klinger Oscillator: {str(e)}")

def calculate_vortex_indicator(data: pd.DataFrame, period: int = 14) -> tuple:
    """Calculate Vortex Indicator (VI).
    Detects the start of a new trend.
    Usage: Buy when VI+ > VI-."""
    try:
        # Calculate True Range
        tr = talib.TRANGE(data['high'].values, data['low'].values, data['close'].values)
        
        # Calculate +VM and -VM
        vm_plus = np.abs(data['high'].values - data['low'].shift(1).values)
        vm_minus = np.abs(data['low'].values - data['high'].shift(1).values)
        
        # Calculate VI+ and VI-
        vi_plus = talib.SMA(vm_plus, timeperiod=period) / talib.SMA(tr, timeperiod=period)
        vi_minus = talib.SMA(vm_minus, timeperiod=period) / talib.SMA(tr, timeperiod=period)
        
        return pd.Series(vi_plus, index=data.index), pd.Series(vi_minus, index=data.index)
    except Exception as e:
        raise Exception(f"Error calculating Vortex Indicator: {str(e)}")

def calculate_chaikin_money_flow(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Chaikin Money Flow (CMF).
    Measures buying pressure based on volume and closing price.
    Usage: >0 = buying pressure, <0 = selling pressure."""
    try:
        # Calculate Money Flow Multiplier
        mfm = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        mfm = mfm.replace([np.inf, -np.inf], 0)
        
        # Calculate Money Flow Volume
        mfv = mfm * data['volume']
        
        # Calculate CMF
        cmf = talib.SMA(mfv, timeperiod=period) / talib.SMA(data['volume'], timeperiod=period)
        
        return pd.Series(cmf, index=data.index)
    except Exception as e:
        raise Exception(f"Error calculating Chaikin Money Flow: {str(e)}")

def calculate_donchian_channels(data: pd.DataFrame, period: int = 20) -> tuple:
    """Calculate Donchian Channels.
    Shows highest high / lowest low over X periods.
    Usage: Buy on breakout above channel."""
    try:
        upper = data['high'].rolling(window=period).max()
        lower = data['low'].rolling(window=period).min()
        middle = (upper + lower) / 2
        
        return (
            pd.Series(upper, index=data.index),
            pd.Series(middle, index=data.index),
            pd.Series(lower, index=data.index)
        )
    except Exception as e:
        raise Exception(f"Error calculating Donchian Channels: {str(e)}")

def calculate_dpo(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Detrended Price Oscillator (DPO).
    Filters out long-term trends to find short-term cycles.
    Usage: Buy on dip below 0, sell on peak above 0."""
    try:
        # Calculate SMA
        sma = talib.SMA(data['close'].values, timeperiod=period)
        
        # Calculate DPO
        dpo = data['close'].values - np.roll(sma, period//2 + 1)
        
        return pd.Series(dpo, index=data.index)
    except Exception as e:
        raise Exception(f"Error calculating DPO: {str(e)}")

def calculate_williams_r(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Williams %R.
    Shows how overbought or oversold an asset is.
    Usage: < -80 = oversold, > -20 = overbought."""
    try:
        williams_r = talib.WILLR(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            timeperiod=period
        )
        return pd.Series(williams_r, index=data.index)
    except Exception as e:
        raise Exception(f"Error calculating Williams %R: {str(e)}")

def calculate_schaff_trend_cycle(data: pd.DataFrame, fast_period: int = 23, slow_period: int = 50) -> pd.Series:
    """Calculate Schaff Trend Cycle (STC).
    Fast trend detector, better than MACD.
    Usage: Buy above 25, sell below 75."""
    try:
        # Calculate MACD
        macd, signal, _ = calculate_macd(data, fast_period, slow_period)
        
        # Calculate first stochastic
        stoch1 = 100 * (macd - macd.rolling(window=slow_period).min()) / \
                 (macd.rolling(window=slow_period).max() - macd.rolling(window=slow_period).min())
        
        # Calculate second stochastic
        stoch2 = 100 * (stoch1 - stoch1.rolling(window=slow_period).min()) / \
                 (stoch1.rolling(window=slow_period).max() - stoch1.rolling(window=slow_period).min())
        
        return pd.Series(stoch2, index=data.index)
    except Exception as e:
        raise Exception(f"Error calculating Schaff Trend Cycle: {str(e)}") 