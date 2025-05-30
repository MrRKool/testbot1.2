import numpy as np
import pandas as pd
import pandas_ta as ta

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    try:
        return data.ta.rsi(length=period)
    except Exception as e:
        raise Exception(f"Error calculating RSI: {str(e)}")

def calculate_macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
    """Calculate MACD indicator."""
    try:
        macd = data.ta.macd(fast=fast_period, slow=slow_period, signal=signal_period)
        return (
            macd[f'MACD_{fast_period}_{slow_period}_{signal_period}'],
            macd[f'MACDs_{fast_period}_{slow_period}_{signal_period}'],
            macd[f'MACDh_{fast_period}_{slow_period}_{signal_period}']
        )
    except Exception as e:
        raise Exception(f"Error calculating MACD: {str(e)}")

def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> tuple:
    """Calculate Bollinger Bands."""
    try:
        bbands = data.ta.bbands(length=period, std=std_dev)
        return (
            bbands[f'BBU_{period}_{std_dev}'],
            bbands[f'BBM_{period}_{std_dev}'],
            bbands[f'BBL_{period}_{std_dev}']
        )
    except Exception as e:
        raise Exception(f"Error calculating Bollinger Bands: {str(e)}")

def calculate_ema(data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate EMA indicator."""
    try:
        return data.ta.ema(length=period)
    except Exception as e:
        raise Exception(f"Error calculating EMA: {str(e)}")

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR indicator."""
    try:
        return data.ta.atr(length=period)
    except Exception as e:
        raise Exception(f"Error calculating ATR: {str(e)}")

def calculate_adx(data: pd.DataFrame, period: int = 14) -> tuple:
    """Calculate ADX indicator."""
    try:
        adx = data.ta.adx(length=period)
        return (
            adx[f'ADX_{period}'],
            adx[f'DMP_{period}'],
            adx[f'DMN_{period}']
        )
    except Exception as e:
        raise Exception(f"Error calculating ADX: {str(e)}")

def calculate_stochastic_rsi(data: pd.DataFrame, rsi_period: int = 14, stoch_period: int = 14, k_period: int = 3, d_period: int = 3) -> tuple:
    """Calculate Stochastic RSI."""
    try:
        stoch_rsi = data.ta.stochrsi(length=rsi_period, rsi_length=stoch_period, k=k_period, d=d_period)
        return (
            stoch_rsi[f'STOCHk_{rsi_period}_{stoch_period}_{k_period}'],
            stoch_rsi[f'STOCHd_{rsi_period}_{stoch_period}_{d_period}']
        )
    except Exception as e:
        raise Exception(f"Error calculating Stochastic RSI: {str(e)}")

def calculate_obv(data: pd.DataFrame, smoothing: int = 5) -> pd.Series:
    """Calculate On Balance Volume with smoothing."""
    try:
        obv = data.ta.obv()
        if smoothing > 1:
            obv = obv.rolling(window=smoothing).mean()
        return obv
    except Exception as e:
        raise Exception(f"Error calculating OBV: {str(e)}")

def calculate_klinger_oscillator(data: pd.DataFrame, short_period: int = 34, long_period: int = 55) -> pd.Series:
    """Calculate Klinger Oscillator."""
    try:
        return data.ta.kvo(short=short_period, long=long_period)
    except Exception as e:
        raise Exception(f"Error calculating Klinger Oscillator: {str(e)}")

def calculate_vortex_indicator(data: pd.DataFrame, period: int = 14) -> tuple:
    """Calculate Vortex Indicator (VI).
    Detects the start of a new trend.
    Usage: Buy when VI+ > VI-."""
    try:
        vi = data.ta.vortex(length=period)
        return vi[f'VMP_{period}'], vi[f'VMN_{period}']
    except Exception as e:
        raise Exception(f"Error calculating Vortex Indicator: {str(e)}")

def calculate_chaikin_money_flow(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Chaikin Money Flow (CMF).
    Measures buying pressure based on volume and closing price.
    Usage: >0 = buying pressure, <0 = selling pressure."""
    try:
        return data.ta.cmf(length=period)
    except Exception as e:
        raise Exception(f"Error calculating Chaikin Money Flow: {str(e)}")

def calculate_donchian_channels(data: pd.DataFrame, period: int = 20) -> tuple:
    """Calculate Donchian Channels.
    Identifies highest high and lowest low over a period.
    Usage: Breakout above upper channel = bullish, below lower channel = bearish."""
    try:
        dc = data.ta.donchian(length=period)
        return dc[f'DCB_{period}'], dc[f'DCM_{period}'], dc[f'DCT_{period}']
    except Exception as e:
        raise Exception(f"Error calculating Donchian Channels: {str(e)}")

def calculate_dpo(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Detrended Price Oscillator (DPO).
    Filters out long-term trends to find short-term cycles.
    Usage: Buy on dip below 0, sell on peak above 0."""
    try:
        return data.ta.dpo(length=period)
    except Exception as e:
        raise Exception(f"Error calculating DPO: {str(e)}")

def calculate_williams_r(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Williams %R.
    Shows how overbought or oversold an asset is.
    Usage: < -80 = oversold, > -20 = overbought."""
    try:
        return data.ta.willr(length=period)
    except Exception as e:
        raise Exception(f"Error calculating Williams %R: {str(e)}")

def calculate_schaff_trend_cycle(data: pd.DataFrame, fast_period: int = 23, slow_period: int = 50) -> pd.Series:
    """Calculate Schaff Trend Cycle (STC).
    Combines elements of MACD and stochastics.
    Usage: > 75 = bullish, < 25 = bearish."""
    try:
        return data.ta.stc(fast=fast_period, slow=slow_period)
    except Exception as e:
        raise Exception(f"Error calculating Schaff Trend Cycle: {str(e)}") 