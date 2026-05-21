"""
market_data.py — yfinance data fetching with Streamlit cache.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
from modules.scoring import add_indicators


@st.cache_data(ttl=300)
def get_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Download OHLCV data and attach technical indicators.
    Returns empty DataFrame on failure.
    """
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        # Flatten MultiIndex columns (yfinance sometimes returns them)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df = add_indicators(df)
        return df
    except Exception:
        return pd.DataFrame()


def get_current_price(ticker: str) -> float | None:
    try:
        info = yf.Ticker(ticker).fast_info
        return float(info.last_price)
    except Exception:
        return None
