"""
scoring.py — Technical indicator calculations, swing trading score engine.
"""

import pandas as pd
import numpy as np


# ─── Safe helpers ─────────────────────────────────────────────────────────────

def safe_float(val, default=0.0) -> float:
    try:
        v = float(val)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except (TypeError, ValueError):
        return default


# ─── Indicator engine ─────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to OHLCV DataFrame.
    Drops rows only where core OHLCV columns are NaN, then uses min_periods=1
    on long-window rolling calculations so short periods don't crash.
    """
    if df is None or df.empty:
        return df

    core_cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df.dropna(subset=core_cols).copy()

    if df.empty:
        return df

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ── EMAs
    df["EMA9"]  = close.ewm(span=9,  adjust=False).mean()
    df["EMA21"] = close.ewm(span=21, adjust=False).mean()
    df["MA50"]  = close.rolling(50, min_periods=1).mean()

    # ── RSI 14
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ── MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # ── Bollinger Bands (20, 2)
    bb_mid       = close.rolling(20, min_periods=1).mean()
    bb_std       = close.rolling(20, min_periods=1).std()
    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Lower"] = bb_mid - 2 * bb_std
    bb_range       = (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)
    df["BB_PctB"]  = (close - df["BB_Lower"]) / bb_range

    # ── ATR 14
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(com=13, adjust=False).mean()

    # ── Relative Volume vs 20-day avg
    avg_vol = vol.rolling(20, min_periods=1).mean().replace(0, np.nan)
    df["RelVol"] = vol / avg_vol

    # ── 52-week high/low — use min_periods=1 so short periods don't fail
    df["High52w"] = high.rolling(252, min_periods=1).max()
    df["Low52w"]  = low.rolling(252,  min_periods=1).min()

    return df


# ─── Scoring engine ───────────────────────────────────────────────────────────

def score_stock(row: dict, news_sentiment: str = "Neutral",
                insider_signal: str = "Neutral") -> tuple[int, list[str]]:
    """
    Returns (score 0-100, list_of_reason_strings).
    row should contain the latest-bar indicator values.
    """
    score   = 50
    reasons = []

    close   = safe_float(row.get("Close"))
    ema9    = safe_float(row.get("EMA9"))
    ema21   = safe_float(row.get("EMA21"))
    ma50    = safe_float(row.get("MA50"))
    rsi     = safe_float(row.get("RSI", 50))
    macd    = safe_float(row.get("MACD"))
    macd_sig= safe_float(row.get("MACD_Signal"))
    macd_h  = safe_float(row.get("MACD_Hist"))
    pctb    = safe_float(row.get("BB_PctB", 0.5))
    relvol  = safe_float(row.get("RelVol", 1.0))
    high52  = safe_float(row.get("High52w"))
    low52   = safe_float(row.get("Low52w"))

    # ── EMA trend
    if ema9 and ema21:
        if ema9 > ema21:
            score += 15
            reasons.append("+15  EMA9 above EMA21 — bullish momentum alignment")
        else:
            score -= 10
            reasons.append("-10  EMA9 below EMA21 — bearish momentum alignment")

    # ── MA50
    if ma50 and close:
        if close > ma50:
            score += 10
            reasons.append("+10  Price above MA50 — trading with the trend")
        else:
            score -= 10
            reasons.append("-10  Price below MA50 — trading against the trend")

    # ── RSI
    if rsi > 75:
        score -= 15
        reasons.append(f"-15  RSI {rsi:.1f} — overbought, avoid chasing")
    elif rsi < 30:
        score -= 5
        reasons.append(f"-5   RSI {rsi:.1f} — deeply oversold, wait for stabilisation")
    elif 30 <= rsi < 40:
        score += 8
        reasons.append(f"+8   RSI {rsi:.1f} — oversold bounce candidate")
    elif 40 <= rsi <= 60:
        score += 15
        reasons.append(f"+15  RSI {rsi:.1f} — ideal swing entry zone")
    elif 60 < rsi <= 70:
        score += 5
        reasons.append(f"+5   RSI {rsi:.1f} — mildly elevated but workable")

    # ── MACD
    if macd > macd_sig and macd_h > 0:
        score += 15
        reasons.append("+15  MACD above signal & histogram positive — momentum accelerating")
    elif macd > macd_sig:
        score += 8
        reasons.append("+8   MACD above signal — mild bullish momentum")
    elif macd < macd_sig and macd_h < 0:
        score -= 10
        reasons.append("-10  MACD below signal & histogram negative — momentum declining")
    elif macd < macd_sig:
        score -= 5
        reasons.append("-5   MACD below signal — mild bearish pressure")

    # ── Bollinger %B
    if pctb > 1.0:
        score -= 10
        reasons.append(f"-10  BB %B {pctb:.2f} — overextended above upper band")
    elif pctb < 0.0:
        score -= 8
        reasons.append(f"-8   BB %B {pctb:.2f} — breakdown below lower band")
    elif 0.2 <= pctb <= 0.5:
        score += 10
        reasons.append(f"+10  BB %B {pctb:.2f} — pullback entry zone")
    elif 0.5 < pctb <= 0.8:
        score += 5
        reasons.append(f"+5   BB %B {pctb:.2f} — momentum zone")

    # ── Relative Volume
    if relvol >= 2.0:
        score += 10
        reasons.append(f"+10  Relative volume {relvol:.1f}x — strong conviction")
    elif relvol >= 1.5:
        score += 7
        reasons.append(f"+7   Relative volume {relvol:.1f}x — above-average activity")
    elif relvol >= 1.0:
        score += 3
        reasons.append(f"+3   Relative volume {relvol:.1f}x — average activity")
    else:
        score -= 5
        reasons.append(f"-5   Relative volume {relvol:.1f}x — below-average volume")

    # ── 52-week high proximity
    if high52 and close:
        pct_from_high = (high52 - close) / high52 * 100 if high52 > 0 else 100
        if pct_from_high <= 5:
            score += 5
            reasons.append(f"+5   Within {pct_from_high:.1f}% of 52-week high — breakout zone")
        elif pct_from_high <= 15:
            score += 3
            reasons.append(f"+3   Within {pct_from_high:.1f}% of 52-week high")
        elif pct_from_high >= 40:
            score -= 5
            reasons.append(f"-5   {pct_from_high:.1f}% below 52-week high — deep drawdown")

    # ── News sentiment
    ns = str(news_sentiment).lower()
    if "bull" in ns:
        score += 8
        reasons.append("+8   News sentiment: Bullish")
    elif "bear" in ns:
        score -= 8
        reasons.append("-8   News sentiment: Bearish")

    # ── Insider signal
    ins = str(insider_signal).lower()
    if "bull" in ins or "buy" in ins:
        score += 8
        reasons.append("+8   Insider activity: Buying detected")
    elif "bear" in ins or "sell" in ins:
        score -= 5
        reasons.append("-5   Insider activity: Selling detected")

    score = max(0, min(100, score))
    return score, reasons


def get_signal(score: int) -> str:
    if score >= 72:
        return "Strong swing candidate"
    elif score >= 58:
        return "Potential swing setup — watch closely"
    elif score >= 45:
        return "Neutral — needs more confirmation"
    else:
        return "Weak setup — avoid or wait"


def get_signal_color(signal: str) -> str:
    mapping = {
        "Strong swing candidate":              "#00c853",
        "Potential swing setup — watch closely": "#ffab00",
        "Neutral — needs more confirmation":   "#90a4ae",
        "Weak setup — avoid or wait":          "#ef5350",
    }
    return mapping.get(signal, "#90a4ae")
