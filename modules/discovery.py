"""
discovery.py — Sector scanner preset lists, quick stats, display builder.
"""

import pandas as pd
from modules.market_data import get_data
from modules.scoring import score_stock, get_signal, safe_float

# ─── Sector preset lists ──────────────────────────────────────────────────────

DISCOVERY_LISTS: dict[str, list[str]] = {
    "AI & Machine Learning": [
        "NVDA", "MSFT", "GOOGL", "META", "PLTR",
        "AI",   "IONQ", "SOUN",  "BBAI", "ARTY",
    ],
    "Cybersecurity": [
        "CRWD", "PANW", "FTNT", "ZS",   "S",
        "OKTA", "CYBR", "TENB", "QLYS", "VRNS",
    ],
    "Semiconductors": [
        "NVDA", "AMD",  "INTC", "QCOM", "AVGO",
        "AMAT", "LRCX", "MU",   "MRVL", "ARM",
    ],
    "Defense": [
        "LMT",  "RTX",  "NOC",  "GD",   "BA",
        "HII",  "LDOS", "SAIC", "AXON", "CACI",
    ],
    "Biotech & Pharma": [
        "MRNA", "BNTX", "GILD", "BIIB", "REGN",
        "VRTX", "ABBV", "LLY",  "NVO",  "PFE",
    ],
    "Cloud & SaaS": [
        "CRM",  "NOW",  "SNOW", "DDOG", "MDB",
        "NET",  "CFLT", "GTLB", "HUBS", "ZM",
    ],
    "Mega-Cap Tech": [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META",
        "TSLA", "NFLX", "UBER", "SHOP",  "SPOT",
    ],
    "Financials": [
        "JPM", "GS",  "MS",  "BAC", "WFC",
        "C",   "BX",  "KKR", "V",   "MA",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "OXY", "SLB",
        "HAL", "EOG", "DVN", "MPC", "VLO",
    ],
    "Consumer & Retail": [
        "AMZN", "WMT",  "TGT",  "COST", "NKE",
        "LULU", "SBUX", "MCD",  "CMG",  "DPZ",
    ],
    "Major ETFs": [
        "SPY", "QQQ", "IWM", "DIA",  "ARKK",
        "XLK", "XLF", "SMH", "SOXX", "GLD",
    ],
}


def _ema_trend(row: dict) -> str:
    e9  = safe_float(row.get("EMA9"))
    e21 = safe_float(row.get("EMA21"))
    if e9 and e21:
        return "↑ Bullish" if e9 > e21 else "↓ Bearish"
    return "N/A"


def _macd_signal_label(row: dict) -> str:
    m  = safe_float(row.get("MACD"))
    ms = safe_float(row.get("MACD_Signal"))
    mh = safe_float(row.get("MACD_Hist"))
    if m > ms and mh > 0:
        return "↑ Strong"
    elif m > ms:
        return "↑ Weak"
    elif m < ms and mh < 0:
        return "↓ Strong"
    return "↓ Weak"


def _bb_position(row: dict) -> str:
    pb = safe_float(row.get("BB_PctB", 0.5))
    if pb > 1.0:
        return "Overextended"
    elif pb < 0.0:
        return "Breakdown"
    elif 0.2 <= pb <= 0.5:
        return "Pullback zone"
    elif 0.5 < pb <= 0.8:
        return "Momentum zone"
    return f"{pb:.2f}"


def _top_reason(reasons: list) -> str:
    for r in reasons:
        if r.startswith("+"):
            return r.split("  ", 1)[-1].strip() if "  " in r else r
    return reasons[0] if reasons else ""


def scan_category(sector: str, period: str = "3mo") -> pd.DataFrame:
    """Scan all tickers in a sector and return a results DataFrame."""
    tickers = DISCOVERY_LISTS.get(sector, [])
    rows = []
    for ticker in tickers:
        df = get_data(ticker, period=period)
        if df is None or df.empty:
            continue
        latest = df.iloc[-1].to_dict()
        close  = safe_float(latest.get("Close"))
        rsi    = safe_float(latest.get("RSI", 50))
        relvol = safe_float(latest.get("RelVol", 1.0))
        score, reasons = score_stock(latest)
        signal = get_signal(score)
        rows.append({
            "Ticker":          ticker,
            "Close":           round(close, 2),
            "RSI":             round(rsi,   1),
            "EMA Trend":       _ema_trend(latest),
            "MACD Signal":     _macd_signal_label(latest),
            "BB Position":     _bb_position(latest),
            "Rel Volume":      round(relvol, 2),
            "Research Score":  score,
            "Signal":          signal,
            "Top Reason":      _top_reason(reasons),
            "_latest":         latest,
            "_reasons":        reasons,
        })

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values("Research Score", ascending=False).reset_index(drop=True)
    return df_out


def get_quick_stats(ticker: str, period: str = "3mo") -> dict | None:
    df = get_data(ticker, period=period)
    if df is None or df.empty:
        return None
    latest = df.iloc[-1].to_dict()
    score, reasons = score_stock(latest)
    return {
        "ticker":  ticker,
        "score":   score,
        "signal":  get_signal(score),
        "latest":  latest,
        "reasons": reasons,
        "df":      df,
    }


def build_display_df(scan_df: pd.DataFrame) -> pd.DataFrame:
    """Strip internal columns for display."""
    cols = ["Ticker", "Close", "RSI", "EMA Trend", "MACD Signal",
            "BB Position", "Rel Volume", "Research Score", "Signal", "Top Reason"]
    return scan_df[[c for c in cols if c in scan_df.columns]].copy()
