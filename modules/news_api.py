"""
news_api.py — Alpha Vantage news sentiment fetcher and market scanner.
"""

import requests
from collections import defaultdict
from typing import Callable

# ─── Sector map ───────────────────────────────────────────────────────────────

TICKER_SECTOR_MAP: dict[str, str] = {
    # AI/ML
    "NVDA": "AI/ML", "MSFT": "AI/ML", "GOOGL": "AI/ML", "META": "AI/ML",
    "AMZN": "AI/ML", "PLTR": "AI/ML", "AI": "AI/ML", "BBAI": "AI/ML",
    "SOUN": "AI/ML", "ARTY": "AI/ML", "IONQ": "AI/ML", "QUBT": "AI/ML",

    # Cybersecurity
    "CRWD": "Cybersecurity", "PANW": "Cybersecurity", "FTNT": "Cybersecurity",
    "ZS": "Cybersecurity", "S": "Cybersecurity", "OKTA": "Cybersecurity",
    "CYBR": "Cybersecurity", "TENB": "Cybersecurity", "QLYS": "Cybersecurity",
    "VRNS": "Cybersecurity",

    # Semiconductors
    "AMD": "Semiconductors", "INTC": "Semiconductors", "QCOM": "Semiconductors",
    "AVGO": "Semiconductors", "TSM": "Semiconductors", "AMAT": "Semiconductors",
    "LRCX": "Semiconductors", "KLAC": "Semiconductors", "MU": "Semiconductors",
    "MRVL": "Semiconductors", "SMCI": "Semiconductors", "ARM": "Semiconductors",

    # Defense
    "LMT": "Defense", "RTX": "Defense", "NOC": "Defense", "GD": "Defense",
    "BA": "Defense", "HII": "Defense", "LDOS": "Defense", "SAIC": "Defense",
    "AXON": "Defense", "CACI": "Defense",

    # Biotech/Pharma
    "MRNA": "Biotech/Pharma", "BNTX": "Biotech/Pharma", "GILD": "Biotech/Pharma",
    "BIIB": "Biotech/Pharma", "REGN": "Biotech/Pharma", "VRTX": "Biotech/Pharma",
    "ABBV": "Biotech/Pharma", "LLY": "Biotech/Pharma", "NVO": "Biotech/Pharma",
    "PFE": "Biotech/Pharma", "BMY": "Biotech/Pharma", "AMGN": "Biotech/Pharma",

    # Cloud/SaaS
    "CRM": "Cloud/SaaS", "NOW": "Cloud/SaaS", "SNOW": "Cloud/SaaS",
    "DDOG": "Cloud/SaaS", "MDB": "Cloud/SaaS", "NET": "Cloud/SaaS",
    "HCP": "Cloud/SaaS", "CFLT": "Cloud/SaaS", "GTLB": "Cloud/SaaS",
    "HUBS": "Cloud/SaaS",

    # Mega-Cap Tech
    "AAPL": "Mega-Cap Tech", "TSLA": "Mega-Cap Tech", "NFLX": "Mega-Cap Tech",
    "UBER": "Mega-Cap Tech", "SHOP": "Mega-Cap Tech", "SPOT": "Mega-Cap Tech",
    "RBLX": "Mega-Cap Tech", "SNAP": "Mega-Cap Tech", "PINS": "Mega-Cap Tech",
    "LYFT": "Mega-Cap Tech",

    # Financials
    "JPM": "Financials", "GS": "Financials", "MS": "Financials",
    "BAC": "Financials", "WFC": "Financials", "C": "Financials",
    "BX": "Financials", "KKR": "Financials", "V": "Financials",
    "MA": "Financials",

    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "OXY": "Energy",
    "SLB": "Energy", "HAL": "Energy", "EOG": "Energy", "PXD": "Energy",
    "DVN": "Energy", "MPC": "Energy",

    # Consumer/Retail
    "AMZN": "Consumer/Retail", "WMT": "Consumer/Retail", "TGT": "Consumer/Retail",
    "COST": "Consumer/Retail", "NKE": "Consumer/Retail", "LULU": "Consumer/Retail",
    "SBUX": "Consumer/Retail", "MCD": "Consumer/Retail", "CMG": "Consumer/Retail",
    "DPZ": "Consumer/Retail",

    # ETFs
    "SPY": "ETFs", "QQQ": "ETFs", "IWM": "ETFs", "DIA": "ETFs",
    "ARKK": "ETFs", "XLK": "ETFs", "XLF": "ETFs", "SMH": "ETFs",
    "SOXX": "ETFs", "GLD": "ETFs",
}


def _clean_ticker(t: str) -> str | None:
    """Return None if ticker looks like crypto/forex/invalid."""
    if not t:
        return None
    t = t.upper().strip()
    if len(t) > 5:
        return None
    if ":" in t or "." in t:
        return None
    return t


def fetch_news_sentiment(ticker: str, alpha_key: str,
                         increment_fn: Callable) -> dict:
    """
    Fetch news sentiment for a single ticker.
    Increments alpha_vantage usage BEFORE calling.
    Returns dict with keys: sentiment_label, avg_score, article_count, articles.
    """
    increment_fn("alpha_vantage")
    url = (
        "https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT&tickers={ticker}"
        f"&limit=50&apikey={alpha_key}"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"error": str(e)}

    if "feed" not in data:
        return {"error": data.get("Note", data.get("Information", "No data returned"))}

    articles = []
    total_score = 0.0
    for item in data.get("feed", []):
        score = 0.0
        label = "Neutral"
        for ts in item.get("ticker_sentiment", []):
            if ts.get("ticker", "").upper() == ticker.upper():
                score = float(ts.get("ticker_sentiment_score", 0))
                label = ts.get("ticker_sentiment_label", "Neutral")
                break
        articles.append({
            "title":  item.get("title", ""),
            "source": item.get("source", ""),
            "date":   item.get("time_published", "")[:10],
            "url":    item.get("url", ""),
            "score":  score,
            "label":  label,
        })
        total_score += score

    n = len(articles)
    avg = total_score / n if n else 0.0
    if avg > 0.15:
        sentiment_label = "Bullish"
    elif avg < -0.15:
        sentiment_label = "Bearish"
    else:
        sentiment_label = "Neutral"

    return {
        "sentiment_label": sentiment_label,
        "avg_score":       round(avg, 4),
        "article_count":   n,
        "articles":        articles[:10],
    }


def fetch_market_sentiment_scan(alpha_key: str, increment_fn: Callable,
                                pages: int = 3) -> dict:
    """
    Fetch up to `pages` pages of market news across different sort orders.
    Returns deduplicated ticker mention data.
    Increments usage BEFORE each request.
    """
    sort_orders = ["LATEST", "RELEVANCE", "EARLIEST"][:pages]
    seen_urls: set[str] = set()
    ticker_data: dict[str, dict] = defaultdict(lambda: {
        "mentions": 0,
        "total_score": 0.0,
        "headlines": [],
    })

    for sort in sort_orders:
        increment_fn("alpha_vantage")
        url = (
            "https://www.alphavantage.co/query"
            f"?function=NEWS_SENTIMENT&sort={sort}"
            f"&limit=50&apikey={alpha_key}"
        )
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            continue

        for item in data.get("feed", []):
            item_url = item.get("url", "")
            if item_url in seen_urls:
                continue
            seen_urls.add(item_url)

            headline = item.get("title", "")
            source   = item.get("source", "")
            pub_date = item.get("time_published", "")[:10]

            for ts in item.get("ticker_sentiment", []):
                raw_t = ts.get("ticker", "")
                t = _clean_ticker(raw_t)
                if not t:
                    continue
                score = float(ts.get("ticker_sentiment_score", 0))
                td = ticker_data[t]
                td["mentions"]    += 1
                td["total_score"] += score
                if len(td["headlines"]) < 3:
                    td["headlines"].append({
                        "title":  headline,
                        "source": source,
                        "date":   pub_date,
                        "score":  score,
                    })

    # Build ranked list
    rows = []
    for ticker, td in ticker_data.items():
        n = td["mentions"]
        avg = td["total_score"] / n if n else 0.0
        if avg > 0.15:
            lbl = "Bullish"
        elif avg < -0.15:
            lbl = "Bearish"
        else:
            lbl = "Neutral"
        top = td["headlines"][0] if td["headlines"] else {}
        rows.append({
            "Ticker":       ticker,
            "Sector":       TICKER_SECTOR_MAP.get(ticker, "Other"),
            "Mentions":     n,
            "Avg Sentiment": round(avg, 4),
            "Sentiment":    lbl,
            "Top Headline": top.get("title", ""),
            "Source":       top.get("source", ""),
            "Date":         top.get("date", ""),
            "All Headlines": td["headlines"],
        })

    rows.sort(key=lambda r: (-r["Mentions"], -r["Avg Sentiment"]))
    return {"tickers": rows[:20], "article_count": len(seen_urls)}
