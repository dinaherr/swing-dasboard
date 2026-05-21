"""
ai_analysis.py — ChatGPT prompt builder and optional OpenAI API integration.
"""

import requests
from modules.scoring import safe_float

SYSTEM_PROMPT = """You are a swing trading research assistant. Your role is to help
users understand technical and fundamental signals for potential swing trade setups.

IMPORTANT RULES:
- Never say "buy" or "sell" — use language like "the setup may suggest", "signals indicate",
  "historically this pattern has", "consider monitoring"
- Always include risk disclaimers
- Frame everything as research and education, not financial advice
- Be balanced — include both bullish and bearish perspectives
- Use hedged language throughout: "may indicate", "signals suggest", "historically"
"""


def build_chatgpt_prompt(ticker: str, row: dict, score: int, signal: str,
                         reasons: list, news: dict = None,
                         insider: dict = None) -> str:
    close   = safe_float(row.get("Close"))
    rsi     = safe_float(row.get("RSI", 50))
    ema9    = safe_float(row.get("EMA9"))
    ema21   = safe_float(row.get("EMA21"))
    ma50    = safe_float(row.get("MA50"))
    macd    = safe_float(row.get("MACD"))
    macd_s  = safe_float(row.get("MACD_Signal"))
    pctb    = safe_float(row.get("BB_PctB", 0.5))
    atr     = safe_float(row.get("ATR"))
    relvol  = safe_float(row.get("RelVol", 1.0))
    high52  = safe_float(row.get("High52w"))
    low52   = safe_float(row.get("Low52w"))

    reasons_text = "\n".join(f"  {r}" for r in reasons)

    news_block = ""
    if news and "error" not in news:
        news_block = f"""
NEWS SENTIMENT
  Label:         {news.get('sentiment_label', 'N/A')}
  Avg Score:     {news.get('avg_score', 0):.4f}
  Article Count: {news.get('article_count', 0)}
  Top Headlines:
"""
        for a in news.get("articles", [])[:3]:
            news_block += f"    - {a.get('title', '')} ({a.get('source', '')} {a.get('date', '')})\n"

    insider_block = ""
    if insider and "error" not in insider:
        insider_block = f"""
INSIDER ACTIVITY (last 90 days)
  Signal:    {insider.get('signal', 'N/A')}
  Buys:      {insider.get('buys', 0)}
  Sells:     {insider.get('sells', 0)}
"""

    prompt = f"""=== SWING TRADING RESEARCH REQUEST ===
Ticker: {ticker}

TECHNICAL SNAPSHOT
  Close Price:   ${close:.2f}
  EMA9:          ${ema9:.2f}
  EMA21:         ${ema21:.2f}
  MA50:          ${ma50:.2f}
  RSI (14):      {rsi:.1f}
  MACD:          {macd:.4f}
  MACD Signal:   {macd_s:.4f}
  BB %B:         {pctb:.2f}
  ATR (14):      ${atr:.2f}
  Relative Vol:  {relvol:.2f}x
  52-Week High:  ${high52:.2f}
  52-Week Low:   ${low52:.2f}

RESEARCH SCORE: {score}/100 — {signal}
SCORING BREAKDOWN:
{reasons_text}
{news_block}{insider_block}

=== PLEASE PROVIDE A SWING TRADING RESEARCH SUMMARY ===

Structure your response as follows:

1. BULLISH FACTORS
   What technical or sentiment signals may suggest bullish momentum?

2. BEARISH RISKS
   What signals or conditions raise concerns or suggest caution?

3. SWING ENTRY / EXIT IDEAS (educational estimates only)
   Based on the technical picture, what levels might a swing trader monitor?
   (Entry zone, potential stop-loss area, potential target area)
   Note: These are educational pattern observations, not financial advice.

4. WHAT TO WATCH
   Key events, levels, or conditions to monitor over the next 1-4 weeks.

5. RISK FACTORS
   Market, sector, or stock-specific risks to be aware of.

IMPORTANT: Use hedged language throughout. Never say "buy" or "sell". 
Frame all observations as research for educational purposes only.
This is not financial advice.
"""
    return prompt


def generate_ai_analysis(prompt: str, openai_key: str,
                         increment_fn) -> str:
    """
    Send prompt to OpenAI gpt-4o-mini and return response text.
    Increments openai usage BEFORE the request.
    """
    increment_fn("openai")
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type":  "application/json",
    }
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": 1000,
        "temperature": 0.3,
    }
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers, json=body, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"OpenAI API error: {e}"
