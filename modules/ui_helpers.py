"""
ui_helpers.py — Reusable Streamlit UI components.
"""

import streamlit as st
from modules.database import get_remaining, get_usage


# ─── API budget sidebar ───────────────────────────────────────────────────────

def render_api_budget_sidebar():
    """Shows live API budget bars in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 API Budget")

    services = [
        ("Alpha Vantage", "alpha_vantage", 25),
        ("Finnhub",       "finnhub",       25),
        ("OpenAI",        "openai",        50),
    ]

    for label, key, limit in services:
        used      = get_usage(key)
        remaining = get_remaining(key, limit)
        pct       = used / limit if limit else 0

        color = "#4caf50"
        if remaining <= 5:
            color = "#f44336"
        elif remaining <= 10:
            color = "#ff9800"

        st.sidebar.markdown(
            f"**{label}** — {remaining}/{limit} remaining"
        )
        st.sidebar.progress(min(pct, 1.0))

        if remaining <= 5:
            st.sidebar.error(f"⚠️ {label}: Only {remaining} requests left today!")
        elif remaining <= 10:
            st.sidebar.warning(f"⚠️ {label}: {remaining} requests left — use carefully")


# ─── Request gate ─────────────────────────────────────────────────────────────

def render_request_gate(service: str, cost: int, daily_limit: int,
                        label: str) -> bool:
    """
    Warns user about request cost. Returns True if user confirms.
    """
    remaining = get_remaining(service, daily_limit)
    if remaining < cost:
        st.error(
            f"❌ Not enough {label} requests remaining. "
            f"You need {cost} but only have {remaining} left today."
        )
        return False

    st.info(
        f"This action will use **{cost}** {label} request(s). "
        f"You have **{remaining}** remaining today."
    )
    return True


def render_openai_gate(daily_limit: int = 50) -> bool:
    remaining = get_remaining("openai", daily_limit)
    if remaining <= 0:
        st.error("❌ OpenAI requests exhausted for today.")
        return False
    st.info(f"This will use 1 OpenAI request. **{remaining}** remaining today.")
    return True


# ─── Signal banner ────────────────────────────────────────────────────────────

def render_signal_banner(signal: str, score: int):
    color_map = {
        "Strong swing candidate":              "#1b5e20",
        "Potential swing setup — watch closely": "#e65100",
        "Neutral — needs more confirmation":   "#37474f",
        "Weak setup — avoid or wait":          "#b71c1c",
    }
    bg = color_map.get(signal, "#37474f")
    st.markdown(
        f"""<div style="background:{bg};padding:12px 18px;border-radius:8px;
        color:white;font-weight:700;font-size:1.1rem;margin-bottom:8px;">
        🎯 Research Score: {score}/100 — {signal}
        </div>""",
        unsafe_allow_html=True,
    )


# ─── Disclaimer ───────────────────────────────────────────────────────────────

def render_disclaimer(compact: bool = False):
    msg = (
        "⚠️ **Disclaimer:** All scores, signals, and analysis are for "
        "**educational and research purposes only**. "
        "This is **not financial advice**. Past performance does not guarantee "
        "future results. Always do your own research before making any investment decision."
    )
    if compact:
        st.caption(
            "⚠️ Educational/research purposes only. Not financial advice."
        )
    else:
        st.warning(msg)


# ─── Sentiment colour helpers ─────────────────────────────────────────────────

def colour_sentiment(val: str) -> str:
    """For use with DataFrame.style.map()"""
    val = str(val).lower()
    if "bull" in val:
        return "color: #00c853; font-weight: bold"
    elif "bear" in val:
        return "color: #ef5350; font-weight: bold"
    return "color: #90a4ae"


def colour_signal(val: str) -> str:
    """For use with DataFrame.style.map()"""
    if "Strong" in val:
        return "color: #00c853; font-weight: bold"
    elif "Potential" in val:
        return "color: #ffab00; font-weight: bold"
    elif "Neutral" in val:
        return "color: #90a4ae"
    return "color: #ef5350"
