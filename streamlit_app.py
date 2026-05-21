"""
app.py — AI Swing Trading Research Dashboard
Orchestration only. All business logic lives in modules/.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

from modules.database  import init_db, increment_usage, get_remaining, get_usage
from modules.scoring   import score_stock, get_signal, get_signal_color, safe_float
from modules.market_data import get_data, get_current_price
from modules.news_api  import fetch_news_sentiment, fetch_market_sentiment_scan, TICKER_SECTOR_MAP
from modules.ai_analysis import build_chatgpt_prompt, generate_ai_analysis
from modules.ui_helpers  import (
    render_api_budget_sidebar, render_request_gate, render_openai_gate,
    render_signal_banner, render_disclaimer, colour_sentiment, colour_signal,
)
from modules.discovery import (
    DISCOVERY_LISTS, scan_category, get_quick_stats, build_display_df,
)

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Swing Trading Research Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  h1, h2, h3 { font-family: 'Space Mono', monospace; }

  .metric-card {
    background: #1e2130;
    border: 1px solid #2d3149;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
  }
  .metric-label { color: #7b8cde; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
  .metric-value { color: #e8eaf6; font-size: 1.8rem; font-weight: 700; font-family: 'Space Mono', monospace; }

  .setup-box {
    background: #0d1117;
    border-left: 4px solid #7b8cde;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 12px 0;
  }
  .stTabs [data-baseweb="tab"] { font-family: 'Space Mono', monospace; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ─── Init ─────────────────────────────────────────────────────────────────────

init_db()

# ─── Secrets (graceful fallback) ──────────────────────────────────────────────

def _secret(key: str, fallback: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return fallback

ALPHA_KEY   = _secret("ALPHA_VANTAGE_API_KEY")
FINNHUB_KEY = _secret("FINNHUB_API_KEY")
OPENAI_KEY  = _secret("OPENAI_API_KEY")

# ─── Sidebar watchlist ────────────────────────────────────────────────────────

st.sidebar.title("📈 Swing Dashboard")
st.sidebar.markdown("---")

watchlist_raw = st.sidebar.text_input(
    "Watchlist tickers",
    value="NVDA, CRWD, PANW, AMD, SPY",
    help="Comma-separated tickers for the Deep Dive tab",
)
WATCHLIST = [t.strip().upper() for t in watchlist_raw.split(",") if t.strip()]

period = st.sidebar.selectbox(
    "Time period",
    options=["3mo", "6mo", "1y", "2y"],
    index=2,
)

render_api_budget_sidebar()

st.sidebar.markdown("---")
render_disclaimer(compact=True)

# ─── Header ──────────────────────────────────────────────────────────────────

st.title("📈 AI Swing Trading Research Dashboard")
st.caption(
    "Educational swing trading research tool. "
    "All scores and signals are for research only. Not financial advice."
)

# ─── Tabs ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "🌐 Market Sentiment Scanner",
    "🔬 Swing Trading Deep Dive",
    "🗂️ Sector Scanner",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Market Sentiment Scanner
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("🌐 Market Sentiment Scanner")
    st.caption(
        "Scans recent market news via Alpha Vantage to surface the most-mentioned "
        "tickers and their sentiment. Costs up to 3 Alpha Vantage requests."
    )
    render_disclaimer(compact=True)

    # Cached wrapper (30 min)
    @st.cache_data(ttl=1800)
    def cached_market_sentiment_scan(key: str, pages: int):
        return fetch_market_sentiment_scan(key, increment_usage, pages=pages)

    col_btn, col_info = st.columns([2, 3])
    with col_btn:
        run_scan = st.button(
            "🔍 Run Market Sentiment Scan",
            type="primary",
            disabled=(not ALPHA_KEY),
            help="Costs up to 3 Alpha Vantage requests",
        )
    with col_info:
        av_remaining = get_remaining("alpha_vantage", 25)
        st.metric("Alpha Vantage Requests Remaining", f"{av_remaining} / 25")

    if not ALPHA_KEY:
        st.warning("⚠️ ALPHA_VANTAGE_API_KEY not set in Streamlit secrets.")

    if run_scan and ALPHA_KEY:
        if av_remaining < 3:
            st.error("❌ Need at least 3 Alpha Vantage requests. Check your budget.")
        else:
            with st.spinner("Fetching market news across 3 sort orders..."):
                result = cached_market_sentiment_scan(ALPHA_KEY, 3)
            st.session_state["sentiment_scan"] = result
            st.success(
                f"✅ Scanned {result.get('article_count', 0)} unique articles. "
                f"Found {len(result.get('tickers', []))} tickers."
            )

    if "sentiment_scan" in st.session_state:
        result  = st.session_state["sentiment_scan"]
        tickers = result.get("tickers", [])

        if not tickers:
            st.info("No ticker data found — try running the scan again.")
        else:
            df_tickers = pd.DataFrame(tickers)

            # ── Sector summary table
            st.subheader("📊 Sector Summary")
            sector_groups = df_tickers.groupby("Sector").agg(
                Total_Mentions  = ("Mentions",      "sum"),
                Tickers_Covered = ("Ticker",        "count"),
                Avg_Sentiment   = ("Avg Sentiment", "mean"),
            ).reset_index().rename(columns={
                "Total_Mentions":  "Total Mentions",
                "Tickers_Covered": "Tickers Covered",
                "Avg_Sentiment":   "Avg Sentiment",
            })
            sector_groups["Sentiment"] = sector_groups["Avg Sentiment"].apply(
                lambda x: "Bullish" if x > 0.15 else ("Bearish" if x < -0.15 else "Neutral")
            )
            sector_groups = sector_groups.sort_values("Total Mentions", ascending=False)
            sector_groups["Avg Sentiment"] = sector_groups["Avg Sentiment"].round(4)

            styled_sectors = sector_groups.style.map(
                colour_sentiment, subset=["Sentiment"]
            ).format({"Avg Sentiment": "{:.4f}"})
            st.dataframe(styled_sectors, use_container_width=True, hide_index=True)

            # Bar chart
            fig_bar = go.Figure(go.Bar(
                x=sector_groups["Sector"],
                y=sector_groups["Total Mentions"],
                marker_color=[
                    "#00c853" if s == "Bullish" else
                    "#ef5350" if s == "Bearish" else "#90a4ae"
                    for s in sector_groups["Sentiment"]
                ],
            ))
            fig_bar.update_layout(
                title="Total Mentions by Sector",
                xaxis_title="Sector", yaxis_title="Mentions",
                height=350,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="#e8eaf6"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── Sector filter
            st.subheader("📋 Top 20 Tickers by Mention Count")
            sectors_available = ["All"] + sorted(df_tickers["Sector"].unique().tolist())
            sector_filter = st.selectbox("Filter by sector", sectors_available)

            df_display = df_tickers.copy()
            if sector_filter != "All":
                df_display = df_display[df_display["Sector"] == sector_filter]

            display_cols = ["Ticker","Sector","Mentions","Avg Sentiment","Sentiment",
                            "Top Headline","Source","Date"]
            styled_tickers = df_display[display_cols].style.map(
                colour_sentiment, subset=["Sentiment"]
            ).format({"Avg Sentiment": "{:.4f}"})
            st.dataframe(styled_tickers, use_container_width=True, hide_index=True)

            # ── Ticker selector
            st.subheader("🔎 Ticker Detail")
            selected_t = st.selectbox(
                "Select a ticker for news context",
                [r["Ticker"] for r in tickers],
                key="t1_selected",
            )

            if selected_t:
                sel_row = next((r for r in tickers if r["Ticker"] == selected_t), None)
                if sel_row:
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Mentions",      sel_row["Mentions"])
                    col_b.metric("Avg Sentiment", f"{sel_row['Avg Sentiment']:.4f}")
                    col_c.metric("Sentiment",     sel_row["Sentiment"])

                    st.markdown(f"**Sector:** {sel_row['Sector']}")
                    st.markdown("**Headlines:**")
                    for h in sel_row.get("All Headlines", []):
                        st.markdown(
                            f"- [{h['title']}]({''}) · *{h['source']}* · {h['date']}"
                        )

                    if st.button("⚡ Quick Technical Scan", key="t1_quick"):
                        with st.spinner(f"Fetching {selected_t} data..."):
                            df_q = get_data(selected_t, period="3mo")
                        if df_q is not None and not df_q.empty:
                            latest_q = df_q.iloc[-1].to_dict()
                            sc, _ = score_stock(latest_q)
                            sig    = get_signal(sc)
                            render_signal_banner(sig, sc)

                            c1, c2, c3 = st.columns(3)
                            c1.metric("Close",  f"${safe_float(latest_q.get('Close')):.2f}")
                            c2.metric("RSI",    f"{safe_float(latest_q.get('RSI', 50)):.1f}")
                            c3.metric("Score",  f"{sc}/100")

                            # Mini candlestick
                            fig_mini = go.Figure()
                            fig_mini.add_trace(go.Candlestick(
                                x=df_q.index,
                                open=df_q["Open"], high=df_q["High"],
                                low=df_q["Low"],   close=df_q["Close"],
                                name=selected_t,
                            ))
                            fig_mini.add_trace(go.Scatter(
                                x=df_q.index, y=df_q["EMA9"],
                                line=dict(color="#00bcd4", width=1.5), name="EMA9",
                            ))
                            fig_mini.add_trace(go.Scatter(
                                x=df_q.index, y=df_q["EMA21"],
                                line=dict(color="#ff9800", width=1.5), name="EMA21",
                            ))
                            fig_mini.update_layout(
                                title=f"{selected_t} — 3-Month Chart",
                                xaxis_rangeslider_visible=False,
                                height=400,
                                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                font=dict(color="#e8eaf6"),
                            )
                            st.plotly_chart(fig_mini, use_container_width=True)
                        else:
                            st.error("Could not fetch data for this ticker.")

                    # Watchlist helper
                    if st.button("➕ Add to Watchlist", key="t1_watchlist"):
                        new_list = list(dict.fromkeys(WATCHLIST + [selected_t]))
                        st.code(", ".join(new_list), language=None)
                        st.caption("Copy the above and paste into the sidebar Watchlist field.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Swing Trading Deep Dive
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("🔬 Swing Trading Deep Dive")
    render_disclaimer(compact=True)

    # ── Signal alignment summary
    st.subheader("📊 Signal Alignment Summary — Watchlist")

    @st.cache_data(ttl=300)
    def build_summary(tickers: tuple, per: str):
        rows = []
        for t in tickers:
            df = get_data(t, period=per)
            if df is None or df.empty:
                continue
            latest = df.iloc[-1].to_dict()
            sc, _  = score_stock(latest)
            sig    = get_signal(sc)

            ema_t = ("↑ Bull" if safe_float(latest.get("EMA9")) > safe_float(latest.get("EMA21"))
                     else "↓ Bear")
            macd  = latest.get("MACD", 0)
            macs  = latest.get("MACD_Signal", 0)
            macd_lbl = "↑" if safe_float(macd) > safe_float(macs) else "↓"
            rsi   = safe_float(latest.get("RSI", 50))

            rows.append({
                "Ticker":         t,
                "Research Score": sc,
                "Signal":         sig,
                "EMA Trend":      ema_t,
                "MACD":           macd_lbl,
                "RSI":            round(rsi, 1),
                "News Sentiment": "—",
            })
        return sorted(rows, key=lambda r: -r["Research Score"])

    summary_rows = build_summary(tuple(WATCHLIST), period)
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        styled_sum = (
            df_sum.style
            .map(colour_signal, subset=["Signal"])
            .format({"Research Score": "{}",  "RSI": "{:.1f}"})
        )
        st.dataframe(styled_sum, use_container_width=True, hide_index=True)
    else:
        st.info("No data loaded. Check your watchlist tickers.")

    st.markdown("---")

    # ── Deep dive selector
    st.subheader("🎯 Deep Dive — Select a Ticker")
    dd_ticker = st.selectbox("Ticker", WATCHLIST, key="dd_ticker")

    if dd_ticker:
        with st.spinner(f"Loading {dd_ticker}..."):
            df_dd = get_data(dd_ticker, period=period)

        if df_dd is None or df_dd.empty:
            st.error(f"Could not load data for {dd_ticker}.")
        else:
            latest_dd = df_dd.iloc[-1].to_dict()

            # Optional: news & insider state
            news_dd    = st.session_state.get(f"news_{dd_ticker}", None)
            insider_dd = st.session_state.get(f"insider_{dd_ticker}", None)

            news_sent   = news_dd["sentiment_label"]    if news_dd    and "error" not in news_dd    else "Neutral"
            insider_sig = insider_dd.get("signal","Neutral") if insider_dd and "error" not in insider_dd else "Neutral"

            score_dd, reasons_dd = score_stock(latest_dd, news_sent, insider_sig)
            signal_dd = get_signal(score_dd)

            # ── 4 metric cards
            render_signal_banner(signal_dd, score_dd)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Research Score", f"{score_dd}/100")
            m2.metric("RSI",  f"{safe_float(latest_dd.get('RSI',50)):.1f}")
            m3.metric("Close", f"${safe_float(latest_dd.get('Close')):.2f}")
            m4.metric("ATR",  f"${safe_float(latest_dd.get('ATR')):.2f}")

            # ── Candlestick chart
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                row_heights=[0.6, 0.2, 0.2],
                vertical_spacing=0.03,
                subplot_titles=(f"{dd_ticker} Price + EMAs", "MACD", "RSI"),
            )

            fig.add_trace(go.Candlestick(
                x=df_dd.index, open=df_dd["Open"], high=df_dd["High"],
                low=df_dd["Low"], close=df_dd["Close"], name=dd_ticker,
            ), row=1, col=1)

            for col_name, clr, width, nm in [
                ("EMA9", "#00bcd4", 1.5, "EMA9"),
                ("EMA21","#ff9800", 1.5, "EMA21"),
                ("MA50", "#ab47bc", 1.5, "MA50"),
            ]:
                if col_name in df_dd.columns:
                    fig.add_trace(go.Scatter(
                        x=df_dd.index, y=df_dd[col_name],
                        line=dict(color=clr, width=width), name=nm,
                    ), row=1, col=1)

            # BB bands
            if "BB_Upper" in df_dd.columns:
                fig.add_trace(go.Scatter(
                    x=df_dd.index, y=df_dd["BB_Upper"],
                    line=dict(color="#546e7a", width=1, dash="dot"), name="BB Upper",
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df_dd.index, y=df_dd["BB_Lower"],
                    line=dict(color="#546e7a", width=1, dash="dot"), name="BB Lower",
                    fill="tonexty", fillcolor="rgba(84,110,122,0.05)",
                ), row=1, col=1)

            # MACD
            if "MACD" in df_dd.columns:
                colors_macd = ["#00c853" if v >= 0 else "#ef5350"
                               for v in df_dd["MACD_Hist"].fillna(0)]
                fig.add_trace(go.Bar(
                    x=df_dd.index, y=df_dd["MACD_Hist"],
                    marker_color=colors_macd, name="Histogram",
                ), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=df_dd.index, y=df_dd["MACD"],
                    line=dict(color="#00bcd4", width=1.2), name="MACD",
                ), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=df_dd.index, y=df_dd["MACD_Signal"],
                    line=dict(color="#ff9800", width=1.2), name="Signal",
                ), row=2, col=1)

            # RSI
            if "RSI" in df_dd.columns:
                fig.add_trace(go.Scatter(
                    x=df_dd.index, y=df_dd["RSI"],
                    line=dict(color="#7b8cde", width=1.5), name="RSI",
                ), row=3, col=1)
                for lvl, clr in [(70, "#ef5350"), (30, "#00c853"), (50, "#546e7a")]:
                    fig.add_hline(y=lvl, line_dash="dot", line_color=clr,
                                  row=3, col=1, line_width=1)

            fig.update_layout(
                height=700,
                xaxis_rangeslider_visible=False,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="#e8eaf6"),
                legend=dict(bgcolor="#1e2130", bordercolor="#2d3149"),
                margin=dict(l=50, r=20, t=60, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Score breakdown
            with st.expander("📋 Score Breakdown", expanded=False):
                for r in reasons_dd:
                    clr = "#00c853" if r.startswith("+") else "#ef5350"
                    st.markdown(f"<span style='color:{clr}'>{r}</span>",
                                unsafe_allow_html=True)

            # ── ATR context
            atr_val  = safe_float(latest_dd.get("ATR"))
            close_dd = safe_float(latest_dd.get("Close"))
            st.markdown(
                f"**ATR (14):** ${atr_val:.2f} — "
                f"represents ~{atr_val/close_dd*100:.1f}% of current price. "
                "Use for approximate stop-loss and target sizing."
            )

            # ── Suggested swing setup
            entry    = close_dd
            stop     = round(entry - atr_val, 2)
            target   = round(entry + 2 * atr_val, 2)
            rr_ratio = round((target - entry) / (entry - stop), 1) if (entry - stop) > 0 else "N/A"

            st.markdown(
                f"""<div class="setup-box">
                <b>📐 Suggested Swing Setup (Educational Estimates Only)</b><br><br>
                🟢 <b>Entry Zone:</b> Near ${entry:.2f} (current price) or pullback to EMA9 (${safe_float(latest_dd.get("EMA9")):.2f})<br>
                🔴 <b>Stop-Loss Idea:</b> ~${stop:.2f} (1× ATR below entry — adjust to your risk tolerance)<br>
                🎯 <b>Target Idea:</b> ~${target:.2f} (2× ATR above entry = ~{rr_ratio}:1 reward/risk)<br><br>
                <small>⚠️ These are educational estimates based on ATR pattern analysis,
                not financial advice. Use your own risk management rules.</small>
                </div>""",
                unsafe_allow_html=True,
            )

            # ── News Sentiment (manual, costs 1 AV request)
            with st.expander("📰 News Sentiment (costs 1 Alpha Vantage request)", expanded=False):
                av_rem = get_remaining("alpha_vantage", 25)
                st.caption(f"Alpha Vantage requests remaining: **{av_rem} / 25**")
                if not ALPHA_KEY:
                    st.warning("ALPHA_VANTAGE_API_KEY not set.")
                elif st.button("Fetch News Sentiment", key=f"news_btn_{dd_ticker}"):
                    if av_rem < 1:
                        st.error("No Alpha Vantage requests remaining.")
                    else:
                        with st.spinner("Fetching news..."):
                            nd = fetch_news_sentiment(dd_ticker, ALPHA_KEY, increment_usage)
                        st.session_state[f"news_{dd_ticker}"] = nd
                        st.rerun()

                if news_dd:
                    if "error" in news_dd:
                        st.error(f"Error: {news_dd['error']}")
                    else:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Sentiment",    news_dd["sentiment_label"])
                        c2.metric("Avg Score",    f"{news_dd['avg_score']:.4f}")
                        c3.metric("Articles",     news_dd["article_count"])
                        st.markdown("**Top Headlines:**")
                        for a in news_dd.get("articles", [])[:3]:
                            s_lbl = a["label"]
                            clr   = "#00c853" if "Bull" in s_lbl else ("#ef5350" if "Bear" in s_lbl else "#90a4ae")
                            st.markdown(
                                f"- <span style='color:{clr}'>[{s_lbl}]</span> "
                                f"**{a['title']}** · *{a['source']}* · {a['date']}",
                                unsafe_allow_html=True,
                            )

            # ── Insider Activity (Finnhub)
            with st.expander("🏦 Insider Activity (costs 1 Finnhub request)", expanded=False):
                fh_rem = get_remaining("finnhub", 25)
                st.caption(f"Finnhub requests remaining: **{fh_rem} / 25**")
                if not FINNHUB_KEY:
                    st.warning("FINNHUB_API_KEY not set.")
                elif st.button("Fetch Insider Activity", key=f"insider_btn_{dd_ticker}"):
                    if fh_rem < 1:
                        st.error("No Finnhub requests remaining.")
                    else:
                        with st.spinner("Fetching insider data..."):
                            increment_usage("finnhub")
                            try:
                                url = (
                                    f"https://finnhub.io/api/v1/stock/insider-transactions"
                                    f"?symbol={dd_ticker}&token={FINNHUB_KEY}"
                                )
                                resp = requests.get(url, timeout=10)
                                raw  = resp.json().get("data", [])
                                buys  = [r for r in raw if r.get("transactionType","") == "P - Purchase"]
                                sells = [r for r in raw if r.get("transactionType","") == "S - Sale"]
                                if len(buys) > len(sells):
                                    ins_signal = "Bullish"
                                elif len(sells) > len(buys):
                                    ins_signal = "Bearish"
                                else:
                                    ins_signal = "Neutral"
                                insider_result = {
                                    "signal": ins_signal,
                                    "buys":   len(buys),
                                    "sells":  len(sells),
                                    "raw":    raw[:20],
                                }
                            except Exception as e:
                                insider_result = {"error": str(e)}
                        st.session_state[f"insider_{dd_ticker}"] = insider_result
                        st.rerun()

                if insider_dd:
                    if "error" in insider_dd:
                        st.error(f"Error: {insider_dd['error']}")
                    else:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Insider Signal", insider_dd["signal"])
                        c2.metric("Buys (90d)",   insider_dd["buys"])
                        c3.metric("Sells (90d)",  insider_dd["sells"])
                        if insider_dd.get("raw"):
                            df_ins = pd.DataFrame(insider_dd["raw"])
                            show_cols = [c for c in
                                         ["name","transactionType","share","change","filingDate"]
                                         if c in df_ins.columns]
                            st.dataframe(df_ins[show_cols].head(10),
                                         use_container_width=True, hide_index=True)

            # ── ChatGPT prompt
            with st.expander("🤖 ChatGPT Research Prompt (always free)", expanded=False):
                prompt_text = build_chatgpt_prompt(
                    dd_ticker, latest_dd, score_dd, signal_dd, reasons_dd,
                    news_dd, insider_dd,
                )
                st.text_area(
                    "Copy this prompt → paste into ChatGPT",
                    value=prompt_text,
                    height=400,
                    key=f"prompt_{dd_ticker}",
                )
                st.caption(
                    "Paste the above into ChatGPT (or Claude) for a detailed "
                    "swing trading research analysis. For educational purposes only."
                )

            # ── Optional OpenAI API
            with st.expander("⚡ OpenAI API Analysis (costs 1 OpenAI request)", expanded=False):
                if not OPENAI_KEY:
                    st.warning("OPENAI_API_KEY not set in Streamlit secrets.")
                else:
                    if render_openai_gate(daily_limit=50):
                        if st.button("Generate AI Analysis", key=f"ai_btn_{dd_ticker}"):
                            with st.spinner("Generating analysis..."):
                                prompt_ai = build_chatgpt_prompt(
                                    dd_ticker, latest_dd, score_dd, signal_dd, reasons_dd,
                                    news_dd, insider_dd,
                                )
                                analysis = generate_ai_analysis(prompt_ai, OPENAI_KEY, increment_usage)
                            st.session_state[f"ai_analysis_{dd_ticker}"] = analysis

                    if f"ai_analysis_{dd_ticker}" in st.session_state:
                        st.markdown(st.session_state[f"ai_analysis_{dd_ticker}"])
                        render_disclaimer(compact=True)

    render_disclaimer()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Sector Scanner
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("🗂️ Sector Scanner")
    st.caption(
        "Scan entire sectors using yfinance only — no paid API calls. "
        "Sorts by Research Score to surface top swing setups."
    )
    render_disclaimer(compact=True)

    col_sec, col_per = st.columns([2, 1])
    with col_sec:
        selected_sector = st.selectbox(
            "Select a sector to scan",
            list(DISCOVERY_LISTS.keys()),
            key="sector_select",
        )
    with col_per:
        scan_period = st.selectbox(
            "Scan period",
            ["3mo", "6mo", "1y"],
            index=0,
            key="scan_period",
        )

    if st.button("🔍 Scan Sector", type="primary", key="scan_btn"):
        with st.spinner(f"Scanning {selected_sector}... ({len(DISCOVERY_LISTS[selected_sector])} tickers)"):
            scan_result = scan_category(selected_sector, period=scan_period)
        st.session_state["sector_scan_result"] = scan_result
        st.session_state["sector_scan_name"]   = selected_sector

    if "sector_scan_result" in st.session_state:
        df_scan = st.session_state["sector_scan_result"]
        sector_name = st.session_state.get("sector_scan_name", "")

        if df_scan.empty:
            st.info("No results returned — all tickers may have failed to load.")
        else:
            st.subheader(f"📊 {sector_name} — Scan Results")
            df_disp = build_display_df(df_scan)

            styled_scan = (
                df_disp.style
                .map(colour_signal, subset=["Signal"])
                .format({
                    "Close":          "${:.2f}",
                    "RSI":            "{:.1f}",
                    "Rel Volume":     "{:.2f}x",
                    "Research Score": "{}",
                })
            )
            st.dataframe(styled_scan, use_container_width=True, hide_index=True)

            # ── Ticker selector
            st.subheader("🔎 Quick Preview")
            t3_selected = st.selectbox(
                "Select a ticker to preview",
                df_scan["Ticker"].tolist(),
                key="t3_selected",
            )

            if t3_selected:
                t3_row = df_scan[df_scan["Ticker"] == t3_selected].iloc[0]
                latest_t3 = t3_row.get("_latest", {})
                sc_t3     = t3_row["Research Score"]
                sig_t3    = t3_row["Signal"]

                render_signal_banner(sig_t3, sc_t3)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Close",  f"${safe_float(latest_t3.get('Close')):.2f}")
                c2.metric("RSI",    f"{safe_float(latest_t3.get('RSI', 50)):.1f}")
                c3.metric("ATR",    f"${safe_float(latest_t3.get('ATR')):.2f}")
                c4.metric("RelVol", f"{safe_float(latest_t3.get('RelVol', 1)):.2f}x")

                with st.expander("Score Reasons", expanded=False):
                    reasons_t3 = t3_row.get("_reasons", [])
                    for r in reasons_t3:
                        clr = "#00c853" if r.startswith("+") else "#ef5350"
                        st.markdown(f"<span style='color:{clr}'>{r}</span>",
                                    unsafe_allow_html=True)

                if st.button("📤 Send to Deep Dive", key="t3_send"):
                    if t3_selected not in WATCHLIST:
                        new_wl = WATCHLIST + [t3_selected]
                        st.code(", ".join(new_wl), language=None)
                        st.caption(
                            f"Copy the above and paste into the **Watchlist tickers** "
                            f"field in the sidebar to add {t3_selected} to your watchlist."
                        )
                    else:
                        st.info(f"{t3_selected} is already in your watchlist.")

    render_disclaimer()
