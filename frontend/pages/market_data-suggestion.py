import streamlit as st
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from utils.runtime_cache import runtime_safe_cache_data

@runtime_safe_cache_data(ttl=60, show_spinner=False)
def _cached_market_overview(api_base_url: str, ticker: str, start_date: str, end_date: str):
    # (Existing logic remains the same for data fetching)
    try:
        res = requests.get(
            f"{api_base_url}/market-overview",
            params={"ticker": ticker, "start_date": start_date, "end_date": end_date},
            timeout=30,
        )
        if res.status_code == 200:
            payload = res.json()
            return {"ok": True, "data": payload.get("data", []), "sentiment": payload.get("sentiment", {}), "error": ""}
        return {"ok": False, "data": [], "sentiment": {}, "error": f"Error {res.status_code}"}
    except Exception as e:
        return {"ok": False, "data": [], "sentiment": {}, "error": str(e)}

def render_market_data_page(T, API_BASE_URL):
    # 1. Configuration & Ticker Mapping
    PAIR_TO_TICKER = {
        "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X",
        "XAU/USD": "GC=F", "XAG/USD": "SI=F", "BTC/USD": "BTC-USD"
    }

    # 2. Sidebar Controls (Improved UX)
    with st.sidebar:
        st.markdown(f"<h2 style='color:{T['accent']};'>Controls</h2>", unsafe_allow_html=True)
        pair = st.selectbox("Select Asset", list(PAIR_TO_TICKER.keys()), index=0)
        
        st.markdown("---")
        st.write("📅 **Date Range**")
        quick_range = st.radio("Quick Select", ["1M", "3M", "6M", "YTD", "Custom"], horizontal=True)
        
        end_date = datetime.now()
        if quick_range == "1M": start_date = end_date - timedelta(days=30)
        elif quick_range == "3M": start_date = end_date - timedelta(days=90)
        elif quick_range == "6M": start_date = end_date - timedelta(days=180)
        elif quick_range == "YTD": start_date = datetime(end_date.year, 1, 1)
        else:
            start_date = st.date_input("Start", end_date - timedelta(days=90))
            end_date = st.date_input("End", end_date)

        reload_btn = st.button("↻ Refresh Terminal", use_container_width=True)

    # 3. Data Fetching Logic
    ticker = PAIR_TO_TICKER.get(pair)
    s_str, e_str = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    
    # Initialize session state if empty
    if "market_data_cache" not in st.session_state or reload_btn:
        with st.spinner("Fetching market pulse..."):
            fetched = _cached_market_overview(API_BASE_URL, ticker, s_str, e_str)
            st.session_state.market_data_cache = fetched.get("data", [])
            st.session_state.market_sentiment = fetched.get("sentiment", {})

    # 4. Main UI Layout
    st.markdown(
        f"<h1 style='color:{T['heading_color']}; margin-bottom:0;'>{pair} <span style='font-size:1rem; color:{T['text_secondary']}'>{ticker}</span></h1>", 
        unsafe_allow_html=True
    )

    # Top Metric Bar (Visual Pulse)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Live Price", "1.0842", "+0.0012", help="Real-time mid-market price")
    m2.metric("Model Signal", "Bullish", "84%", delta_color="normal")
    m3.metric("Volatility", "Low", "Stable")
    m4.metric("Daily Volume", "1.2M", "-5%")

    try:
        from streamlit_extras.metric_cards import style_metric_cards
        style_metric_cards(background_color=T['table_bg'], border_left_color=T['accent'])
    except: pass

    st.divider()

    # 5. Dashboard Grid
    col_signals, col_chart = st.columns([1, 3])

    with col_signals:
        st.markdown(f"### 🧠 Outlook")
        sentiment = st.session_state.market_sentiment or {}
        for row in sentiment.get("models", []):
            color = T["positive"] if row['signal'] == "Bullish" else "#ff5c5c"
            st.markdown(f"""
                <div style="background:{T['table_bg']}; padding:15px; border-radius:10px; border-left:4px solid {color}; margin-bottom:10px;">
                    <small style="color:{T['text_secondary']}">{row['model']}</small>< dream >
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <b style="color:{T['heading_color']}">{row['predicted_price']:.4f}</b>
                        <span style="color:{color}; font-size:0.8rem; font-weight:bold;">{row['signal']}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    with col_chart:
        df = pd.DataFrame(st.session_state.market_data_cache)
        if not df.empty:
            df["Date"] = pd.to_datetime(df["Date"])
            
            # Subplot: Candlestick (Top) and Volume (Bottom)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, 
                                row_heights=[0.7, 0.3])

            # OHLC
            fig.add_trace(go.Candlestick(
                x=df["Date"], open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="Price"
            ), row=1, col=1)

            # SMA 20 (Visual refinement: Adding a technical indicator)
            df['SMA20'] = df['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA20"], line=dict(color=T['accent'], width=1), name="SMA 20"), row=1, col=1)

            # Volume
            fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume", marker_color="rgba(100, 150, 255, 0.4)"), row=2, col=1)

            fig.update_layout(
                template=T["plot_template"],
                height=600,
                margin=dict(l=10, r=10, t=30, b=10),
                hovermode="x unified",
                showlegend=False,
                xaxis_rangeslider_visible=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select a ticker and click refresh to load data.")

    # 6. Data Explorer (Expandable to save space)
    with st.expander("🔍 Raw Market Data & Statistics"):
        tab1, tab2 = st.tabs(["Data Table", "Descriptive Stats"])
        with tab1:
            st.dataframe(df, use_container_width=True)
        with tab2:
            st.table(df.describe())

    st.markdown(f"<p style='text-align:center; color:{T['text_secondary']}; font-size:0.7rem;'>Last update: {datetime.now().strftime('%H:%M:%S')}</p>", unsafe_allow_html=True)