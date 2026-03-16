"""
Market Data Page - Display historical OHLCV data with charts and Live Price
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from utils.runtime_cache import runtime_safe_cache_data


@runtime_safe_cache_data(ttl=60, show_spinner=False)
def _cached_market_overview(api_base_url: str, ticker: str, start_date: str, end_date: str):
    try:
        res = requests.get(
            f"{api_base_url}/market-overview",
            params={"ticker": ticker, "start_date": start_date, "end_date": end_date},
            timeout=30,
        )
        if res.status_code == 200:
            payload = res.json()
            return {
                "ok": True,
                "data": payload.get("data", []),
                "sentiment": payload.get("sentiment", {}),
                "error": "",
            }
        detail = "Unknown error"
        try:
            detail = res.json().get("detail", res.text)
        except Exception:
            detail = res.text
        return {"ok": False, "data": [], "sentiment": {}, "error": f"Failed to fetch data ({res.status_code}): {detail}"}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "data": [], "sentiment": {}, "error": "Connection failed. Backend service unavailable."}
    except requests.exceptions.Timeout:
        return {"ok": False, "data": [], "sentiment": {}, "error": "Request timed out. Try a smaller date range."}
    except Exception as e:
        return {"ok": False, "data": [], "sentiment": {}, "error": f"Unexpected error: {str(e)}"}


def render_market_data_page(T, API_BASE_URL):
    """Render the Market Data page."""
    
    # Currency pair mapping
    PAIR_TO_TICKER = {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "USDJPY=X",
        "USD/CHF": "USDCHF=X",
        "AUD/USD": "AUDUSD=X",
        "USD/CAD": "USDCAD=X",
        "NZD/USD": "NZDUSD=X",
        "EUR/GBP": "EURGBP=X",
        "EUR/JPY": "EURJPY=X",
        "GBP/JPY": "GBPJPY=X",
        "AUD/JPY": "AUDJPY=X",
        "CAD/JPY": "CADJPY=X",
        "CHF/JPY": "CHFJPY=X",
        "EUR/CHF": "EURCHF=X",
        "GBP/CHF": "GBPCHF=X",
        "EUR/AUD": "EURAUD=X",
        "EUR/CAD": "EURCAD=X",
        "GBP/AUD": "GBPAUD=X",
        "GBP/CAD": "GBPCAD=X",
        "AUD/NZD": "AUDNZD=X",
        "USD/SGD": "USDSGD=X",
        "USD/HKD": "USDHKD=X",
        "USD/MXN": "USDMXN=X",
        "USD/ZAR": "USDZAR=X",
        "XAU/USD": "GC=F",
        "XAG/USD": "SI=F",
    }
    
    # Default date range (last 90 days)
    default_end = datetime.now()
    default_start = default_end - timedelta(days=90)

    if "market_data_cache" not in st.session_state:
        st.session_state.market_data_cache = []
    if "market_data_error" not in st.session_state:
        st.session_state.market_data_error = ""
    if "market_data_query" not in st.session_state:
        st.session_state.market_data_query = None
    if "market_sentiment" not in st.session_state:
        st.session_state.market_sentiment = {}
    
    # Page Header
    st.markdown(
        f"<h1 style='color:{T['heading_color']}; font-size:2rem; margin-bottom:4px;'>📊 Market Data</h1>"
        f"<p style='color:{T['text_secondary']}; font-size:0.9rem; margin-top:0;'>View historical OHLCV data with interactive charts</p>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <style>
        .market-hero {{
            background: linear-gradient(120deg, {T['table_bg']} 0%, {T['plot_bg']} 100%);
            border: 1px solid {T['card_border']};
            border-radius: 16px;
            padding: 18px 20px;
            margin-bottom: 14px;
            box-shadow: 0 12px 28px {T['card_shadow']};
        }}
        .market-side-card {{
            background: {T['table_bg']};
            border: 1px solid {T['card_border']};
            border-radius: 14px;
            padding: 14px;
            margin-bottom: 10px;
        }}
        .market-overall {{
            border-left: 4px solid {T['accent']};
            background: {T['table_bg']};
            border-radius: 12px;
            padding: 12px 14px;
            margin-bottom: 12px;
        }}
        div[data-testid="stMetric"] {{
            min-height: 130px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    
    # ─── Live Price Section ─────────────────────────────────────────────────────────
    st.markdown(f"<h3 style='color:{T['subheading_color']};'>📈 Live Price</h3>", unsafe_allow_html=True)

    # Live Price Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Live Price", "1.0842", "+0.0012")
    col2.metric("Model Accuracy", "84.2%", "+1.2%")
    col3.metric("Predicted Trend", "Bullish ↑", delta_color="normal")
    col4.metric("Volatility Index", "Low", "Stable")

    try:
        from streamlit_extras.metric_cards import style_metric_cards
        style_metric_cards(
            background_color=T['table_bg'],
            border_left_color=T['accent'],
            border_color=T['card_border'],
            box_shadow=True
        )
    except Exception:
        pass
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    # Controls
    ctl1, ctl2, ctl3, ctl4 = st.columns([1.3, 1, 1, 0.9])

    with ctl1:
        pair = st.selectbox(
            "Currency Pair",
            list(PAIR_TO_TICKER.keys()),
            key="market_pair_select",
        )

    with ctl2:
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            key="market_start_date",
        )

    with ctl3:
        end_date = st.date_input(
            "End Date",
            value=default_end,
            key="market_end_date",
        )

    with ctl4:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        reload_btn = st.button(
            "↻ Refresh Data",
            use_container_width=True,
            key="fetch_market_data_btn",
        )

    ticker = PAIR_TO_TICKER.get(pair, "EURUSD=X")
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    current_query = (ticker, start_date_str, end_date_str)

    auto_load = st.session_state.market_data_query != current_query
    should_fetch = reload_btn or auto_load

    if should_fetch:
        with st.spinner(f"Loading {pair} data..."):
            fetched = _cached_market_overview(API_BASE_URL, ticker, start_date_str, end_date_str)
            st.session_state.market_data_query = current_query
            st.session_state.market_data_cache = fetched.get("data", [])
            st.session_state.market_sentiment = fetched.get("sentiment", {})
            st.session_state.market_data_error = fetched.get("error", "")

    st.markdown("<br>", unsafe_allow_html=True)

    layout_left, layout_main = st.columns([1.05, 2.25])

    with layout_left:
        st.markdown(f"<h4 style='color:{T['subheading_color']}; margin-bottom:6px;'>🧠 Model Outlook</h4>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='color:{T['text_secondary']}; margin-top:0; font-size:0.85rem;'>One-day directional signal by model</p>",
            unsafe_allow_html=True,
        )

        sentiment = st.session_state.market_sentiment or {}
        model_signals = sentiment.get("models", [])
        threshold_pct = float(sentiment.get("flat_threshold_pct", 0.001))

        for row in model_signals:
            model_name = row.get("model", "Model")
            trend = row.get("signal", "No Signal")
            pred_price = row.get("predicted_price")
            change_pct = row.get("change_pct")

            trend_color = T["positive"] if trend == "Bullish" else ("#ff5c5c" if trend == "Bearish" else T["text_secondary"])
            price_txt = f"{float(pred_price):.5f}" if pred_price is not None else "N/A"
            delta_txt = f"{float(change_pct) * 100:+.2f}%" if change_pct is not None else "-"

            st.markdown(
                f"""
                <div class='market-side-card'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <span style='color:{T['heading_color']}; font-weight:700;'>{model_name}</span>
                        <span style='color:{trend_color}; font-weight:700;'>{trend}</span>
                    </div>
                    <div style='margin-top:8px; color:{T['text_primary']}; font-size:1.05rem; font-weight:600;'>
                        {price_txt}
                    </div>
                    <div style='margin-top:4px; color:{T['text_secondary']}; font-size:0.85rem;'>
                        vs last close: {delta_txt}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if model_signals:
            overall_label = sentiment.get("overall", "No Signal")
            if overall_label == "Bullish":
                overall_color = T["positive"]
            elif overall_label == "Bearish":
                overall_color = "#ff5c5c"
            else:
                overall_color = T["text_secondary"]

            st.markdown(
                f"""
                <div class='market-overall'>
                    <p style='margin:0; color:{T['text_secondary']}; font-size:0.82rem;'>Overall Model Sentiment</p>
                    <p style='margin:4px 0 0 0; color:{overall_color}; font-size:1.2rem; font-weight:800;'>{overall_label}</p>
                    <p style='margin:4px 0 0 0; color:{T['text_secondary']}; font-size:0.78rem;'>Adaptive flatish band: +/-{threshold_pct * 100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with layout_main:
        data_container = st.container(border=True)

        with data_container:
            market_data = st.session_state.market_data_cache or []

            if st.session_state.market_data_error:
                st.error(f"❌ {st.session_state.market_data_error}")

            if market_data:
                df = pd.DataFrame(market_data)

                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])

                numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                st.success(f"✅ Loaded {len(df)} records for {pair}")

                st.markdown(f"<h4 style='color:{T['subheading_color']};'>📈 {pair} Price Chart</h4>", unsafe_allow_html=True)

                if all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
                    fig = go.Figure(
                        data=[
                            go.Candlestick(
                                x=df["Date"],
                                open=df["Open"],
                                high=df["High"],
                                low=df["Low"],
                                close=df["Close"],
                                name=pair,
                            )
                        ]
                    )

                    fig.update_layout(
                        title=f"{pair} - OHLC Candlestick Chart",
                        template=T["plot_template"],
                        paper_bgcolor=T["plot_paper"],
                        plot_bgcolor=T["plot_bg"],
                        xaxis=dict(
                            title="Date",
                            gridcolor=T["plot_grid"],
                            color=T["plot_axis_color"],
                            rangeslider=dict(visible=False),
                        ),
                        yaxis=dict(
                            title="Price",
                            gridcolor=T["plot_grid"],
                            color=T["plot_axis_color"],
                        ),
                        legend=dict(
                            bgcolor=T["plot_legend_bg"],
                            bordercolor=T["plot_legend_border"],
                            font=dict(color=T["text_primary"]),
                        ),
                        margin=dict(l=20, r=20, t=50, b=20),
                        height=450,
                    )

                    st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"<h5 style='color:{T['subheading_color']};'>📉 Close Price Trend</h5>", unsafe_allow_html=True)

                if "Close" in df.columns and "Date" in df.columns:
                    fig_line = go.Figure()

                    fig_line.add_trace(
                        go.Scatter(
                            x=df["Date"],
                            y=df["Close"],
                            mode="lines",
                            name="Close Price",
                            line=dict(color=T["accent"], width=2.3),
                            fill="tozeroy",
                            fillcolor="rgba(0, 212, 255, 0.1)",
                        )
                    )

                    fig_line.update_layout(
                        title=f"{pair} - Close Price",
                        template=T["plot_template"],
                        paper_bgcolor=T["plot_paper"],
                        plot_bgcolor=T["plot_bg"],
                        xaxis=dict(
                            title="Date",
                            gridcolor=T["plot_grid"],
                            color=T["plot_axis_color"],
                        ),
                        yaxis=dict(
                            title="Price",
                            gridcolor=T["plot_grid"],
                            color=T["plot_axis_color"],
                        ),
                        legend=dict(
                            bgcolor=T["plot_legend_bg"],
                            bordercolor=T["plot_legend_border"],
                            font=dict(color=T["text_primary"]),
                        ),
                        margin=dict(l=20, r=20, t=50, b=20),
                        height=350,
                    )

                    st.plotly_chart(fig_line, use_container_width=True)

                if "Volume" in df.columns:
                    st.markdown(f"<h5 style='color:{T['subheading_color']};'>📊 Trading Volume</h5>", unsafe_allow_html=True)

                    fig_vol = go.Figure()

                    fig_vol.add_trace(
                        go.Bar(
                            x=df["Date"],
                            y=df["Volume"],
                            name="Volume",
                            marker_color=T["accent2"],
                        )
                    )

                    fig_vol.update_layout(
                        title=f"{pair} - Trading Volume",
                        template=T["plot_template"],
                        paper_bgcolor=T["plot_paper"],
                        plot_bgcolor=T["plot_bg"],
                        xaxis=dict(
                            title="Date",
                            gridcolor=T["plot_grid"],
                            color=T["plot_axis_color"],
                        ),
                        yaxis=dict(
                            title="Volume",
                            gridcolor=T["plot_grid"],
                            color=T["plot_axis_color"],
                        ),
                        legend=dict(
                            bgcolor=T["plot_legend_bg"],
                            bordercolor=T["plot_legend_border"],
                            font=dict(color=T["text_primary"]),
                        ),
                        margin=dict(l=20, r=20, t=50, b=20),
                        height=300,
                    )

                    st.plotly_chart(fig_vol, use_container_width=True)

                st.markdown(f"<h5 style='color:{T['subheading_color']};'>📋 Data Table</h5>", unsafe_allow_html=True)

                df_display = df.copy()
                if "Date" in df_display.columns:
                    df_display["Date"] = df_display["Date"].dt.strftime("%Y-%m-%d")

                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if col in df_display.columns:
                        df_display[col] = df_display[col].round(4)

                st.dataframe(df_display, use_container_width=True, height=300)

                st.markdown(f"<h5 style='color:{T['subheading_color']};'>📊 Statistics</h5>", unsafe_allow_html=True)

                stat_cols = st.columns(4)

                if "Close" in df.columns:
                    close_stats = df["Close"].describe()
                    stat_cols[0].metric("Mean Close", f"{close_stats['mean']:.4f}")
                    stat_cols[1].metric("Min Close", f"{close_stats['min']:.4f}")
                    stat_cols[2].metric("Max Close", f"{close_stats['max']:.4f}")
                    stat_cols[3].metric("Std Dev", f"{close_stats['std']:.4f}")

                if "Volume" in df.columns:
                    vol_stats = df["Volume"].describe()
                    stat_cols2 = st.columns(4)
                    stat_cols2[0].metric("Mean Volume", f"{vol_stats['mean']:.0f}")
                    stat_cols2[1].metric("Min Volume", f"{vol_stats['min']:.0f}")
                    stat_cols2[2].metric("Max Volume", f"{vol_stats['max']:.0f}")
                    stat_cols2[3].metric("Total Volume", f"{df['Volume'].sum():.0f}")
            else:
                st.markdown(
                    f"""
                    <div style='text-align:center; padding:60px 20px; color:{T['text_secondary']};'>
                        <p style='font-size:3rem; margin-bottom:10px;'>📊</p>
                        <p>No market data available for this selection.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    
    # Info Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style='background:{T['table_bg']}; border:1px solid {T['card_border']}; 
                    border-radius:12px; padding:20px;'>
            <h4 style='color:{T['heading_color']}; margin-top:0; font-size:1.1rem;'>📈 Chart Types</h4>
            <ul style='color:{T['text_primary']}; padding-left:20px; font-size:0.95rem; line-height:1.6; margin-bottom:0;'>
                <li><b>Candlestick Chart</b> - Shows Open, High, Low, Close prices</li>
                <li><b>Line Chart</b> - Shows closing price trends over time</li>
                <li><b>Volume Chart</b> - Shows trading volume</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background:{T['table_bg']}; border:1px solid {T['card_border']}; 
                    border-radius:12px; padding:20px;'>
            <h4 style='color:{T['heading_color']}; margin-top:0; font-size:1.1rem;'>💡 Tips</h4>
            <ul style='color:{T['text_primary']}; padding-left:20px; font-size:0.95rem; line-height:1.6; margin-bottom:0;'>
                <li>Use a narrower date range for faster loading</li>
                <li>Hover over charts for detailed tooltips</li>
                <li>Click legend items to toggle series visibility</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
