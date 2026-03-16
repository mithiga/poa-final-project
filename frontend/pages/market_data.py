"""
Market Data Page - Premium market terminal style with synced charts and model sentiment.
"""

from datetime import date, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

from utils.runtime_cache import runtime_safe_cache_data


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
    "BTC/USD": "BTC-USD",
}


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
        return {
            "ok": False,
            "data": [],
            "sentiment": {},
            "error": f"Failed to fetch data ({res.status_code}): {detail}",
        }
    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "data": [],
            "sentiment": {},
            "error": "Connection failed. Backend service unavailable.",
        }
    except requests.exceptions.Timeout:
        return {
            "ok": False,
            "data": [],
            "sentiment": {},
            "error": "Request timed out. Try a smaller date range.",
        }
    except Exception as exc:
        return {
            "ok": False,
            "data": [],
            "sentiment": {},
            "error": f"Unexpected error: {str(exc)}",
        }


@runtime_safe_cache_data(ttl=60, show_spinner=False)
def _cached_ticker_snapshots(api_base_url: str, tickers: tuple[str, ...]):
    snapshots = {}
    try:
        joined = ",".join(tickers)
        res = requests.get(
            f"{api_base_url}/market-ticker-tape",
            params={"tickers": joined, "lookback_days": 7},
            timeout=15,
        )
        if res.status_code == 200:
            items = res.json().get("items", [])
            for item in items:
                ticker = item.get("ticker")
                if not ticker:
                    continue
                snapshots[ticker] = {
                    "price": float(item.get("price", 0.0)),
                    "delta_pct": float(item.get("delta_pct", 0.0)),
                }
            return snapshots
    except Exception:
        pass

    # Backward-compatible fallback if the new endpoint is not available.
    end_date = date.today()
    start_date = end_date - timedelta(days=7)
    for ticker in tickers:
        try:
            res = requests.get(
                f"{api_base_url}/market-overview",
                params={
                    "ticker": ticker,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                },
                timeout=12,
            )
            if res.status_code != 200:
                continue

            rows = res.json().get("data", [])
            if len(rows) < 2:
                continue

            latest = rows[-1]
            prev = rows[-2]
            latest_close = float(latest.get("Close", 0.0))
            prev_close = float(prev.get("Close", latest_close))
            delta_pct = ((latest_close - prev_close) / prev_close) * 100 if prev_close else 0.0
            snapshots[ticker] = {"price": latest_close, "delta_pct": delta_pct}
        except Exception:
            continue

    return snapshots


def _confidence_from_models(models):
    if not models:
        return 0.0

    votes = {"Bullish": 0, "Bearish": 0, "Neutral": 0}
    weighted = []

    for row in models:
        signal = row.get("signal", "Neutral")
        if signal not in votes:
            signal = "Neutral"
        votes[signal] += 1

        change_pct = row.get("change_pct")
        if change_pct is not None:
            weighted.append(abs(float(change_pct)) * 100)

    majority_votes = max(votes.values())
    majority_ratio = majority_votes / len(models)
    magnitude_score = min((sum(weighted) / len(weighted)) if weighted else 0.0, 8.0) / 8.0

    return round(((majority_ratio * 0.75) + (magnitude_score * 0.25)) * 100, 1)


def _range_to_dates(selected_range):
    end_dt = date.today()
    if selected_range == "1D":
        return end_dt - timedelta(days=1), end_dt
    if selected_range == "1W":
        return end_dt - timedelta(days=7), end_dt
    if selected_range == "1M":
        return end_dt - timedelta(days=30), end_dt
    if selected_range == "3M":
        return end_dt - timedelta(days=90), end_dt
    if selected_range == "YTD":
        return date(end_dt.year, 1, 1), end_dt
    return end_dt - timedelta(days=90), end_dt


def _render_top_ticker(T, snapshots):
    if not snapshots:
        return

    reverse_map = {v: k for k, v in PAIR_TO_TICKER.items()}
    items = []
    for ticker, payload in snapshots.items():
        pair_label = reverse_map.get(ticker, ticker)
        price = payload.get("price", 0.0)
        delta_pct = payload.get("delta_pct", 0.0)
        trend_color = T["positive"] if delta_pct >= 0 else T.get("danger", "#d64545")
        items.append(
            f"<span class='md-item'><b>{pair_label}</b> {price:,.5f} "
            f"<span style='color:{trend_color};'>{delta_pct:+.2f}%</span></span>"
        )

    line = "".join(items + items)
    st.markdown(
        f"""
        <style>
        .md-ticker-wrap {{
            position: sticky;
            top: 0;
            z-index: 80;
            margin-bottom: 14px;
            border: 1px solid {T['card_border']};
            border-radius: 12px;
            background: linear-gradient(90deg, {T['table_bg']} 0%, {T['plot_bg']} 100%);
            overflow: hidden;
            box-shadow: 0 8px 22px {T['card_shadow']};
        }}
        .md-track {{
            white-space: nowrap;
            display: inline-block;
            padding: 10px 0;
            animation: market-scroll 40s linear infinite;
        }}
        .md-item {{
            display: inline-block;
            margin-right: 34px;
            color: {T['text_primary']};
            font-size: 0.9rem;
        }}
        @keyframes market-scroll {{
            0% {{ transform: translateX(0); }}
            100% {{ transform: translateX(-50%); }}
        }}
        </style>
        <div class='md-ticker-wrap'>
            <div class='md-track'>{line}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_market_data_page(T, API_BASE_URL):
    """Render the Market Data page."""

    if "market_data_cache" not in st.session_state:
        st.session_state.market_data_cache = []
    if "market_data_error" not in st.session_state:
        st.session_state.market_data_error = ""
    if "market_data_query" not in st.session_state:
        st.session_state.market_data_query = None
    if "market_sentiment" not in st.session_state:
        st.session_state.market_sentiment = {}

    snapshot_tickers = (
        "EURUSD=X",
        "GBPUSD=X",
        "USDJPY=X",
        "GC=F",
        "SI=F",
        "BTC-USD",
    )
    snapshots = _cached_ticker_snapshots(API_BASE_URL, snapshot_tickers)
    _render_top_ticker(T, snapshots)

    bearish_color = T.get("danger", "#d64545")

    st.markdown(
        f"<h1 style='color:{T['heading_color']}; margin-bottom:0;'>Market Data Terminal</h1>"
        f"<p style='color:{T['text_secondary']}; margin-top:0.25rem;'>"
        f"Technical charting, volume context, and model consensus in one view.</p>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(f"### Controls")
        pair = st.selectbox("Asset", list(PAIR_TO_TICKER.keys()), index=0)
        quick_range = st.radio("Quick Range", ["1D", "1W", "1M", "3M", "YTD", "Custom"], horizontal=True, index=4)

        if quick_range == "Custom":
            default_start = date.today() - timedelta(days=90)
            start_date = st.date_input("Start Date", value=default_start)
            end_date = st.date_input("End Date", value=date.today())
        else:
            start_date, end_date = _range_to_dates(quick_range)
            st.caption(f"Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        reload_btn = st.button("Refresh Terminal", use_container_width=True)

    ticker = PAIR_TO_TICKER.get(pair, "EURUSD=X")
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    query = (ticker, start_str, end_str)

    should_fetch = reload_btn or st.session_state.market_data_query != query
    chart_placeholder = st.empty()
    sentiment_placeholder = st.empty()

    if should_fetch:
        with sentiment_placeholder.container():
            st.info("Updating model outlook...")
        with chart_placeholder.container():
            st.info("Loading chart frames and price feed...")

        fetched = _cached_market_overview(API_BASE_URL, ticker, start_str, end_str)
        st.session_state.market_data_query = query
        st.session_state.market_data_cache = fetched.get("data", [])
        st.session_state.market_sentiment = fetched.get("sentiment", {})
        st.session_state.market_data_error = fetched.get("error", "")

        chart_placeholder.empty()
        sentiment_placeholder.empty()

    df = pd.DataFrame(st.session_state.market_data_cache or [])
    if not df.empty and "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sentiment = st.session_state.market_sentiment or {}
    models = sentiment.get("models", [])
    overall_signal = sentiment.get("overall", "Neutral")
    confidence = _confidence_from_models(models)

    live_price = "N/A"
    live_delta = "-"
    has_volume_data = False
    if not df.empty and "Close" in df.columns:
        latest_close = df["Close"].iloc[-1]
        live_price = f"{latest_close:,.5f}"
        if len(df) > 1:
            prev_close = df["Close"].iloc[-2]
            pct = ((latest_close - prev_close) / prev_close) * 100 if prev_close else 0.0
            live_delta = f"{pct:+.2f}%"
        if "Volume" in df.columns:
            vol_series = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
            has_volume_data = bool(vol_series.gt(0).any())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Live Price", live_price, live_delta)
    m2.metric("Model Signal", overall_signal, f"{confidence:.1f}% confidence")
    m3.metric("Quick Range", quick_range, f"{len(df)} rows")
    m4.metric("Asset / Ticker", pair, ticker)

    layout_left, layout_main = st.columns([1, 2.6])

    with layout_left:
        st.markdown("### Model Outlook")

        gauge_value = confidence if overall_signal != "Neutral" else min(confidence, 50.0)
        gauge_bar = T["positive"] if overall_signal == "Bullish" else (bearish_color if overall_signal == "Bearish" else T["accent"])

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=gauge_value,
                title={"text": "Signal Confidence"},
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": gauge_bar},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 35], "color": "rgba(128,128,128,0.2)"},
                        {"range": [35, 70], "color": "rgba(255,170,0,0.2)"},
                        {"range": [70, 100], "color": "rgba(0,200,120,0.2)"},
                    ],
                },
            )
        )
        gauge.update_layout(
            template=T["plot_template"],
            height=250,
            margin=dict(l=10, r=10, t=35, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(gauge, use_container_width=True)

        threshold_pct = float(sentiment.get("flat_threshold_pct", 0.001)) * 100
        signal_color = T["positive"] if overall_signal == "Bullish" else (bearish_color if overall_signal == "Bearish" else T["text_secondary"])
        st.markdown(
            f"<p style='margin-top:0; color:{signal_color}; font-weight:700;'>Overall: {overall_signal}</p>"
            f"<p style='margin-top:-8px; color:{T['text_secondary']}; font-size:0.85rem;'>"
            f"Adaptive neutral band: +/-{threshold_pct:.2f}%</p>",
            unsafe_allow_html=True,
        )

        for row in models:
            model_name = row.get("model", "Model")
            trend = row.get("signal", "Neutral")
            trend_color = T["positive"] if trend == "Bullish" else (bearish_color if trend == "Bearish" else T["text_secondary"])
            predicted_price = row.get("predicted_price")
            price_label = f"{float(predicted_price):,.5f}" if predicted_price is not None else "N/A"
            change_pct = row.get("change_pct")
            conf_label = f"{abs(float(change_pct)) * 100:.1f}%" if change_pct is not None else "-"

            st.markdown(
                f"""
                <div style="background:{T['table_bg']}; border:1px solid {T['card_border']}; border-radius:12px; padding:12px; margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; color:{T['heading_color']};">{model_name}</span>
                        <span style="color:{trend_color}; font-weight:700;">{trend}</span>
                    </div>
                    <div style="margin-top:6px; color:{T['text_primary']};">Predicted: <b>{price_label}</b></div>
                    <div style="margin-top:4px; color:{T['text_secondary']}; font-size:0.82rem;">Confidence: {conf_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with layout_main:
        if st.session_state.market_data_error:
            st.error(st.session_state.market_data_error)

        if df.empty:
            st.info("No market data available. Use quick ranges or press Refresh Terminal.")
        else:
            if "Close" in df.columns:
                df["SMA20"] = df["Close"].rolling(window=20, min_periods=1).mean()

            if has_volume_data:
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.04,
                    row_heights=[0.72, 0.28],
                    subplot_titles=(f"{pair} ({ticker}) OHLC + SMA(20)", "Volume"),
                )
            else:
                fig = make_subplots(
                    rows=1,
                    cols=1,
                    subplot_titles=(f"{pair} ({ticker}) OHLC + SMA(20)",),
                )

            fig.add_trace(
                go.Candlestick(
                    x=df["Date"],
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="OHLC",
                ),
                row=1,
                col=1,
            )

            if "SMA20" in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["Date"],
                        y=df["SMA20"],
                        mode="lines",
                        line=dict(color=T["accent"], width=1.8),
                        name="SMA 20",
                    ),
                    row=1,
                    col=1,
                )

            if has_volume_data:
                vol_color = [
                    T["positive"] if close_val >= open_val else bearish_color
                    for close_val, open_val in zip(df["Close"], df["Open"])
                ]
                fig.add_trace(
                    go.Bar(
                        x=df["Date"],
                        y=df["Volume"],
                        marker_color=vol_color,
                        name="Volume",
                        opacity=0.55,
                    ),
                    row=2,
                    col=1,
                )

            fig.update_xaxes(matches="x", showspikes=True, spikemode="across", spikesnap="cursor")
            fig.update_yaxes(tickformat=".4f", rangemode="nonnegative", row=1, col=1)
            if has_volume_data:
                fig.update_yaxes(rangemode="tozero", row=2, col=1)
            fig.update_layout(
                template=T["plot_template"],
                hovermode="x unified",
                showlegend=True,
                legend=dict(orientation="h", y=1.05, x=0),
                xaxis_rangeslider_visible=False,
                height=650,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

            if not has_volume_data:
                st.caption("Volume is unavailable for this asset from Yahoo Finance. This is common for spot forex symbols because there is no centralized exchange volume feed.")

            with st.expander("Raw Market Data & Statistics"):
                tab1, tab2 = st.tabs(["Data Table", "Descriptive Stats"])
                with tab1:
                    display_df = df.copy()
                    display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
                    st.dataframe(display_df, use_container_width=True, height=320)
                with tab2:
                    desc = df.describe(include="all").transpose()
                    st.dataframe(desc, use_container_width=True, height=320)

    st.markdown(
        f"<p style='text-align:center; color:{T['text_secondary']}; font-size:0.75rem;'>"
        f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        unsafe_allow_html=True,
    )
