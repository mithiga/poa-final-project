"""
About Page - Product overview and current project status.
"""

from datetime import datetime

import requests
import streamlit as st

from utils.runtime_cache import runtime_safe_cache_data


@runtime_safe_cache_data(ttl=20, show_spinner=False)
def _cached_status(api_base_url: str):
    try:
        res = requests.get(f"{api_base_url}/status", timeout=8)
        if res.status_code == 200:
            return {"ok": True, "payload": res.json(), "error": ""}
        return {"ok": False, "payload": {}, "error": f"Status endpoint returned {res.status_code}"}
    except Exception as exc:
        return {"ok": False, "payload": {}, "error": str(exc)}


@runtime_safe_cache_data(ttl=60, show_spinner=False)
def _cached_training_period(api_base_url: str, ticker: str):
    try:
        res = requests.get(f"{api_base_url}/training-period", params={"ticker": ticker}, timeout=8)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return {"start": "N/A", "end": "N/A", "ticker": ticker}


@runtime_safe_cache_data(ttl=60, show_spinner=False)
def _cached_tape_health(api_base_url: str):
    try:
        res = requests.get(
            f"{api_base_url}/market-ticker-tape",
            params={"tickers": "EURUSD=X,GC=F,BTC-USD", "lookback_days": 7},
            timeout=10,
        )
        if res.status_code == 200:
            return {"ok": True, "count": len(res.json().get("items", []))}
    except Exception:
        pass
    return {"ok": False, "count": 0}


def render_about_page(T, API_BASE_URL, BACKEND_MODE):
    """Render an updated About page aligned to the current project state."""

    status_wrapper = _cached_status(API_BASE_URL)
    status_payload = status_wrapper.get("payload", {})

    models = status_payload.get("models_available", [])
    tickers = status_payload.get("tickers_trained", [])

    tape_health = _cached_tape_health(API_BASE_URL)
    example_ticker = tickers[0] if tickers else "EURUSD=X"
    period = _cached_training_period(API_BASE_URL, example_ticker)

    st.markdown(
        f"""
        <style>
        .about-hero {{
            background: linear-gradient(120deg, {T['table_bg']} 0%, {T['plot_bg']} 100%);
            border: 1px solid {T['card_border']};
            border-radius: 14px;
            padding: 16px;
            margin-bottom: 14px;
            box-shadow: 0 10px 24px {T['card_shadow']};
        }}
        .about-card {{
            background: {T['table_bg']};
            border: 1px solid {T['card_border']};
            border-radius: 12px;
            padding: 16px;
            height: 100%;
        }}
        .about-kicker {{
            color: {T['accent']};
            font-size: 0.8rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<h1 style='color:{T['heading_color']}; margin-bottom:0;'>About ForexAI Pro</h1>"
        f"<p style='color:{T['text_secondary']}; margin-top:0.3rem;'>"
        f"Current build summary, architecture, and live capability status.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Backend Mode", BACKEND_MODE)
    c2.metric("System", status_payload.get("status", "offline").title())
    c3.metric("Models Available", len(models))
    c4.metric("Trained Tickers", len(tickers))

    st.markdown(
        f"""
        <div class="about-hero">
            <div class="about-kicker">Product Snapshot</div>
            <p style="color:{T['text_primary']}; margin:0; line-height:1.6;">
                ForexAI Pro is now centered around a <b>Market Data Terminal</b> landing experience:
                sticky live ticker tape, quick-range controls, synchronized OHLC + volume charting,
                and model sentiment confidence visualization. Model workflows are consolidated under
                <b>Model Management</b> for forecasting, training, and hyperparameter tuning.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"""
            <div class="about-card">
                <div class="about-kicker">Architecture</div>
                <ul style="color:{T['text_primary']}; margin:0; line-height:1.7;">
                    <li>Frontend: Streamlit multi-page UI via a minimal app router.</li>
                    <li>Backend: FastAPI inference and training services.</li>
                    <li>Deployment: single-process embedded mode or external API mode.</li>
                    <li>Data: yfinance market feed with backend fallback for non-trading windows.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        ticker_bar_status = "Active" if tape_health.get("ok") and tape_health.get("count", 0) > 0 else "Unavailable"
        status_color = T["positive"] if ticker_bar_status == "Active" else T.get("danger", "#d64545")
        st.markdown(
            f"""
            <div class="about-card">
                <div class="about-kicker">Latest Enhancements</div>
                <ul style="color:{T['text_primary']}; margin:0; line-height:1.7;">
                    <li>Market Data is the default home experience.</li>
                    <li>Dedicated backend endpoint for ticker tape snapshots.</li>
                    <li>Fallback fetch windows reduce weekend/no-session failures.</li>
                    <li>Ticker tape endpoint health: <b style="color:{status_color};">{ticker_bar_status}</b>.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="about-card" style="margin-top:12px;">
            <div class="about-kicker">Training Coverage</div>
            <p style="color:{T['text_primary']}; margin:0; line-height:1.6;">
                Example ticker: <b>{period.get('ticker', example_ticker)}</b><br>
                Training window: <b>{period.get('start', 'N/A')}</b> to <b>{period.get('end', 'N/A')}</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not status_wrapper.get("ok"):
        st.warning(f"Live status endpoint note: {status_wrapper.get('error', 'Unknown issue')}")

    st.markdown(
        f"<p style='text-align:center; color:{T['text_secondary']}; font-size:0.78rem;'>"
        f"About page updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        unsafe_allow_html=True,
    )
