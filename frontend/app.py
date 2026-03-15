"""
ForexAI Pro - Main Application Entry Point
Multi-page Streamlit application for Forex AI Inference Engine
"""

import streamlit as st
import requests
from datetime import datetime, timedelta

from utils.theme import THEMES, get_theme_css, get_navbar_toggle_html
from utils.backend_adapter import configure_backend

try:
    from streamlit_extras.metric_cards import style_metric_cards
    HAS_METRIC_CARDS = True
except ImportError:
    HAS_METRIC_CARDS = False

# ─── API Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL, BACKEND_MODE = configure_backend()

# ─── Page Config ───────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="ForexAI Pro", page_icon="📈")

# ─── Theme State ───────────────────────────────────────────────────────────────────
params = st.query_params
if "theme" in params and params["theme"] in THEMES:
    st.session_state.theme = params["theme"]
elif "theme" not in st.session_state:
    st.session_state.theme = "dark"

T = THEMES.get(st.session_state.theme, THEMES["dark"])

# ─── Inject Theme CSS + Floating Toggle ─────────────────────────────────────────
st.markdown(get_theme_css(T), unsafe_allow_html=True)
st.markdown(get_navbar_toggle_html(T), unsafe_allow_html=True)


# ─── Shared API Functions ───────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_available_models():
    """Fetch list of supported model names."""
    try:
        res = requests.get(f"{API_BASE_URL}/available_models", timeout=10)
        if res.status_code == 200:
            return res.json().get("models", [])
    except Exception:
        pass
    return ["ARIMA", "SARIMAX", "SARIMA", "LSTM", "GRU", "Prophet", "LightGBM", "LinearRegression", "RandomForest"]


@st.cache_data(ttl=60, show_spinner=False)
def fetch_available_tickers():
    """Fetch list of tickers with trained models."""
    try:
        res = requests.get(f"{API_BASE_URL}/available_tickers", timeout=10)
        if res.status_code == 200:
            return res.json().get("tickers", [])
    except Exception:
        pass
    return []


@st.cache_data(ttl=5, show_spinner=False)
def fetch_system_status():
    """Fetch system status."""
    try:
        res = requests.get(f"{API_BASE_URL}/status", timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        return {"status": "error", "message": str(e), "models_available": [], "tickers_trained": []}
    return {"status": "unknown", "message": "Unable to connect to backend", "models_available": [], "tickers_trained": []}


def check_backend_connection():
    """Check if backend is running."""
    status = fetch_system_status()
    return status.get("status") == "operational"


# ─── Sidebar Navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<h2 style='color:{T['accent']}; font-size:1.4rem; margin-bottom:0;'>📈 ForexAI Pro</h2>"
        f"<p style='color:{T['text_secondary']}; font-size:0.75rem; margin-top:4px;'>Powered by ML Inference Engine</p>",
        unsafe_allow_html=True
    )
    st.divider()
    
    # Navigation
    st.markdown(f"<p style='color:{T['text_secondary']}; font-size:0.8rem; font-weight:700;'>NAVIGATION</p>", unsafe_allow_html=True)
    
    # Page selection via radio buttons
    page = st.radio(
        "Go to",
        ["Home", "Model Management", "Market Data", "System Status"],
        label_visibility="collapsed"
    )
    
    st.divider()
    st.caption(f"Backend mode: {BACKEND_MODE}")
    
    # Quick Stats in Sidebar
    st.markdown(f"<p style='color:{T['text_secondary']}; font-size:0.8rem; font-weight:700;'>QUICK STATS</p>", unsafe_allow_html=True)
    
    # Fetch system status for sidebar
    status_data = fetch_system_status()
    trained_count = len(status_data.get("tickers_trained", []))
    models_count = len(status_data.get("models_available", []))
    
    st.metric("Trained Tickers", trained_count)
    st.metric("Available Models", models_count)
    
    st.divider()
    
    # System Status Indicator
    is_connected = check_backend_connection()
    if is_connected:
        st.markdown(
            f"<div style='background:{T['status_bg']}; border:1px solid {T['status_border']}; "
            f"border-radius:8px; padding:10px; text-align:center;'>"
            f"<span style='color:{T['status_color']}; font-weight:700; font-size:0.85rem;'>● SYSTEM OPERATIONAL</span>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='background:rgba(255,82,82,0.1); border:1px solid #ff5252; "
            f"border-radius:8px; padding:10px; text-align:center;'>"
            f"<span style='color:#ff5252; font-weight:700; font-size:0.85rem;'>● BACKEND OFFLINE</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='color:#ff5252; font-size:0.75rem; text-align:center;'>Backend service unavailable</p>",
            unsafe_allow_html=True
        )


# ─── Page Routing ───────────────────────────────────────────────────────────────
if page == "Home":
    # ─── Home Page ─────────────────────────────────────────────────────────────
    st.markdown(
        f"<h1 style='color:{T['heading_color']}; font-size:2rem; margin-bottom:4px;'>📉 Forex Predictive Analytics</h1>"
        f"<p style='color:{T['text_secondary']}; font-size:0.9rem; margin-top:0;'>Real-time AI-powered currency forecasting dashboard</p>",
        unsafe_allow_html=True
    )
    st.divider()
    
    # Welcome Message
    st.markdown(f"""
    <div style='background:{T['table_bg']}; border:1px solid {T['card_border']}; 
                border-radius:12px; padding:20px; margin-bottom:20px;'>
        <h3 style='color:{T['heading_color']}; margin-top:0;'>Welcome to ForexAI Pro</h3>
        <p style='color:{T['text_primary']};'>
            This application provides comprehensive Forex forecasting capabilities powered by 
            machine learning models. Navigate through the sections below to explore all features.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style='background:{T['table_bg']}; border:1px solid {T['card_border']}; 
                    border-radius:12px; padding:20px; height:100%;'>
            <h4 style='color:{T['accent']}; margin-top:0;'>🔮 Prediction</h4>
            <p style='color:{T['text_primary']}; font-size:0.9rem;'>
                Generate forecasts using pre-trained models. Select your currency pair, 
                choose a model, and get AI-powered predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background:{T['table_bg']}; border:1px solid {T['card_border']}; 
                    border-radius:12px; padding:20px; height:100%;'>
            <h4 style='color:{T['accent']}; margin-top:0;'>🎓 Training</h4>
            <p style='color:{T['text_primary']}; font-size:0.9rem;'>
                Train individual models or all supported models on historical data. 
                Monitor training progress and view performance metrics.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='background:{T['table_bg']}; border:1px solid {T['card_border']}; 
                    border-radius:12px; padding:20px; height:100%;'>
            <h4 style='color:{T['accent']}; margin-top:0;'>📊 Market Data</h4>
            <p style='color:{T['text_primary']}; font-size:0.9rem;'>
                View historical OHLCV data with interactive charts. Select date ranges 
                and visualize price trends.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown(f"""
        <div style='background:{T['table_bg']}; border:1px solid {T['card_border']}; 
                    border-radius:12px; padding:20px; height:100%;'>
            <h4 style='color:{T['accent']}; margin-top:0;'>⚙️ System Status</h4>
            <p style='color:{T['text_primary']}; font-size:0.9rem;'>
                Monitor system health, view trained models, and check available tickers. 
                Get detailed system information.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div style='background:{T['table_bg']}; border:1px solid {T['card_border']}; 
                    border-radius:12px; padding:20px; height:100%;'>
            <h4 style='color:{T['accent']}; margin-top:0;'>🧠 Model Management</h4>
            <p style='color:{T['text_primary']}; font-size:0.9rem;'>
                Browse all available models and see which tickers have trained models. 
                View model details and training information.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown(f"""
        <div style='background:{T['table_bg']}; border:1px solid {T['card_border']}; 
                    border-radius:12px; padding:20px; height:100%;'>
            <h4 style='color:{T['accent']}; margin-top:0;'>📅 Training Period</h4>
            <p style='color:{T['text_primary']}; font-size:0.9rem;'>
                View the training period for any ticker. See when models were trained 
                and the data range used.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    
    # Quick Actions
    st.markdown(
        f"<h3 style='color:{T['heading_color']}; margin-top:0;'>🚀 Quick Actions</h3>",
        unsafe_allow_html=True
    )
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔮 Generate Forecast", use_container_width=True, key="home_forecast"):
            st.session_state['page_redirect'] = "🔮 Prediction"
            st.rerun()
    
    with col2:
        if st.button("🎓 Train Model", use_container_width=True, key="home_train"):
            st.session_state['page_redirect'] = "🎓 Training"
            st.rerun()
    
    with col3:
        if st.button("📊 View Market Data", use_container_width=True, key="home_market"):
            st.session_state['page_redirect'] = "📊 Market Data"
            st.rerun()
    
    with col4:
        if st.button("⚙️ Check Status", use_container_width=True, key="home_status"):
            st.session_state['page_redirect'] = "⚙️ System Status"
            st.rerun()
    
    # System Status Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📈 System Overview")
    
    status = fetch_system_status()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("System Status", status.get("status", "Unknown").title())
    col2.metric("Available Models", len(status.get("models_available", [])))
    col3.metric("Trained Tickers", len(status.get("tickers_trained", [])))
    col4.metric("Backend", "Online ✓" if is_connected else "Offline ✗")
    
    if HAS_METRIC_CARDS:
        try:
            style_metric_cards(
                background_color=T['table_bg'],
                border_left_color=T['accent'],
                border_color=T['card_border'],
                box_shadow=True
            )
        except Exception:
            pass
    
    # Recent Tickers
    if status.get("tickers_trained"):
        st.markdown(f"<p style='color:{T['text_secondary']}; font-size:0.9rem;'>Trained Tickers: {', '.join(status['tickers_trained'][:5])}{'...' if len(status.get('tickers_trained', [])) > 5 else ''}</p>", unsafe_allow_html=True)

elif page == "Model Management":
    # Import and run the Model Management page (merged from Prediction and Training)
    from pages.model_management import render_model_management_page
    render_model_management_page(T, API_BASE_URL)

elif page == "Market Data":
    # Import and run the Market Data page
    from pages.market_data import render_market_data_page
    render_market_data_page(T, API_BASE_URL)

elif page == "System Status":
    # Import and run the System Status page
    from pages.system_status import render_system_status_page
    render_system_status_page(T, API_BASE_URL)
