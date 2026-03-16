"""
ForexAI Pro - Minimal frontend shell with sidebar navigation and page routing.
"""

import requests
import streamlit as st
try:
    from streamlit_option_menu import option_menu
except Exception:
    option_menu = None

from utils.backend_adapter import configure_backend
from utils.runtime_cache import runtime_safe_cache_data
from utils.theme import THEMES, get_navbar_toggle_html, get_theme_css


API_BASE_URL, BACKEND_MODE = configure_backend()

st.set_page_config(layout="wide", page_title="ForexAI Pro", page_icon="📈")

params = st.query_params
if "theme" in params and params["theme"] in THEMES:
    st.session_state.theme = params["theme"]
elif "theme" not in st.session_state:
    st.session_state.theme = "dark"

T = THEMES.get(st.session_state.theme, THEMES["dark"])
st.markdown(get_theme_css(T), unsafe_allow_html=True)
st.markdown(get_navbar_toggle_html(T), unsafe_allow_html=True)


@runtime_safe_cache_data(ttl=5, show_spinner=False)
def _cached_system_status(api_base_url: str):
    try:
        res = requests.get(f"{api_base_url}/status", timeout=8)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return {"status": "offline", "models_available": [], "tickers_trained": []}


PAGES = ["Market Data", "Model Management", "System Status", "About"]
PAGE_OPTIONS = ["market_data", "model_management", "system_status", "about"]
PAGE_LABELS = {
    "market_data": "📊 Market Data Terminal",
    "model_management": "🧠 Model Management",
    "system_status": "⚙️ System Status",
    "about": "ℹ️ About Project",
}

with st.sidebar:
    st.markdown(
        f"<h2 style='color:{T['accent']}; margin-bottom:0;'>📈 ForexAI Pro</h2>"
        f"<p style='color:{T['text_secondary']}; font-size:0.78rem; margin-top:4px;'>"
        f"AI Forex Terminal</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <style>
        .sidebar-nav-label {{
            color: {T['text_secondary']};
            font-size: 0.76rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 0 0 0.3rem 0;
            font-weight: 700;
        }}
        .about-badge {{
            border: 1px solid {T['card_border']};
            border-radius: 10px;
            background: linear-gradient(110deg, {T['table_bg']} 0%, {T['plot_bg']} 100%);
            padding: 8px 10px;
            margin-top: 8px;
            color: {T['text_secondary']};
            font-size: 0.8rem;
        }}
        </style>
        <p class='sidebar-nav-label'>Navigation</p>
        """,
        unsafe_allow_html=True,
    )

    if option_menu is not None:
        page_display_options = [PAGE_LABELS.get(key, key) for key in PAGE_OPTIONS]
        selected_display = option_menu(
            menu_title=None,
            options=page_display_options,
            icons=["graph-up-arrow", "cpu", "activity", "info-circle"],
            default_index=0,
            styles={
                "container": {
                    "padding": "6px",
                    "background-color": T["table_bg"],
                    "border": f"1px solid {T['card_border']}",
                    "border-radius": "12px",
                },
                "icon": {
                    "color": T["text_secondary"],
                    "font-size": "1rem",
                },
                "nav-link": {
                    "font-size": "0.9rem",
                    "text-align": "left",
                    "margin": "4px 0",
                    "padding": "10px 12px",
                    "border-radius": "8px",
                    "color": T["text_primary"],
                    "border": "1px solid transparent",
                    "--hover-color": T["alert_bg"],
                },
                "nav-link-selected": {
                    "background-color": T["accent2"],
                    "color": "white",
                    "font-weight": "600",
                    "border": f"1px solid {T['accent']}",
                },
            },
        )
        selected_index = page_display_options.index(selected_display)
        page_key = PAGE_OPTIONS[selected_index]
    else:
        # Fallback when the optional dependency is unavailable.
        page_key = st.radio(
            "Navigation",
            PAGE_OPTIONS,
            index=0,
            format_func=lambda value: PAGE_LABELS.get(value, value),
            label_visibility="collapsed",
        )

    if page_key == "about":
        st.markdown("<div class='about-badge'>Project architecture and release status</div>", unsafe_allow_html=True)

    st.divider()
    st.caption(f"Backend mode: {BACKEND_MODE}")

    status = _cached_system_status(API_BASE_URL)
    models_count = len(status.get("models_available", []))
    tickers_count = len(status.get("tickers_trained", []))

    st.metric("Models", models_count)
    st.metric("Tickers", tickers_count)

    is_online = status.get("status") == "operational"
    if is_online:
        st.success("System operational")
    else:
        st.error("Backend offline")


if page_key == "market_data":
    from pages.market_data import render_market_data_page

    render_market_data_page(T, API_BASE_URL)
elif page_key == "model_management":
    from pages.model_management import render_model_management_page

    render_model_management_page(T, API_BASE_URL)
elif page_key == "system_status":
    from pages.system_status import render_system_status_page

    render_system_status_page(T, API_BASE_URL)
else:
    from pages.about import render_about_page

    render_about_page(T, API_BASE_URL, BACKEND_MODE)
