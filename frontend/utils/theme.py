"""
Theme definitions and CSS generation for ForexAI Pro.

Usage in app.py:
    from utils.theme import THEMES, get_theme_css, get_navbar_toggle_html

    T = THEMES[st.session_state.theme]
    st.markdown(get_theme_css(T), unsafe_allow_html=True)
    st.markdown(get_navbar_toggle_html(T), unsafe_allow_html=True)
"""

# ─── Theme Color Palettes ────────────────────────────────────────────────────────
THEMES = {
    "dark": {
        "app_bg":            "linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%)",
        "sidebar_bg":        "linear-gradient(180deg, #0d1b2a 0%, #112240 100%)",
        "sidebar_border":    "#1e3a5f",
        "text_primary":      "#e0e6f0",
        "text_secondary":    "#64b5f6",
        "accent":            "#00d4ff",
        "accent2":           "#0077b6",
        "accent3":           "#00b4d8",
        "positive":          "#00e676",
        "card_bg":           "rgba(13, 27, 42, 0.9), rgba(17, 34, 64, 0.9)",
        "card_border":       "#1e3a5f",
        "card_shadow":       "rgba(0, 212, 255, 0.08)",
        "card_hover_shadow": "rgba(0, 212, 255, 0.2)",
        "tab_bg":            "rgba(13, 27, 42, 0.8)",
        "tab_border":        "#1e3a5f",
        "tab_text":          "#64b5f6",
        "table_bg":          "rgba(13, 27, 42, 0.8)",
        "table_border":      "#1e3a5f",
        "table_th_bg":       "linear-gradient(135deg, #0d1b2a, #112240)",
        "table_th_color":    "#64b5f6",
        "table_td_color":    "#e0e6f0",
        "hr_color":          "#1e3a5f",
        "alert_bg":          "rgba(0, 119, 182, 0.15)",
        "alert_border":      "#0077b6",
        "alert_color":       "#90caf9",
        "plot_paper":        "rgba(10, 14, 26, 0)",
        "plot_bg":           "rgba(13, 27, 42, 0.6)",
        "plot_grid":         "rgba(30, 58, 95, 0.5)",
        "plot_axis_color":   "#64b5f6",
        "plot_legend_bg":    "rgba(13, 27, 42, 0.8)",
        "plot_legend_border":"#1e3a5f",
        "plot_template":     "plotly_dark",
        "status_bg":         "rgba(0,230,118,0.1)",
        "status_border":     "#00e676",
        "status_color":      "#00e676",
        "heading_color":     "#e0f7fa",
        "subheading_color":  "#64b5f6",
        "navbar_bg":         "#0d1b2a",
        "navbar_border":     "#1e3a5f",
        "navbar_text":       "#64b5f6",
        "navbar_icon":       "#00d4ff",
        "toggle_icon":       "☀️",
        "toggle_label":      "Switch to Light Mode",
        "name":              "dark",
    },
    "light": {
        "navbar_icon":       "#0077b6",
        "toggle_icon":       "🌙",
        "toggle_label":      "Switch to Dark Mode",
        "name":              "light",
        "app_bg":            "linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 50%, #f5f7ff 100%)",
        "sidebar_bg":        "linear-gradient(180deg, #ffffff 0%, #e8f0fe 100%)",
        "sidebar_border":    "#c5d8f5",
        "text_primary":      "#1a2340",
        "text_secondary":    "#1565c0",
        "accent":            "#0077b6",
        "accent2":           "#0077b6",
        "accent3":           "#00b4d8",
        "positive":          "#2e7d32",
        "card_bg":           "rgba(255, 255, 255, 0.95), rgba(232, 240, 254, 0.95)",
        "card_border":       "#c5d8f5",
        "card_shadow":       "rgba(0, 119, 182, 0.08)",
        "card_hover_shadow": "rgba(0, 119, 182, 0.25)",
        "tab_bg":            "rgba(255, 255, 255, 0.9)",
        "tab_border":        "#c5d8f5",
        "tab_text":          "#1565c0",
        "table_bg":          "rgba(255, 255, 255, 0.95)",
        "table_border":      "#c5d8f5",
        "table_th_bg":       "linear-gradient(135deg, #e8f0fe, #dce8fd)",
        "table_th_color":    "#1565c0",
        "table_td_color":    "#1a2340",
        "hr_color":          "#c5d8f5",
        "alert_bg":          "rgba(21, 101, 192, 0.08)",
        "alert_border":      "#1565c0",
        "alert_color":       "#1565c0",
        "plot_paper":        "rgba(240, 244, 255, 0)",
        "plot_bg":           "rgba(255, 255, 255, 0.8)",
        "plot_grid":         "rgba(197, 216, 245, 0.8)",
        "plot_axis_color":   "#1565c0",
        "plot_legend_bg":    "rgba(255, 255, 255, 0.9)",
        "plot_legend_border":"#c5d8f5",
        "plot_template":     "plotly_white",
        "status_bg":         "rgba(46,125,50,0.1)",
        "status_border":     "#2e7d32",
        "status_color":      "#2e7d32",
        "heading_color":     "#0d1b2a",
        "subheading_color":  "#1565c0",
        "navbar_bg":         "#ffffff",
        "navbar_border":     "#c5d8f5",
        "navbar_text":       "#1565c0",
        "navbar_icon":       "#0077b6",
        "toggle_icon":       "🌙",
        "toggle_label":      "Switch to Dark Mode",
    },
}


def get_theme_css(T: dict) -> str:
    """
    Generate the full theme-driven CSS string for injection via st.markdown.

    Args:
        T: Active theme dictionary from THEMES.

    Returns:
        HTML <style> block as a string.
    """
    return f"""
    <style>
    /* ── Override Streamlit CSS variables ── */
    :root {{
        --background-color: {T['app_bg']} !important;
        --secondary-background-color: {T['table_bg']} !important;
        --text-color: {T['text_primary']} !important;
        --font: sans-serif;
    }}

    /* ── Global Background & Text ── */
    .stApp,
    .stApp > div,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > section,
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    .main .block-container {{
        background: {T['app_bg']} !important;
        color: {T['text_primary']} !important;
    }}

    /* ── All text elements (scoped to app, not fixed overlays) ── */
    .stApp p, .stApp span, .stApp label, .stApp li {{
        color: {T['text_primary']};
    }}
    .stMarkdown, .stMarkdown p, .stMarkdown span {{
        color: {T['text_primary']} !important;
    }}

    /* ── Protect fixed-position theme toggle from color overrides ── */
    a#theme-toggle-btn {{
        color: {T['navbar_icon']} !important;
        text-decoration: none !important;
        background: {T['navbar_bg']} !important;
        border: 1px solid {T['navbar_border']} !important;
    }}

    /* ── Input fields (selectbox, text input, number input) ── */
    [data-testid="stSelectbox"] > div > div,
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {{
        background: {T['table_bg']} !important;
        color: {T['text_primary']} !important;
        border-color: {T['card_border']} !important;
    }}
    [data-testid="stNumberInput"] input::placeholder,
    [data-testid="stTextInput"] input::placeholder {{
        color: {T['text_secondary']} !important;
        opacity: 1 !important;
    }}

    /* ── Dropdown option list ── */
    [data-baseweb="popover"] ul,
    [data-baseweb="menu"] {{
        background: {T['table_bg']} !important;
        border: 1px solid {T['card_border']} !important;
    }}
    [data-baseweb="menu"] li,
    [data-baseweb="option"] {{
        color: {T['text_primary']} !important;
        background: {T['table_bg']} !important;
    }}
    [data-baseweb="option"]:hover {{
        background: {T['tab_bg']} !important;
    }}
    [data-baseweb="option"][aria-selected="true"] {{
        background: {T['accent']} !important;
        color: {T['text_primary']} !important;
    }}

    /* ── Native HTML select/option styling (handles some Streamlit widgets) ── */
    select,
    select option {{
        background: {T['table_bg']} !important;
        color: {T['text_primary']} !important;
    }}

    /* ── BaseWeb/Streamlit option styling (various DOM patterns) ── */
    div[role="option"],
    div[class*="menu"] div[role="option"],
    div[class*="menu"] li,
    div[class*="menu"] div[class*="item"],
    div[class*="menu"] div[class*="option"] {{
        background: {T['table_bg']} !important;
        color: {T['text_primary']} !important;
    }}

    div[role="option"][aria-selected="true"],
    div[class*="menu"] div[aria-selected="true"],
    div[class*="menu"] li[aria-selected="true"] {{
        background: {T['accent']} !important;
        color: {T['text_primary']} !important;
    }}

    /* ── Radio buttons ── */
    [data-testid="stRadio"] label {{
        color: {T['text_primary']} !important;
    }}
    [data-testid="stRadio"] [data-baseweb="radio"] div {{
        border-color: {T['accent']} !important;
    }}

    /* ── Slider ── */
    [data-testid="stSlider"] label {{
        color: {T['text_primary']} !important;
    }}
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
        background: {T['accent']} !important;
    }}

    /* ── Checkbox ── */
    [data-testid="stCheckbox"] label {{
        color: {T['text_primary']} !important;
    }}

    /* ── Containers / Expanders ── */
    [data-testid="stExpander"],
    [data-testid="stExpander"] > div {{
        background: {T['table_bg']} !important;
        border: 1px solid {T['card_border']} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stExpander"] summary {{
        color: {T['text_primary']} !important;
    }}

    /* ── Code blocks ── */
    [data-testid="stCode"] pre,
    .stCode pre {{
        background: {T['table_bg']} !important;
        color: {T['text_primary']} !important;
        border: 1px solid {T['card_border']} !important;
    }}

    /* ── Spinner text ── */
    [data-testid="stSpinner"] p {{
        color: {T['text_secondary']} !important;
    }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background: {T['sidebar_bg']};
        border-right: 1px solid {T['sidebar_border']};
    }}
    [data-testid="stSidebar"] * {{
        color: {T['text_primary']};
    }}
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSlider label {{
        color: {T['text_secondary']} !important;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }}

    /* ── Metric Cards ── */
    [data-testid="stMetric"] {{
        background: linear-gradient(135deg, {T['card_bg']});
        border: 1px solid {T['card_border']};
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 20px {T['card_shadow']};
        transition: box-shadow 0.3s ease;
    }}
    [data-testid="stMetric"]:hover {{
        box-shadow: 0 4px 30px {T['card_hover_shadow']};
        border-color: {T['accent']};
    }}
    [data-testid="stMetricLabel"] {{
        color: {T['text_secondary']} !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {T['heading_color']} !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricDelta"] {{
        color: {T['positive']} !important;
        font-size: 0.85rem !important;
    }}

    /* ── Buttons (general) ── */
    .stButton > button {{
        background: linear-gradient(135deg, {T['accent2']}, {T['accent3']});
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.05em;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px {T['card_shadow']};
    }}
    .stButton > button:hover {{
        background: linear-gradient(135deg, {T['accent3']}, {T['accent']});
        box-shadow: 0 6px 25px {T['card_hover_shadow']};
        transform: translateY(-1px);
    }}

    /* ── Minimalistic Theme Toggle Button (last column) ── */
    div[data-testid="column"]:last-child .stButton > button {{
        background: transparent !important;
        border: 1px solid {T['card_border']} !important;
        border-radius: 50% !important;
        width: 36px !important;
        height: 36px !important;
        min-height: 36px !important;
        padding: 0 !important;
        font-size: 1.1rem !important;
        box-shadow: none !important;
        color: {T['navbar_icon']} !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: border-color 0.2s ease, background 0.2s ease !important;
        transform: none !important;
    }}
    div[data-testid="column"]:last-child .stButton > button:hover {{
        background: {T['tab_bg']} !important;
        border-color: {T['accent']} !important;
        box-shadow: 0 0 8px {T['card_shadow']} !important;
        transform: none !important;
    }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        background: {T['tab_bg']};
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
        border: 1px solid {T['tab_border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {T['tab_text']};
        font-weight: 600;
        border-radius: 8px;
        padding: 8px 20px;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {T['accent2']}, {T['accent3']}) !important;
        color: #ffffff !important;
    }}

    /* ── Tables ── */
    .stTable table {{
        background: {T['table_bg']};
        border: 1px solid {T['table_border']};
        border-radius: 10px;
        overflow: hidden;
    }}
    .stTable th {{
        background: {T['table_th_bg']};
        color: {T['table_th_color']} !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.8rem;
        border-bottom: 1px solid {T['table_border']} !important;
    }}
    .stTable td {{
        color: {T['table_td_color']} !important;
        border-bottom: 1px solid {T['table_border']} !important;
    }}

    /* ── Divider ── */
    hr {{
        border-color: {T['hr_color']} !important;
    }}

    /* ── Alert / Info boxes ── */
    .stAlert {{
        background: {T['alert_bg']} !important;
        border: 1px solid {T['alert_border']} !important;
        border-radius: 8px !important;
        color: {T['alert_color']} !important;
    }}

    /* ── Headings ── */
    h1, h2, h3 {{
        color: {T['heading_color']} !important;
    }}

    /* ── Top Navigation / Header Bar ── */
    header[data-testid="stHeader"] {{
        background: {T['navbar_bg']} !important;
        border-bottom: 1px solid {T['navbar_border']} !important;
    }}
    [data-testid="stToolbar"] {{
        background: {T['navbar_bg']} !important;
    }}
    [data-testid="stToolbar"] button,
    [data-testid="stToolbar"] svg {{
        color: {T['navbar_icon']} !important;
        fill: {T['navbar_icon']} !important;
    }}
    [data-testid="stDecoration"] {{
        background: {T['navbar_bg']} !important;
    }}

    /* ── Hamburger / Main Menu ── */
    #MainMenu button,
    #MainMenu svg {{
        color: {T['navbar_icon']} !important;
        fill: {T['navbar_icon']} !important;
    }}

    /* ── Main Menu Popover ── */
    [data-testid="stMainMenuPopover"],
    [data-testid="stMainMenuPopover"] ul {{
        background: {T['navbar_bg']} !important;
        border: 1px solid {T['navbar_border']} !important;
    }}
    [data-testid="stMainMenuPopover"] li,
    [data-testid="stMainMenuPopover"] span {{
        color: {T['text_primary']} !important;
    }}
    [data-testid="stMainMenuPopover"] li:hover {{
        background: {T['tab_bg']} !important;
    }}

    /* ── Status Widget (running indicator) ── */
    [data-testid="stStatusWidget"] {{
        color: {T['navbar_text']} !important;
    }}
    [data-testid="stStatusWidget"] svg {{
        fill: {T['navbar_icon']} !important;
    }}
    </style>
    """


def get_navbar_toggle_html(T: dict) -> str:
    """
    Returns a fixed-position HTML button that sits in the top-right corner,
    visually inside the Streamlit nav bar. Clicking it navigates to
    ?theme=light or ?theme=dark, which app.py reads via st.query_params.

    Args:
        T: Active theme dictionary from THEMES.

    Returns:
        HTML string with the floating toggle button.
    """
    current_theme = T.get('name', 'dark')
    next_theme = "light" if T.get('name', 'dark') == "dark" else "dark"
    icon = T.get('toggle_icon', '☀️')
    label = T.get('toggle_label', '')
    bg = T['navbar_bg']
    border = T['navbar_border']
    color = T['navbar_icon']
    hover_bg = T['tab_bg']

    return f"""
    <a id="theme-toggle-btn" href="?theme={next_theme}" title="{label}" style="
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 9999;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: {bg};
        border: 1px solid {border};
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
        line-height: 1;
        text-decoration: none;
        color: {color};
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        opacity: 0.75;
        transition: opacity 0.2s ease, box-shadow 0.2s ease;
        cursor: pointer;
    " onmouseover="this.style.opacity='1'; this.style.boxShadow='0 6px 24px rgba(0,0,0,0.4)';"
       onmouseout="this.style.opacity='0.75'; this.style.boxShadow='0 4px 16px rgba(0,0,0,0.3)';">
        {icon}
    </a>
    """
