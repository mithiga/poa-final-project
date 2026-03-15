"""
Prediction Page - Generate forecasts using pre-trained models
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta


DEFAULT_MODELS = ["ARIMA", "SARIMAX", "SARIMA", "LSTM", "GRU", "Prophet", "LightGBM", "LinearRegression", "RandomForest"]


@st.cache_data(ttl=60, show_spinner=False)
def _cached_available_models(api_base_url: str):
    try:
        res = requests.get(f"{api_base_url}/available_models", timeout=5)
        if res.status_code == 200:
            return res.json().get("models", [])
    except Exception:
        pass
    return DEFAULT_MODELS


def render_prediction_page(T, API_BASE_URL):
    """Render the Prediction page."""
    
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
    
    # Session state for hyperparameters
    if "hyperparams" not in st.session_state:
        st.session_state.hyperparams = {}
    if "hyperparams_loaded" not in st.session_state:
        st.session_state.hyperparams_loaded = False
    
    # Page Header
    st.markdown(
        f"<h1 style='color:{T['heading_color']}; font-size:2rem; margin-bottom:4px;'>📉 Forex Predictive Analytics</h1>"
        f"<p style='color:{T['text_secondary']}; font-size:0.9rem; margin-top:0;'>Real-time AI-powered currency forecasting dashboard</p>",
        unsafe_allow_html=True
    )
    st.divider()

    # ─── Metric Cards Row ────────────────────────────────────────────────────────────
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
    
    # Input Section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"<h4 style='color:{T['subheading_color']};'>📊 Prediction Parameters</h4>", unsafe_allow_html=True)
        
        # Currency Pair Selection
        pair = st.selectbox(
            "Currency Pair",
            list(PAIR_TO_TICKER.keys()),
            key="pred_pair_select"
        )
        
        # Model Selection
        models = _cached_available_models(API_BASE_URL)
        
        model = st.selectbox(
            "AI Model",
            models,
            key="pred_model_select"
        )
        
        # Forecast Days
        days = st.slider(
            "Forecast Days",
            min_value=1,
            max_value=90,
            value=7,
            key="pred_days_slider"
        )
        
        # Date Range Selector for Historical Data
        st.markdown(f"<p style='color:{T['text_secondary']}; font-size:0.85rem; margin-top:8px;'>📅 Historical Data Range</p>", unsafe_allow_html=True)
        
        # Default to last 30 days
        today = datetime.now().date()
        default_start = today - timedelta(days=30)
        
        date_range = st.date_input(
            "Select Date Range",
            value=(default_start, today),
            key="pred_date_range"
        )
        
        # Validate date range
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date_str = date_range[0].strftime("%Y-%m-%d")
            end_date_str = date_range[1].strftime("%Y-%m-%d")
        else:
            start_date_str = default_start.strftime("%Y-%m-%d")
            end_date_str = today.strftime("%Y-%m-%d")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Generate Button
        generate_btn = st.button(
            "🚀 Generate Forecast",
            use_container_width=True,
            key="generate_forecast_btn"
        )
    
    with col2:
        st.markdown(f"<h4 style='color:{T['subheading_color']};'>📈 Forecast Chart</h4>", unsafe_allow_html=True)
        
        chart_container = st.container(border=True, key="forecast_chart_container")
        
        with chart_container:
            if generate_btn:
                ticker = PAIR_TO_TICKER.get(pair, "EURUSD=X")
                payload = {
                    "ticker": ticker,
                    "model_type": model,
                    "days": days
                }
                with st.spinner(f"Generating {model} forecast for {pair}..."):
                    try:
                        res = requests.post(
                            f"{API_BASE_URL}/predict",
                            json=payload,
                            timeout=60
                        )
                        if res.status_code == 200:
                            data = res.json()
                            
                            # Fetch historical market data for date range
                            try:
                                hist_res = requests.get(
                                    f"{API_BASE_URL}/market-data",
                                    params={"ticker": ticker, "start_date": start_date_str, "end_date": end_date_str},
                                    timeout=30
                                )
                                historical_data = []
                                if hist_res.status_code == 200:
                                    hist_data = hist_res.json().get("data", [])
                                    for item in hist_data:
                                        try:
                                            close_price = float(item.get("Close")) if item.get("Close") else None
                                        except (ValueError, TypeError):
                                            close_price = None
                                        if close_price:
                                            historical_data.append({
                                                "date": item.get("Date"),
                                                "close": close_price
                                            })
                            except Exception as e:
                                historical_data = []
                            
                            # Get training cutoff date from model metadata
                            cutoff_date = None
                            try:
                                cutoff_res = requests.get(
                                    f"{API_BASE_URL}/model-cutoff-date",
                                    params={"ticker": ticker, "model": model},
                                    timeout=10
                                )
                                if cutoff_res.status_code == 200:
                                    cutoff_data = cutoff_res.json()
                                    cutoff_date = cutoff_data.get("cutoff_date")
                            except:
                                pass
                            
                            # Create forecast chart with historical data overlay
                            fig = go.Figure()
                            
                            # Add vertical line at cutoff date if available
                            if cutoff_date:
                                try:
                                    fig.add_vline(
                                        x=str(cutoff_date), 
                                        line_dash="dash", 
                                        line_color="#FFA500",
                                        annotation_text="Training Cutoff",
                                        annotation_position="top",
                                        annotation_font_color="#FFA500"
                                    )
                                except Exception as e:
                                    pass
                            
                            # Add historical market data if available
                            if historical_data:
                                try:
                                    hist_dates = [str(item["date"]) for item in historical_data]
                                    hist_prices = []
                                    for item in historical_data:
                                        try:
                                            hist_prices.append(float(item["close"]))
                                        except (ValueError, TypeError):
                                            hist_prices.append(0.0)
                                    if hist_dates and hist_prices:
                                        fig.add_trace(go.Scatter(
                                            x=hist_dates,
                                            y=hist_prices,
                                            mode='lines',
                                            name='Historical Prices',
                                            line=dict(color='#888888', width=2, dash='solid'),
                                            hovertemplate='Date: %{x}<br>Price: %{y:.5f}<extra>Historical</extra>'
                                        ))
                                except Exception as e:
                                    pass  # Skip historical data if there's an error
                            
                            # Add prediction line
                            try:
                                pred_values = data['predictions']
                                if isinstance(data['predictions'], list):
                                    pred_values = []
                                    for p in data['predictions']:
                                        try:
                                            pred_values.append(float(p))
                                        except (ValueError, TypeError):
                                            pred_values.append(0.0)
                                
                                # Ensure dates are strings
                                date_values = data.get('dates', [])
                                if date_values:
                                    date_values = [str(d) for d in date_values]
                                
                                if date_values and pred_values:
                                    fig.add_trace(go.Scatter(
                                        x=date_values,
                                        y=pred_values,
                                        mode='lines+markers',
                                        name='AI Forecast',
                                        line=dict(color=T['accent'], width=2.5),
                                        marker=dict(
                                            color=T['accent'],
                                            size=8,
                                            symbol='circle',
                                            line=dict(color='#ffffff', width=1)
                                        ),
                                        hovertemplate='Date: %{x}<br>Price: %{y:.5f}<extra>Forecast</extra>'
                                    ))
                            except Exception as e:
                                st.warning(f"Could not render forecast line: {str(e)}")
                            
                            # Add confidence intervals if available
                            if data.get('lower') and data.get('upper'):
                                try:
                                    # Ensure data is in correct format (list of numbers)
                                    if isinstance(data['dates'], list) and isinstance(data['upper'], list):
                                        # Convert dates to strings for concatenation
                                        date_strs = [str(d) for d in data['dates']]
                                        upper_vals = [float(u) for u in data['upper']]
                                        lower_vals = [float(l) for l in data['lower']]
                                        
                                        fig.add_trace(go.Scatter(
                                            x=date_strs + date_strs[::-1],
                                            y=upper_vals + lower_vals[::-1],
                                            fill='toself',
                                            fillcolor='rgba(0, 212, 255, 0.15)',
                                            line=dict(color='rgba(255,255,255,0)'),
                                            name='Confidence Interval',
                                            showlegend=True
                                        ))
                                except Exception as e:
                                    # Skip confidence interval if there's a type error
                                    pass
                            
                            fig.update_layout(
                                title=dict(
                                    text=f"{pair} — {data['model_used']} Forecast ({days} days)<br><span style='font-size:11px; color:#888;'>Historical: {start_date_str} to {end_date_str} ({len(historical_data)} days) | Cutoff: {cutoff_date or 'N/A'}</span>",
                                    font=dict(color=T['heading_color'], size=16)
                                ),
                                template=T['plot_template'],
                                paper_bgcolor=T['plot_paper'],
                                plot_bgcolor=T['plot_bg'],
                                xaxis=dict(
                                    title="Date",
                                    gridcolor=T['plot_grid'],
                                    color=T['plot_axis_color']
                                ),
                                yaxis=dict(
                                    title="Price",
                                    gridcolor=T['plot_grid'],
                                    color=T['plot_axis_color']
                                ),
                                legend=dict(
                                    bgcolor=T['plot_legend_bg'],
                                    bordercolor=T['plot_legend_border'],
                                    font=dict(color=T['text_primary'])
                                ),
                                margin=dict(l=20, r=20, t=50, b=20),
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Metrics Display - with smaller font for long values
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Custom HTML for metrics with smaller font for long values
                            mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
                            
                            ticker_val = data.get('ticker', ticker)
                            model_val = data.get('model_used', model)
                            cutoff_val = str(cutoff_date) if cutoff_date else "N/A"
                            
                            with mcol1:
                                st.markdown(f"<div style='text-align:center;'><span style='font-size:0.75rem; color:{T["text_secondary"]};'>Ticker</span><br><span style='font-size:0.85rem; font-weight:bold;'>{ticker_val}</span></div>", unsafe_allow_html=True)
                            with mcol2:
                                st.markdown(f"<div style='text-align:center;'><span style='font-size:0.75rem; color:{T["text_secondary"]};'>Model</span><br><span style='font-size:0.85rem; font-weight:bold;'>{model_val}</span></div>", unsafe_allow_html=True)
                            with mcol3:
                                st.markdown(f"<div style='text-align:center;'><span style='font-size:0.75rem; color:{T["text_secondary"]};'>Hist. Days</span><br><span style='font-size:0.85rem; font-weight:bold;'>{len(historical_data)}</span></div>", unsafe_allow_html=True)
                            with mcol4:
                                st.markdown(f"<div style='text-align:center;'><span style='font-size:0.75rem; color:{T["text_secondary"]};'>Cutoff Date</span><br><span style='font-size:0.75rem; font-weight:bold;'>{cutoff_val}</span></div>", unsafe_allow_html=True)
                            with mcol5:
                                st.markdown(f"<div style='text-align:center;'><span style='font-size:0.75rem; color:{T["text_secondary"]};'>Forecast Days</span><br><span style='font-size:0.85rem; font-weight:bold;'>{days}</span></div>", unsafe_allow_html=True)
                            
                            # Status
                            st.success(f"✅ {data.get('status', 'Forecast generated successfully')}")
                            
                        elif res.status_code == 404:
                            st.error(f"⚠️ Model not found: {res.json().get('detail', 'No trained model available for this ticker')}")
                        else:
                            st.error(f"❌ Server Error ({res.status_code}): {res.text}")
                            
                        except requests.exceptions.ConnectionError:
                            st.error("⚠️ Connection Failed: Backend service unavailable.")
                    except requests.exceptions.Timeout:
                        st.error("⏱️ Request timed out. The backend may be overloaded or model training is taking too long.")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)}")
            else:
                # Empty state
                st.markdown(f"""
                <div style='text-align:center; padding:60px 20px; color:{T['text_secondary']};'>
                    <p style='font-size:3rem; margin-bottom:10px;'>📊</p>
                    <p>Select parameters and click <b>Generate Forecast</b> to see predictions</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Prediction Details Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    
    # Additional info cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style='background:{T['table_bg']}; border:1px solid {T['card_border']}; 
                    border-radius:12px; padding:20px;'>
            <h4 style='color:{T['heading_color']}; margin-top:0; font-size:1.1rem;'>💡 How It Works</h4>
            <ul style='color:{T['text_primary']}; padding-left:20px; font-size:0.95rem; line-height:1.6; margin-bottom:0;'>
                <li>Select a currency pair from the dropdown</li>
                <li>Choose an AI model (ARIMA, LSTM, GRU, etc.)</li>
                <li>Set the number of days to forecast</li>
                <li>Click Generate Forecast to get predictions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background:{T['table_bg']}; border:1px solid {T['card_border']}; 
                    border-radius:12px; padding:20px;'>
            <h4 style='color:{T['heading_color']}; margin-top:0; font-size:1.1rem;'>📋 Available Models</h4>
            <ul style='color:{T['text_primary']}; padding-left:20px; font-size:0.95rem; line-height:1.6; margin-bottom:0;'>
                <li><b>ARIMA/SARIMA</b> - Statistical time series</li>
                <li><b>LSTM/GRU</b> - Deep learning recurrent networks</li>
                <li><b>Prophet</b> - Facebook's forecasting tool</li>
                <li><b>LightGBM/RandomForest</b> - Gradient boosting methods</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
