"""
Model Management Page - Train models, generate forecasts, and manage trained tickers
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


@st.cache_data(ttl=60, show_spinner=False)
def _cached_available_tickers(api_base_url: str):
    try:
        res = requests.get(f"{api_base_url}/available_tickers", timeout=5)
        if res.status_code == 200:
            return res.json().get("tickers", [])
    except Exception:
        pass
    return []


@st.cache_data(ttl=60, show_spinner=False)
def _cached_hyperparameters(api_base_url: str, model: str):
    try:
        hp_response = requests.get(f"{api_base_url}/hyperparameters", params={"model": model}, timeout=5)
        if hp_response.status_code == 200:
            hp_data = hp_response.json()
            return hp_data.get("hyperparameters", [])
    except Exception:
        pass
    return []


def render_model_management_page(T, API_BASE_URL):
    """Render the Model Management page."""
    
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
    
    # Default date range (last 2 years for training)
    default_end = datetime.now()
    default_start = default_end - timedelta(days=730)
    
    # Session state for hyperparameters
    if "hyperparams" not in st.session_state:
        st.session_state.hyperparams = {}
    if "hyperparams_loaded" not in st.session_state:
        st.session_state.hyperparams_loaded = False
    
    # Page Header
    st.markdown(
        f"<h1 style='color:{T['heading_color']}; font-size:2rem; margin-bottom:4px;'>🧠 Model Management</h1>"
        f"<p style='color:{T['text_secondary']}; font-size:0.9rem; margin-top:0;'>Train ML models and generate AI-powered forecasts</p>",
        unsafe_allow_html=True
    )
    st.divider()
    
    # Main Tabs - Merged from Prediction and Training
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Forecast",
        "Train Single",
        "Train All",
        "Hyperparameters",
        "Available Models"
    ])
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1: FORECAST (with date range approach from prediction.py)
    # ═══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown(f"<h3 style='color:{T['subheading_color']};'>🔮 Generate Forecast</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{T['text_secondary']};'>Generate AI-powered predictions using pre-trained models</p>", unsafe_allow_html=True)
        
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
            default_hist_start = today - timedelta(days=30)
            
            date_range = st.date_input(
                "Select Date Range",
                value=(default_hist_start, today),
                key="pred_date_range"
            )
            
            # Validate date range
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date_str = date_range[0].strftime("%Y-%m-%d")
                end_date_str = date_range[1].strftime("%Y-%m-%d")
            else:
                start_date_str = default_hist_start.strftime("%Y-%m-%d")
                end_date_str = today.strftime("%Y-%m-%d")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Generate Button
            generate_btn = st.button(
                "Generate Forecast",
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
                                        pass
                                
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
                                        if isinstance(data['dates'], list) and isinstance(data['upper'], list):
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
                                
                                # Metrics Display
                                st.markdown("<br>", unsafe_allow_html=True)
                                
                                mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
                                
                                ticker_val = data.get('ticker', ticker)
                                model_val = data.get('model_used', model)
                                cutoff_val = str(cutoff_date) if cutoff_date else "N/A"
                                
                                with mcol1:
                                    st.markdown(f"<div style='text-align:center;'><span style='font-size:0.75rem; color:{T['text_secondary']};'>Ticker</span><br><span style='font-size:0.85rem; font-weight:bold;'>{ticker_val}</span></div>", unsafe_allow_html=True)
                                with mcol2:
                                    st.markdown(f"<div style='text-align:center;'><span style='font-size:0.75rem; color:{T['text_secondary']};'>Model</span><br><span style='font-size:0.85rem; font-weight:bold;'>{model_val}</span></div>", unsafe_allow_html=True)
                                with mcol3:
                                    st.markdown(f"<div style='text-align:center;'><span style='font-size:0.75rem; color:{T['text_secondary']};'>Hist. Days</span><br><span style='font-size:0.85rem; font-weight:bold;'>{len(historical_data)}</span></div>", unsafe_allow_html=True)
                                with mcol4:
                                    st.markdown(f"<div style='text-align:center;'><span style='font-size:0.75rem; color:{T['text_secondary']};'>Cutoff Date</span><br><span style='font-size:0.75rem; font-weight:bold;'>{cutoff_val}</span></div>", unsafe_allow_html=True)
                                with mcol5:
                                    st.markdown(f"<div style='text-align:center;'><span style='font-size:0.75rem; color:{T['text_secondary']};'>Forecast Days</span><br><span style='font-size:0.85rem; font-weight:bold;'>{days}</span></div>", unsafe_allow_html=True)
                                
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
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2: TRAIN SINGLE MODEL
    # ═══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown(f"<h3 style='color:{T['subheading_color']};'>📊 Train Single Model</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{T['text_secondary']};'>Train a specific model with custom parameters</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"<h4 style='color:{T['subheading_color']};'>📋 Training Parameters</h4>", unsafe_allow_html=True)
            
            # Currency Pair Selection
            pair = st.selectbox(
                "Currency Pair",
                list(PAIR_TO_TICKER.keys()),
                key="train_single_pair"
            )
            
            # Model Selection
            models = _cached_available_models(API_BASE_URL)
            
            model = st.selectbox(
                "Model to Train",
                models,
                key="train_single_model"
            )
            
            # Date Range
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date = st.date_input("Start Date", value=default_start, key="train_single_start")
            with col_date2:
                end_date = st.date_input("End Date", value=default_end, key="train_single_end")
            
            # Train Size
            train_size = st.slider(
                "Train Size (%)",
                min_value=50,
                max_value=95,
                value=80,
                key="train_single_size"
            ) / 100.0
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            train_btn = st.button(
                "Train Model",
                use_container_width=True,
                key="train_single_btn"
            )
        
        with col2:
            st.markdown(f"<h4 style='color:{T['subheading_color']};'>📈 Training Results</h4>", unsafe_allow_html=True)
            
            results_container = st.container(border=True, key="train_single_results")
            
            with results_container:
                if train_btn:
                    ticker = PAIR_TO_TICKER.get(pair, "EURUSD=X")
                    payload = {
                        "ticker": ticker,
                        "model": model,
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d"),
                        "train_size": train_size
                    }
                    
                    with st.spinner(f"Training {model} model for {pair}..."):
                        try:
                            res = requests.post(f"{API_BASE_URL}/train", json=payload, timeout=300)
                            
                            if res.status_code == 200:
                                data = res.json()
                                st.success(f"✅ {data.get('message', 'Training completed')}")
                                
                                mcol1, mcol2, mcol3 = st.columns(3)
                                mcol1.metric("RMSE", f"{data.get('rmse', 'N/A')}")
                                mcol2.metric("MAE", f"{data.get('mae', 'N/A')}")
                                mcol3.metric("MAPE", f"{data.get('mape', 'N/A')}")
                            else:
                                st.error(f"❌ Training failed: {res.text}")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.markdown(f"""
                    <div style='text-align:center; padding:50px 20px; color:{T['text_secondary']};'>
                        <p style='font-size:3rem; margin-bottom:10px;'>🎓</p>
                        <p>Configure and click <b>Train Model</b></p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3: TRAIN ALL MODELS
    # ═══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown(f"<h3 style='color:{T['subheading_color']};'>🚀 Train All Models</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{T['text_secondary']};'>Train all supported models simultaneously for comparison</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"<h4 style='color:{T['subheading_color']};'>📋 Training Parameters</h4>", unsafe_allow_html=True)
            
            pair = st.selectbox(
                "Currency Pair",
                list(PAIR_TO_TICKER.keys()),
                key="train_all_pair"
            )
            
            # Show available models
            models = [m for m in _cached_available_models(API_BASE_URL) if m != "SARIMA"]
            
            st.markdown(f"<p style='color:{T['accent']}; font-size:0.9rem;'><b>{len(models)} Models:</b> {', '.join(models)}</p>", unsafe_allow_html=True)
            
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date = st.date_input("Start Date", value=default_start, key="train_all_start")
            with col_date2:
                end_date = st.date_input("End Date", value=default_end, key="train_all_end")
            
            train_size = st.slider(
                "Train Size (%)",
                min_value=50,
                max_value=95,
                value=80,
                key="train_all_size"
            ) / 100.0
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            train_all_btn = st.button(
                "Train All Models",
                use_container_width=True,
                key="train_all_btn"
            )
        
        with col2:
            st.markdown(f"<h4 style='color:{T['subheading_color']};'>📈 Training Results</h4>", unsafe_allow_html=True)
            
            results_container = st.container(border=True, key="train_all_results")
            
            with results_container:
                if train_all_btn:
                    ticker = PAIR_TO_TICKER.get(pair, "EURUSD=X")
                    payload = {
                        "ticker": ticker,
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d"),
                        "train_size": train_size
                    }
                    
                    with st.spinner(f"Training all {len(models)} models for {pair}..."):
                        try:
                            res = requests.post(f"{API_BASE_URL}/train_all", json=payload, timeout=600)
                            
                            if res.status_code == 200:
                                data = res.json()
                                st.success(f"✅ All models trained successfully!")
                                
                                if "evaluation" in data:
                                    eval_data = data["evaluation"]
                                    metrics_list = []
                                    for model_name, metrics in eval_data.items():
                                        if isinstance(metrics, dict):
                                            param_source = metrics.get("parameter_source", "default")
                                            metrics_list.append({
                                                "Model": model_name,
                                                "RMSE": f"{metrics.get('rmse', 'N/A'):.4f}" if isinstance(metrics.get('rmse'), (int, float)) else "N/A",
                                                "MAE": f"{metrics.get('mae', 'N/A'):.4f}" if isinstance(metrics.get('mae'), (int, float)) else "N/A",
                                                "MAPE": f"{metrics.get('mape', 'N/A'):.2f}%" if isinstance(metrics.get('mape'), (int, float)) else "N/A",
                                                "Params": "Saved Best CV" if param_source == "saved_best_cv" else "Default",
                                            })
                                    
                                    if metrics_list:
                                        df = pd.DataFrame(metrics_list)
                                        st.dataframe(df, use_container_width=True, hide_index=True)
                            else:
                                st.error(f"❌ Training failed: {res.text}")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.markdown(f"""
                    <div style='text-align:center; padding:50px 20px; color:{T['text_secondary']};'>
                        <p style='font-size:3rem; margin-bottom:10px;'>🚀</p>
                        <p>Configure and click <b>Train All Models</b></p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4: HYPERPARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown(f"<h3 style='color:{T['subheading_color']};'>⚙️ Hyperparameter Tuning</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{T['text_secondary']};'>Fine-tune model hyperparameters for optimal performance</p>", unsafe_allow_html=True)
        
        # Model selection for hyperparameters
        hp_model = st.selectbox(
            "Select Model",
            ["ARIMA", "SARIMAX", "LSTM", "GRU", "Prophet", "LightGBM", "LinearRegression", "RandomForest"],
            key="hp_model_select"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Fetch hyperparameters from API
        try:
            hyperparameters = _cached_hyperparameters(API_BASE_URL, hp_model)
            if hyperparameters:
                
                st.markdown(f"**Available Parameters for {hp_model}:**", unsafe_allow_html=True)
                
                # Create input fields for each hyperparameter
                hp_values = {}
                for hp in hyperparameters:
                    hp_name = hp.get("name", "")
                    hp_type = hp.get("type", "float")
                    hp_default = hp.get("default", 0)
                    hp_min = hp.get("min")
                    hp_max = hp.get("max")
                    hp_step = hp.get("step", 1)
                    hp_options = hp.get("options", [])
                    hp_desc = hp.get("description", "")
                    
                    col_hp1, col_hp2 = st.columns([3, 1])
                    
                    with col_hp1:
                        value = None
                        # Allow exact value or range for numeric parameters
                        if hp_type in ("int", "float"):
                            mode = st.radio(
                                f"{hp_name} mode",
                                ["Exact", "Range"],
                                horizontal=True,
                                key=f"hp_{hp_name}_mode"
                            )

                            if mode == "Exact":
                                if hp_type == "int":
                                    value = st.number_input(
                                        f"**{hp_name}** - {hp_desc}",
                                        min_value=int(hp_min) if hp_min is not None else 0,
                                        max_value=int(hp_max) if hp_max is not None else 1000,
                                        value=int(hp_default),
                                        step=int(hp_step),
                                        key=f"hp_{hp_name}"
                                    )
                                else:
                                    value = st.number_input(
                                        f"**{hp_name}** - {hp_desc}",
                                        min_value=float(hp_min) if hp_min is not None else 0.0,
                                        max_value=float(hp_max) if hp_max is not None else 100.0,
                                        value=float(hp_default),
                                        step=float(hp_step),
                                        key=f"hp_{hp_name}"
                                    )
                            else:
                                # Range (grid search)
                                if hp_type == "int":
                                    min_val = st.number_input(
                                        f"{hp_name} min",
                                        min_value=int(hp_min) if hp_min is not None else 0,
                                        max_value=int(hp_max) if hp_max is not None else 1000,
                                        value=int(hp_min) if hp_min is not None else 0,
                                        step=1,
                                        key=f"hp_{hp_name}_min"
                                    )
                                    max_val = st.number_input(
                                        f"{hp_name} max",
                                        min_value=min_val,
                                        max_value=int(hp_max) if hp_max is not None else 1000,
                                        value=int(hp_max) if hp_max is not None else min_val + 1,
                                        step=1,
                                        key=f"hp_{hp_name}_max"
                                    )
                                    step_val = st.number_input(
                                        f"{hp_name} step",
                                        min_value=1,
                                        max_value=100,
                                        value=int(hp_step),
                                        step=1,
                                        key=f"hp_{hp_name}_step"
                                    )
                                else:
                                    min_val = st.number_input(
                                        f"{hp_name} min",
                                        min_value=float(hp_min) if hp_min is not None else 0.0,
                                        max_value=float(hp_max) if hp_max is not None else 100.0,
                                        value=float(hp_min) if hp_min is not None else 0.0,
                                        step=float(hp_step),
                                        key=f"hp_{hp_name}_min"
                                    )
                                    max_val = st.number_input(
                                        f"{hp_name} max",
                                        min_value=min_val,
                                        max_value=float(hp_max) if hp_max is not None else 100.0,
                                        value=float(hp_max) if hp_max is not None else min_val + 1.0,
                                        step=float(hp_step),
                                        key=f"hp_{hp_name}_max"
                                    )
                                    step_val = st.number_input(
                                        f"{hp_name} step",
                                        min_value=0.0,
                                        max_value=100.0,
                                        value=float(hp_step),
                                        step=float(hp_step),
                                        key=f"hp_{hp_name}_step"
                                    )

                                value = {
                                    "range": [min_val, max_val, step_val]
                                }

                        elif hp_type == "categorical":
                            value = st.selectbox(
                                f"**{hp_name}** - {hp_desc}",
                                options=hp_options,
                                index=hp_options.index(hp_default) if hp_default in hp_options else 0,
                                key=f"hp_{hp_name}"
                            )
                        elif hp_type == "bool":
                            value = st.checkbox(
                                f"**{hp_name}** - {hp_desc}",
                                value=bool(hp_default),
                                key=f"hp_{hp_name}"
                            )
                        else:
                            value = st.text_input(hp_name, str(hp_default), key=f"hp_{hp_name}")
                        
                        hp_values[hp_name] = value
                    
                    with col_hp2:
                        st.caption(f"Default: {hp_default}")
                
                st.markdown("<hr>", unsafe_allow_html=True)
                
                # Training configuration
                col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
                with col_cfg1:
                    hp_pair = st.selectbox("Currency Pair", list(PAIR_TO_TICKER.keys()), key="hp_pair")
                with col_cfg2:
                    hp_start = st.date_input("Start Date", value=default_start, key="hp_start")
                with col_cfg3:
                    hp_end = st.date_input("End Date", value=default_end, key="hp_end")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Train with custom hyperparameters button
                hp_train_btn = st.button(
                    "🎯 Train with Custom Hyperparameters",
                    use_container_width=True,
                    key="hp_train_btn"
                )
                
                if hp_train_btn:
                    with st.spinner(f"Training {hp_model} with custom hyperparameters..."):
                        try:
                            ticker = PAIR_TO_TICKER.get(hp_pair, "EURUSD=X")
                            payload = {
                                "ticker": ticker,
                                "model": hp_model,
                                "start_date": hp_start.strftime("%Y-%m-%d"),
                                "end_date": hp_end.strftime("%Y-%m-%d"),
                                "train_size": 0.8,
                                "hyperparameters": hp_values
                            }

                            # Use tuning endpoint to run CV search and persist best params in backend.
                            res = requests.post(f"{API_BASE_URL}/train_with_tuning", json=payload, timeout=300)
                            
                            if res.status_code == 200:
                                data = res.json()
                                st.success(f"✅ Training completed with custom hyperparameters!")
                                
                                mcol1, mcol2, mcol3 = st.columns(3)
                                mcol1.metric("RMSE", f"{data.get('rmse', 'N/A')}")
                                mcol2.metric("MAE", f"{data.get('mae', 'N/A')}")
                                mcol3.metric("MAPE", f"{data.get('mape', 'N/A')}")
                            else:
                                st.error(f"❌ Training failed: {res.text}")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            else:
                st.warning(f"Could not load hyperparameters for {hp_model}")
        except Exception as e:
            st.error(f"❌ Error loading hyperparameters: {str(e)}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 5: AVAILABLE MODELS (from model_management.py)
    # ═══════════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown(f"<h3 style='color:{T['subheading_color']};'>📋 Available Models & Trained Tickers</h3>", unsafe_allow_html=True)
        
        # Fetch data
        with st.spinner("Fetching model information..."):
            # Fetch available models
            models = _cached_available_models(API_BASE_URL)
            tickers = _cached_available_tickers(API_BASE_URL)
        
        # Display models
        if models:
            st.markdown(f"<p style='color:{T['text_secondary']}; margin-bottom:15px;'>The following ML models are available for training and prediction:</p>", unsafe_allow_html=True)
            
            # Display as grid
            cols = st.columns(3)
            for i, model in enumerate(models):
                with cols[i % 3]:
                    model_category = ""
                    deep_learning = ["LSTM", "GRU"]
                    statistical = ["ARIMA", "SARIMAX", "SARIMA", "Prophet"]
                    tree_based = ["LightGBM", "RandomForest", "LinearRegression"]
                    
                    if model in deep_learning:
                        model_category = "🔮 Deep Learning"
                    elif model in statistical:
                        model_category = "📊 Statistical"
                    elif model in tree_based:
                        model_category = "🌲 Tree-Based"
                    else:
                        model_category = "⚙️ Other"
                    
                    st.markdown(f"""
                    <div style='background:{T['table_bg']}; border:1px solid {T['card_border']}; 
                                border-radius:12px; padding:15px; margin-bottom:10px;'>
                        <h4 style='color:{T['accent']}; margin:0 0 5px 0;'>{model}</h4>
                        <p style='color:{T['text_secondary']}; font-size:0.85rem; margin:0;'>{model_category}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Model comparison table
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color:{T['subheading_color']};'>📊 Model Comparison</h4>", unsafe_allow_html=True)
            
            comparison_data = []
            for model in models:
                deep_learning = ["LSTM", "GRU"]
                statistical = ["ARIMA", "SARIMAX", "SARIMA", "Prophet"]
                tree_based = ["LightGBM", "RandomForest", "LinearRegression"]
                
                if model in deep_learning:
                    category = "Deep Learning"
                    best_for = "Complex patterns, long-term trends"
                    speed = "Slow"
                elif model in statistical:
                    category = "Statistical"
                    best_for = "Seasonal data, short-term"
                    speed = "Fast"
                elif model in tree_based:
                    category = "Tree-Based"
                    best_for = "Feature-rich data"
                    speed = "Medium"
                else:
                    category = "Other"
                    best_for = "General use"
                    speed = "Medium"
                
                comparison_data.append({
                    "Model": model,
                    "Category": category,
                    "Best For": best_for,
                    "Speed": speed
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        
        # Display trained tickers
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:{T['subheading_color']};'>📈 Trained Tickers</h4>", unsafe_allow_html=True)
        
        if tickers:
            TICKER_TO_PAIR = {
                "EURUSD=X": "EUR/USD",
                "GBPUSD=X": "GBP/USD",
                "USDJPY=X": "USD/JPY",
                "USDCHF=X": "USD/CHF",
                "AUDUSD=X": "AUD/USD",
                "USDCAD=X": "USD/CAD",
                "NZDUSD=X": "NZD/USD",
                "EURGBP=X": "EUR/GBP",
                "EURJPY=X": "EUR/JPY",
                "GBPJPY=X": "GBP/JPY",
            }
            
            st.markdown(f"<p style='color:{T['text_secondary']}; margin-bottom:15px;'>The following currency pairs have trained models:</p>", unsafe_allow_html=True)
            
            cols = st.columns(4)
            for i, ticker in enumerate(tickers):
                pair = TICKER_TO_PAIR.get(ticker, ticker)
                with cols[i % 4]:
                    st.markdown(f"""
                    <div style='background:{T['table_bg']}; border:1px solid {T['card_border']}; 
                                border-radius:12px; padding:15px; margin-bottom:10px; text-align:center;'>
                        <h4 style='color:{T['accent']}; margin:0;'>{pair}</h4>
                        <p style='color:{T['text_secondary']}; font-size:0.75rem; margin:5px 0 0 0;'>{ticker}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            col_stat1, col_stat2 = st.columns(2)
            col_stat1.metric("Total Trained Tickers", len(tickers))
            col_stat2.metric("Available for Prediction", len(tickers))
        else:
            st.info("No tickers trained yet. Use the Train Single or Train All tabs to train models.")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BOTTOM SECTIONS: How It Works and Available Models (from prediction.py)
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    
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
                <li>Select historical date range for comparison</li>
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
