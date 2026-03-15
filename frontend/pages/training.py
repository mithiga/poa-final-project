"""
Training Page - Train models on historical data
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
def _cached_hyperparameters(api_base_url: str, model: str):
    try:
        hp_response = requests.get(f"{api_base_url}/hyperparameters", params={"model": model}, timeout=5)
        if hp_response.status_code == 200:
            hp_data = hp_response.json()
            return hp_data.get("hyperparameters", [])
    except Exception:
        pass
    return []


def render_training_page(T, API_BASE_URL):
    """Render the Training page."""
    
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
    
    # Default date range (last 2 years)
    default_end = datetime.now()
    default_start = default_end - timedelta(days=730)
    
    # Page Header
    st.markdown(
        f"<h1 style='color:{T['heading_color']}; font-size:2rem; margin-bottom:4px;'>🎓 Model Training</h1>"
        f"<p style='color:{T['text_secondary']}; font-size:0.9rem; margin-top:0;'>Train ML models on historical forex data</p>",
        unsafe_allow_html=True
    )
    st.divider()
    
    # ─── Main Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Quick Predict", 
        "📊 Train Single", 
        "🚀 Train All",
        "⚙️ Hyperparameters"
    ])
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1: QUICK PREDICT
    # ═══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown(f"<h3 style='color:{T['subheading_color']};'>🔮 Quick Forecast</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{T['text_secondary']};'>Generate immediate predictions using pre-trained models</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"<h4 style='color:{T['subheading_color']};'>📊 Parameters</h4>", unsafe_allow_html=True)
            
            # Currency Pair Selection
            pair = st.selectbox(
                "Currency Pair",
                list(PAIR_TO_TICKER.keys()),
                key="quick_pair_select"
            )
            
            # Model Selection
            models = _cached_available_models(API_BASE_URL)
            
            model = st.selectbox(
                "Model",
                models,
                key="quick_model_select"
            )
            
            # Forecast Days
            days = st.slider(
                "Forecast Days",
                min_value=1,
                max_value=90,
                value=7,
                key="quick_days_slider"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Predict Button
            predict_btn = st.button(
                "🔮 Generate Forecast",
                use_container_width=True,
                key="quick_predict_btn"
            )
        
        with col2:
            st.markdown(f"<h4 style='color:{T['subheading_color']};'>📈 Forecast Chart</h4>", unsafe_allow_html=True)
            
            chart_container = st.container(border=True, key="quick_chart_container")
            
            with chart_container:
                if predict_btn:
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
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=data['dates'],
                                    y=data['predictions'],
                                    mode='lines+markers',
                                    name='AI Forecast',
                                    line=dict(color=T['accent'], width=2.5),
                                    marker=dict(
                                        color=T['accent'],
                                        size=8,
                                        symbol='circle',
                                        line=dict(color='#ffffff', width=1)
                                    ),
                                    fill='tozeroy',
                                    fillcolor="rgba(0, 212, 255, 0.1)"
                                ))
                                
                                fig.update_layout(
                                    title=dict(
                                        text=f"{pair} — {data['model_used']} Forecast ({days} days)",
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
                                    height=350
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Metrics
                                mcol1, mcol2, mcol3 = st.columns(3)
                                mcol1.metric("RMSE", f"{data['metrics'].get('RMSE', 'N/A')}")
                                mcol2.metric("MAE", f"{data['metrics'].get('MAE', 'N/A')}")
                                mcol3.metric("Status", data.get('status', 'Success'))
                                
                            elif res.status_code == 404:
                                st.error(f"⚠️ No trained model found for {pair}. Please train a model first.")
                            else:
                                st.error(f"❌ Server Error ({res.status_code}): {res.text}")
                                
                        except requests.exceptions.ConnectionError:
                            st.error("⚠️ Connection Failed: Backend service unavailable.")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.markdown(f"""
                    <div style='text-align:center; padding:60px 20px; color:{T['text_secondary']};'>
                        <p style='font-size:3rem; margin-bottom:10px;'>🔮</p>
                        <p>Select parameters and click <b>Generate Forecast</b></p>
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
                "🎓 Train Model",
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
                "🚀 Train All Models",
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
                        if hp_type == "int":
                            value = st.number_input(
                                f"**{hp_name}** - {hp_desc}",
                                min_value=int(hp_min) if hp_min is not None else 0,
                                max_value=int(hp_max) if hp_max is not None else 1000,
                                value=int(hp_default),
                                step=int(hp_step),
                                key=f"hp_{hp_name}"
                            )
                        elif hp_type == "float":
                            value = st.number_input(
                                f"**{hp_name}** - {hp_desc}",
                                min_value=float(hp_min) if hp_min is not None else 0.0,
                                max_value=float(hp_max) if hp_max is not None else 100.0,
                                value=float(hp_default),
                                step=float(hp_step),
                                key=f"hp_{hp_name}"
                            )
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
                
                # Train button
                col_btn1, col_btn2 = st.columns([3, 1])
                with col_btn1:
                    if st.button("🚀 Train with Tuned Parameters", use_container_width=True, key="train_tuned_btn"):
                        ticker = PAIR_TO_TICKER.get(hp_pair, "EURUSD=X")
                        
                        payload = {
                            "ticker": ticker,
                            "model": hp_model,
                            "start_date": hp_start.strftime("%Y-%m-%d"),
                            "end_date": hp_end.strftime("%Y-%m-%d"),
                            "train_size": 0.8,
                            "hyperparameters": hp_values
                        }
                        
                        with st.spinner(f"Training {hp_model} with custom hyperparameters..."):
                            try:
                                train_response = requests.post(f"{API_BASE_URL}/train_with_tuning", json=payload, timeout=120)
                                if train_response.status_code == 200:
                                    result = train_response.json()
                                    st.success(f"✅ Training Complete! RMSE: {result.get('rmse', 'N/A'):.4f}, MAE: {result.get('mae', 'N/A'):.4f}")
                                else:
                                    st.error(f"Training failed: {train_response.text}")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                
                with col_btn2:
                    if st.button("🔄 Reset", use_container_width=True, key="reset_hp_btn"):
                        st.rerun()
        
            else:
                st.warning(f"Could not load hyperparameters for {hp_model}")
        except Exception as e:
            st.error(f"Could not load hyperparameters: {str(e)}")
            st.info("Make sure the backend service is available and try again.")
