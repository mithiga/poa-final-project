"""
System Status Page - Display system status and information
"""

import streamlit as st
import requests
import plotly.graph_objects as go
from datetime import datetime


def render_system_status_page(T, API_BASE_URL):
    """Render the System Status page."""
    
    # Page Header
    st.markdown(
        f"<h1 style='color:{T['heading_color']}; font-size:2rem; margin-bottom:4px;'>⚙️ System Status</h1>"
        f"<p style='color:{T['text_secondary']}; font-size:0.9rem; margin-top:0;'>Monitor system health and view available resources</p>",
        unsafe_allow_html=True
    )
    st.divider()
    
    # Refresh Button
    col_refresh, col_space = st.columns([1, 4])
    with col_refresh:
        refresh_btn = st.button("🔄 Refresh Status", use_container_width=True)
    
    # Fetch Status
    if refresh_btn or 'system_status' not in st.session_state or 'status_timestamp' not in st.session_state:
        with st.spinner("Fetching system status..."):
            try:
                res = requests.get(f"{API_BASE_URL}/status", timeout=10)
                if res.status_code == 200:
                    st.session_state.system_status = res.json()
                    st.session_state.status_timestamp = datetime.now()
                else:
                    st.session_state.system_status = {
                        "status": "error",
                        "message": f"Server returned {res.status_code}",
                        "models_available": [],
                        "tickers_trained": []
                    }
            except requests.exceptions.ConnectionError:
                st.session_state.system_status = {
                    "status": "offline",
                    "message": "Unable to connect to backend",
                    "models_available": [],
                    "tickers_trained": []
                }
            except Exception as e:
                st.session_state.system_status = {
                    "status": "error",
                    "message": str(e),
                    "models_available": [],
                    "tickers_trained": []
                }
    
    # Display Status
    if 'system_status' in st.session_state:
        status = st.session_state.system_status
        
        # Status Banner
        status_color = T['positive'] if status.get('status') == 'operational' else '#ff5252'
        status_bg = T['status_bg'] if status.get('status') == 'operational' else 'rgba(255,82,82,0.1)'
        status_text = "OPERATIONAL" if status.get('status') == 'operational' else "ISSUE DETECTED"
        
        st.markdown(f"""
        <div style='background:{status_bg}; border:1px solid {status_color}; 
                    border-radius:12px; padding:20px; text-align:center; margin-bottom:20px;'>
            <h2 style='color:{status_color}; margin:0; font-size:1.5rem;'>● {status_text}</h2>
            <p style='color:{T['text_primary']}; margin-top:10px;'>{status.get('message', 'System is running normally')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        models_count = len(status.get('models_available', []))
        tickers_count = len(status.get('tickers_trained', []))
        
        col1.metric("System Status", status.get('status', 'Unknown').title())
        col2.metric("Available Models", models_count)
        col3.metric("Trained Tickers", tickers_count)
        
        if 'status_timestamp' in st.session_state:
            timestamp = st.session_state.status_timestamp
            col4.metric("Last Updated", timestamp.strftime("%H:%M:%S"))
        
        # Detailed Information
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown(f"<h4 style='color:{T['subheading_color']};'>🧠 Available Models</h4>", unsafe_allow_html=True)
            
            models = status.get('models_available', [])
            if models:
                # Display as cards
                for i, model in enumerate(models):
                    st.markdown(f"""
                    <div style='background:{T['table_bg']}; border:1px solid {T['card_border']}; 
                                border-radius:8px; padding:12px; margin-bottom:8px;'>
                        <span style='color:{T['accent']}; font-weight:600;'>{model}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No models available. Train models in the Training page.")
        
        with col_info2:
            st.markdown(f"<h4 style='color:{T['subheading_color']};'>📈 Trained Tickers</h4>", unsafe_allow_html=True)
            
            tickers = status.get('tickers_trained', [])
            if tickers:
                # Display as cards
                for ticker in tickers:
                    st.markdown(f"""
                    <div style='background:{T['table_bg']}; border:1px solid {T['card_border']}; 
                                border-radius:8px; padding:12px; margin-bottom:8px;'>
                        <span style='color:{T['accent']}; font-weight:600;'>{ticker}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No tickers trained yet. Train models in the Training page.")
        
        # System Health Visualization
        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()
        
        st.markdown(f"<h4 style='color:{T['subheading_color']};'>📊 System Health</h4>", unsafe_allow_html=True)
        
        # Create health gauge
        fig = go.Figure()
        
        # Calculate health score (simple metric based on available resources)
        health_score = min(100, (models_count * 10) + (tickers_count * 15))
        
        fig.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = health_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "System Health Score"},
            delta = {'reference': 50, 'increasing': {'color': T['positive']}},
            gauge = {
                'axis': {'range': [0, 100], 'tickcolor': T['text_secondary']},
                'bar': {'color': T['accent']},
                'bgcolor': T['table_bg'],
                'borderwidth': 2,
                'bordercolor': T['card_border'],
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(255, 82, 82, 0.3)'},
                    {'range': [30, 60], 'color': 'rgba(255, 193, 7, 0.3)'},
                    {'range': [60, 100], 'color': 'rgba(0, 230, 118, 0.3)'}
                ],
            }
        ))
        
        fig.update_layout(
            paper_bgcolor=T['plot_paper'],
            font={'color': T['text_primary']},
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Resource Summary
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.markdown(f"""
            <div style='background:{T['table_bg']}; border:1px solid {T['card_border']}; 
                        border-radius:12px; padding:20px; text-align:center;'>
                <h3 style='color:{T['accent']}; margin:0;'>{models_count}</h3>
                <p style='color:{T['text_secondary']}; margin:5px 0 0 0;'>Models Available</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_res2:
            st.markdown(f"""
            <div style='background:{T['table_bg']}; border:1px solid {T['card_border']}; 
                        border-radius:12px; padding:20px; text-align:center;'>
                <h3 style='color:{T['accent']}; margin:0;'>{tickers_count}</h3>
                <p style='color:{T['text_secondary']}; margin:5px 0 0 0;'>Trained Tickers</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_res3:
            uptime_status = "Online" if status.get('status') != 'offline' else "Offline"
            st.markdown(f"""
            <div style='background:{T['table_bg']}; border:1px solid {T['card_border']}; 
                        border-radius:12px; padding:20px; text-align:center;'>
                <h3 style='color:{T['positive'] if uptime_status == 'Online' else '#ff5252'}; margin:0;'>●</h3>
                <p style='color:{T['text_secondary']}; margin:5px 0 0 0;'>Backend {uptime_status}</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Empty state
        st.markdown(f"""
        <div style='text-align:center; padding:60px 20px; color:{T['text_secondary']};'>
            <p style='font-size:3rem; margin-bottom:10px;'>⚙️</p>
            <p>Click <b>Refresh Status</b> to view system information</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Info Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style='background:{T['table_bg']}; border:1px solid {T['card_border']}; 
                    border-radius:12px; padding:20px;'>
            <h4 style='color:{T['heading_color']}; margin-top:0; font-size:1.1rem;'>💡 About System Status</h4>
            <p style='color:{T['text_primary']}; font-size:0.95rem; line-height:1.6; margin-bottom:0;'>
                The system status page shows real-time information about the Forex AI engine,
                including available models and trained tickers. A health score is calculated
                based on the number of trained models and tickers.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background:{T['table_bg']}; border:1px solid {T['card_border']}; 
                    border-radius:12px; padding:20px;'>
            <h4 style='color:{T['heading_color']}; margin-top:0; font-size:1.1rem;'>🔧 Troubleshooting</h4>
            <ul style='color:{T['text_primary']}; padding-left:20px; font-size:0.95rem; line-height:1.6; margin-bottom:0;'>
                <li>If backend shows offline, ensure the backend service is available</li>
                <li>Train models to increase the health score</li>
                <li>Check the Training page to add more tickers</li>
                <li>Use the refresh button to update status</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
