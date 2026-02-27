import streamlit as st
import cv2
from detector import TrafficMonitor
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import os

# Page config
st.set_page_config(
    page_title="AI Traffic Monitor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme with neon accents
st.markdown("""
<style>
    * {
        font-family: 'Courier New', monospace;
    }
    
    body {
        background: linear-gradient(135deg, #0a0e27 0%, #16213e 100%);
        color: #00ff88;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #16213e 100%);
    }
    
    /* Hero Section */
    .hero-text {
        font-size: 3.5em;
        font-weight: 900;
        background: linear-gradient(90deg, #00ff88, #00d9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin: 40px 0;
        text-shadow: 0 0 30px rgba(0, 255, 136, 0.5);
        letter-spacing: 2px;
    }
    
    .hero-subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #00d9ff;
        margin-bottom: 30px;
        letter-spacing: 1px;
    }
    
    /* Stat Cards */
    .stat-card {
        background: rgba(16, 32, 64, 0.8);
        border: 2px solid #00ff88;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        box-shadow: 0 0 40px rgba(0, 255, 136, 0.6);
        transform: translateY(-5px);
        border-color: #00d9ff;
    }
    
    .stat-value {
        font-size: 2.5em;
        color: #00ff88;
        font-weight: bold;
        margin: 10px 0;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    
    .stat-label {
        color: #00d9ff;
        font-size: 0.9em;
        letter-spacing: 1px;
    }
    
    /* Feature Cards */
    .feature-card {
        background: rgba(16, 32, 64, 0.9);
        border-left: 4px solid #00ff88;
        padding: 25px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.2);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        border-left: 4px solid #00d9ff;
        box-shadow: 0 8px 25px rgba(0, 255, 136, 0.3);
        transform: translateX(5px);
    }
    
    .feature-icon {
        font-size: 2em;
        margin-bottom: 10px;
    }
    
    .feature-title {
        color: #00ff88;
        font-size: 1.3em;
        font-weight: bold;
        margin: 10px 0;
        letter-spacing: 1px;
    }
    
    .feature-desc {
        color: #b0b0b0;
        font-size: 0.95em;
        line-height: 1.6;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 2em;
        color: #00ff88;
        border-bottom: 2px solid #00d9ff;
        padding-bottom: 15px;
        margin-top: 40px;
        margin-bottom: 30px;
        letter-spacing: 2px;
    }
    
    /* Process Steps */
    .process-step {
        background: rgba(16, 32, 64, 0.8);
        border: 2px solid #00d9ff;
        border-radius: 50%;
        width: 120px;
        height: 120px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 20px auto;
        font-size: 3em;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.4);
    }
    
    .process-label {
        text-align: center;
        color: #00ff88;
        font-weight: bold;
        margin-top: 10px;
        letter-spacing: 1px;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(90deg, #00ff88, #00d9ff);
        color: #0a0e27;
        border: none;
        font-weight: bold;
        padding: 12px 30px;
        border-radius: 8px;
        font-size: 1em;
        letter-spacing: 1px;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 0 40px rgba(0, 255, 136, 0.8);
        transform: scale(1.05);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #00d9ff;
        padding: 40px 0;
        border-top: 1px solid rgba(0, 255, 136, 0.3);
        margin-top: 60px;
        font-size: 0.9em;
        letter-spacing: 1px;
    }
    
    /* Tab styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(16, 32, 64, 0.6);
        border: 1px solid #00d9ff;
        border-radius: 6px;
        color: #00d9ff;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(0, 255, 136, 0.2);
        border: 2px solid #00ff88;
        color: #00ff88;
    }
    
    /* Alert styling */
    .alert-high {
        background: rgba(255, 0, 0, 0.1);
        border-left: 4px solid #ff0055;
        color: #ff0055;
    }
    
    .alert-medium {
        background: rgba(255, 200, 0, 0.1);
        border-left: 4px solid #ffc800;
        color: #ffc800;
    }
    
    .alert-low {
        background: rgba(0, 255, 136, 0.1);
        border-left: 4px solid #00ff88;
        color: #00ff88;
    }
    
    /* Metric styling for IN/OUT counters */
    [data-testid="metric-container"] {
        background: rgba(16, 32, 64, 0.9) !important;
        border: 2px solid #00ff88 !important;
        border-radius: 12px !important;
        padding: 20px 10px !important;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.3) !important;
        text-align: center !important;
    }
    
    [data-testid="metric-container"] > div:first-child {
        font-size: 1.2em !important;
        color: #00d9ff !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
    }
    
    [data-testid="metric-container"] > div:last-child {
        font-size: 2.8em !important;
        color: #00ff88 !important;
        font-weight: 900 !important;
        text-shadow: 0 0 15px rgba(0, 255, 136, 0.6) !important;
        margin-top: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ============ HERO SECTION ============
st.markdown('<h1 class="hero-text">🚦 AI TRAFFIC MONITOR</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Command Center • Real-Time Analytics • Smart City Intelligence</p>', unsafe_allow_html=True)

# Animated separator
st.markdown("---")

# ============ LIVE STATS BAR ============
st.markdown('<h2 class="section-header">📊 LIVE STATISTICS</h2>', unsafe_allow_html=True)

# Initialize sample data (will be real from detector)
if 'live_stats' not in st.session_state:
    st.session_state.live_stats = {
        'vehicles_detected': 0,
        'avg_speed': 0,
        'incidents': 0,
        'congestion': 0
    }

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">🚗 VEHICLES DETECTED</div>
        <div class="stat-value">{st.session_state.live_stats['vehicles_detected']}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">⚡ AVG SPEED</div>
        <div class="stat-value">{st.session_state.live_stats['avg_speed']:.1f}</div>
        <span style="color: #00d9ff; font-size: 0.8em;">km/h</span>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">⚠️ INCIDENTS</div>
        <div class="stat-value">{st.session_state.live_stats['incidents']}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">🔴 CONGESTION</div>
        <div class="stat-value">{st.session_state.live_stats['congestion']}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============ FEATURES GRID ============
st.markdown('<h2 class="section-header">✨ CORE FEATURES</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

features = [
    {"icon": "🎯", "title": "Real-Time Detection", "desc": "YOLOv8 powered multi-object detection with sub-10ms latency"},
    {"icon": "📈", "title": "Speed Analysis", "desc": "Accurate vehicle speed calculation using optical flow tracking"},
    {"icon": "🚨", "title": "Incident Alerts", "desc": "Automatic detection of accidents, violations & anomalies"},
    {"icon": "🔍", "title": "License Plate Recognition", "desc": "OCR-based plate identification for vehicle tracking"},
    {"icon": "🗺️", "title": "Heatmap Analytics", "desc": "Spatial visualization of traffic density & patterns"},
    {"icon": "🤖", "title": "AI Predictions", "desc": "Machine learning forecasts for congestion & flow patterns"}
]

for i, feature in enumerate(features):
    if i % 3 == 0:
        col1, col2, col3 = st.columns(3)
    
    with [col1, col2, col3][i % 3]:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{feature['icon']}</div>
            <div class="feature-title">{feature['title']}</div>
            <div class="feature-desc">{feature['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ============ HOW IT WORKS ============
st.markdown('<h2 class="section-header">🔄 HOW IT WORKS</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    st.markdown(f"""
    <div class="process-step">📹</div>
    <div class="process-label">CAMERA INPUT</div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("<div style='text-align: center; color: #00ff88; font-size: 2em; margin-top: 40px;'>→</div>", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="process-step">⚙️</div>
    <div class="process-label">AI PROCESSING</div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("<div style='text-align: center; color: #00ff88; font-size: 2em; margin-top: 40px;'>→</div>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    st.markdown(f"""
    <div class="process-step">💡</div>
    <div class="process-label">INSIGHTS</div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============ DASHBOARD PREVIEW & LIVE MONITORING ============
st.markdown('<h2 class="section-header">📡 DASHBOARD & MONITORING</h2>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🎬 Live Feed", "📊 Analytics", "🗺️ Map View", "🚨 Alerts"])

# Session state
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
if 'monitor' not in st.session_state:
    st.session_state.monitor = None

with tab1:
    st.subheader("Live Traffic Camera Feed")
    
    # Initialize session state for video file
    if 'uploaded_videos' not in st.session_state:
        st.session_state.uploaded_videos = None
    if 'selected_video' not in st.session_state:
        st.session_state.selected_video = None
    if 'video_source' not in st.session_state:
        st.session_state.video_source = None
    
    # Find local video files
    local_video_files = []
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    for file in os.listdir('.'):
        if file.lower().endswith(video_extensions):
            local_video_files.append(file)
    
    # Video file upload
    st.markdown("**📁 Upload Video(s)**")
    uploaded_files = st.file_uploader(
        "Choose video file(s) at least one video (MP4, AVI, MOV, MKV)",
        type=['mp4', 'avi', 'mov', 'mkv'],
        accept_multiple_files=True,
        key="video_uploader"
    )
    
    if uploaded_files:
        st.session_state.uploaded_videos = uploaded_files
    
    # Combine local and uploaded videos
    all_videos = []
    video_labels = []
    
    # Add local videos
    for idx, local_video in enumerate(local_video_files):
        all_videos.append(('local', local_video))
        video_labels.append(f"📂 {local_video}")
    
    # Add uploaded videos
    if st.session_state.uploaded_videos:
        for idx, uploaded_video in enumerate(st.session_state.uploaded_videos):
            all_videos.append(('uploaded', uploaded_video))
            video_labels.append(f"⬆️ {uploaded_video.name}")
    
    if all_videos:
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_idx = st.selectbox(
                "Select a video to analyze",
                range(len(all_videos)),
                format_func=lambda i: video_labels[i],
                key="video_selector"
            )
            video_type, selected_file = all_videos[selected_idx]
            st.session_state.selected_video = selected_file
            st.session_state.video_source = video_type
        
        # Display video info
        source_emoji = "📂" if st.session_state.video_source == 'local' else "⬆️"
        st.info(f"{source_emoji} Selected: {selected_file.name if hasattr(selected_file, 'name') else selected_file}")
    
    video_placeholder = st.empty()
    
    # Centered stats display
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        fps_metric = st.empty()
    
    with stat_col2:
        active_metric = st.empty()
    
    with stat_col3:
        in_metric = st.empty()
    
    with stat_col4:
        out_metric = st.empty()
    
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("🚀 Start Monitoring", key="start", type="primary", disabled=(st.session_state.selected_video is None))
    with col2:
        stop_button = st.button("⏹️ Stop", key="stop")
    
    if start_button:
        st.session_state.monitoring = True
    if stop_button:
        st.session_state.monitoring = False
    
    if st.session_state.monitoring and st.session_state.selected_video is not None:
        if st.session_state.monitor is None:
            with st.spinner("🔄 Loading AI model..."):
                st.session_state.monitor = TrafficMonitor()
        
        # Handle both local and uploaded videos
        if st.session_state.video_source == 'local':
            video_path = st.session_state.selected_video
            video_name = st.session_state.selected_video
        else:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(st.session_state.selected_video.read())
                video_path = tmp_file.name
            video_name = st.session_state.selected_video.name
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                st.error("❌ Cannot open video!")
                st.session_state.monitoring = False
            else:
                success_placeholder = st.empty()
                
                # Get video properties for optimal FPS playback
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                if video_fps <= 0:
                    video_fps = 30  # Default fallback FPS
                
                frame_delay = 1 / video_fps  # Time between frames in seconds
                last_frame_time = time.time()
                
                success_placeholder.success(f"✅ Monitoring active! Playing: {video_name} @ {video_fps:.1f} FPS")
                
                frame_count = 0
                while st.session_state.monitoring and cap.isOpened():
                    ret, frame = cap.read()
                    
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    
                    # Process frame
                    frame_start_time = time.time()
                    annotated_frame, stats = st.session_state.monitor.process_frame(frame)
                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    processing_time = time.time() - frame_start_time
                    
                    # Display frame and metrics
                    video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                    
                    fps_metric.metric("🎯 FPS", f"{stats['fps']:.1f}")
                    active_metric.metric("🚗 Active", stats['active_vehicles'])
                    in_metric.metric("⬇️ Entered", stats['total_in'])
                    out_metric.metric("⬆️ Exited", stats['total_out'])
                    
                    st.session_state.live_stats['vehicles_detected'] = stats['active_vehicles']
                    
                    # Dynamic frame timing: maintain video's original FPS
                    # Calculate remaining time to meet target frame rate
                    elapsed_time = time.time() - frame_start_time
                    remaining_delay = max(0, frame_delay - elapsed_time)
                    
                    if remaining_delay > 0:
                        time.sleep(remaining_delay)
                    
                    frame_count += 1
                    
                    if not st.session_state.monitoring:
                        break
                
                cap.release()
                success_placeholder.info("⏹️ Monitoring stopped")
        finally:
            # Clean up temporary file if it was an uploaded video
            if st.session_state.video_source == 'uploaded' and os.path.exists(video_path):
                os.remove(video_path)
    elif st.session_state.monitoring and st.session_state.selected_video is None:
        st.error("❌ Please select a video first!")
        st.session_state.monitoring = False
    else:
        if local_video_files:
            st.info("👈 Select a video and click 'Start Monitoring' to begin")
        else:
            st.info("👈 Upload video(s) and click 'Start Monitoring' to begin")

with tab2:
    st.subheader("Traffic Analytics Dashboard")
    
    # Generate sample data
    times = pd.date_range(start='2025-02-25 08:00', periods=24, freq='H')
    traffic_data = pd.DataFrame({
        'Time': times,
        'Vehicles': np.random.randint(20, 150, 24),
        'Avg Speed': np.random.randint(30, 80, 24),
        'Congestion %': np.random.randint(10, 90, 24)
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_vehicles = px.line(
            traffic_data, x='Time', y='Vehicles',
            title='Vehicles Over Time',
            labels={'Vehicles': 'Count'}
        )
        fig_vehicles.update_layout(
            template='plotly_dark',
            singlelinecolor='#00ff88',
            paper_bgcolor='rgba(16, 32, 64, 0.8)',
            plot_bgcolor='rgba(26, 42, 74, 0.8)',
            font=dict(color='#00ff88')
        )
        st.plotly_chart(fig_vehicles, use_container_width=True)
    
    with col2:
        fig_speed = px.area(
            traffic_data, x='Time', y='Avg Speed',
            title='Average Speed Over Time'
        )
        fig_speed.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(16, 32, 64, 0.8)',
            plot_bgcolor='rgba(26, 42, 74, 0.8)',
            font=dict(color='#00ff88')
        )
        st.plotly_chart(fig_speed, use_container_width=True)
    
    fig_congestion = px.bar(
        traffic_data, x='Time', y='Congestion %',
        title='Traffic Congestion Level',
        color='Congestion %',
        color_continuous_scale='Reds'
    )
    fig_congestion.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(16, 32, 64, 0.8)',
        plot_bgcolor='rgba(26, 42, 74, 0.8)',
        font=dict(color='#00ff88')
    )
    st.plotly_chart(fig_congestion, use_container_width=True)

with tab3:
    st.subheader("Traffic Density Heatmap")
    
    # Generate heatmap data
    heatmap_data = np.random.rand(20, 30) * 100
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        colorscale='Reds',
        colorbar=dict(title="Density %")
    ))
    fig_heat.update_layout(
        title='Real-Time Traffic Density Map',
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        template='plotly_dark',
        paper_bgcolor='rgba(16, 32, 64, 0.8)',
        plot_bgcolor='rgba(26, 42, 74, 0.8)',
        font=dict(color='#00ff88')
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with tab4:
    st.subheader("Active Alerts Feed")
    
    alerts = [
        {"type": "high", "time": datetime.now() - timedelta(minutes=2), "message": "Heavy congestion detected on Route 1"},
        {"type": "high", "time": datetime.now() - timedelta(minutes=5), "message": "Accident reported at intersection 5"},
        {"type": "medium", "time": datetime.now() - timedelta(minutes=15), "message": "Speed violation: Vehicle exceeded 80 km/h"},
        {"type": "low", "time": datetime.now() - timedelta(minutes=30), "message": "Traffic flow normal on secondary roads"},
    ]
    
    for alert in alerts:
        alert_class = f"alert-{alert['type']}"
        st.markdown(f"""
        <div style="background: rgba(16, 32, 64, 0.8); border-left: 4px solid {'#ff0055' if alert['type'] == 'high' else '#ffc800' if alert['type'] == 'medium' else '#00ff88'}; padding: 15px; border-radius: 6px; margin: 10px 0; color: {'#ff0055' if alert['type'] == 'high' else '#ffc800' if alert['type'] == 'medium' else '#00ff88'};">
            <strong>⏰ {alert['time'].strftime('%H:%M:%S')}</strong> | {alert['message']}
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ============ FOOTER ============
st.markdown("""
<div class="footer">
    <p>🚦 AI Traffic Monitoring System | Powered by YOLOv8 & Streamlit</p>
    <p>© 2025 Smart City Analytics. All Rights Reserved.</p>
    <p style="margin-top: 20px; color: #00d9ff; letter-spacing: 2px;">[ LIVE COMMAND CENTER ]</p>
</div>
""", unsafe_allow_html=True)