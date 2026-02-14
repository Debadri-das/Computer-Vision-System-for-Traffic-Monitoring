import streamlit as st
import cv2
from detector import TrafficMonitor
import time

# Page config
st.set_page_config(
    page_title="AI Traffic Monitor",
    page_icon="🚦",
    layout="wide"
)

# Title
st.title("🚦 AI Traffic Monitoring System")
st.markdown("**Real-time vehicle detection, tracking & counting using YOLOv8**")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Control Panel")
video_source = st.sidebar.selectbox(
    "Video Source",
    ["traffic.mp4", "Webcam"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 Click 'Start Monitoring' to begin")

# Main layout
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📹 Live Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("📊 Statistics")
    fps_metric = st.empty()
    active_metric = st.empty()
    in_metric = st.empty()
    out_metric = st.empty()
    total_metric = st.empty()

# Buttons
start_button = st.button("🚀 Start Monitoring", type="primary")
stop_button = st.button("⏹️ Stop")

# Session state
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
if 'monitor' not in st.session_state:
    st.session_state.monitor = None

if start_button:
    st.session_state.monitoring = True
if stop_button:
    st.session_state.monitoring = False

# Main loop
if st.session_state.monitoring:
    # Initialize
    if st.session_state.monitor is None:
        with st.spinner("🔄 Loading AI model..."):
            st.session_state.monitor = TrafficMonitor()
    
    # Open video
    cap = cv2.VideoCapture("traffic.mp4" if video_source == "traffic.mp4" else 0)
    
    if not cap.isOpened():
        st.error("❌ Cannot open video!")
        st.session_state.monitoring = False
    else:
        st.success("✅ Monitoring active!")
        
        while st.session_state.monitoring and cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                continue
            
            # Process frame
            annotated_frame, stats = st.session_state.monitor.process_frame(frame)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display
            video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
            
            # Update metrics
            fps_metric.metric("🎯 FPS", f"{stats['fps']:.1f}")
            active_metric.metric("🚗 Active", stats['active_vehicles'])
            in_metric.metric("⬇️ Entered", stats['total_in'])
            out_metric.metric("⬆️ Exited", stats['total_out'])
            total_metric.metric("📈 Total", stats['total_crossed'])
            
            time.sleep(0.03)
            
            if not st.session_state.monitoring:
                break
        
        cap.release()
        st.info("⏹️ Monitoring stopped")
else:
    st.info("👈 Click 'Start Monitoring' to begin")

# Footer
st.markdown("---")
st.markdown("**Features:** Real-time Detection • Multi-Object Tracking • Vehicle Counting • FPS Monitoring")
