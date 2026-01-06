import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import json
import pandas as pd
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI-Powered Video Content Product Recommendation System",
    layout="wide",
    page_icon="🛍️",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Hide the default Streamlit header and footer */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom Font */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Card Styling */
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    
    /* Green Highlight */
    [data-testid="stMetricValue"] {
        color: #00FF7F !important;
        font-size: 36px !important;
    }
    
    /* Toast Styling */
    div[data-testid="stToast"] {
        background-color: #2b313e;
        border: 1px solid #444;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Real-Time Detection Engine")
    st.caption("v3.0 - MVP (Cloud Optimized)")
    st.divider()
    conf_threshold = st.slider("AI Sensitivity", 0.3, 1.0, 0.50, 0.05)
    
    # PERFORMANCE CONTROL (Kept from your code)
    st.subheader("🚀 Performance Mode")
    frame_skip = st.slider("Frame Skip (Higher = Smoother)", 2, 10, 3)
    
    # NEW: COOLDOWN CONTROL (Added)
    st.subheader("⏱️ Alert Settings")
    cooldown = st.slider("Cooldown Timer (Sec)", 1, 10, 5, help="Wait this many seconds before showing the same item again.")

# --- RESOURCE LOADING (Kept your robust version) ---
@st.cache_resource
def load_resources():
    # 1. Define Standard Paths
    local_windows_path = r"C:\Users\Naveen Prasad\Documents\Project_data\RTPD_v2.pt"
    cloud_filename = "RTPD_v2.pt"
    
    # 2. SMART SEARCH LOGIC
    model_path = None
    files = os.listdir(os.getcwd())
    
    if os.path.exists(cloud_filename):
        model_path = cloud_filename
    elif any(f.lower() == cloud_filename.lower() for f in files):
        actual_name = next(f for f in files if f.lower() == cloud_filename.lower())
        model_path = actual_name
    elif os.path.exists(local_windows_path):
        model_path = local_windows_path
    else:
        st.error(f"❌ Critical Error: Could not find '{cloud_filename}' in {os.getcwd()}")
        st.stop()

    model = YOLO(model_path)
    
    db = {}
    db_file = "inventory.json"
    if os.path.exists(db_file):
        with open(db_file, 'r') as f:
            db = json.load(f)
            
    return model, db

model, PRODUCT_DB = load_resources()

if not model:
    st.error("⚠️ System Error: Model file not found.")
    st.stop()

# --- MAIN DASHBOARD ---
col_logo, col_title = st.columns([0.1, 0.9])
with col_logo:
    st.markdown("# 🛍️")
with col_title:
    st.markdown("""
    <h1 style='margin-bottom: 0px;'>ShopVision Pro</h1>
    <p style='color: #888; margin-top: 0px; font-size: 18px;'>
        AI-Powered Video Commerce Engine • v3.0
    </p>
    """, unsafe_allow_html=True)

st.divider()

uploaded_file = st.file_uploader("Upload Video Stream", type=["mp4", "mov", "avi"])

# Initialize Session States
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_seen' not in st.session_state: # NEW: Memory for cooldowns
    st.session_state.last_seen = {}

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    col_video, col_live = st.columns([0.65, 0.35])

    with col_video:
        st.subheader("Input Stream")
        video_window = st.empty()
    
    with col_live:
        st.subheader("Live Market Data")
        live_alert = st.empty()
        with live_alert.container():
             st.info("Waiting for video start...")

    start_btn = st.button("▶️ Analyze Stream", type="primary")
    
    if start_btn:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 30
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            
            # Use your Slider logic for skipping
            if frame_count % frame_skip != 0: 
                continue 

            frame = cv2.resize(frame, (640, 480))
            annotated_frame = frame.copy()

            results = model.predict(frame, conf=conf_threshold, verbose=False)
            
            if results[0].boxes:
                for box in results[0].boxes:
                    # Geometry
                    x, y, w, h = box.xywh[0].cpu().numpy()
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    aspect_ratio = h / w
                    
                    if aspect_ratio > 2.7:
                        subtype = "Bottle"
                        box_color = (0, 0, 255) 
                    else:
                        subtype = "Can"
                        box_color = (0, 255, 0) 
                        
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 3)
                    
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id] 
                    
                    # Database Lookup
                    lookup_key = f"{label}_{subtype}"
                    matched_product = PRODUCT_DB.get(lookup_key)
                    
                    if not matched_product:
                         for db_key in PRODUCT_DB:
                            if lookup_key.lower() == db_key.lower():
                                matched_product = PRODUCT_DB[db_key]
                                break

                    if matched_product:
                        product_name = matched_product.get('name', f"Unknown {label}")
                        price = matched_product.get('price', 'N/A')
                        url = matched_product.get('url', '#')
                        
                        # --- NEW COOLDOWN LOGIC ---
                        # Calculate current time in seconds
                        current_time_sec = frame_count / fps
                        last_time = st.session_state.last_seen.get(product_name, -100)
                        
                        # Only proceed if enough time has passed
                        if (current_time_sec - last_time) > cooldown:
                            
                            # Update memory
                            st.session_state.last_seen[product_name] = current_time_sec
                            
                            # 1. Update Live Card
                            with live_alert.container():
                                st.success(f"**Recommended:** {product_name}")
                                st.metric("Best Price", price, f"Variant: {subtype}")
                            
                            # 2. Show Toast (New Feature)
                            st.toast(f"✅ Found {product_name}!", icon="🛒")

                            # 3. Add to History Table
                            entry_id = f"{frame_count}_{product_name}"
                            st.session_state.history.append({
                                "id": entry_id,
                                "Time": f"{current_time_sec:.1f}s",
                                "Product": product_name,
                                "Price": price,
                                "Vendor": "Partner Store", 
                                "Link": url 
                            })

            video_window.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        cap.release()

    # --- RESULTS TABLE ---
    if st.session_state.history:
        st.divider()
        st.subheader("🛒 Recommended Products")
        df = pd.DataFrame(st.session_state.history)
        
        if not df.empty:
            st.dataframe(
                df[["Time", "Product", "Price", "Vendor", "Link"]],
                column_config={
                    "Link": st.column_config.LinkColumn(
                        "Action", display_text="Buy Now 🔗", validate="^https://.*"
                    ),
                },
                hide_index=True,
                use_container_width=True
            )
