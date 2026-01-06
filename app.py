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
    page_title="ShopVision Pro",
    layout="wide",
    page_icon="🛍️",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (FIXED FOR MENU ARROW + CLEAN UI) ---
st.markdown("""
<style>
    /* 1. Hide the "Deploy" button and the "Three Dots" menu */
    [data-testid="stToolbar"] {
        visibility: hidden;
    }
    
    /* 2. Hide the colorful top decoration bar */
    [data-testid="stDecoration"] {
        visibility: hidden;
    }

    /* 3. TARGET THE HEADER: 
       Make it transparent so clicks pass through to the arrow. */
    [data-testid="stHeader"] {
        background-color: transparent;
        color: transparent; 
    }

    /* 4. FORCE THE ARROW TO BE VISIBLE */
    section[data-testid="stSidebar"] > div > div {
        visibility: visible;
    }
    [data-testid="stSidebarCollapsedControl"] {
        visibility: visible !important;
        display: block !important;
        color: white !important;
        background-color: rgba(100, 100, 100, 0.4); 
        border-radius: 50%;
        padding: 5px;
        z-index: 999999; /* Force top layer */
    }
    
    /* 5. Font & Card Styling */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
    }
    [data-testid="stMetricValue"] {
        color: #00FF7F !important;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Real-Time Detection Engine")
    st.caption("v3.2 - Stable Release")
    st.divider()
    conf_threshold = st.slider("AI Sensitivity", 0.3, 1.0, 0.50, 0.05)
    
    st.subheader("🚀 Performance Mode")
    frame_skip = st.slider("Frame Skip (Higher = Smoother)", 2, 10, 3)
    
    st.subheader("⏱️ Alert Settings")
    cooldown = st.slider("Cooldown Timer (Sec)", 1, 10, 5, help="Wait this many seconds before showing the same item again.")

# --- RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    local_windows_path = r"C:\Users\Naveen Prasad\Documents\Project_data\RTPD_v2.pt"
    cloud_filename = "RTPD_v2.pt"
    
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
        AI-Powered Video Commerce Engine • v3.2
    </p>
    """, unsafe_allow_html=True)

st.divider()

uploaded_file = st.file_uploader("Upload Video Stream", type=["mp4", "mov", "avi"])

# Initialize Session States
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_seen' not in st.session_state:
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
             st.info("Ready to analyze. Click Start.")

    start_btn = st.button("▶️ Analyze Stream", type="primary")
    
    if start_btn:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps == 0: fps = 30
        
        with live_alert.container():
             st.spinner("Processing video feed...")
        
        frame_count = 0
        detections_found = False 
        progress_bar = st.progress(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            if total_frames > 0:
                progress_bar.progress(min(frame_count / total_frames, 1.0))

            if frame_count % frame_skip != 0: 
                continue 

            # --- OPTIMIZATION LOGIC ---
            # 1. Resize (Safe Mode): Only resize if huge. 
            # If your old code worked with resize, you can uncomment the next line. 
            # But usually, keeping original res is safer for detection.
            # frame = cv2.resize(frame, (640, 480)) 

            # 2. AI INFERENCE (THE FIX)
            # We pass the raw 'frame' (BGR). We DO NOT convert to RGB for the model.
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            
            annotated_frame = frame.copy()
            
            if results[0].boxes:
                detections_found = True
                for box in results[0].boxes:
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
                        
                        current_time_sec = frame_count / fps
                        last_time = st.session_state.last_seen.get(product_name, -100)
                        
                        if (current_time_sec - last_time) > cooldown:
                            st.session_state.last_seen[product_name] = current_time_sec
                            
                            with live_alert.container():
                                st.success(f"**Recommended:** {product_name}")
                                st.metric("Best Price", price, f"Variant: {subtype}")
                            
                            st.toast(f"✅ Found {product_name}!", icon="🛒")

                            entry_id = f"{frame_count}_{product_name}"
                            st.session_state.history.append({
                                "id": entry_id,
                                "Time": f"{current_time_sec:.1f}s",
                                "Product": product_name,
                                "Price": price,
                                "Vendor": "Partner Store", 
                                "Link": url 
                            })

            # 3. DISPLAY (Convert to RGB for Human Eyes only)
            video_window.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        cap.release()
        progress_bar.empty()

        with live_alert.container():
            if detections_found:
                 st.success("✅ Analysis Complete.")
                 st.info("Scroll down to view shopping list.")
            else:
                 st.warning("⚠️ Analysis Complete: No products detected.")
                 st.caption("Try lowering the AI Sensitivity slider or using a different video.")

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
