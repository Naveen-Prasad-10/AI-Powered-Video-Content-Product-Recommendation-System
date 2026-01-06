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
    
    /* Custom Font for the whole app */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Card Styling for Metrics */
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    
    /* "Best Price" Green Highlight */
    [data-testid="stMetricValue"] {
        color: #00FF7F !important;
        font-size: 36px !important;
    }
</style>
""", unsafe_allow_html=True)
# --- CSS FOR PERFORMANCE ---
st.markdown("""
<style>
    header {visibility: hidden;}
    .block-container {padding-top: 2rem;}
    [data-testid="stMetricValue"] { color: #00FF7F !important; }
    /* Force images to load faster */
    img {
        image-rendering: -webkit-optimize-contrast;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Real-Time Detection Engine")
    st.caption("v1.0 - MVP (Cloud Optimized)")
    st.divider()
    conf_threshold = st.slider("AI Sensitivity", 0.3, 1.0, 0.50, 0.05)
    
    # NEW: Performance Control
    st.divider()
    st.subheader("🚀 Performance Mode")
    frame_skip = st.slider("Frame Skip (Higher = Smoother)", 2, 10, 3)
    st.info(f"Skipping {frame_skip-1} frames. Displaying ~{30//frame_skip} FPS.")

# --- RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    # --- DEBUGGING: Print what the server sees ---
    files = os.listdir(os.getcwd())
    st.write(f"📂 Files detected on server: {files}")
    # ---------------------------------------------

    # 1. Define Standard Paths
    local_windows_path = r"C:\Users\Naveen Prasad\Documents\Project_data\RTPD_v2.pt"
    cloud_filename = "RTPD_v2.pt"
    
    # 2. SMART SEARCH LOGIC
    model_path = None
    
    # Check 1: Exact Match (Cloud)
    if os.path.exists(cloud_filename):
        model_path = cloud_filename
    
    # Check 2: Case Insensitive Search (Cloud Fallback)
    elif any(f.lower() == cloud_filename.lower() for f in files):
        # Find the actual name of the file on the server
        actual_name = next(f for f in files if f.lower() == cloud_filename.lower())
        model_path = actual_name
        st.warning(f"⚠️ Found file, but case didn't match. Used: '{actual_name}'")
        
    # Check 3: Local Windows Path
    elif os.path.exists(local_windows_path):
        model_path = local_windows_path
    
    else:
        st.error(f"❌ Critical Error: Could not find '{cloud_filename}' in {os.getcwd()}")
        st.stop()

    # Load Model
    model = YOLO(model_path)
    
    # Load Database
    db = {}
    db_file = "inventory.json"
    if os.path.exists(db_file):
        with open(db_file, 'r') as f:
            db = json.load(f)
    else:
        st.warning("⚠️ Inventory.json not found. Using empty database.")
            
    return model, db

model, PRODUCT_DB = load_resources()

if not model:
    st.error("⚠️ System Error: Model file not found.")
    st.stop()

# --- MAIN DASHBOARD ---
# --- HERO SECTION ---
col_logo, col_title = st.columns([0.1, 0.9])

with col_logo:
    # You can replace this with st.image("logo.png") if you have one
    st.markdown("# 🛍️")

with col_title:
    st.markdown("""
    <h1 style='margin-bottom: 0px;'>ShopVision Pro</h1>
    <p style='color: #888; margin-top: 0px; font-size: 18px;'>
        AI-Powered Video Commerce Engine • v3.0
    </p>
    """, unsafe_allow_html=True)

st.divider() # Adds a clean horizontal line
st.markdown("Upload a video stream to detect products and retrieve **real-time pricing**.")

uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi"])

if 'history' not in st.session_state:
    st.session_state.history = []

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    col_video, col_live = st.columns([0.65, 0.35])

    with col_video:
        st.subheader("Input Stream")
        # Use a placeholder for the video to prevent "stuttering" stack up
        video_window = st.empty()
    
    with col_live:
        st.subheader("Live Market Data")
        live_alert = st.empty()
        
        # --- EMPTY STATE (Shows before video starts) ---
        with live_alert.container():
            st.markdown("""
            <div style='text-align: center; color: #666; padding: 50px;'>
                <h3>Waiting for video...</h3>
                <p>Upload a file to start tracking real-time prices.</p>
            </div>
            """, unsafe_allow_html=True)

    start_btn = st.button("▶️ Analyze Stream", type="primary")
    
    if start_btn:
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            
            # OPTIMIZATION 1: SKIP FRAMES
            # If slider is 3, we process frame 3, 6, 9...
            if frame_count % frame_skip != 0: 
                continue 

            # OPTIMIZATION 2: RESIZE
            # Resize 4K/1080p videos to 640p for fast web rendering
            frame = cv2.resize(frame, (640, 480))

            results = model.predict(frame, conf=conf_threshold, verbose=False)
            annotated_frame = frame.copy()
            
            if results[0].boxes:
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
                        
                        with live_alert.container():
                            st.success(f"**Recommended:** {product_name}")
                            st.metric("Best Price", price, f"Variant: {subtype}")
                        
                        # Use time.time() for unique ID instead of frame math
                        timestamp = frame_count
                        entry_id = f"{timestamp}_{product_name}"
                        
                        already_logged = any(x['id'] == entry_id for x in st.session_state.history)
                        
                        if not already_logged:
                            st.session_state.history.append({
                                "id": entry_id,
                                "Time (Frame)": timestamp,
                                "Product": product_name,
                                "Price": price,
                                "Vendor": "Partner Store", 
                                "Link": url 
                            })

            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # OPTIMIZATION 3: DISPLAY
            # .image() sends data to browser. Smaller image = faster.
            video_window.image(frame_rgb, use_container_width=True)

        cap.release()

    if st.session_state.history:
        st.divider()
        st.subheader("🛒 Recommended Products")
        
        df = pd.DataFrame(st.session_state.history)
        
        # Check if df is empty before rendering
        if not df.empty:
            st.dataframe(
                df[["Time (Frame)", "Product", "Price", "Vendor", "Link"]],
                column_config={
                    "Link": st.column_config.LinkColumn(
                        "Action",
                        display_text="Buy Now 🔗", 
                        validate="^https://.*"
                    ),
                },
                hide_index=True,
                use_container_width=True
            )
