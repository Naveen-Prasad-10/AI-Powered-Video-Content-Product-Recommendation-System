import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import json
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI-Powered Video Content Product Recommendation System (MVP)",
    layout="wide",
    page_icon="🛍️",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    header {visibility: hidden;}
    .block-container {padding-top: 2rem;}
    [data-testid="stMetricValue"] { color: #00FF7F !important; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Real-Time Detection Engine")
    st.divider()
    conf_threshold = st.slider("AI Sensitivity", 0.3, 1.0, 0.60, 0.05)

# --- RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    local_path = r"path/to/your/RTPD_v2.pt"
    
    if os.path.exists(local_path):
        model_path = local_path
    elif os.path.exists("RTPD_v2.pt"):
        model_path = "RTPD_v2.pt"
    else:
        return None, {}

    model = YOLO(model_path)
    
    db = {}
    if os.path.exists("inventory.json"):
        with open("inventory.json", 'r') as f:
            db = json.load(f)
            
    return model, db

model, PRODUCT_DB = load_resources()

if not model:
    st.error("⚠️ System Error: Model file not found.")
    st.stop()

# --- MAIN DASHBOARD ---
# UPDATED TITLE HERE
st.title("Real-Time Product Detection & Recommendation")
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
        video_window = st.empty()
    
    with col_live:
        st.subheader("Recommendation Feed")
        live_alert = st.empty()

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
            if frame_count % (fps // 2) != 0: 
                continue 

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
                        
                        timestamp = frame_count // fps
                        entry_id = f"{timestamp}_{product_name}"
                        
                        already_logged = any(x['id'] == entry_id for x in st.session_state.history)
                        
                        if not already_logged:
                            st.session_state.history.append({
                                "id": entry_id,
                                "Time (s)": timestamp,
                                "Product": product_name,
                                "Price": price,
                                "Vendor": "Partner Store", 
                                "Link": url 
                            })

            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_window.image(frame_rgb, use_container_width=True)

        cap.release()

    if st.session_state.history:
        st.divider()
        st.subheader("🛒 Recommended Products")
        
        df = pd.DataFrame(st.session_state.history)
        
        st.dataframe(
            df[["Time (s)", "Product", "Price", "Vendor", "Link"]],
            column_config={
                "Link": st.column_config.LinkColumn(
                    "Action",
                    display_text="Buy Now 🔗", 
                    validate="^https://.*"
                ),
                "Time (s)": st.column_config.NumberColumn(format="%ds")
            },
            hide_index=True,
            use_container_width=True
        )
