import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import json
import pandas as pd
import time
from datetime import datetime, timezone
from optimizer import rank_vendors

# Scraper is optional — gracefully skip if dependencies aren't installed
try:
    from scraper import run as _scrape_inventory
    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False

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

    /* ── NDU Smart Recommendation Cards ─────────────────── */
    .ndu-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #00FF7F;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 10px;
        box-shadow: 0 4px 18px rgba(0, 255, 127, 0.12);
    }
    .ndu-badge {
        background: #00FF7F;
        color: #000;
        font-weight: 700;
        font-size: 11px;
        padding: 3px 10px;
        border-radius: 20px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .utility-score {
        font-size: 24px;
        font-weight: 800;
        color: #00FF7F;
        font-family: 'Courier New', monospace;
    }
    .alt-vendor {
        background: rgba(255, 255, 255, 0.04);
        border-left: 3px solid #f0a500;
        border-radius: 6px;
        padding: 8px 12px;
        margin-top: 10px;
        font-size: 12px;
        color: #bbb;
    }
    .why-pill {
        background: rgba(0, 255, 127, 0.08);
        border-radius: 6px;
        padding: 5px 10px;
        font-size: 12px;
        color: #ddd;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Real-Time Detection Engine")
    st.caption("v4.0 — NDU Engine")
    st.divider()
    conf_threshold = st.slider("AI Sensitivity", 0.3, 1.0, 0.50, 0.05)

    st.subheader("🚀 Performance Mode")
    frame_skip = st.slider("Frame Skip (Higher = Smoother)", 2, 10, 3)

    st.subheader("⏱️ Alert Settings")
    cooldown = st.slider("Cooldown Timer (Sec)", 1, 10, 5,
                         help="Wait this many seconds before showing the same item again.")

    st.divider()

    # —— Price Freshness Panel ———————————————————————————
    st.subheader("💰 Price Data")
    _db_for_date = json.load(open("inventory.json")) if os.path.exists("inventory.json") else {}
    if _db_for_date:
        _sample = next(iter(_db_for_date.values()))
        _scraped_at_str = _sample.get("scraped_at", "")
        if _scraped_at_str:
            try:
                _scraped_dt  = datetime.fromisoformat(_scraped_at_str)
                _age_hours   = (datetime.now() - _scraped_dt).total_seconds() / 3600
                _age_label   = f"{_age_hours:.0f}h ago" if _age_hours >= 1 else "just now"
                st.caption(f"📅 Last refreshed: {_scraped_at_str[:10]}  ({_age_label})")
                if _age_hours > 24:
                    st.warning("⚠️ Price data is over 24h old. Click Refresh.", icon="📅")
            except ValueError:
                st.caption(f"📅 Last refreshed: {_scraped_at_str[:10]}")

    if SCRAPER_AVAILABLE:
        if st.button("🔄 Refresh Prices", help="Re-scrape Amazon & Bing and update inventory.json"):
            with st.spinner("🔄 Fetching latest prices from Amazon & Bing... (this takes ~30s)"):
                _scrape_inventory()
            st.cache_resource.clear()   # force model + DB reload
            st.success("✅ Prices updated!")
            st.rerun()
    else:
        st.caption("⚠️ Scraper unavailable (`pip install requests beautifulsoup4`)")

    st.divider()
    with st.expander("⚙️ NDU Weight Tuner", expanded=False):
        st.caption("Adjust utility function weights. Changes apply on the next detection event.")
        ndu_wp = st.slider("Price Weight (wₚ)",  0.10, 0.80, 0.40, 0.05,
                           help="Exponential penalty on high-price vendors (γ = 2.0)")
        ndu_wt = st.slider("Speed Weight (wₜ)",  0.10, 0.80, 0.35, 0.05,
                           help="Exponential penalty on slow-delivery vendors (δ = 1.5)")
        ndu_wr = st.slider("Rating Weight (wᵣ)", 0.10, 0.80, 0.25, 0.05,
                           help="Logarithmic reward for highly-rated vendors")
        # Always normalise weights so they sum to 1.0
        _w_total = ndu_wp + ndu_wt + ndu_wr
        ndu_wp /= _w_total
        ndu_wt /= _w_total
        ndu_wr /= _w_total
        st.caption(f"Current (normalised): wₚ={ndu_wp:.2f}  wₜ={ndu_wt:.2f}  wᵣ={ndu_wr:.2f}  |  γ=2.0  δ=1.5")

# --- RESOURCE LOADING ---
@st.cache_resource
def load_resources():

    local_windows_path = r"C:\Users\Naveen Prasad\Documents\Project_data\RTPD_v3_2.pt"
    cloud_filename = "RTPD_v3_2.pt"
    
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

if model is None:
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
        AI-Powered Video Commerce Engine • v4.0 — NDU Engine
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
    # Fix #1: Use suffix so OpenCV recognises the container format;
    # close immediately so Windows allows VideoCapture to open it.
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()
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

            # --- AI INFERENCE ---
            # We pass the raw 'frame' (BGR).
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            
            annotated_frame = frame.copy()
            
            if results[0].boxes:
                detections_found = True
                for box in results[0].boxes:
                    # 1. Geometry Extraction
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w = x2 - x1
                    h = y2 - y1
                    aspect_ratio = h / w

                    # 2. Get Class Name First
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id] 

                    # 3. Dynamic Subtype Logic (Geometric Logic)
                    # ... inside the loop ...
                    if label == "dove":
                        # Dove Logic: Soap is wide/square, Shampoo is tall
                        if aspect_ratio > 1.5:
                            subtype = "Shampoo"   
                            box_color = (203, 192, 255) 
                        else:
                            subtype = "Soap"      
                            box_color = (255, 255, 255)
                    
                    elif label in ["pepsi", "cocacola", "coca-cola"]:
                        # Soda Logic: Bottles are tall, Cans are short
                        if aspect_ratio > 2.7:
                            subtype = "Bottle"
                            box_color = (255, 0, 0) # Blue (BGR)
                        else:
                            subtype = "Can"
                            box_color = (0, 255, 0) # Green (BGR)
                    
                    else:
                        # Fallback for anything else
                        subtype = "Product"
                        box_color = (0, 165, 255) # Orange

                    # 4. Draw Box
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 3)
                    
                    # 5. Database Lookup
                    # Fix #2: Normalise label so "coca-cola" and "cocacola" both
                    # resolve to "coca_cola" matching the inventory.json keys.
                    label_norm = label.replace("-", "_").replace(" ", "_")
                    lookup_key = f"{label_norm}_{subtype}"

                    # Case-insensitive match against DB keys
                    matched_product = None
                    for db_key in PRODUCT_DB:
                        if lookup_key.lower() == db_key.lower():
                            matched_product = PRODUCT_DB[db_key]
                            break
                    
                    if matched_product:
                        product_name = matched_product.get('name', f"Unknown {label}")
                        vendors      = matched_product.get('vendors', [])

                        current_time_sec = frame_count / fps
                        last_time = st.session_state.last_seen.get(product_name, -100)

                        if vendors and (current_time_sec - last_time) > cooldown:
                            # ── NDU Ranking ───────────────────────────────
                            ranked    = rank_vendors(vendors, wp=ndu_wp, wt=ndu_wt, wr=ndu_wr)
                            winner    = ranked[0]
                            runner_up = ranked[1] if len(ranked) > 1 else None

                            price         = f"\u20b9{winner['price']:.0f}"
                            utility_score = winner['utility_score']
                            why_string    = winner.get('why', 'Best weighted balance')

                            # 2nd-place alt block
                            alt_html = ""
                            if runner_up:
                                alt_html = (
                                    f'<div class="alt-vendor">'
                                    f'\U0001f948 <strong>2nd:</strong> {runner_up["vendor_name"]} &nbsp;'
                                    f'\u2014 \u20b9{runner_up["price"]:.0f} &nbsp;&bull;&nbsp; '
                                    f'{runner_up["delivery_time"]} min &nbsp;&bull;&nbsp; '
                                    f'U&thinsp;=&thinsp;{runner_up["utility_score"]:.4f}'
                                    f'</div>'
                                )

                            st.session_state.last_seen[product_name] = current_time_sec

                            # ── Smart Recommendation Card ─────────────────
                            with live_alert.container():
                                st.markdown(f"""
<div class="ndu-card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <span class="ndu-badge">\U0001f3c6 NDU Rank #1</span>
    <span class="utility-score">{utility_score:.4f}</span>
  </div>
  <div style="font-size:12px;color:#888;margin-bottom:2px;">{product_name} &middot; {subtype}</div>
  <div style="font-size:20px;font-weight:700;color:#fff;margin-bottom:3px;">{winner['vendor_name']}</div>
  <div style="font-size:22px;font-weight:600;color:#00FF7F;margin-bottom:5px;">{price}</div>
  <div style="font-size:12px;color:#999;margin-bottom:8px;">
    \U0001f69a {winner['delivery_time']} min &nbsp;&bull;&nbsp; \u2b50 {winner['rating']}
  </div>
  <div class="why-pill">\U0001f4a1 {why_string}</div>
  {alt_html}
</div>
                                """, unsafe_allow_html=True)

                            st.toast(f"\u2705 Found {product_name}!", icon="\U0001f6d2")

                            st.session_state.history.append({
                                "Time":        f"{current_time_sec:.1f}s",
                                "Product":     product_name,
                                "Vendor":      winner['vendor_name'],
                                "Price":       price,
                                "U_Score":     f"{utility_score:.4f}",
                                "Why":         why_string,
                                "Alt. Vendor": runner_up['vendor_name'] if runner_up else "—",
                                "Alt. Price":  f"\u20b9{runner_up['price']:.0f}" if runner_up else "—",
                                "Link":        winner.get('url', '#'),
                            })

            # 6. DISPLAY (Convert to RGB for Human Eyes only)
            # Fix: use_container_width deprecated post-2025 → width='stretch'
            video_window.image(
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                width="stretch"
            )

        cap.release()
        os.unlink(video_path)  # Fix #1: delete temp file after processing
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
        st.subheader("🛒 NDU Smart Recommendations")
        df = pd.DataFrame(st.session_state.history)

        if not df.empty:
            # Keep only the most recent detection per product — re-detecting the
            # same item adds no new information to the shopping list.
            df = df.drop_duplicates(subset=["Product"], keep="last")

            desired_cols = ["Time", "Product", "Vendor", "Price", "U_Score",
                            "Why", "Alt. Vendor", "Alt. Price", "Link"]
            cols_to_show = [c for c in desired_cols if c in df.columns]
            st.dataframe(
                df[cols_to_show],
                column_config={
                    "U_Score": st.column_config.TextColumn(
                        "Utility Score", help="NDU objective score (higher = better)"
                    ),
                    "Why": st.column_config.TextColumn(
                        "Why It Won", help="Primary factors driving the NDU ranking"
                    ),
                    "Link": st.column_config.LinkColumn(
                        "Buy Now", display_text="Buy Now 🔗", validate="^https://.*"
                    ),
                },
                hide_index=True,
                width="stretch",
            )
