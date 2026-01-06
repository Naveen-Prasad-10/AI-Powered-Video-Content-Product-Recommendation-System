import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import json
import pandas as pd
import subprocess  # <--- NEEDED FOR THE FIX

# ... (Keep your imports and Page Config same as before) ...

# --- NEW FUNCTION: VIDEO SANITIZER ---
def sanitize_video(input_path):
    """
    Force-converts video to a Linux-friendly format (H.264/MP4) 
    using system FFMPEG. Fixes 'No frames found' errors on Colab.
    """
    output_path = input_path.replace(".mp4", "_fixed.mp4")
    
    # Simple FFMPEG command to re-encode video
    # -y = overwrite, -vcodec libx264 = standard format, -preset ultrafast = speed over size
    command = [
        "ffmpeg", "-y", 
        "-i", input_path,
        "-vcodec", "libx264",
        "-acodec", "aac", 
        "-preset", "ultrafast", 
        output_path
    ]
    
    try:
        # Run conversion silently
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
    except Exception as e:
        st.warning(f"⚠️ Video conversion failed. Trying original file. Error: {e}")
        return input_path

# ... (Load resources function stays the same) ...
model, PRODUCT_DB = load_resources()

# ... (UI Setup stays the same) ...

if uploaded_file:
    # 1. Save the Raw Upload
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") 
    tfile.write(uploaded_file.read())
    raw_video_path = tfile.name

    col_video, col_live = st.columns([0.65, 0.35])

    # 2. RUN THE SANITIZER (The Fix)
    with col_live:
        with st.spinner("Preparing video engine..."):
            video_path = sanitize_video(raw_video_path)

    # 3. DEBUG: Show user if we swapped files (Optional, good for you to know)
    # if video_path != raw_video_path:
    #     st.toast("✅ Video optimized for Linux", icon="🔧")

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
