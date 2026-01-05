"""
AI-Powered Video Content Product Recommendation System
------------------------------------------------------
"""
import cv2
import torch
import webbrowser
import time
import json
import os
from ultralytics import YOLO

# --- SYSTEM CONFIGURATION ---
CONF_THRESHOLD = 0.70       
RATIO_THRESHOLD = 2.7       
MIN_ASPECT_RATIO = 1.50     
JSON_FILE = "inventory.json"

def load_inventory():
    if not os.path.exists(JSON_FILE): return {}
    with open(JSON_FILE, 'r') as f:
        try: return json.load(f)
        except: return {}

def draw_smart_label(img, text, x, y, bg_color=(0, 0, 0), txt_color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, 0.6, 2)
    cv2.rectangle(img, (x, y - text_h - 10), (x + text_w + 10, y + 5), bg_color, -1)
    cv2.putText(img, text, (x + 5, y), font, 0.6, txt_color, 2)
    return y + 35 

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 AI-Powered Video Content Product Recommendation System Active on {device.upper()}")
    
    product_db = load_inventory()

    # ⚠️ YOUR LOCAL PATH
    model_path = r"C:\Users\Naveen Prasad\Documents\Project_data\RTPD_v2.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at: {model_path}")
        return
        
    model = YOLO(model_path) 
    cap = cv2.VideoCapture(0)
    
    print("🎥 Stream Started. Controls: [SPACE] to Buy | [Q] to Quit")

    while True:
        success, frame = cap.read()
        if not success: break

        results = model.predict(frame, conf=CONF_THRESHOLD, device=device, verbose=False)
        annotated_frame = frame.copy()
        current_product_link = None 

        if results[0].boxes:
            for box in results[0].boxes:
                x, y, w, h = box.xywh[0].cpu().numpy()
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                aspect_ratio = h / w
                
                if aspect_ratio < MIN_ASPECT_RATIO: continue 
                
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 3)

                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                subtype = "Bottle" if aspect_ratio > RATIO_THRESHOLD else "Can"
                ratio_color = (0, 0, 255) if subtype == "Bottle" else (0, 255, 0)

                lookup_key = f"{class_name}_{subtype}"
                info = product_db.get(lookup_key, {"name": f"Unknown: {class_name} {subtype}", "price": "N/A", "url": None})
                current_product_link = info['url']

                start_x, current_y = int(x1), int(y1) - 20
                current_y = draw_smart_label(annotated_frame, info['name'], start_x, current_y, bg_color=(0,0,0), txt_color=(0, 255, 255))
                current_y = draw_smart_label(annotated_frame, f"Price: {info['price']}", start_x, current_y, bg_color=(0, 100, 0))
                cv2.putText(annotated_frame, f"Ratio: {aspect_ratio:.2f}", (int(x1), int(y2) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ratio_color, 2)
                
                if current_product_link:
                    cv2.putText(annotated_frame, "[SPACE] to Buy", (int(x - 50), int(y2) + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Real-Time Product Detection & Recommendation", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == 32: 
            if current_product_link:
                webbrowser.open(current_product_link)
                time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
