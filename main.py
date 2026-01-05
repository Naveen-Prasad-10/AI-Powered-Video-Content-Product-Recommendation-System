"""
Real-Time Product Recognition & Augmented Commerce (VidRecAI)
------------------------------------------------------------
A Computer Vision application that detects soft drink brands (Pepsi/Coke)
and their subtypes (Can/Bottle) using YOLOv12-Nano and geometric heuristics.

Features:
- Real-time inference on webcam feed.
- Geometric logic to distinguish Cans vs. Bottles.
- "Smart Label" UI overlay with Price and Name.
- [SPACE] key trigger to open purchase links.
"""

import time
import webbrowser
import cv2
import torch
from ultralytics import YOLO

# --- CONFIGURATION CONSTANTS ---
CONF_THRESHOLD = 0.70       # Confidence required to accept a detection
RATIO_THRESHOLD = 2.7       # Aspect Ratio boundary: Ratio > 2.7 is a Bottle
MIN_ASPECT_RATIO = 1.50     # Face Filter: Ignore objects "squarer" than this

# --- MOCK PRODUCT DATABASE ---
# Maps (Class Name, Subtype) -> Metadata
PRODUCT_DB = {
    ("pepsi", "Can"): {
        "name": "Pepsi Can (330ml)",
        "price": "Rs.40",
        "url": "https://www.bigbasket.com/pd/40327060/pepsi-soft-drink-330-ml"
    },
    ("pepsi", "Bottle"): {
        "name": "Pepsi Bottle (750ml)",
        "price": "Rs.37",
        "url": "https://www.bigbasket.com/pd/251047/pepsi-soft-drink-750-ml/"
    },
    ("coca_cola", "Can"): {
        "name": "Coke Can (300ml)",
        "price": "Rs.34",
        "url": "https://www.bigbasket.com/pd/100401160/coca-cola-soft-drink-original-taste-300-ml-can/"
    },
    ("coca_cola", "Bottle"): {
        "name": "Coke Bottle (750ml)",
        "price": "Rs.31",
        "url": "https://www.bigbasket.com/pd/251023/coca-cola-soft-drink-original-taste-750-ml-pet-bottle/"
    }
}


def draw_smart_label(img, text, x, y, bg_color=(0, 0, 0), txt_color=(255, 255, 255)):
    """
    Draws text with a solid background box to ensure readability.
    Returns: The Y-coordinate for the next line of text.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Calculate text size to dynamically size the background box
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw filled rectangle (Background)
    cv2.rectangle(img, (x, y - text_h - 10), (x + text_w + 10, y + 5), bg_color, -1)
    
    # Draw text on top
    cv2.putText(img, text, (x + 5, y), font, font_scale, txt_color, thickness)
    
    return y + 35  # Return new Y position for next line


def main():
    # 1. Hardware Initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Commerce System Active on {device.upper()}")
    
    # 2. Load Model
    # NOTE: When providing a Windows file path, use the 'r' prefix (e.g., r"C:\Path\To\Model.pt")
    # to treat backslashes as literal characters and avoid unicode errors.
    model = YOLO("path/to/your/RTPD_v2.pt") 

    # 3. Open Video Stream
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Webcam not found.")
        return

    print("🎥 Camera Active. Point at a product.")
    print("👉 Controls: [SPACE] to Buy | [Q] to Quit")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 4. Inference
        # verbose=False suppresses the default library terminal output
        results = model.predict(frame, conf=CONF_THRESHOLD, device=device, verbose=False)
        
        # Create a clean copy for drawing custom UI (avoids double-drawing)
        annotated_frame = frame.copy()
        current_product_link = None 

        if results[0].boxes:
            for box in results[0].boxes:
                # A. Extract Geometry
                # xywh: Center X, Center Y, Width, Height
                # xyxy: Top-Left X, Top-Left Y, Bottom-Right X, Bottom-Right Y
                x, y, w, h = box.xywh[0].cpu().numpy()
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                aspect_ratio = h / w
                
                # B. Geometric Filter (Face Rejection)
                # Ignore objects that are too square (likely faces or random clutter)
                if aspect_ratio < MIN_ASPECT_RATIO:
                    continue 
                
                # --- Valid Product Detected ---
                
                # C. Draw Bounding Box
                # Cyan color (BGR: 255, 255, 0)
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 3)

                # D. Classification Logic
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                # Subtype Logic based on Aspect Ratio
                if aspect_ratio > RATIO_THRESHOLD:
                    subtype = "Bottle"
                    ratio_color = (0, 0, 255) # Red text for debug
                else:
                    subtype = "Can"
                    ratio_color = (0, 255, 0) # Green text for debug

                # E. Retrieve Metadata
                # Default to "Unknown" if combination not found in DB
                info = PRODUCT_DB.get((class_name, subtype), {
                    "name": f"{class_name} {subtype}",
                    "price": "N/A",
                    "url": None
                })
                current_product_link = info['url']

                # F. Render UI Layer
                current_y = int(y1) - 20
                start_x = int(x1)

                # Label 1: Product Name (Black Box, Cyan Text)
                current_y = draw_smart_label(annotated_frame, info['name'], start_x, current_y, 
                                             bg_color=(0, 0, 0), txt_color=(0, 255, 255))
                
                # Label 2: Price (Dark Green Box, White Text)
                current_y = draw_smart_label(annotated_frame, f"Price: {info['price']}", start_x, current_y, 
                                             bg_color=(0, 100, 0))
                
                # Debug Info: Show Ratio at bottom of box
                cv2.putText(annotated_frame, f"Ratio: {aspect_ratio:.2f}", (int(x1), int(y2) + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, ratio_color, 2)
                
                # Call-to-Action
                cv2.putText(annotated_frame, "[SPACE] to Buy", (int(x) - 50, int(y2) + 55), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 5. Display Output
        cv2.imshow("VidRecAI - Smart Shopping Assistant", annotated_frame)

        # 6. Input Handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        elif key == 32: # SPACEBAR
            if current_product_link:
                print(f"🔗 Opening External Link: {current_product_link}")
                webbrowser.open(current_product_link)
                time.sleep(0.5) # Debounce to prevent multiple tabs

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✅ System Shutdown Successfully.")

if __name__ == "__main__":
    main()
