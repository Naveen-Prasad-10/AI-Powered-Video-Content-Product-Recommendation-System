# AI-Powered Video Content Product Recommendation System

> **Status:** Functional MVP (v2)
> **Target:** Real-time retail detection on consumer hardware

## Project Overview
This project focuses on the development of an AI-powered system that analyzes video content in real time to identify consumer products and provide relevant product recommendations. By combining computer vision–based object detection with a backend recommendation pipeline, the system aims to bridge the gap between visual media and actionable product information.

The core functionality involves detecting products appearing in video streams (such as advertisements, reviews, or user-generated content) using a deep learning–based object detection model. Once a product is identified, the system maps the detection to a structured product database and retrieves associated metadata, including product name, pricing details, and purchase links. These recommendations are then presented to the user in an intuitive and interactive manner.

## Technical Implementation (MVP Scope)
While the long-term vision encompasses broad product discovery, the current release focuses on a specialized implementation for **High-Speed Retail Detection**.



### Key Capabilities
* **Real-Time Inference:** Utilizes a fine-tuned **YOLOv12-Nano** architecture to achieve **30+ FPS** on standard consumer GPUs (e.g., NVIDIA RTX series).
* **Multi-Brand Logic:** Successfully distinguishes between **Pepsi** and **Coca-Cola** products.
* **Geometric Sub-typing:** Differentiates between *Cans* and *Bottles* using a custom **Aspect Ratio Algorithm** (Ratio > 2.7 classified as Bottle) rather than computationally expensive segmentation masks.
* **False Positive Rejection:** Implements a "Negative Sample" training strategy to explicitly ignore human faces and background clutter.
* **Augmented Commerce:** Integrated `Spacebar` trigger that instantly opens the purchase URL for the detected product.

### System Pipeline
The system operates on a hybrid **Detect → Analyze → Augment** architecture:
1.  **Input:** Raw video feed (Webcam or Video File).
2.  **Detection:** YOLOv12n identifies brand bounding boxes.
3.  **Logic Layer:** A Python script filters low-confidence detections and calculates object geometry (Height/Width).
4.  **Output:** OpenCV renders the "Smart Label" overlay and listens for user interaction.

## Repository Structure

| File | Description |
| :--- | :--- |
| `yolov12n_multi_v2.pt` | The fine-tuned model weights (Pepsi vs. Coke). |
| `detect.py` | Main inference script. Handles webcam streaming, geometric logic, and UI. |
| `train.py` | Optimized training script with thermal safety limits (thread clamping). |
| `requirements.txt` | Project dependencies (Ultralytics, OpenCV, Pandas, etc.). |

## 💻 Quick Start 

### 1. Install Dependencies
Ensure you have Python installed, then run:

 `pip install -r requirements.txt` 

### 2. Download Model Weights
Because the model weights are binary files, they are stored in the **Releases** section to keep the repository lightweight.

1.  Navigate to the **[Releases](../../releases)** page of this repository.
2.  Download the latest model file: **`yolov12n_multi_v2.pt`**.
3.  Move the downloaded file into the **root directory** of this project (the same folder where `detect.py` is located).

### 3. Run Inference
Start the real-time detection on your primary webcam:
python detect.py
### 4. Controls
Spacebar: Open the purchase link for the currently detected item in your default browser.

Q: Quit the application.
