# 🛍️ ShopVision Pro: Real-Time Detection & Multi-Criteria Vendor Optimization

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Framework](https://img.shields.io/badge/Framework-FastAPI-009688) ![AI](https://img.shields.io/badge/AI-YOLOv12--Nano-green) ![Math](https://img.shields.io/badge/Math-NumPy-blueviolet) ![Status](https://img.shields.io/badge/Status-Research%20Prototype%20v4.0-orange)

ShopVision Pro is an AI-powered Decision Support System that bridges the gap between visual media and actionable consumer intelligence. 

Moving beyond standard object detection, this system utilizes a decoupled "Eye-to-Brain" architecture. It detects consumer products in real-time video feeds and passes them through a custom mathematical optimization algorithm to instantly calculate and recommend the optimal purchasing path across multiple e-commerce vendors.

---

## ✨ Key Research Innovations (v4.0)

### 1. 🧠 The "Eye": Optimized Perception Layer
* **Model:** Fine-tuned **YOLOv12-Nano** model detects broad brand classes (e.g., Pepsi, Coca-Cola) running efficiently on consumer-grade edge hardware.
* **Agnostic NMS & Color Correction:** Implements strict BGR-to-RGB conversion pipelines and Non-Maximum Suppression to eliminate false-positive ghosting.
* **Geometric Sub-typing:** Custom Aspect Ratio Algorithm mathematically distinguishes between sub-variants without inflating the neural network class count:
  * *Ratio > 2.7* ➔ Classified as **Bottle**.
  * *Ratio < 2.7* ➔ Classified as **Can**.

## 📸 Interface Preview
<img width="1812" height="665" alt="Screenshot 2026-01-05 214649" src="docs/images/v3-Web App Dashboard/Screenshot 2026-01-05 214649.png" />


**U = [W_p × e^(-γ × P)] + [W_t × e^(-δ × T)] + [W_r × ln(1 + R)]**

* **P, T, R:** Min-Max normalized values for Price, Delivery Time, and Rating.
* **W (Weights):** Dynamic weights adjusted based on the product profile (e.g., high delivery weight for impulse items like cold soda; high price weight for bulk items like shampoo).
* **γ, δ (Decay Constants):** Controls how aggressively a vendor is penalized for slow delivery or high prices.

### 3. ⚡ High-Speed FastAPI Engine
* **Streaming Architecture:** Migrated from Streamlit to a native FastAPI asynchronous backend, ensuring <16ms latency for the video generator loop.
* **Persistent Session Memory:** Tracks object history dynamically to prevent shopping cart duplication loops during streaming video playback.

### 4. 🛒 Smart Recommendation Dashboard
* **Shoppable Timeline:** Detected items populate a clean HTML/JS sidebar.
* **Mathematical Transparency:** Instead of just a "Buy" button, the UI presents the #1 mathematically ranked vendor alongside the logic of why it won the utility calculation.

## 🛠️ Technical Stack

* **Frontend:** Streamlit (Python-based Web Framework)
* **AI Engine:** Ultralytics YOLOv12-Nano (Custom Fine-Tune)
* **Computer Vision:** OpenCV (Headless)
* **Data Handling:** Pandas (DataFrames) & JSON
* **Media Pipeline:** YT-DLP (Video Extraction)

---

## 🛠️ Technical Stack

* **Backend/API:** FastAPI & Uvicorn (Asynchronous Web Server)
* **AI Engine:** Ultralytics YOLOv12-Nano (PyTorch)
* **Computer Vision:** OpenCV (Headless) & PIL
* **Mathematical Optimization:** NumPy
* **Frontend:** HTML5, CSS3, Vanilla JavaScript (Server-Side Rendered)

---

## 💻 Quick Start (Local Deployment)

### 1. Clone the Repository
Clone the repository to your local machine. This package includes the required API architecture and the `inventory.json` simulated market cache.

### 2. Install Dependencies
Ensure you have Python 3.9+ installed, then run:
`pip install -r requirements.txt`

### 3. Ensure Model Weights
Place your fine-tuned model weights (e.g., `RTPD_v2.pt`) in the root directory. If not found, the system will safely fallback to a default `yolov8n.pt` model.

### 4. Run the Engine
Launch the Uvicorn ASGI server to start the application:
`python -m uvicorn main:app --reload`

Navigate to `http://127.0.0.1:8000` in your browser to access the live dashboard.

---

## 🚀 Future Roadmap
* **Dynamic Review Clustering:** Implementation of K-Means clustering (using `scikit-learn` and text embeddings) to autonomously group and display qualitative user reviews (e.g., "Shipping Issues" vs. "Great Value").
* **Live API Integration:** Replacing the `inventory.json` static cache with live scraping endpoints to feed real-time pricing data into the NDU equation.

> **📝 License:** This project is open-source and available under the MIT License.
