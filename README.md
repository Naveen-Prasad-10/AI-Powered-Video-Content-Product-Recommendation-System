# 🛍️ AI-Powered Video Content Product Recommendation System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red) ![YOLO](https://img.shields.io/badge/AI-YOLOv12--Nano-green) ![Status](https://img.shields.io/badge/Status-Prototype%20v3.0-orange)

This project focuses on the development of an AI-powered system that analyzes video content in real-time to identify consumer products and provide relevant purchasing options.

By combining computer vision–based object detection with a backend data handler, the system aims to bridge the gap between visual media and actionable product information. The current release (**v3**) introduces a web-based dashboard for video analysis.

> **🚀 Live Demo:** https://ai-powered-video-content-appuct-recommendation-system-bhwz7ikh.streamlit.app/

---

## 📸 Interface Preview
<img width="1812" height="665" alt="Screenshot 2026-01-05 214649" src="docs/images/v3-Web App Dashboard/Screenshot 2026-01-05 214649.png" />
<img width="1812" height="665" alt="Screenshot 2026-01-05 214649" src=docs/images/v4/Screenshot 2026-03-01 214817.png" />

### ✨ Key Features (v3.0)

### 1. 📹 Multi-Source Analysis
* **Video Upload:** Drag-and-drop support for MP4, MOV, and AVI files.
* **YouTube Integration:** Fetches and processes video directly from YouTube links (using `yt-dlp`).

### 2. 🧠 Hybrid Detection Logic
* **Brand Recognition:** Fine-tuned **YOLOv12-Nano** model detects broad brand classes (e.g., Pepsi, Coke).
* **Geometric Sub-typing:** Custom **Aspect Ratio Algorithm** mathematically distinguishes between sub-variants without needing extra training classes:
    * *Ratio > 2.7* ➔ Classified as **Bottle**.
    * *Ratio < 2.7* ➔ Classified as **Can**.

### 3. 📝 Multi-Vendor Link Support (Prototype)
* **Static Database:** Maps detected products to a local JSON registry (`inventory.json`).
* **Vendor Selection:** Demonstrates the architecture for handling multiple sellers (e.g., BigBasket vs. Amazon).
* *Note: Live price comparison across real-time vendor data is planned for v4.*

### 4. 🛒 The "Shoppable Timeline"
* **Session History:** Builds a persistent shopping list as the video plays.
* **Interactive UI:** detected items appear in a clean data table with **direct "Purchase" buttons**.

---

## 🛠️ Technical Stack

* **Frontend:** Streamlit (Python-based Web Framework)
* **AI Engine:** Ultralytics YOLOv12-Nano (Custom Fine-Tune)
* **Computer Vision:** OpenCV (Headless)
* **Data Handling:** Pandas (DataFrames) & JSON
* **Media Pipeline:** YT-DLP (Video Extraction)

---

## 💻 Quick Start (for local testing)
### 1. Clone the Repository
The repo contains the requirements,model weights as well as the necessary web framework to run the project locally, at lower latency.

### 2. Install Dependencies
Ensure you have Python installed, then run:

 `pip install -r requirements.txt` 

### 2. Download Model Weights

### 3. Run the Dashboard
Launch the web server locally:
`streamlit run app.py` or `python -m streamlit run app.py`
### Future Roadmap
Competitive Pricing Engine: Integrate live scraping APIs (e.g., RapidAPI) to compare prices in real-time.

User Accounts: Allow users to save their "Wishlist" across sessions.

Mobile App: Convert the Streamlit prototype into a native React Native application.

> 📝 License: 
This project is open-source and available under the MIT License.

