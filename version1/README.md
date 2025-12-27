# Real-Time Product Detector- Version 1 (YOLOv12-Nano)

This project explores the development of a real-time object detection system capable of identifying **Pepsi-branded products** from a live camera feed. The system is based on the **YOLOv12-Nano** architecture and is being developed incrementally, with an emphasis on understanding the complete pipeline from data collection and training to real-time inference.

## Current Status (MVP)
The current version represents an early **Minimum Viable Product (MVP)**. The model was trained and tested locally using an NVIDIA RTX 3050 GPU and is intended as a functional baseline rather than a finalized solution.

- **Model Architecture:** YOLOv12-Nano (fine-tuned)
- **Dataset:** Custom dataset consisting of **90 manually annotated images**
  - Includes different Pepsi product types such as cans, bottles, cups, and popcorn buckets
  - Human faces are present as 'negative samples' in the dataset to observe and reduce unintended detections
- **Validation Set:** 5 images
- **Test Set:** 4 images
- **Training Setup:** 100 training epochs
- **Inference Behaviour:** The model is able to detect Pepsi products in a live webcam feed with confidence values typically above **0.75**.
- **Observed Behaviour:**
  - Detections remain stable even when products are partially covered
  - No incorrect detections on human faces were observed during testing
## Repository Structure

- **RTPD_version1.pt**  
  Fine-tuned YOLOv12-Nano model weights trained on a custom dataset over 100 epochs.  
  This file contains the parameters required by `run_stable.py` to initialize the model for real-time inference.

- **train_1.py**  
  Training script used to fine-tune the YOLOv12-Nano model on the custom dataset.

- **run_1.py**  
  Script for performing real-time product detection using a live webcam feed.

- **requirements.txt**  
  Lists the Python dependencies required to run the training and inference scripts.

    
## Scope and Current Limitations

At this stage, the model treats all Pepsi products as a **single class (`pepsi`)**. Although the dataset contains multiple product variants, the current focus has been on achieving reliable real-time detection before introducing finer-grained classification. Due to the limited size of the dataset, performance metrics are currently observational rather than statistically comprehensive.

## 🎯 Project Goals & Roadmap
As development continues, the project will be expanded in the following directions:

- [ ] **Sub-type Recognition:** Differentiating between packaging formats such as cans, bottles, and cups
- [ ] **Database Integration:** Map detected Class IDs to a backend database to retrieve metadata (Price, Product Name).
- [ ] **Web Framework:** Display "Purchase Links" and product info overlays on the live video stream.
