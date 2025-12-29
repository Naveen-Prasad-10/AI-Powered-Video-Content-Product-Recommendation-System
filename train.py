"""
YOLOv12-Nano Training Script
----------------------------
Standard training pipeline for the VidRecAI product detection model.
This script is configured for CUDA-enabled GPUs and uses standard
hyperparameters for balancing training speed and model convergence.
"""

from ultralytics import YOLO

def main():
    """
    Main training entry point.
    Ensures safe multiprocessing on Windows via the __main__ guard.
    """
    
    # 1. Load Pretrained Model
    # We use 'yolo12n.pt' (Nano) for real-time edge performance.
    model = YOLO("yolo12n.pt")

    print("🚀 Starting Model Training (100 Epochs)...")

    # 2. Execute Training
    results = model.train(
        # ---------------------------------------------------------
        # ⚠️ CONFIGURATION NOTE:
        # When pasting your Windows file path below, ensure you keep the 'r' prefix
        # (e.g., data=r"C:\Path\To\data.yaml") to avoid unicode errors.
        # ---------------------------------------------------------
        data=r"path/to/your/data.yaml",
        
        # Training Duration
        epochs=100,
        
        # Image Settings
        imgsz=640,
        batch=16,       # Standard batch size for 6GB+ VRAM
        device=0,       # Use primary GPU
        
        # Output Naming
        name="vidrecai_standard_v1",

        # Hyperparameters & System Settings
        workers=4,      # optimized for 4-core+ CPUs
        patience=0,     # Disable early stopping
        close_mosaic=10,# Disable mosaic augmentation for final epochs
        lr0=0.01,       # Initial Learning Rate
        lrf=0.01,       # Final Learning Rate (Cosine Decay)
        augment=True,   # Default augmentation enabled
    )

    print("✅ Training completed successfully.")
    print("   Weights saved to: runs/detect/vidrecai_standard_v1")


if __name__ == "__main__":
    main()
