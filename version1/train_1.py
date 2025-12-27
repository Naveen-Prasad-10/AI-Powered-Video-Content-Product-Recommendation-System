from ultralytics import YOLO


def main():
    """
    Train a YOLOv12-Nano model on a custom dataset.

    Note:
    On Windows, the training logic must be placed inside a function
    and protected by the '__main__' guard to avoid multiprocessing issues.
    """

    # Load pretrained YOLOv12-Nano model
    model = YOLO("yolo12n.pt")

    print("Starting model training (100 epochs)...")

    results = model.train(
        data="path/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        name="pepsi_overkill",

        # Windows-safe training settings
        workers=4,
        patience=0,
        close_mosaic=10,
        lr0=0.01,
        lrf=0.01,
        augment=True,
    )

    print("Training completed successfully.")


if __name__ == "__main__":
    main()
