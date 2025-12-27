from ultralytics import YOLO
import cv2
import torch

# Inference configuration
CONF_THRESHOLD = 0.75


def main():
    """
    Run real-time object detection using a trained YOLOv12-Nano model
    on a live webcam feed. Safe for iGPU / CPU-only systems
    """

    # Select computation device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference on {device.upper()}.")

    # Load trained model
    model = YOLO("pepsi_final.pt")

    # Initialize webcam capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    print("Webcam active. Press 'Q' to exit.")

    while True:
        # Capture a single frame from the webcam
        success, frame = cap.read()
        if not success:
            break

        # Run model inference on the frame
        results = model.predict(
            frame,
            conf=CONF_THRESHOLD,
            device=device,
            verbose=False
        )

        # Render detections on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("Pepsi Detector (Press Q to Exit)", annotated_frame)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated successfully.")


if __name__ == "__main__":
    main()
