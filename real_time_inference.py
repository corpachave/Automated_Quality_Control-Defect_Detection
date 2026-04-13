import os
import time
import argparse
import cv2
import numpy as np
import tensorflow as tf # type: ignore
from ultralytics import YOLO #type: ignore

# SETTINGS
CLASS_MODEL_PATH = "visionspec_qc_model.keras"
DETECTION_MODEL_PATH = "runs/detect/pcb_yolov8n/weights/best.pt"
IMAGE_SIZE = (224, 224)
THRESHOLD = 0.5

# LOAD MODELS

## CLASSIFICATION MODEL
def load_classification_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Classification model not found: {model_path}")
    return tf.keras.models.load_model(model_path)

## DETECTION MODEL
def load_detection_model(model_path: str):
    if not os.path.exists(model_path):
        print(f"Warning: Detection model not found at {model_path}. Localization will be disabled.")
        return None
    return YOLO(model_path)

# PREPROCESSING AND INFERENCE
def preprocess_frame(frame: np.ndarray):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# CLASSIFICATION AND DETECTION
def predict_class(frame: np.ndarray, model):
    image = preprocess_frame(frame)
    score = float(model.predict(image, verbose=0)[0][0])
    label = "DEFECT" if score >= THRESHOLD else "PASS"
    return label, score

# DRAW DETECTION BOXES
def draw_detection_results(frame: np.ndarray, results):
    annotated = frame.copy()
    if results is None:
        return annotated

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            score = float(box.conf[0])
            cls = int(box.cls[0])
            label = result.names[cls] if cls in result.names else str(cls)
            text = f"{label} {score:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return annotated

# ADD OVERLAY (Status + FPS)
def add_status_overlay(frame: np.ndarray, label: str, score: float, fps: float):
    status = f"Status: {label} ({score:.2f})"
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame

# LIVE INFERENCE (Webcam / Video)
def run_live_inference(source: int | str, class_model, detect_model):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, score = predict_class(frame, class_model)
        frame_out = frame.copy()

        if label == "DEFECT" and detect_model is not None:
            results = detect_model(frame)
            frame_out = draw_detection_results(frame_out, results)

        current_time = time.time()
        fps = 1.0 / max((current_time - prev_time), 1e-6)
        prev_time = current_time

        frame_out = add_status_overlay(frame_out, label, score, fps)
        cv2.imshow("PCB Inspection", frame_out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# IMAGE INFERENCE (Single Image)
def run_image_inference(image_path: str, class_model, detect_model):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path not found: {image_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Unable to load image: {image_path}")

    label, score = predict_class(frame, class_model)
    frame_out = frame.copy()

    if label == "DEFECT" and detect_model is not None:
        results = detect_model(frame)
        frame_out = draw_detection_results(frame_out, results)

    frame_out = add_status_overlay(frame_out, label, score, 0.0)
    cv2.imshow("PCB Inspection", frame_out)
    print(f"Image inference result: {label} ({score:.4f})")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run live PCB defect classification and localization.")
    parser.add_argument("--source", default="0", help="Video source: 0 for webcam or path to video/image")
    parser.add_argument("--class_model", default=CLASS_MODEL_PATH, help="Path to classification model")
    parser.add_argument("--detect_model", default=DETECTION_MODEL_PATH, help="Path to detection model weights")

    args = parser.parse_args()
    class_model = load_classification_model(args.class_model)
    detect_model = load_detection_model(args.detect_model)

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    if isinstance(source, str) and os.path.isfile(source):
        run_image_inference(source, class_model, detect_model)
    else:
        run_live_inference(source, class_model, detect_model)
