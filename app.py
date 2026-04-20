import os
import streamlit as st
import requests
from PIL import Image
import io
import cv2
import numpy as np
import time
import tensorflow as tf
from ultralytics import YOLO

st.title("VisionSpec QC - PCB Inspection")

# Settings
CLASS_MODEL_PATH = "visionspec_qc_model.keras"
DETECTION_MODEL_PATH = "runs/detect/pcb_yolov8n/weights/best.pt"
IMAGE_SIZE = (224, 224)
THRESHOLD = 0.7

# Load models
@st.cache_resource
def load_models():
    class_model = tf.keras.models.load_model(CLASS_MODEL_PATH)
    detect_model = YOLO(DETECTION_MODEL_PATH) if os.path.exists(DETECTION_MODEL_PATH) else None
    return class_model, detect_model

option = st.selectbox("Choose input method", ["Upload Image", "Webcam Capture", "Live Webcam"])

# === Upload Image ===
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload PCB Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")
        file_bytes = uploaded_file.getvalue()

        if st.button("Run Inspection"):
            with st.spinner("Running inspection..."):
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    files={"file": file_bytes}
                )
                data = response.json()

            status = data["status"]
            detections = data["detections"]
            inference_time = data["inference_time"]

            if status == "PASS":
                st.success(f"Status: {status}")
            else:
                st.error(f"Status: {status}")

            st.write(f"Inference Time: {inference_time}")

            if detections:
                st.write("Detections:")
                st.table(detections)

                img_array = np.array(image)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                for det in detections:
                    x1, y1, x2, y2 = map(int, det["bbox"])
                    label = det["label"]
                    conf = det["confidence"]
                    cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img_array, f"{label} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                st.image(img_array, caption="Detected Defects")
            else:
                st.write("No defects detected.")

# === Webcam Capture (Single Image) ===
elif option == "Webcam Capture":
    camera_input = st.camera_input("Capture PCB Image")
    if camera_input:
        image = Image.open(camera_input)
        st.image(image, caption="Captured Image")
        file_bytes = camera_input.getvalue()

        if st.button("Run Inspection"):
            with st.spinner("Running inspection..."):
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    files={"file": file_bytes}
                )
                data = response.json()

            status = data["status"]
            detections = data["detections"]
            inference_time = data["inference_time"]

            if status == "PASS":
                st.success(f"Status: {status}")
            else:
                st.error(f"Status: {status}")

            st.write(f"Inference Time: {inference_time}")

            if detections:
                st.write("Detections:")
                st.table(detections)

                img_array = np.array(image)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                for det in detections:
                    x1, y1, x2, y2 = map(int, det["bbox"])
                    label = det["label"]
                    conf = det["confidence"]
                    cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img_array, f"{label} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                st.image(img_array, caption="Detected Defects")
            else:
                st.write("No defects detected.")

# === Live Webcam (Real-time) ===
elif option == "Live Webcam":
    st.info("Starting live webcam inference... Click 'Start' to begin, 'Stop' to end.")
    
    if "live_running" not in st.session_state:
        st.session_state.live_running = False
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Live Inference"):
            st.session_state.live_running = True
    with col2:
        if st.button("Stop"):
            st.session_state.live_running = False
    
    if st.session_state.live_running:
        try:
            class_model, detect_model = load_models()
            
            # Placeholder for video feed
            frame_placeholder = st.empty()
            status_placeholder = st.empty()
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot access webcam")
                st.session_state.live_running = False
            else:
                prev_time = time.time()
                fps = 0.0
                
                while st.session_state.live_running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Classification
                    img_resized = cv2.resize(frame, IMAGE_SIZE)
                    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    img_normalized = img_rgb.astype("float32") / 255.0
                    img_normalized = np.expand_dims(img_normalized, axis=0)
                    
                    score = float(class_model.predict(img_normalized, verbose=0)[0][0])
                    label = "DEFECT" if score >= THRESHOLD else "PASS"
                    
                    frame_out = frame.copy()
                    
                    # Detection if DEFECT
                    if label == "DEFECT" and detect_model is not None:
                        results = detect_model(frame)
                        for result in results:
                            boxes = result.boxes
                            for box in boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                conf = float(box.conf[0])
                                cls = int(box.cls[0])
                                det_label = result.names[cls] if cls in result.names else str(cls)
                                text = f"{det_label} {conf:.2f}"
                                cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(frame_out, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                    
                    # FPS calculation
                    current_time = time.time()
                    fps = 1.0 / max((current_time - prev_time), 1e-6)
                    prev_time = current_time
                    
                    # Overlay status and FPS
                    color = (0, 255, 0) if label == "PASS" else (0, 0, 255)
                    cv2.putText(frame_out, f"Status: {label} ({score:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame_out, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Display frame
                    frame_out_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_out_rgb, caption="Live PCB Inspection", channels="RGB")
                    
                    # Update status
                    status_placeholder.write(f"**Status:** {label} | **Confidence:** {score:.4f} | **FPS:** {fps:.1f}")
                    
                    # Check stop button
                    if not st.session_state.live_running:
                        break
                
                cap.release()
                st.session_state.live_running = False
                st.success("Live inference stopped.")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.live_running = False