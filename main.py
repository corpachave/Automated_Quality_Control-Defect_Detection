from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from ultralytics import YOLO
import time

app = FastAPI()

model = YOLO("runs/detect/pcb_yolov8n/weights/best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(img)

    end_time = time.time()
    inference_time = f"{(end_time - start_time)*1000:.1f}ms"

    detections = []
    for box in results[0].boxes:
        cls = int(box.cls)
        label = results[0].names[cls]
        detections.append({
            "label": label,
            "confidence": float(box.conf),
            "bbox": box.xyxy.tolist()[0]
        })

    status = "DEFECT" if len(detections) > 0 else "PASS"

    return {
        "status": status,
        "detections": detections,
        "inference_time": inference_time
    }