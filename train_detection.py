import os
import argparse
from ultralytics import YOLO # type: ignore

# CONFIG

DATA_CONFIG = "data.yaml"
MODEL_NAME = "yolov8n.pt"
RUN_NAME = "pcb_yolov8n"
PROJECT_DIR = os.path.abspath("runs/detect")
OUTPUT_WEIGHTS = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")

# TRAINING FUNCTION

def train_detection(data_cfg: str, epochs: int, batch: int, imgsz: int, device: str):
    model = YOLO(MODEL_NAME)

    print(f"Training YOLOv8 detection model with data config: {data_cfg}")
    print(f"Save directory: {PROJECT_DIR}/{RUN_NAME}")

    model.train(
        data=data_cfg,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=PROJECT_DIR,
        name=RUN_NAME,
        device=device,
        save=True,
        patience=10,
        cache=True,
    )

    if os.path.exists(OUTPUT_WEIGHTS):
        print(f"Detection model training completed. Best weights saved to: {OUTPUT_WEIGHTS}")
    else:
        print("Training completed, but best weights were not found. Check the ultralytics logs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 detection model for PCB defects.")
    parser.add_argument("--data", default=DATA_CONFIG, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size for detection")
    parser.add_argument("--device", default="0", help="Compute device, e.g. 0 or cpu")

    args = parser.parse_args()

    train_detection(
        data_cfg=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )
