# Convert COCO to YOLO
from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="raw_data/PCB_Defect/annotation",
    save_dir="raw_data/converted_PCB_Defect"
)

