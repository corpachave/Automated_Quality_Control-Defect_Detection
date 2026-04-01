# Convert COCO to YOLO
from ultralytics.data.utils import convert_coco # type: ignore
import os

labels_dir = "raw_data/PCB_Defect/annotation"
save_dir = "raw_data/converted_PCB_Defect"

# Check if save_dir exists and is not empty
if os.path.exists(save_dir) and os.listdir(save_dir):
    print(f"Skipping conversion: '{save_dir}' already contains files.")
else:
    os.makedirs(save_dir, exist_ok=True)
    convert_coco(
        labels_dir=labels_dir,
        save_dir=save_dir
    )
    print("Conversion completed.")
    print(dir(ultralytics.data)) # type: ignore