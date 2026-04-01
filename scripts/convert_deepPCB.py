import os
import cv2

INPUT_DIR = "raw_data/DeepPCB-master/PCBData"
OUTPUT_IMG = "data/detection/images/train"
OUTPUT_LBL = "data/detection/labels/train"

os.makedirs(OUTPUT_IMG, exist_ok=True)
os.makedirs(OUTPUT_LBL, exist_ok=True)

# Class mapping to unify with other datasets
CLASS_MAP = {
    1: 0,  # open
    2: 1,  # short
    3: 2,  # mouse bite
    4: 3,  # spur
    5: 4,  # pinhole
    6: 5   # spurious copper
}

def convert_bbox(x1, y1, x2, y2, w, h):
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    width = (x2 - x1) / w
    height = (y2 - y1) / h
    return x_center, y_center, width, height


for group in os.listdir(INPUT_DIR):
    group_path = os.path.join(INPUT_DIR, group)

    if not os.path.isdir(group_path):
        continue

    # Extract folder name (00041 from group00041)
    folder_name = group.replace("group", "")

    img_folder = os.path.join(group_path, folder_name)
    lbl_folder = os.path.join(group_path, f"{folder_name}_not")

    if not os.path.exists(img_folder) or not os.path.exists(lbl_folder):
        continue

    img_files = [f for f in os.listdir(img_folder) if f.endswith(".jpg")]

    for file in img_files:

        # ONLY use defect images
        if "_test" not in file:
            continue

        img_path = os.path.join(img_folder, file)

        # Extract base ID (00041000)
        base_name = file.split("_")[0]

        txt_path = os.path.join(lbl_folder, base_name + ".txt")

        if not os.path.exists(txt_path):
            continue

        image = cv2.imread(img_path)

        # Safety check
        if image is None:
            print(f"Failed to read image: {img_path}")
            continue

        h, w = image.shape[:2]

        yolo_lines = []

        with open(txt_path, "r") as f:
            for line in f.readlines():
                x1, y1, x2, y2, cls = map(int, line.strip().split())

                # Apply class mapping
                cls = CLASS_MAP.get(cls, cls)

                x_center, y_center, bw, bh = convert_bbox(x1, y1, x2, y2, w, h)

                yolo_lines.append(
                    f"{cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"
                )

        # Save image
        out_img_path = os.path.join(OUTPUT_IMG, file)
        cv2.imwrite(out_img_path, image)

        # Save label
        out_lbl_path = os.path.join(OUTPUT_LBL, file.replace(".jpg", ".txt"))
        with open(out_lbl_path, "w") as f:
            f.write("\n".join(yolo_lines))

print("DeepPCB Conversion Completed Successfully")