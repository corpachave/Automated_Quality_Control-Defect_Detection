import os

LABEL_DIR = "data/detection/labels"

CLASS_MAP = {
    0: 0,  # open
    1: 1,  # short
    2: 3,  # spur
    3: 2,  # mouse bite
    4: 4,  # pinhole
    5: 5,  # spurious
    6: 5,
    7: 5,
    8: 5
}

for split in ["train", "val"]:
    folder = os.path.join(LABEL_DIR, split)

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        new_lines = []

        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                cls = int(parts[0])

                cls = CLASS_MAP.get(cls, 5)
                new_lines.append(" ".join([str(cls)] + parts[1:]))

        with open(path, "w") as f:
            f.write("\n".join(new_lines))

print("Class mapping fixed")