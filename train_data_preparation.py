import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# CONFIGURATION

DATA_DIR = "data/classification"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# DATA AUGMENTATION

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# LOAD DATA

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

# PRINT CLASS MAPPING

print("\nClass Indices:")
print(train_generator.class_indices)   # Example output: {'defect': 0, 'pass': 1}

# VISUALIZE AUGMENTED DATA

def visualize_augmentation(generator):
    images, labels = next(generator)

    plt.figure(figsize=(10, 10))

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        
        label = "DEFECT" if labels[i] == 0 else "PASS"
        plt.title(label)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# RUN VISUALIZATION

visualize_augmentation(train_generator)

print("\n Augmentation Visualization Completed")