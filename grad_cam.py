import numpy as np
import cv2
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import matplotlib.pyplot as plt


# CONFIG

MODEL_PATH = "visionspec_qc_model.keras"
IMG_PATH = r"data\\classification\\defect\\00041003_test.jpg"
IMG_SIZE = (224, 224)


# LOAD MODEL

model = load_model(MODEL_PATH)


# FIND BASE MODEL (MobileNetV2)

base_model = None
for layer in model.layers:
    if "mobilenet" in layer.name.lower():
        base_model = layer
        break

if base_model is None:
    raise ValueError("MobileNetV2 base model not found!")

print("Base model:", base_model.name)


# PREPROCESS IMAGE

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found at path: {img_path}")
    img = cv2.resize(img, IMG_SIZE)
    img_array = img / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


# LAST CONV LAYER

last_conv_layer_name = "Conv_1"
print("Using Conv Layer:", last_conv_layer_name)

def make_gradcam_heatmap(img_array, model, base_model, last_conv_layer_name):

    # Ensure model is built
    _ = model(img_array)

    # Get last conv layer from base model
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    # Create a model that maps input -> conv output
    conv_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    # Create classifier model (conv output -> final prediction)
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input

    # Rebuild top layers manually
    for layer in model.layers[1:]:  # skip base_model
        x = layer(x)

    classifier_model = tf.keras.Model(classifier_input, x)

    
    # GRAD-CAM FUNCTION

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)

        predictions = classifier_model(conv_outputs)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# OVERLAY FUNCTION

def overlay_heatmap(original_img, heatmap):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    return overlay


# RUN

original_img, img_array = preprocess_image(IMG_PATH)

heatmap = make_gradcam_heatmap(
    img_array, model, base_model, last_conv_layer_name
)

overlay = overlay_heatmap(original_img, heatmap)


# DISPLAY

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Heatmap")
plt.imshow(heatmap, cmap='jet')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()