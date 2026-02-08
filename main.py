import numpy as np
import tensorflow as tf
from pathlib import Path

# tkinter + PIL imports
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk


# -----------------------------
# Paths
# -----------------------------
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "animalImages"
model_path = base_dir / "animal_detector_mobilenet.keras"

# -----------------------------
# Load model ONLY (no training)
# -----------------------------
if not model_path.exists():
    raise FileNotFoundError(
        f"Model not found:\n{model_path}\n"
        f"You said you don't want retraining, so place your saved model here."
    )

model = tf.keras.models.load_model(model_path)

# ⭐ Automatically adapt to model input size
h, w = model.input_shape[1], model.input_shape[2]
img_size = (int(h), int(w))

# Build class names from folder structure
animal_names = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])

print("Model loaded")
print("Expected input:", model.input_shape)
print("Using img_size:", img_size)
print("Classes detected:", len(animal_names))


# -----------------------------
# Prediction
# -----------------------------
def predict_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    x = tf.keras.utils.img_to_array(img)
    x = tf.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_conf = float(np.max(probs))
    pred_label = animal_names[pred_idx]

    return pred_label, pred_conf, img


# -----------------------------
# Tkinter UI
# -----------------------------
root = tk.Tk()
root.title("Animal Detector")
root.geometry("420x560")

title_label = Label(root, text="Animal Detector", font=("Arial", 16, "bold"))
title_label.pack(pady=(10, 0))

sub_label = Label(
    root,
    text=f"Model input size: {img_size[0]}×{img_size[1]}",
    font=("Arial", 11)
)
sub_label.pack(pady=(0, 10))

img_label = Label(root)
img_label.pack(pady=10)

result_label = Label(
    root,
    text="Click + to select an image",
    font=("Arial", 14),
    justify="center"
)
result_label.pack(pady=10)


def open_image():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    label, conf, pil_img = predict_image(file_path)

    display_img = pil_img.resize((280, 280))
    tk_img = ImageTk.PhotoImage(display_img)

    img_label.config(image=tk_img)
    img_label.image = tk_img

    result_label.config(
        text=f"Prediction: {label}\nConfidence: {conf*100:.2f}%"
    )


Button(
    root,
    text="➕ Open Image",
    command=open_image,
    font=("Arial", 12),
    width=20
).pack(pady=15)

root.mainloop()
