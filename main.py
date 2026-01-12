import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

#tkinter + PIL imports
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# resolve project paths
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "animalImages"
model_path = base_dir / "animal_detector_mobilenet.keras"
test_image_path = base_dir / "test.jpg"

# config
img_size = (224, 224)
batch_size = 64
seed = 123
epochs_head = 8
epochs_fine = 5

# set false to load if model exists
force_retrain = False

loaded = False
model = None
animal_names = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])

if model_path.exists() and not force_retrain:
    model = tf.keras.models.load_model(model_path)
    loaded = True

else:
    # build datasets only if training
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )

    animal_names = train_ds.class_names
    num_classes = len(animal_names)

    # build base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    # build full model
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    # phase 1: head training
    base_model.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(train_ds, validation_data=val_ds, epochs=epochs_head, verbose=1)

    # phase 2: fine-tune top layers
    base_model.trainable = True

    # keep batchnorm frozen for stability
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    n_unfreeze = 20
    for layer in base_model.layers[:-n_unfreeze]:
        layer.trainable = False
    for layer in base_model.layers[-n_unfreeze:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-6),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(train_ds, validation_data=val_ds, epochs=epochs_fine, verbose=1)

    model.save(model_path)

#prediction helper for tkinter
def predict_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    x = tf.keras.utils.img_to_array(img)
    x = tf.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_conf = float(np.max(probs))
    pred_label = animal_names[pred_idx]

    return pred_label, pred_conf, img


# ---------------- TKINTER UI (NEW, NO ORIGINAL COMMENTS MODIFIED) ---------------- #

root = tk.Tk()
root.title("Animal Detector")
root.geometry("420x520")

img_label = Label(root)
img_label.pack(pady=10)

result_label = Label(
    root,
    text="Click + to select an image",
    font=("Arial", 14)
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
        text=f"Prediction: {label}\nConfidence: {conf:.3f}"
    )

Button(
    root,
    text="➕ Open Image",
    command=open_image,
    font=("Arial", 12),
    width=20
).pack(pady=15)

root.mainloop()
