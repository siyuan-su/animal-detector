import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

#resolve project paths and key directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "animalImages"
MODEL_PATH = BASE_DIR / "animal_detector_mobilenet.keras"
TEST_IMAGE_PATH = DATA_DIR / "fly" / "0d1a3fdd51.jpg"

#image and training configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
SEED = 123
EPOCHS_HEAD = 8
EPOCHS_FINE = 5

FORCE_RETRAIN = True

#create training and validation datasets from directory structure
trainDS = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

valDS = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

#class names map output indices to animal labels
animalNames = trainDS.class_names
numClasses = len(animalNames)

loaded = False
#load existing model to avoid retraining
if MODEL_PATH.exists() and not FORCE_RETRAIN:
    model = tf.keras.models.load_model(MODEL_PATH)
    loaded = True
else:
#MobileNetV2 acts as a pretrained feature extractor
#it uses depthwise separable convolutions to efficiently learn visual patterns
    # build base model
    base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
    )

    # build full model (base + head)
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x)  # let keras handle training/inference mode
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(numClasses, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    # --------------------
    # phase 1: train head only
    # --------------------
    base_model.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentrgopy",
        metrics=["accuracy"]
    )

    history_head = model.fit(
        trainDS,
        validation_data=valDS,
        epochs=EPOCHS_HEAD,
        verbose=1
    )

    # --------------------
    # phase 2: fine-tune top layers
    # --------------------
    base_model.trainable = True

    # freeze most layers, unfreeze only the top N layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-6),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_fine = model.fit(
        trainDS,
        validation_data=valDS,
        epochs=EPOCHS_FINE,
        verbose=1
    )

    model.save(MODEL_PATH)

#prepare the external image for prediction
testIMG = tf.keras.utils.load_img(TEST_IMAGE_PATH, target_size=IMG_SIZE)
testIMG_arr = tf.keras.utils.img_to_array(testIMG)
testIMG_arr = tf.expand_dims(testIMG_arr, axis=0)
testIMG_arr = tf.keras.applications.mobilenet_v2.preprocess_input(testIMG_arr)
#load an external image for inference testing

#run inference and extract predicted class

# predict
preds = model.predict(testIMG_arr, verbose=0)
probs = preds[0]

pred_idx = int(np.argmax(probs))
pred_conf = float(np.max(probs))

print("\nloaded model?" , loaded)
print("predicted:", animalNames[pred_idx], "| max prob:", pred_conf)

# print specific class probabilities if they exist
if "fly" in animalNames:
    fly_i = animalNames.index("fly")
    print("p(fly):", float(probs[fly_i]))
else:
    print("fly not found in class list")

if "bear" in animalNames:
    bear_i = animalNames.index("bear")
    print("p(bear):", float(probs[bear_i]))
else:
    print("bear not found in class list")

# top-5 diagnostic (shows what the model is choosing between)
top5 = np.argsort(probs)[-5:][::-1]
print("\ntop 5 predictions:")
for i in top5:
    print(animalNames[i], float(probs[i]))

# display image
plt.imshow(testIMG)
plt.title(f"predicted: {animalNames[pred_idx]}  ({pred_conf:.3f})")
plt.axis("off")
plt.show()