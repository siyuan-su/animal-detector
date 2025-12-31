import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

#resolve project paths and key directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "animalImages"
MODEL_PATH = BASE_DIR / "animal_detector_mobilenet.keras"
TEST_IMAGE_PATH = BASE_DIR / "test.jpg"

#image and training configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
SEED = 123
EPOCHS = 10

#load an external image for inference testing
testIMG = tf.keras.utils.load_img(TEST_IMAGE_PATH, target_size=IMG_SIZE)

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

#load existing model to avoid retraining
if MODEL_PATH.exists():
    print("loading saved model")
    model = tf.keras.models.load_model(MODEL_PATH)

else:
    print("training new model")

    #MobileNetV2 acts as a pretrained feature extractor
    #it uses depthwise separable convolutions to efficiently learn visual patterns
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    #fine tuning setup: freeze most layers, train higher-level features
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    #build classification head on top of pretrained features
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(numClasses, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    #compile with a low learning rate for stable fine tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    #train the model and save for future use
    model.fit(
        trainDS,
        validation_data=valDS,
        epochs=EPOCHS,
        verbose=1
    )

    model.save(MODEL_PATH)
    print("model saved")

#prepare the external image for prediction
testIMG_arr = tf.keras.utils.img_to_array(testIMG)
testIMG_arr = tf.expand_dims(testIMG_arr, axis=0)
testIMG_arr = tf.keras.applications.mobilenet_v2.preprocess_input(testIMG_arr)

#run inference and extract predicted class
preds = model.predict(testIMG_arr, verbose=0)
pred_idx = int(np.argmax(preds[0]))

print("predicted animal:", animalNames[pred_idx])

#display the image with prediction result
plt.imshow(testIMG)
plt.title(f"predicted: {animalNames[pred_idx]}")
plt.axis("off")
plt.show()
