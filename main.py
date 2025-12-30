import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from pathlib import Path

#Path(__file__) finds the path of the current file (.../Animal-Detector/main.py)
#.resolve() cleans it up
#.parent is the parent file ("Animal-Detector")
BASE_DIR = Path(__file__).resolve().parent 
#DATA_DIR becomes the BASE_DIR/animalImages = Animal-Detector/animalImages
DATA_DIR = BASE_DIR / "animalImages"

#Image configs for training
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

#training amount
EPOCHS = 100

#training the model
#trainig data set that is used to teach the model patterns
trainDS = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split = 0.2,
    subset = "training",
    seed = SEED,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE
)

#validation data set that is used to fine-tune the models settings
valDS = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split = 0.2,
    subset = "validation",
    seed = SEED,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE
)

#saved data from previous training
MODEL_PATH = BASE_DIR / "animal_detector.keras"

#stores the names of the data set into animal_names alphabetically
animalNames = trainDS.class_names
numClasses = len(animalNames)

#performance optimizations
AUTOTUNE = tf.data.AUTOTUNE
trainDS = trainDS.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
valDS = valDS.cache().prefetch(AUTOTUNE)

#sequential model is one that uses layers to change a specific input(essentially a function)
if MODEL_PATH.exists():
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = tf.keras.Sequential([
        #prepping
        tf.keras.layers.Input(shape = (224, 224, 3)), #declares expected image parameters
        tf.keras.layers.Rescaling(1.0/255), #normalizes the pixels of the image to float numbers from 0-1

        #mapping and efficiency
        #utilizes Conv2d, set to using 32 different filters for every 3x3 pixel grid while using relu to create feature maps
        tf.keras.layers.Conv2D(32, (3, 3), activation = "relu"),
        #takes the most prominent patterns from a 2x2 pixel the feature map for efficiency
        tf.keras.layers.MaxPooling2D((2,2)), 
        tf.keras.layers.GlobalAveragePooling2D(), #efficiency

        #building neurons
        tf.keras.layers.Dense(128, activation = "relu"), #"shrinks" vector to 128 new numbers by combining them and doing magic with predetermined weights using relu
        tf.keras.layers.Dropout(0.5), #takes away 50% of the units randomly to improve generalization and forces the model to be robust

        #assigning probability
        tf.keras.layers.Dense(numClasses, activation = "softmax") #converts raw values into 90 probabilities(all values sum to 1)
    ])

#compiling the model(rules for training)
model.compile(
    #Adam updates the weights
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
    #compares multiple "classes" probabilities with the correct answer
    loss = "sparse_categorical_crossentropy",
    #percent of predicitons where the top predicted class equals the correct label
    metrics = ["accuracy"],
)

#actually trains the model
#for each epoch, the model sees all training images once in batches(32)
#for each batch, it compiles the sequential
#after each epoch the model is evaluated on validation data
history = model.fit( #history stores the training loss per epoch, validation loss per epoch, and accuracy metrics
    trainDS,
    validation_data = valDS,
    epochs = EPOCHS,
)

#saves model to file in Animal-Detector
model.save(MODEL_PATH)

acc = history.history["accuracy"] #training accuracy
val_acc = history.history["val_accuracy"] #validation accuracy
loss = history.history["loss"] #training loss
val_loss = history.history["val_loss"] #validation loss

#displays graphs
plt.figure()
plt.plot(acc, label="train accuracy")
plt.plot(val_acc, label="val accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure()
plt.plot(loss, label="train loss")
plt.plot(val_loss, label="val loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

for images, labels in valDS.take(1):
    preds = model.predict(images) #predicts image        
    pred_idx = int(np.argmax(preds[0])) #get predicted class index for the first image
    true_idx = int(labels[0].numpy()) #get the true class index for first iamge

    print("\nPredicted:", animalNames[pred_idx], f"(index {pred_idx})")
    print("Actual:   ", animalNames[true_idx], f"(index {true_idx})")

    #updates graph
    plt.figure()
    plt.imshow(images[0].numpy().astype("float32"))
    plt.title(f"Pred: {animalNames[pred_idx]} | True: {animalNames[true_idx]}")
    plt.axis("off")
    plt.show()

#loads the previously saved model
loaded_model = tf.keras.models.load_model(BASE_DIR / "animal_detector.keras")
loaded_labels = (BASE_DIR / "animalNames.txt").read_text(encoding="utf-8").splitlines()

#checks for prediction
for images, labels in valDS.take(1):
    preds = loaded_model.predict(images)
    pred_idx = int(np.argmax(preds[0]))
    print("\nLoaded model predicts:", loaded_labels[pred_idx])