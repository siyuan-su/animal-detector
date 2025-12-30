import PIL
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

#initializes dataset as a list of images with corresponding labels
dataSet = tf.keras.utils.image_dataset_from_directory(DATA_DIR, seed = SEED, image_size = IMG_SIZE, batch_size = BATCH_SIZE)

#stores the names of the data set into animal_names alphabetically
animal_names = dataSet.class_names

#testing for accurate number of animals and including ever animal name
print("animal #", len(animal_names))
print("animal names:", animal_names)

#each element in dataSet as said before has a corresponding image and label
#dataSet.take(1) represents the first batch(32 as set in BATCH_SIZE) of the randomly sorted list, dataSet(randomly sorted based on SEED = 123)
#essentially, it is saying "for each corresponding image and label per element of the first batch in dataSet"
for images, labels in dataSet.take(1):
    print("Numeric labels: ", labels[:10].numpy()) #print the first 10 labels
    #list comprehension, essentially for each i from labels[0]-labels[10] find the corresponding animal_names[i]
    print("Mapped labels: ", [animal_names[i] for i in labels[:10].numpy()]) 
