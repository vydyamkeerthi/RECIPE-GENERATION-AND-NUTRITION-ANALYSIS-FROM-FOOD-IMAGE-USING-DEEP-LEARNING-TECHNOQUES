import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import keras
import pandas as pd
import tensorflow as tf

from keras.models import Model
from keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.optimizers import Adam

# Paths and Parameters
image_dir = 'food-101/images' 
models_filename = 'v8_vgg16_model_1.h5'

image_size = (224, 224)
batch_size = 16
epochs = 80
tf.config.threading.set_intra_op_parallelism_threads(4)  
tf.config.threading.set_inter_op_parallelism_threads(2)

# Data Augmentation for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    horizontal_flip=True,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=20,
    validation_split=0.2  # ✅ Added validation split
)

# Load training images
train_generator = train_datagen.flow_from_directory(
    image_dir,
    target_size=image_size,
    batch_size=batch_size, 
    class_mode="categorical",
    subset='training'  # ✅ Training subset
)

# Load validation images
val_generator = train_datagen.flow_from_directory(
    image_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # ✅ Validation subset
)

num_of_classes = len(train_generator.class_indices)

# Load VGG16 model without the top layer
base_model = VGG16(weights=None, include_top=False, input_shape=(image_size[0], image_size[1], 3))

# Add custom layers on top of VGG16
x = base_model.output
x = Flatten()(x)
x = Dense(202, activation="relu")(x)
x = Dense(202, activation="relu")(x)
predictions = Dense(101, activation="softmax")(x)

# Compile the model
model_final = Model(inputs=base_model.input, outputs=predictions)
model_final.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=["accuracy"])

# Load pretrained weights
model_final.load_weights(models_filename)

# Evaluate model performance on validation set
val_loss, val_acc = model_final.evaluate(val_generator, steps=100)
print("Validation Loss: ", val_loss)
print("Validation Accuracy: {:.2f}%".format(val_acc * 100))

model_final.save('vgg16_food101_trained_before.h5')
print("Model saved successfully.")

# Human evaluation loop
for _ in range(10):  # Run for 10 images
    image, classifier = next(val_generator)  # ✅ Use validation data for human eval
    image, classifier = image[0], classifier[0]  # Take first image in batch

    true_label = list(val_generator.class_indices.keys())[np.argmax(classifier)]
    predicted = model_final.predict(np.expand_dims(image, axis=0))
    predicted_label = list(val_generator.class_indices.keys())[np.argmax(predicted[0])]

    plt.imshow(image)
    plt.axis("off")
    plt.show()

    print(f'Correct label: {true_label}')
    print(f'CNN Prediction: {predicted_label}\n')

# Save the trained model
model_final.save('vgg16_food101_trained.h5')
print("Model saved successfully.")
