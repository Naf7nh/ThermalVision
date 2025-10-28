#%% Loading Image Data without Shuffling
import os
import string
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
import csv
from shutil import copyfile
from numpy import hstack
from tensorflow.keras import optimizers, losses
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
import time

def parse_data_from_input(filename):
  """
  Parses the images and labels from a CSV file
  
  Args:
    filename (string): path to the CSV file
    
  Returns:
    images, labels: tuple of numpy arrays containing the images and labels
  """
  with open(filename) as file:
    # Use csv.reader, passing in the appropriate delimiter
    # Remember that csv.reader can be iterated and returns one line in each iteration
    csv_reader = csv.reader(file, delimiter=',')
    # next(csv_reader)
    
    labels = []
    images = []
    count = 0

    for row in csv_reader:
      count = count + 1
      labels.append(row[0])
      images.append(row[1:])
    
    labels = np.asarray(labels, dtype = np.float64)
    images = np.asarray(images, dtype = np.float64)
    images = np.reshape(images, (count, 60, 80))

    return images, labels

def train_val_generators(training_images, training_labels, validation_images, validation_labels):
  """
  Returns:
    train_generator, validation_generator - tuple containing the generators
  """
  training_images = np.expand_dims(training_images, axis = 3)
  validation_images = np.expand_dims(validation_images, axis = 3)

  train_datagen = ImageDataGenerator(
      rescale = 1./255.,           
      rotation_range=25,            
      width_shift_range=0.05,        
      height_shift_range=0.05,       
      shear_range=0.05,              
      zoom_range=0,               
      horizontal_flip= True,         
      vertical_flip= False,           
      fill_mode='nearest')


  train_generator = train_datagen.flow(x=training_images,
                                        y=training_labels,
                                        batch_size=16)       

  validation_datagen = ImageDataGenerator(rescale = 1./255.)        # was 65535

  validation_generator = validation_datagen.flow(x=validation_images,
                                                  y=validation_labels,
                                                  batch_size=16)     # was 16

  return train_generator, validation_generator


def create_model():   
    model = tf.keras.models.Sequential([
        # This is the first convolution
        tf.keras.layers.Conv2D(8, (3,3), activation='relu', input_shape=(60, 80, 1)), # Input shape = (Rows, columns, and channel)
        tf.keras.layers.MaxPooling2D(2,2),                     
        # tf.keras.layers.Dropout(0.1),
        # The second convolution
        tf.keras.layers.Conv2D(16, (3,3), activation='relu'),   
        tf.keras.layers.MaxPooling2D(2,2),                      
        # tf.keras.layers.Dropout(0.2),
        # third convolution
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),  
        tf.keras.layers.MaxPooling2D(2,2),                      
        # tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),  # Dropout for regularization
        tf.keras.layers.Dense(512, activation='relu'),     
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    model.summary()
    
    # optimizer = RMSprop(learning_rate = 0.00004)       
    optimizer = Adam(learning_rate = 0.0009105)       
    
    model.compile(optimizer = optimizer,
                  loss = 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])      
  
    return model


TRAINING_FILE = os.path.join(os.getcwd(), 'training_data.csv')
VALIDATION_FILE = os.path.join(os.getcwd(), 'validation_data.csv')

# ----------------- test parse_data_from_input function ----------------------
training_images, training_labels = parse_data_from_input(TRAINING_FILE)
validation_images, validation_labels = parse_data_from_input(VALIDATION_FILE)

# ------------------------ Test your generators ------------------------------
train_generator, validation_generator = train_val_generators(training_images, training_labels, validation_images, validation_labels)


start = time.time()

fig1, ax1 = plt.subplots()
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax2 = ax1.twinx()    #Change to 2nd axis
ax2.set_ylabel('Loss')


for i in range(1):
    model = create_model()
    
    if i == 0:
        color = 'r'
    if i == 1:
        color = 'b'
    if i == 2:
        color = 'g'
    
    history = model.fit(train_generator,
                        epochs=65,             
                        validation_data=validation_generator)
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    graph_label = "Trial " + str(i + 1)
    epochs = range(len(acc))
    ax1.plot(epochs, acc, color, label=graph_label)       #BLUE IS VALIDATION ON GRAPH
    ax2.plot(epochs, loss, color)
    ax1.plot(epochs, val_acc, 'b') 
    ax2.plot(epochs, val_loss, 'b')

# This section below saves and converts the model into a format suitable for running on the Raspberry Pi 

Model_number = 1                                                                                       #Change this number each time
model.save('XXX/XXXX/XXXX/my_modelv'+str(Model_number), save_format='tf')                              #Change the directory to point to the folder where the models are saved
saved_model_path = 'XXX/XXXX/XXXX/my_modelv'+str(Model_number)                          

# Convert the SavedModel to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
output_file_path = 'XXX/XXXX/XXXX/modelv'+str(Model_number)+'.tflite'                        #Change this file name each time
try:
    with open(output_file_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TensorFlow Lite model saved successfully to {output_file_path}")
except Exception as e:
    print(f"Error saving TensorFlow Lite model: {e}")





