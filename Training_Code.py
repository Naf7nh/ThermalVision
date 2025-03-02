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
      rescale = 1./255.,           # was 65535
      rotation_range=25,            # best was 20, was 5, previous set 40, then 20, then 10 and 0(maybe switch back???)
      width_shift_range=0.05,        # best was 0, was 0.05, previous 0.1
      height_shift_range=0.05,       # best was 0, was 0.05, previous 0.2
      shear_range=0.05,              # best was 0, was 0.05, previous 0
      zoom_range=0,               # best was 0
      horizontal_flip= True,         # was True
      vertical_flip= False,           # best was False, was True
      fill_mode='nearest')


  train_generator = train_datagen.flow(x=training_images,
                                        y=training_labels,
                                        batch_size=16)       # was 40 w/ 0.8 ,,,,was 16

  validation_datagen = ImageDataGenerator(rescale = 1./255.)        # was 65535

  validation_generator = validation_datagen.flow(x=validation_images,
                                                  y=validation_labels,
                                                  batch_size=16)     # was 16

  return train_generator, validation_generator


def create_model():   
    model = tf.keras.models.Sequential([
        # This is the first convolution
        tf.keras.layers.Conv2D(8, (3,3), activation='relu', input_shape=(60, 80, 1)), # Input shape = (Rows, columns, and channel)
        tf.keras.layers.MaxPooling2D(2,2),                     # was(2,2)
        # tf.keras.layers.Dropout(0.1),
        # The second convolution
        tf.keras.layers.Conv2D(16, (3,3), activation='relu'),   # was 64, (3,3)
        tf.keras.layers.MaxPooling2D(2,2),                      # was (2,2)
        # tf.keras.layers.Dropout(0.2),
        # third convolution
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),  # was 64, (3,3)
        tf.keras.layers.MaxPooling2D(2,2),                      # was (2,2)
        # tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),  # Dropout for regularization
        tf.keras.layers.Dense(512, activation='relu'),     # 512 best, then 256
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    model.summary()
    
    # optimizer = RMSprop(learning_rate = 0.00004)       # was 5e-5 w/ 65535
    optimizer = Adam(learning_rate = 0.0009105)       # was 5e-5 w/ 65535
    
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

#%% Graph Section - Finished
import time

start = time.time()

fig1, ax1 = plt.subplots()
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax2 = ax1.twinx()    #Change to 2nd axis
ax2.set_ylabel('Loss')

# fig2, ax3 = plt.subplots()
# ax3.set_xlabel('Epochs')
# ax3.set_ylabel(' Accuracy')
# ax4 = ax3.twinx()  # Create a second y-axis for validation loss
# ax4.set_ylabel('Loss')


for i in range(1):
    # need to create a new model each time, since it'll keep training on
    # the same one if not
    model = create_model()
    
    if i == 0:
        color = 'r'
    if i == 1:
        color = 'b'
    if i == 2:
        color = 'g'
    
    history = model.fit(train_generator,
                        epochs=65,             # overfit w/ 222, maybe try around 60
                        validation_data=validation_generator)
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    graph_label = "Trial " + str(i + 1)
    epochs = range(len(acc))
    ax1.plot(epochs, acc, color, label=graph_label)       #BLUE IS VALIDATION ON GRAPH
    ax2.plot(epochs, loss, color)
    # plt.legend(graph_label)
    ax1.plot(epochs, val_acc, 'b') 
    ax2.plot(epochs, val_loss, 'b')
    # ax3.plot(epochs, val_acc, color, label=graph_label) 
    # ax4.plot(epochs, val_loss, color)
    
    
# fig1.legend(loc='center right', bbox_to_anchor=(0.9,0.5))
# fig2.legend(loc='center right', bbox_to_anchor=(0.9,0.5))
# fig1.suptitle('Validation')
# fig2.suptitle('Training')
# plt.show()

end = time.time()
print(end - start)
print("Last 5 acc average: " + str(sum(acc[-5:]*100)/5))
print("Last 5 Val_acc average: " + str(sum(val_acc[-5:]*100)/5))


#%% Confusion Matrix Things - Finished
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay             # This for the confusion matrix
from sklearn.metrics import classification_report
import seaborn as sns

predicted_model = model.predict(validation_images / 255.0)
holder = np.argmax(predicted_model, axis=1)


print(classification_report(validation_labels,holder))

classy = classification_report(validation_labels, holder, output_dict=True)

report_dict = classification_report(validation_labels, holder, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

#%% Plotting the report as a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu")
plt.title("Classification Report Heatmap")
plt.show()


#%% Confusion Matrix Printer
jammer = confusion_matrix(validation_labels, holder)
sns.heatmap(jammer, annot=True, fmt='g',cbar_kws={'label':'Number of Instances'})
plt.figure(2)
plt.ylabel('True',fontsize=13)
plt.xlabel('Predicted',fontsize=13)
plt.title('Validation Confusion Matrix',fontsize=17)
plt.show()

#%% Saving and Converting the Model in One Cell
Model_number = 23

john = model.save('C:/Users/Navian/Desktop/Models/my_modelv'+str(Model_number), save_format='tf')

saved_model_path = 'C:/Users/Navian/Desktop/Models/my_modelv'+str(Model_number)                           #Change this file name each time

# Convert the SavedModel to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
output_file_path = 'C:/Users/Navian/Desktop/Models/modelv'+str(Model_number)+'.tflite'                        #Change this file name each time
try:
    with open(output_file_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TensorFlow Lite model saved successfully to {output_file_path}")
except Exception as e:
    print(f"Error saving TensorFlow Lite model: {e}")


#%% Show Images
images_array = validation_images
def plot_image(index):
    # Ensure the index is within bounds
    if index < 0 or index >= len(images_array):
        print(f"Index {index} is out of bounds. Please choose a valid index.")
        return
    
    image = images_array[index]
    plt.imshow(image, cmap='gray')  # Set cmap='gray' if the images are grayscale
    plt.title(f"Image {index}")
    plt.axis('off')  # Hide axes
    plt.show()

# Example usage
plot_image(107)  # Plot the image at index 0

