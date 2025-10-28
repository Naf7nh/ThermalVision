import os
import string
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
import csv
from shutil import copyfile
from numpy import hstack
from tensorflow.keras import optimizers, losses
from tensorflow.keras.optimizers import RMSprop
import time
import pandas as pd


# Directory Variables
root_path = os.path.join(os.getcwd(), 'Images')
ZERO_SOURCE_DIR = os.path.join(root_path, '0 People')
ONE_SOURCE_DIR = os.path.join(root_path, '1 Person')
TWO_SOURCE_DIR = os.path.join(root_path, '2 People')
THREE_SOURCE_DIR = os.path.join(root_path, '3 People')
FOUR_SOURCE_DIR = os.path.join(root_path, '4 People')

testing_root = os.path.join(os.getcwd(), 'Testing')
TRAINING_DIR = os.path.join(testing_root, 'Training')
VALIDATION_DIR = os.path.join(testing_root, 'Validation')

TRAINING_ZERO_DIR = os.path.join(TRAINING_DIR, 'zero')
VALIDATION_ZERO_DIR = os.path.join(VALIDATION_DIR, 'zero')
TRAINING_ONE_DIR = os.path.join(TRAINING_DIR, 'one')
VALIDATION_ONE_DIR = os.path.join(VALIDATION_DIR, 'one')
TRAINING_TWO_DIR = os.path.join(TRAINING_DIR, 'two')
VALIDATION_TWO_DIR = os.path.join(VALIDATION_DIR, 'two')
TRAINING_THREE_DIR = os.path.join(TRAINING_DIR, 'three')
VALIDATION_THREE_DIR = os.path.join(VALIDATION_DIR, 'three')
TRAINING_FOUR_DIR = os.path.join(TRAINING_DIR, 'four')
VALIDATION_FOUR_DIR = os.path.join(VALIDATION_DIR, 'four')


# Empty directories in case you run this multiple times
if len(os.listdir(TRAINING_ZERO_DIR)) > 0:
  for file in os.scandir(TRAINING_ZERO_DIR):
    os.remove(file.path)
if len(os.listdir(TRAINING_ONE_DIR)) > 0:
  for file in os.scandir(TRAINING_ONE_DIR):
    os.remove(file.path)
if len(os.listdir(TRAINING_TWO_DIR)) > 0:
  for file in os.scandir(TRAINING_TWO_DIR):
    os.remove(file.path)
if len(os.listdir(TRAINING_THREE_DIR)) > 0:
  for file in os.scandir(TRAINING_THREE_DIR):
    os.remove(file.path)
if len(os.listdir(TRAINING_FOUR_DIR)) > 0:
  for file in os.scandir(TRAINING_FOUR_DIR):
    os.remove(file.path)
    
if len(os.listdir(VALIDATION_ZERO_DIR)) > 0:
  for file in os.scandir(VALIDATION_ZERO_DIR):
    os.remove(file.path)
if len(os.listdir(VALIDATION_ONE_DIR)) > 0:
  for file in os.scandir(VALIDATION_ONE_DIR):
    os.remove(file.path)
if len(os.listdir(VALIDATION_TWO_DIR)) > 0:
  for file in os.scandir(VALIDATION_TWO_DIR):
    os.remove(file.path)
if len(os.listdir(VALIDATION_THREE_DIR)) > 0:
  for file in os.scandir(VALIDATION_THREE_DIR):
    os.remove(file.path)
if len(os.listdir(VALIDATION_FOUR_DIR)) > 0:
  for file in os.scandir(VALIDATION_FOUR_DIR):
    os.remove(file.path)

# Other Variables
split_size = 0.8          

# ----------------------------- FUNCTION DEFINITIONS -------------------------

def readpgm(name): #THIS IS FOR P5 IMAGES
    with open(name, 'rb') as f:
        # Read the PGM header
        magic_number = f.readline().strip()
        assert magic_number == b'P2'

        # Read width, height, and max value
        width, height = map(int, f.readline().split())
        max_val = int(f.readline().strip())

        # Read binary pixel data
        pixel_data = np.fromfile(f, dtype=np.uint8)

    # Output the same data as in the P2 format
    return (pixel_data, (height, width), max_val)


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


def split_data(SOURCE_DIR, TRAINING_DIR, VALIDATION_DIR, SPLIT_SIZE):
    """
    Splits the data into train and test sets
      
    Args:
        SOURCE_DIR (string): directory path containing the images
        TRAINING_DIR (string): directory path to be used for training
        VALIDATION_DIR (string): directory path to be used for validation
        SPLIT_SIZE (float): proportion of the dataset to be used for training
        
    Returns:
        None
      """
    
    # source_directory = os.listdir(SOURCE_DIR)
    new_dir = []
    for name in os.listdir(SOURCE_DIR):
        path = SOURCE_DIR + '\\' + name
        size = os.path.getsize(path)
        if size == 0 or size < 0:
            print(str(name) + " is zero length, so ignoring.")
        else:
            new_dir.append(name)
    
    num_files = len(new_dir)
    split = int(num_files * SPLIT_SIZE)
    
    randomized = random.sample(new_dir, num_files)
    training_images = randomized[:split]
    validation_images = randomized[split:]
    
    for filename in training_images:
        copyfile(SOURCE_DIR + '\\' + filename, TRAINING_DIR + '\\' + filename)
    for filename in validation_images:
        copyfile(SOURCE_DIR + '\\' + filename, VALIDATION_DIR + '\\' + filename)
        
# ----------------------------------------------------------------------------

# Splitting data into training and validation sets
split_data(ZERO_SOURCE_DIR, TRAINING_ZERO_DIR, VALIDATION_ZERO_DIR, split_size)
split_data(ONE_SOURCE_DIR, TRAINING_ONE_DIR, VALIDATION_ONE_DIR, split_size)
split_data(TWO_SOURCE_DIR, TRAINING_TWO_DIR, VALIDATION_TWO_DIR, split_size)
split_data(THREE_SOURCE_DIR, TRAINING_THREE_DIR, VALIDATION_THREE_DIR, split_size)
split_data(FOUR_SOURCE_DIR, TRAINING_FOUR_DIR, VALIDATION_FOUR_DIR, split_size)


# ---------- creating CSV files to use in the training/testing process -----------
training = []
for directory in os.listdir(TRAINING_DIR):
    if directory == 'zero':
        dir_zero = os.path.join(TRAINING_DIR, 'zero')
        zero_val = [0]
        zero_val = np.array(zero_val)
        for img_name in os.listdir(dir_zero):
            file = os.path.join(dir_zero, img_name)
            zero_data = readpgm(file)
            total_arr = hstack((zero_val, zero_data[0]))
            training.append(total_arr)
    if directory == 'one':
        dir_one = os.path.join(TRAINING_DIR, 'one')
        one_val = [1]
        one_val = np.array(one_val)
        for img_name in os.listdir(dir_one):
            file = os.path.join(dir_one, img_name)
            one_data = readpgm(file)
            total_arr = hstack((one_val, one_data[0]))
            training.append(total_arr)
    if directory == 'two':
        dir_two = os.path.join(TRAINING_DIR, 'two')
        two_val = [2]
        two_val = np.array(two_val)
        for img_name in os.listdir(dir_two):
            file = os.path.join(dir_two, img_name)
            two_data = readpgm(file)
            total_arr = hstack((two_val, two_data[0]))
            training.append(total_arr)
    if directory == 'three':
        dir_three = os.path.join(TRAINING_DIR, 'three')
        three_val = [3]
        three_val = np.array(three_val)
        for img_name in os.listdir(dir_three):
            file = os.path.join(dir_three, img_name)
            three_data = readpgm(file)
            total_arr = hstack((three_val, three_data[0]))
            training.append(total_arr)
    if directory == 'four':
        dir_four = os.path.join(TRAINING_DIR, 'four')
        four_val = [4]
        four_val = np.array(four_val)
        for img_name in os.listdir(dir_four):
            file = os.path.join(dir_four, img_name)
            four_data = readpgm(file)
            total_arr = hstack((four_val, four_data[0]))
            training.append(total_arr)
                
np.savetxt("training_data.csv", training, delimiter = ',')

validation = []
for directory in os.listdir(VALIDATION_DIR):
    if directory == 'zero':
        dir_zero = os.path.join(VALIDATION_DIR, 'zero')
        zero_val = [0]
        zero_val = np.array(zero_val)
        for img_name in os.listdir(dir_zero):
            file = os.path.join(dir_zero, img_name)
            zero_data = readpgm(file)
            total_arr = hstack((zero_val, zero_data[0]))
            validation.append(total_arr)
    if directory == 'one':
        dir_one = os.path.join(VALIDATION_DIR, 'one')
        one_val = [1]
        one_val = np.array(one_val)
        for img_name in os.listdir(dir_one):
            file = os.path.join(dir_one, img_name)
            one_data = readpgm(file)
            total_arr = hstack((one_val, one_data[0]))
            validation.append(total_arr)
    if directory == 'two':
        dir_two = os.path.join(VALIDATION_DIR, 'two')
        two_val = [2]
        two_val = np.array(two_val)
        for img_name in os.listdir(dir_two):
            file = os.path.join(dir_two, img_name)
            two_data = readpgm(file)
            total_arr = hstack((two_val, two_data[0]))
            validation.append(total_arr)
    if directory == 'three':
        dir_three = os.path.join(VALIDATION_DIR, 'three')
        three_val = [3]
        three_val = np.array(three_val)
        for img_name in os.listdir(dir_three):
            file = os.path.join(dir_three, img_name)
            three_data = readpgm(file)
            total_arr = hstack((three_val, three_data[0]))
            validation.append(total_arr)
    if directory == 'four':
        dir_four = os.path.join(VALIDATION_DIR, 'four')
        four_val = [4]
        four_val = np.array(four_val)
        for img_name in os.listdir(dir_four):
            file = os.path.join(dir_four, img_name)
            four_data = readpgm(file)
            total_arr = hstack((four_val, four_data[0]))
            validation.append(total_arr)
                
np.savetxt("validation_data.csv", validation, delimiter = ',')
