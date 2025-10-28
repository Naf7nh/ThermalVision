# ThermalVision

This repository is for setting up a thermal imaging-based edge computing system using a Raspberry Pi and a FLIR Lepton thermal camera. The objective of this repository is to provide step-by-step guidance on how to set up a thermal imaging-based edge computing system using Raspberry Pi and develop an algorithm for occupancy detection in buildings. This repository comprises three main components: a tutorial detailing the hardware, a script for creating the training and validation datasets, and a script for training a CNN to detect up to 4 people in an image. The published paper for this repository can be found [here](https://doi.org/10.1016/j.buildenv.2025.113871)

## Contents of the Files 

### Tutorial 
The tutorial PDF outlines the system's objective, lists the components used, provides a step-by-step process for assembling the hardware components, and explains the system's configuration on the Raspberry Pi. The following will be found within the tutorial document:
* Comprehensive list of materials needed
* Step-by-step instructions to set up the hardware components
* Step-by-step instructions to set up the software needed
  * Instructions for using the system with C
  * Instructions for using the system with Python **(Recommended)** 
* Recommendations for courses relevant to this work 

### Dataset Creation Script 
This Python script contains the provisions for the following:
* Creating the training and validation dataste directories
* Creating the subfolders for the training and validation datasets
* Functions to read the PGM images and parse that information for each image into CSV files
* Splitting the entire dataset based on the desired training/validation split and randomly assigning images into each dataset

Please ensure that the Images folder is created before running this script. An example of the Image folder is provided below:
- 📂 Images
  - 📂 0 People
  - 📂 1 Person
  - 📂 2 People
  - 📂 3 People
  - 📂 4 People

**IMPORTANT** This script should be used only when splitting the images into the training and validation datasets. Each time this script is executed, the images will be randomly assigned to the training or validation set. No two runs will result in the same allocation. The CSV files generated should be saved and used until a new split is desired. 

### CNN Creation  
The Python file contains the script used to train the algorithm that detects occupants in the designated spaces and includes the following:
* A function to parse the image information from the training and validation dataset CSVs
* Functions to define the validation image augmentation and model creation
* Model training block
* Functionality to save the trained model and output a converted model file suitable for the Raspberry Pi  
