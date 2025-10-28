# ThermalVision

This repository is for setting up a thermal imaging-based edge computing system using a Raspberry Pi and a FLIR Lepton thermal camera. The objective of this repository is to provide step-by-step guidance on how to set up a thermal imaging-based edge computing system using Raspberry Pi and develop an algorithm for occupancy detection in buildings. This repository comprises two main components: a tutorial detailing the hardware portion and the Python code for creating a CNN to detect up to 4 people in an image. The published paper for this repository can be found [here](https://doi.org/10.1016/j.buildenv.2025.113871)

## Contents of the Files 

### Tutorial 
The tutorial PDF outlines the system's objective, lists the components used, provides a step-by-step process for assembling the hardware components, and explains the system's configuration on the Raspberry Pi. The following will be found within the tutorial document:
* Comprehensive list of materials needed
* Step-by-step instructions to set up the hardware components
* Step-by-step instructions to set up the software needed
  * Instructions for using the system with C
  * Instructions for using the system with Python **(Recommended)** 
* Recommendations for courses relevant to this work 

### Training Script 
The Python file contains the script that is used to train the alogrithm that is used to detect occupants in the designated spaces 
