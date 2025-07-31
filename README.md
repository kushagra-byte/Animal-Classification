# Animal-Classification

This project builds a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify animals from images. It includes two main components:

Model Training
Image Classification Script for predicting animal class from a given image.

Installation

pip install tensorflow matplotlib

Model Training
Train your model using:

train_model.py

This will:

Train the model on 80% of the dataset
Validate on 20%

Save:

animal_classifier_model.h5: The trained model

training_accuracy.png: Accuracy plot

 Image Classification
You can predict an animal class using an image with the classify_image.py script.

Usage

python classify_image.py path_to_image.jpg

Output Example

Predicted Animal: dog (96.34%)

How it Works

Loads the trained model (.h5)
Reads the input image
Resizes it to 224x224
Normalizes pixel values
Predicts the most probable class from the available dataset folders
