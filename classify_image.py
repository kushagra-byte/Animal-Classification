import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import sys

# Load the saved model
model = tf.keras.models.load_model("animal_classifier_model.h5")

# Get class labels from dataset directory (alphabetically sorted)
dataset_dir = "./dataset"
class_names = sorted(os.listdir(dataset_dir))

# Function to classify a new image
def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = np.max(predictions)

    print(f"Predicted Animal: {predicted_class} ({confidence * 100:.2f}%)")

# Run from command line
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python classify_image.py path_to_image.jpg")
    else:
        classify_image(sys.argv[1])
