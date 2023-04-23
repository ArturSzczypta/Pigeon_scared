import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, decode_predictions
import os
import csv

# Load the ResNet50V2 model
model = ResNet50V2(weights='imagenet')
def image_recognition(img_folder, img_name):
    # Load and preprocess the image
    img_path = os.path.join(img_folder, img_name)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.resnet_v2.preprocess_input(x)

    # Make predictions
    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=5)[0]

    # return top 5 predictions as a list
    result = []
    for pred in decoded_preds:
        result.append([img_name, pred[1], pred[2]])
    return result

# Get the full path of the 'pictures' folder
folder_path = os.path.join(os.getcwd(), 'samples')
balcony_path = os.path.join(folder_path, 'Balcony')
pigeons_path = os.path.join(folder_path, 'Pigeons')
humans_path = os.path.join(folder_path, 'Humans')
google_path = os.path.join(folder_path, 'Google')

with open('google_pred.csv', 'a', newline='', encoding = 'utf-8') as file:
    writer = csv.writer(file)

    # write the header row to the CSV file
    writer.writerow(['Image', 'Prediction', 'Probability'])

    for filename in os.listdir(google_path):
        prediction = image_recognition(google_path, filename)
        for pred in prediction:
            writer.writerow(pred)
