import urllib.request
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

model_url = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4"
model = hub.KerasLayer(model_url, trainable=False)

classes_url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
classes_file = "ImageNetLabels.txt"
urllib.request.urlretrieve(classes_url, classes_file)

classes = []
with open(classes_file, "r") as f:
    classes = [s.strip() for s in f.readlines()]

def classify_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    features = model(x)
    prediction = tf.keras.activations.softmax(features)
    return classes[np.argmax(prediction)]

image_path = "C:/Users/Artur/Documents/GitHub/Pigeon_scared/pigeon-sample_1.jpg"
result = classify_image(image_path)
print(result)