import tensorflow as tf
import numpy as np
from PIL import Image


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="C:/Users/Artur/Downloads/inception_v3_1_default_1.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the image.
image = Image.open("C:/Users/Artur/Downloads/pigeon-sample_1.jpg").resize((299, 299))
image = np.array(image).astype('float32') / 255.0
image = np.expand_dims(image, axis=0)

# Set the input tensor.
interpreter.set_tensor(input_details[0]['index'], image)

# Run the model.
interpreter.invoke()

# Get the output tensor and print the top 3 predictions.
output_data = interpreter.get_tensor(output_details[0]['index'])
top_k = np.argsort(output_data[0])[::-1][:3]
print("Top 3 predictions:", top_k)

# GitHub yrevar/imagenet1000_clsidx_to_labels.txt
with open('C:/Users/Artur/Downloads/imagenet1000_clsidx_to_labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

class_dict = dict(zip(np.arange(len(labels)), labels))

for i in top_k:
    print(class_dict[i])