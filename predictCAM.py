from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.densenet import preprocess_input
from keras.models import Model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

print("Current working directory:", os.getcwd())

# Update dimensions for DenseNet-121 (or your specific model)
img_width, img_height = 224, 224

# Function to preprocess individual images
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_width, img_height))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Load model and print summary
model_path = 'model.keras'  # Replace with your model path
test_model = load_model(model_path)
test_model.summary()  # This will help you to identify the layer names

# Identify and get the last dense layer
# Replace 'dense_layer_name' with the actual name of your last dense layer
dense_layer_name = 'dense_1'
final_dense = test_model.get_layer(dense_layer_name)
weights = final_dense.get_weights()
if weights:
    weights_dense = weights[0]
else:
    raise ValueError("The dense layer has no weights")

# Identify and get the last conv layer
# Replace 'conv_layer_name' with the name of your last convolutional layer
conv_layer_name = 'conv2d_2'
last_conv_layer = test_model.get_layer(conv_layer_name)

# -----------------------------------------------------------------------------
# Prediction and CAM Function
# -----------------------------------------------------------------------------
def get_cam(image_path, model, last_conv_layer, weights_dense):
    # Use the preprocess_image function
    x = preprocess_image(image_path)

    # Get the prediction
    preds = model.predict(x)
    class_pred = (preds > 0.5).astype(int)[0][0]
    class_names = {0: 'PULMONARY', 1: 'NORMAL'}
    predicted_class = class_names[class_pred]

    # Get output from last conv layer
    last_conv_model = Model(model.input, last_conv_layer.output)
    last_conv_output = last_conv_model.predict(x)[0]  # Shape: (height, width, num_filters)

    # Flatten the output to shape (height * width, num_filters)
    last_conv_output_flattened = last_conv_output.reshape((-1, last_conv_output.shape[-1]))

    # Perform the dot product between the flattened output and dense layer weights
    # Shape of weights_dense is expected to be (num_filters, 1) for binary classification
    heatmap = np.dot(last_conv_output_flattened, weights_dense).reshape((last_conv_output.shape[0], last_conv_output.shape[1]))

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Superimpose the heatmap on original image
    img_original = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (img_original.shape[1], img_original.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img_original

    return predicted_class, preds[0][0], superimposed_img

# -----------------------------------------------------------------------------
# Loop Over Test Images
# -----------------------------------------------------------------------------
basedir = "C:/Users/flxsy/Documents/College Term 1 2023-24/LBYMF3B/Final Project/orig code/PULMONARY DETECTION/data/test/"

# Path to the directory containing images
dir_path = "C:/Users/flxsy/Documents/College Term 1 2023-24/LBYMF3B/Final Project/orig code/PULMONARY DETECTION/data/test/"

# List all files in the directory
print("Files in the directory:")
for file in os.listdir(dir_path):
    print(file)

# Get a list of image file names in the directory
image_files = [f for f in os.listdir(basedir) if f.endswith('.jpeg')]

# Loop Over Test Images
for file_name in image_files:
    print(f'Testing Image: {file_name}')
    path = os.path.join(basedir, file_name)

    predicted_class, probability, cam_image = get_cam(path, test_model, last_conv_layer, weights_dense)

    print(f'Predicted Class: {predicted_class}')
    print(f'Probability: {probability}')

    # Display CAM
    plt.imshow(cam_image[..., ::-1])  # Convert BGR to RGB
    plt.title(f"Class: {predicted_class}, Probability: {probability}")
    plt.show()

    print(' ')