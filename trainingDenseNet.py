from keras.applications import DenseNet121
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from skimage import exposure, img_as_float
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.densenet import preprocess_input
from keras.models import load_model

# Load model architecture from 'model.keras'
model = load_model('path/to/model.keras')

# If this doesn't include weights, load them separately
model.load_weights('path/to/weights.keras')


img_width, img_height = 224, 224


train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 500
epochs = 25
batch_size = 32
early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1, min_lr=0.001)

# Load DenseNet-121 with pre-trained ImageNet weights
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Adding custom layers for binary classification
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global Average Pooling
x = Dense(1024, activation='relu')(x)  # Dense layer
predictions = Dense(1, activation='sigmoid')(x)  # Final output layer for binary classification

model = Model(inputs=base_model.input, outputs=predictions)

#for layer in base_model.layers:
 #   layer.trainable = False
    
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_width, img_height))
    img = img_to_array(img)
    
    # Convert the image to float and rescale it to the range [0, 1]
    img = img_as_float(img)

    # Apply adaptive histogram equalization
    img = exposure.equalize_adapthist(img)
    
    # Normalize the image as expected by DenseNet
    img = preprocess_input(img)
    
    # Add a batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

# Update ImageDataGenerator
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Use DenseNet specific preprocessing
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)  # Use DenseNet specific preprocessing

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),  # Resize images
    batch_size=batch_size,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),  # Resize images
    batch_size=batch_size,
    class_mode='binary')


# Compile the model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Train the model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[early_stop, reduce_lr])

# Save the model
model.save('model.keras')
model.save_weights('weights.keras')
