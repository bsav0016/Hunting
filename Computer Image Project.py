import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from tensorflow.keras import layers


# Set image dimensions
IMG_HEIGHT = 128  # Adjust this based on your images
IMG_WIDTH = 128

# Path to your dataset (adjust paths as necessary)
train_dir = "./Test"
test_dir = "./Validation"

# Image augmentation for training to improve generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values to [0, 1]
    rotation_range=40,           # Random rotations
    width_shift_range=0.2,       # Random width shifts
    height_shift_range=0.2,      # Random height shifts
    shear_range=0.2,             # Random shear transformations
    zoom_range=0.2,              # Random zoom
    horizontal_flip=True,        # Random horizontal flips
    fill_mode='nearest'          # Fill in pixels when transforming
)

# Apply data augmentation to training data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    class_mode='categorical',    # Use 'categorical' for one-hot encoding
    shuffle=True
)

# Test data should only be rescaled (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load test data
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    class_mode='categorical',    # Use 'categorical' for one-hot encoding
    shuffle=False
)

class_indices = train_data.class_indices
class_names = [key for key, value in sorted(class_indices.items(), key=lambda item: item[1])]
print("Train Data Class names:", class_names)

# Define a simpler CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(64, activation='relu'),

    Dense(len(class_names), activation='softmax')  # Output layer with the number of classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Use 'categorical_crossentropy' for one-hot encoded
              metrics=['accuracy'])

# Set up callbacks for learning rate reduction and early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Train the model with the callbacks
history = model.fit(
    train_data,
    epochs=20,
    validation_data=test_data,
    callbacks=[reduce_lr, early_stop]
)

# Optional: Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Optional: Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()