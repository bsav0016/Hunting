import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
import matplotlib.pyplot as plt
import numpy as np

# Set paths to your image directories
base_dir = "./Images"
movement_dir = os.path.join(base_dir, "movement")
no_movement_dir = os.path.join(base_dir, "no_movement")

# Image preprocessing
image_size = (128, 128)  # Resize images for uniformity

data = []
labels = []

# Load images from the 'movement' directory
for filename in os.listdir(movement_dir):
    img_path = os.path.join(movement_dir, filename)
    img = load_img(img_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0
    data.append(img_array)
    labels.append(1)

# Load images from the 'no_movement' directory
for filename in os.listdir(no_movement_dir):
    img_path = os.path.join(no_movement_dir, filename)
    img = load_img(img_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0
    data.append(img_array)
    labels.append(0) 

# Convert to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Split data into training and test sets
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# One-hot encode labels for binary classification (if necessary)
train_labels = to_categorical(train_labels, num_classes=2)
test_labels = to_categorical(test_labels, num_classes=2)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the base model layers
base_model.trainable = True
for layer in base_model.layers[:15]:
    layer.trainable = False

# Build the CNN model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(2, activation='sigmoid')  # For binary classification
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)


# Train the model
history = model.fit(
    train_data,
    train_labels,
    epochs=10,
    validation_data=(test_data, test_labels),
    callbacks=[lr_scheduler, early_stopping],
)

# Save the model
model.save("movement_classifier.keras")

# Plot training history
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(test_data, test_labels)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# Predict on new data
def classify_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=image_size)
    img_array = tf.keras.utils.img_to_array(img) / 255.0  # Normalize
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    return "movement" if prediction[0][0] > 0.5 else "no_movement"

# Example usage
# image_path = "path_to_a_new_image.jpg"
# result = classify_image(image_path)
# print(f"The image is classified as: {result}")
