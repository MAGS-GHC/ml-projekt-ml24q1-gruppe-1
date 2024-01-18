import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
mnist_dataset = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()

# Normalize pixel values to range [0, 1]
x_train_normalized = x_train / 255.0
x_test_normalized = x_test / 255.0

# Model definition for MNIST
model_mnist = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the MNIST model
model_mnist.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the MNIST model
history_mnist = model_mnist.fit(x_train_normalized, y_train, epochs=5, validation_data=(x_test_normalized, y_test))

# Evaluate the MNIST model on test data
test_loss_mnist, test_accuracy_mnist = model_mnist.evaluate(x_test_normalized, y_test)
print(f'MNIST Test accuracy: {test_accuracy_mnist * 100:.2f}%')

# Load the EMNIST dataset (you may need to install the emnist package)
(train_images, train_labels), (_, _) = mnist_dataset.load_data()

concat_imgs = []
concat_labels = []

# Create concatenated images and labels for double digits
for i in range(0, len(train_images) - 1, 2):
    img1, img2 = train_images[i], train_images[i+1]
    concat_img1 = np.concatenate([img1, img2], axis=1)
    resized_img1 = Image.fromarray(concat_img1).resize((32, 32))
    concat_imgs.append(np.asarray(resized_img1).reshape(32, 32, 1))  # Reshape to (32, 32, 1)

    label1, label2 = train_labels[i], train_labels[i+1]
    combined_label = label1 * 10 + label2
    concat_labels.append(combined_label)

# Convert lists to numpy arrays
concat_imgs = np.array(concat_imgs)
concat_labels = np.array(concat_labels)

# Split the EMNIST data into training and validation sets
x_train_emnist, x_val_emnist, y_train_emnist, y_val_emnist = train_test_split(
    concat_imgs, concat_labels, test_size=0.2, random_state=42
)

# Model definition for double digits (CNN)
model_emnist = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(100, activation='softmax')  # Adjust the output size for characters
])

# Compile the EMNIST model
model_emnist.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Increase the number of epochs and the batch size for EMNIST
epochs_emnist = 20  # Adjust the number of epochs
batch_size_emnist = 64  # Adjust the batch size

# Train the EMNIST model with data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,      # Rotate images by up to 10 degrees
    width_shift_range=0.1,  # Shift width by up to 10% of the image width
    height_shift_range=0.1,  # Shift height by up to 10% of the image height
    zoom_range=0.1,          # Zoom in/out by up to 10%
)

history_emnist = model_emnist.fit(
    datagen.flow(x_train_emnist, y_train_emnist, batch_size=batch_size_emnist),
    epochs=epochs_emnist,
    validation_data=(x_val_emnist, y_val_emnist),
)

# Visualize training and validation loss for both models
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_mnist.history['loss'], label='MNIST Training Loss')
plt.plot(history_mnist.history['val_loss'], label='MNIST Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('MNIST Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history_emnist.history['loss'], label='EMNIST Training Loss')
plt.plot(history_emnist.history['val_loss'], label='EMNIST Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('EMNIST Training and Validation Loss')

plt.tight_layout()
plt.show()

# Visualize training and validation accuracy for both models
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_mnist.history['accuracy'], label='MNIST Training Accuracy')
plt.plot(history_mnist.history['val_accuracy'], label='MNIST Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('MNIST Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history_emnist.history['accuracy'], label='EMNIST Training Accuracy')
plt.plot(history_emnist.history['val_accuracy'], label='EMNIST Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('EMNIST Training and Validation Accuracy')

plt.tight_layout()
plt.show()

# Make predictions on test data using both models
predictions_mnist = model_mnist.predict(x_test_normalized)
predictions_emnist = model_emnist.predict(x_val_emnist)

# Visualize some test images and their predicted labels for MNIST
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_mnist[i])
    true_label = y_test[i]
    plt.xlabel(f'Predicted: {predicted_label}\nTrue: {true_label}')
plt.show()

# Visualize some double-digit images and their predicted labels for EMNIST
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_val_emnist[i].reshape(32, 32), cmap=plt.cm.binary)  # Reshape back to (32, 32)
    predicted_label = np.argmax(predictions_emnist[i])
    true_label = y_val_emnist[i]
    plt.xlabel(f'Predicted: {predicted_label}\nTrue: {true_label}')
plt.show()
