import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from sklearn.metrics import confusion_matrix
import seaborn as sns
import sys

# Load the dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Rescale pixel values to [0, 1] range
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape images to add a color channel dimension
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Convert labels to one-hot encoded vectors
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Data augmentation using ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=10,
                             zoom_range=0.10,
                             width_shift_range=0.1,
                             height_shift_range=0.1)

# Define the CNN model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model with data augmentation
batch_size = 64
epochs = 50
history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test))

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model as CNN.h5
model.save('CNN.h5')

# Plot the training and validation loss and accuracy
fig_loss = plt.figure(figsize=(8, 6))
ax_loss = fig_loss.add_subplot(111)
ax_loss.plot(history.history['loss'], label='Training Loss')
ax_loss.plot(history.history['val_loss'], label='Validation Loss')
ax_loss.set_title('Loss History')
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Loss')
ax_loss.legend()

fig_accuracy = plt.figure(figsize=(8, 6))
ax_accuracy = fig_accuracy.add_subplot(111)
ax_accuracy.plot(history.history['accuracy'], label='Training Accuracy')
ax_accuracy.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax_accuracy.set_title('Accuracy History')
ax_accuracy.set_xlabel('Epoch')
ax_accuracy.set_ylabel('Accuracy')
ax_accuracy.legend()

# Save the plots to files
fig_loss.savefig('CNN_Loss.png')
fig_accuracy.savefig('CNN_Accuracy.png')

# Plot the combined graph
fig_combined = Figure(figsize=(8, 6))
ax_combined = fig_combined.add_subplot(111)
ax_combined.plot(history.history['loss'], label='Training Loss')
ax_combined.plot(history.history['val_loss'], label='Validation Loss')
ax_combined.plot(history.history['accuracy'], label='Training Accuracy')
ax_combined.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax_combined.set_title('Training History')
ax_combined.set_xlabel('Epoch')
ax_combined.set_ylabel('Loss/Accuracy')
ax_combined.legend()

# Save the combined plot to a file
fig_combined.savefig('CNN_overall.png')

root = tk.Tk()
root.wm_title("Training History")

canvas = FigureCanvasTkAgg(fig_combined, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

root.mainloop()