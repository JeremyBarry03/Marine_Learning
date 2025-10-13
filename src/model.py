"""
Notes
- Figure out how to visualize the weights 
- Save the model to freeze it/interact with front end (will use mode.save(), model.load(), model.predict())
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from data_processing import test_images, test_labels, train_images, train_labels

num_classes = 26 

"""
Baseline model with no optimizations
Accuracy: 98.47%
"""
# #defining the model and its layers
# model = keras.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     #put all the pixels into 1d array
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(16, activation='relu'),
#     layers.Dense(8, activation='relu'),
#     layers.Dense(num_classes, activation='softmax')
# ])

# #compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# #print a summary of the model architecture
# model.summary()

# num_epochs = 10
# batch_size = 32

# #history = model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size, validation_split=0.2)
# history = model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size, validation_data=(test_images, test_labels))

# test_loss, test_accuracy = model.evaluate(test_images, test_labels)

"""
Optimized Model 1
Optimizations: kernel_initializer
Accuracy: 98.07%
"""
# #defining the model and its layers
# model = keras.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     #put all the pixels into 1d array
#     layers.Flatten(),
#     layers.Dense(64, activation='relu', kernel_initializer="he_normal"),
#     layers.Dense(32, activation='relu', kernel_initializer="he_normal"),
#     layers.Dense(16, activation='relu', kernel_initializer="he_normal"),
#     layers.Dense(8, activation='relu', kernel_initializer="he_normal"),
#     layers.Dense(num_classes, activation='softmax')
# ])

# #compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# #print a summary of the model architecture
# model.summary()

# num_epochs = 10
# batch_size = 32

# #history = model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size, validation_split=0.2)
# history = model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size, validation_data=(test_images, test_labels))

# test_loss, test_accuracy = model.evaluate(test_images, test_labels)

# #print the test accuracy
# print("Test accuracy:", test_accuracy)

"""
Optimized Model 2
Optimizations: kernel_initializer and dropout
Accuracy: 97.89%
"""
# #defining the model and its layers
# model = keras.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     #put all the pixels into 1d array
#     layers.Flatten(),
#     layers.Dropout(0.2),
#     layers.Dense(64, activation='relu', kernel_initializer="he_normal"),
#     layers.Dense(32, activation='relu', kernel_initializer="he_normal"),
#     layers.Dense(16, activation='relu', kernel_initializer="he_normal"),
#     layers.Dense(8, activation='relu', kernel_initializer="he_normal"),
#     layers.Dense(num_classes, activation='softmax')
# ])

# #compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# #print a summary of the model architecture
# model.summary()

# num_epochs = 10
# batch_size = 32

# #history = model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size, validation_split=0.2)
# history = model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size, validation_data=(test_images, test_labels))

# test_loss, test_accuracy = model.evaluate(test_images, test_labels)

# #print the test accuracy
# print("Test accuracy:", test_accuracy)

"""
Optimized Model 3
Accuracy @ 10 Epochs: 98.84%
Accuracy @ 15 Epochs: 98.34%
Accuracy @ 25 Epochs: 99.78%
Optmizations include: kernel_initizalizer, dropout, early stopping, and more layers
"""
# #defining the model and its layers
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3), kernel_initializer="he_normal"),
    layers.MaxPooling2D((5, 5)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer="he_normal"),
    layers.MaxPooling2D((2, 2)),
    #put all the pixels into 1d array
    layers.Flatten(),
    #experiment with dropout
    layers.Dropout(0.2),
    # layers.Dense(1024, activation='relu', kernel_initializer="he_normal"),
    layers.Dense(256, activation='relu', kernel_initializer="he_normal"),
    layers.Dense(64, activation='relu', kernel_initializer="he_normal"),
    layers.Dense(16, activation='relu', kernel_initializer="he_normal"),
    layers.Dense(8, activation='relu', kernel_initializer="he_normal"),
    layers.Dense(num_classes, activation='softmax')
])

#compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#print a summary of the model architecture
model.summary()

#define early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  #monitor validation loss
    patience=4,           #number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  #restore model weights from the epoch with the best value of the monitored quantity
)

#try with minimum 50 epochs, and try different architectures
#5 got the 98%
num_epochs = 50
batch_size = 32

history=model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size, validation_data=(test_images, test_labels))
#model.save("/models/optimized-model2.h5")

test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print("Test accuracy:", test_accuracy)

"""
Graphs and Accuracy for every model
"""
#plot training history
plt.figure(figsize=(12, 5))

#plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Optimized Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

#plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Optimized Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

#show the plots
plt.tight_layout()
plt.show()

