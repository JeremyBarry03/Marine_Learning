import cv2
import numpy as np
import os

#define the directory where your fruit images are stored
#it should be the full pathname apparently

#Test data
test_data_dir = "/Users/meganpitts/Desktop/Intro to AI/Final Project GitHub/B351-final-project/data/test"
test_images = []
test_labels = []

#Train data
train_data_dir = "/Users/meganpitts/Desktop/Intro to AI/Final Project GitHub/B351-final-project/data/train"
train_images = []
train_labels = []

class_mapping = {"apple": 0, "apricot": 1, "banana": 2, "blueberry": 3, "cactus-fruit": 4, 
                 "cantaloupe": 5, "cherry": 6, "dates": 7, "grape": 8, "grapefruit": 9,
                 "guava": 10, "kiwi": 11, "lemon": 12, "lime": 13, "lychee": 14, "mango": 15,
                 "orange": 16, "peach": 17, "pear": 18, "pineapple": 19, "plum": 20, "pomegranate": 21,
                 "raspberry": 22, "strawberry": 23, "tomato": 24, "watermelon": 25}

#Loop through each folder in the test data directory
#something going on where extra files and folders are being created and added
for class_name in os.listdir(test_data_dir):
    class_dir = os.path.join(test_data_dir, class_name)
    if os.path.isdir(class_dir):
        class_label = class_mapping[class_name]
        #Loop through each image
        for image_file in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_file)
            #Load and preprocess the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error loading test image: {image_path}")
                continue  # Skip to the next iteration
            img = cv2.resize(img, (64, 64))  #Resize the image to a consistent size
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #Convert to RGB colorspace
            img = img / 255.0  #Normalize pixel values

            test_images.append(img)
            test_labels.append(class_label)

#Convert the image and label lists to NumPy arrays
test_images = np.array(test_images)
test_labels = np.array(test_labels)

#Print information about the preprocessed data
print("Test data preprocessing complete.")
print("Number of test images:", len(test_images))
print("Shape of test images array:", test_images.shape)
print("Shape of test labels array:", test_labels.shape)


#Loop through each folder in the train data directory
for class_name in os.listdir(train_data_dir):
    class_dir = os.path.join(train_data_dir, class_name)
    if os.path.isdir(class_dir):
        class_label = class_mapping[class_name]
        #Loop through each image
        for image_file in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_file)
            #Load and preprocess the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error loading train image: {image_path}")
                continue  # Skip to the next iteration
            img = cv2.resize(img, (64, 64))  #Resize the image to a consistent size
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #Convert to RGB colorspace
            img = img / 255.0  #Normalize pixel values

            train_images.append(img)
            train_labels.append(class_label)

#Convert the image and label lists to NumPy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)

#Print information about the preprocessed data
print("Training data preprocessing complete.")
print("Number of training images:", len(train_images))
print("Shape of training images array:", train_images.shape)
print("Shape of training labels array:", train_labels.shape)
