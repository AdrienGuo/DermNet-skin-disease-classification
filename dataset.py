import cv2
import os
import random
import numpy as np
import tensorflow as tf
import albumentations as A

IMAGE_SIZE = 128


transform = A.Compose([
    A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
    A.HorizontalFlip(p=1),
    # A.ShiftScaleRotate(p=1),
    # A.RandomBrightnessContrast(p=1),
])

def dataset(dir, num_classes, augment=False, origin=False, aug_data=None, aug_labels=None):
    """
    Args:
        dir: Directory path
        num_classes: Number of classes we want for training
    
    Return:
        X: The images (data)
        y: The classes (labels)
        class_names (list): Contains the name of classes
    """
    train_symptom_dir = os.listdir(dir)
    train_symptom_dir.sort()                # The order returned from os.listdir is random...
    class_names = [train_symptom_dir[i] for i in range(num_classes)]
    
    # X: image list
    # y: label list
    X, y = [], []
    new_images_num = []
    THRESHOLD = 800
    for class_i, class_name in enumerate(class_names):
        images = os.listdir(dir + class_name)
        # If the len(images) less than THRESHOLD, then add (THRESHOLD-len(images)) to this class, otherwise add 0
        new_images_num.append(THRESHOLD-len(images)) if len(images) < THRESHOLD else new_images_num.append(0)
        for image in images:
            image_path = dir + class_name + "/" + image
            img = tf.keras.preprocessing.image.load_img(image_path, color_mode="rgb", target_size= (IMAGE_SIZE, IMAGE_SIZE))
            X.append(np.array(img))
            y.append(class_i)
    
    if origin:
        pass
    # The transformed dataset
    elif (not augment):
        for class_i, num in enumerate(new_images_num):
            if num == 0: continue
            images = os.listdir(dir + class_name)
            for _ in range(num):
                random_picked_image = random.choice(images)         # Randomly pick one from "images" list
                image_path = dir + class_name + "/" + random_picked_image
                img = cv2.imread(image_path)
                transformed_img = transform(image=img)["image"]     # Add the transform to img
                X.append(transformed_img)
                y.append(class_i)
    # The GAN dataset
    elif (augment):
        for idx, data in enumerate(aug_data):
            for i in range(len(data)):
                X.append(data[i])
        for idx, labels in enumerate(aug_labels):
            for i in range(len(labels)):
                y.append(labels[i])

    X = np.array(X)
    y = np.array(y)
    # print("X shape: ", X.shape)
    # print("y shape: ", y.shape)
    print_class_info(class_names, y)

    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    
    # shuffle the dataset (but it also shuffle the test dataset...)
    """
    Somebody says don't use shuffle because it's slow and I also think the setting is so wierd...
    """
    permutation = np.random.permutation(X.shape[0])
    X = X[permutation, :, :, :]
    y = y[permutation]

    return X, y, class_names

def print_class_info(class_names, labels):
    # Print the class labels and the corresponding class names
    unique, counts = np.unique(labels, return_counts=True)
    print("Class Info:")
    for i, (class_i, num) in enumerate(zip(unique, counts)):
        print("Class #{} = {:70} {:>5}".format(class_i, class_names[i], num))
    print("Total number: {}".format(len(labels)))
