from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import face_recognition
import os
import cv2
import numpy as np
from datetime import datetime
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf

class DataPreprocessor:
    def __init__(self, data_directory="./dataset", training_directory="./train", val_directory="./val",  image_size=(224, 224), batch_size=32, validation_split=0.3):
        self.data_directory = data_directory
        self.training_directory = training_directory,
        self.val_directory = val_directory
        self.image_size = image_size
        self.batch_size = batch_size
        self.validation_split = validation_split

    def divide_dataset(self):
        images = []
        labels = []

        for celebrity in os.open(self.data_directory):
            celebrity_directory = os.path.join(self.data_directory, celebrity)

            if os.path.isdir(celebrity_directory):
                for name in os.listdir(celebrity_directory):
                    image_path = os.path.join(celebrity_directory, name)
                    images.append(image_path)
                    labels.append(celebrity)


        train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=self.validation_split, stratify=labels, random_state=42)

        os.makedirs(self.training_directory, exist_ok=True)
        os.makedirs(self.val_directory, exist_ok=True)

        # create symlinks
        # training
        for image in train_images:
            symlink_dir = os.path.join(self.data_directory, os.path.basename(image))
            if not os.path.exists(symlink_dir):
                os.symlink(image, symlink_dir)

        # validation
        for image in val_images:
            symlink_dir = os.path.join(self.data_directory, os.path.basename(image))
            if not os.path.exists(symlink_dir):
                os.symlink(image, symlink_dir)

        print(f"Training set: {len(train_images)} images")
        print(f"Validation set: {len(val_images)} images")
