from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, data_directory="./dataset", training_directory="./train", val_directory="./val",
                 image_size=(224, 224), batch_size=32, validation_split=0.3):
        self.data_directory = data_directory
        self.training_directory = training_directory
        self.val_directory = val_directory
        self.image_size = image_size
        self.batch_size = batch_size
        self.validation_split = validation_split

        self.datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def load_images(self, directory):
        images = []
        labels = []

        for celebrity in os.listdir(directory):
            celebrity_directory = os.path.join(directory, celebrity)

            if os.path.isdir(celebrity_directory):
                for name in os.listdir(celebrity_directory):
                    image_path = os.path.join(celebrity_directory, name)
                    image = cv2.imread(image_path)
                    if image is not None:
                        image = self.resize(image)
                        images.append(image)
                        labels.append(celebrity)
                    else:
                        print(f"Warning: Could not read image {image_path}")

        return images, labels

    def divide_dataset(self):
        images, labels = self.load_images(self.data_directory)

        train_images, val_images, train_labels, val_labels = train_test_split(
            images, labels, test_size=self.validation_split, stratify=labels, random_state=42
        )

        os.makedirs(self.training_directory, exist_ok=True)
        os.makedirs(self.val_directory, exist_ok=True)

        for image in train_images:
            symlink_dir = os.path.join(self.training_directory, os.path.basename(image))
            if not os.path.exists(symlink_dir):
                os.symlink(image, symlink_dir)

        for image in val_images:
            symlink_dir = os.path.join(self.val_directory, os.path.basename(image))
            if not os.path.exists(symlink_dir):
                os.symlink(image, symlink_dir)

        print(f"Training set: {len(train_images)} images")
        print(f"Validation set: {len(val_images)} images")

    def resize(self, image):
        return cv2.resize(image, self.image_size)

    def preprocess(self):
        images, labels = self.load_images(self.training_directory)
        return images, labels

    def generate_pairs(self, images, labels):
        positive_pairs = []
        negative_pairs = []

        label_to_images = {}
        for img, label in zip(images, labels):
            if label not in label_to_images:
                label_to_images[label] = []
            label_to_images[label].append(img)

        for img_list in label_to_images.values():
            for i in range(len(img_list)):
                for j in range(i + 1, len(img_list)):
                    positive_pairs.append((img_list[i], img_list[j]))

        all_labels = list(label_to_images.keys())
        img_to_label = {img: label for img, label in zip(images, labels)}
        for img in images:
            if len(negative_pairs) >= len(positive_pairs):
                break
            label = img_to_label[img]
            negative_label = random.choice([l for l in all_labels if l != label])
            negative_image = random.choice(label_to_images[negative_label])
            negative_pairs.append((img, negative_image))

        min_pairs = min(len(positive_pairs), len(negative_pairs))
        return positive_pairs[:min_pairs], negative_pairs[:min_pairs]

    def data_generator(self, images, labels):
        while True:
            indices = np.arange(len(images))
            np.random.shuffle(indices)

            for start in range(0, len(images), self.batch_size):
                end = min(start + self.batch_size, len(images))
                batch_indices = indices[start:end]

                positive_pairs, negative_pairs = self.generate_pairs(images, labels)

                random.shuffle(positive_pairs)
                random.shuffle(negative_pairs)

                for pos_pair in positive_pairs:
                    yield np.array([pos_pair[0], pos_pair[1]]), np.array([1])

                for neg_pair in negative_pairs:
                    yield np.array([neg_pair[0], neg_pair[1]]), np.array([0])
