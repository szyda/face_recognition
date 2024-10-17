import os
import cv2
import numpy as np
from data_processor import DataPreprocessor
from face_recognizer import FaceRecognizer
from collections import Counter

def main():
    print("Initialize the DataPreprocessor for dataset division")
    data_processor_divide = DataPreprocessor()

    # print("Dividing the dataset into training and validation sets...")
    # data_processor_divide.divide_dataset()

    print("Load and preprocess training data")
    train_images, train_labels = data_processor_divide.load_training_data()

    print("Load and preprocess validation data")
    val_images, val_labels = data_processor_divide.load_validation_data()

    print(f"Loaded {len(train_images)} training images with labels: {Counter(train_labels)}")
    print(f"Loaded {len(val_images)} validation images with labels: {Counter(val_labels)}")

    print("Creating training data generator...")
    train_data_generator = DataPreprocessor(
        images=train_images,
        labels=train_labels,
        image_size=(224, 224),
        batch_size=32,
        shuffle=True
    )

    print("Creating validation data generator...")
    val_data_generator = DataPreprocessor(
        images=val_images,
        labels=val_labels,
        image_size=(224, 224),
        batch_size=32,
        shuffle=False
    )

    print("Initialize face recognizer")
    face_recognizer = FaceRecognizer()

    print("Testing the generator before training...")
    test_batch = train_data_generator.__getitem__(0)
    print("Test batch generated shapes:")
    print(f"Image 1 batch shape: {test_batch[0][0].shape}")
    print(f"Image 2 batch shape: {test_batch[0][1].shape}")
    print(f"Labels batch shape: {test_batch[1].shape}")

    print("Training model ...")
    history = face_recognizer.train(train_data=train_data_generator, val_data=val_data_generator, epochs=5)

    print("Evaluating model ...")
    face_recognizer.evaluate(val_data_generator)

    print("Training and evaluation completed.")

if __name__ == "__main__":
    main()
