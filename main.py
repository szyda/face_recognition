import os
import cv2
import numpy as np
from data_processor import DataProcessor
from face_recognizer import FaceRecognizer
from collections import Counter

def main():
    print("Initializing DataProcessor...")
    data_processor = DataProcessor(data_directory="dataset", image_size=(224, 224), batch_size=32)

    print("Creating training data generator...")
    train_generator = data_processor.data_generator(data_processor.train_pairs, data_processor.train_labels_pair)

    print("Creating validation data generator...")
    val_generator = data_processor.data_generator(data_processor.val_pairs, data_processor.val_labels_pair)

    print("Initializing face recognizer...")
    face_recognizer = FaceRecognizer()

    steps_per_epoch = len(data_processor.train_pairs) // data_processor.batch_size
    validation_steps = len(data_processor.val_pairs) // data_processor.batch_size

    print("Training model...")
    face_recognizer.train(
        train_data=train_generator,
        val_data=val_generator,
        epochs=5,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )

    print("Training completed.")


if __name__ == "__main__":
    main()
