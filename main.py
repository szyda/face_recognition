# TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=1 python3 /home/s/face_recognition/main.py
import os
from itertools import count

from face_recognizer import FaceRecognition
from data_processor import DataProcessor
import random
import numpy as np
import tensorflow as tf

def main():
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    data_directory = "dataset"
    image_size = (224, 224)
    batch_size = 32
    augment = True
    shuffle = True
    validation_split = 0.35
    num_identities_to_use = 2000
    num_images_per_identity = None  # Use all images per identity
    num_pairs_per_identity = 50

    identity_to_images = DataProcessor.load_data(data_directory, num_identities_to_use, num_images_per_identity)
    identities = list(identity_to_images.keys())
    train_identities, val_identities = DataProcessor.split_identities(identities, validation_split)

    print(f"Loaded {len(identity_to_images)} identities from {data_directory}.")
    print(f"Split into {len(train_identities)} training and {len(val_identities)} validation identities.")

    train_generator = DataProcessor(
        identity_to_images=identity_to_images,
        identities=train_identities,
        image_size=image_size,
        batch_size=batch_size,
        num_pairs_per_identity=num_pairs_per_identity,
        augment=augment,
        shuffle=shuffle,
        mode='train',
        seed=42
    )

    val_generator = DataProcessor(
        identity_to_images=identity_to_images,
        identities=val_identities,
        image_size=image_size,
        batch_size=batch_size,
        num_pairs_per_identity=num_pairs_per_identity,
        augment=False,
        shuffle=False,
        mode='validation',
        seed=42
    )

    face_recognizer = FaceRecognition(input_shape=image_size + (3,), learning_rate=0.00005, dropout_rate=0.2)
    history = face_recognizer.train(
        model=face_recognizer.model,
        train_generator=train_generator,
        val_generator=val_generator,
        epochs=10
    )

    file_path = "model.weights.h5"
    face_recognizer.save_model(filepath=file_path)
    print("Training complete.")

    face_recognizer.model.load_weights(file_path)
    print(f"Weights loaded from {file_path}")

    face_recognizer.evaluate(val_generator)


if __name__ == "__main__":
    main()

