# TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=1 python3 /home/s/face_recognition/main.py
import os
from face_recognizer import FaceRecognition
from data_processor import DataProcessor

def main():
    data_directory = "dataset"
    image_size = (224, 224)
    batch_size = 32
    augment = True
    shuffle = True
    validation_split = 0.3
    num_identities_to_use = 1000
    num_images_per_identity = None  # Use all images per identity
    num_pairs_per_identity = 50

    identity_to_images = DataProcessor.load_data(data_directory, num_identities_to_use, num_images_per_identity)
    train_identities, val_identities = DataProcessor.split_identities(identity_to_images.keys(), validation_split)

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
        mode='train'
    )

    val_generator = DataProcessor(
        identity_to_images=identity_to_images,
        identities=val_identities,
        image_size=image_size,
        batch_size=batch_size,
        num_pairs_per_identity=num_pairs_per_identity,
        augment=False,
        shuffle=False,
        mode='validation'
    )

    face_recognizer = FaceRecognition(input_shape=image_size + (3,), learning_rate=0.00005, dropout_rate=0.3)
    history = face_recognizer.train(
        model=face_recognizer.model,
        train_generator=train_generator,
        val_generator=val_generator,
        epochs=15
    )

    face_recognizer.save_model(filepath='model-25-2.weights.h5')
    print("Training complete.")


if __name__ == "__main__":
    main()

