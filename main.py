import os
from datetime import datetime
from data_processor import DataSequenceGenerator
from face_recognizer import FaceRecognition

# TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=1 python3 /home/s/face_recognition/main.py

def main():
    data_directory = "dataset"
    image_size = (224, 224)
    batch_size = 16
    augment = True
    shuffle = True
    validation_split = 0.35
    num_identities_to_use = 17
    num_images_per_identity = 72
    num_pairs_per_identity = 500

    train_generator = DataSequenceGenerator(
        data_directory=data_directory,
        image_size=image_size,
        batch_size=batch_size,
        num_pairs_per_identity=num_pairs_per_identity,
        validation_split=validation_split,
        augment=augment,
        shuffle=shuffle,
        mode='train',
        num_identities_to_use=num_identities_to_use,
        num_images_per_identity=num_images_per_identity
    )

    val_generator = DataSequenceGenerator(
        data_directory=data_directory,
        image_size=image_size,
        batch_size=batch_size,
        num_pairs_per_identity=num_pairs_per_identity,
        validation_split=validation_split,
        augment=False,
        shuffle=False,
        mode='validation',
        num_identities_to_use=num_identities_to_use,
        num_images_per_identity=num_images_per_identity
    )

    face_recognizer = FaceRecognition(input_shape=image_size + (3,), learning_rate=0.0001, dropout_rate=0.4)
    history = face_recognizer.train(model=face_recognizer.model, train_generator=train_generator, val_generator=val_generator, epochs=30)

    face_recognizer.save_model(filepath='celebs-500.weights.h5')

    print("Finish.")

if __name__ == "__main__":
    main()
