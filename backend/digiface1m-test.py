# tests based on digiface-1m P2 dataset
import os
from face_recognizer import FaceRecognition
from data_processor import DataProcessor


def main():
    data_directory = "digiface"
    image_size = (224, 224)
    batch_size = 32
    num_pairs_per_identity = 50
    model_weights_path = "model.weights.h5"

    identity_to_images = DataProcessor.load_data(data_directory)
    identities = list(identity_to_images.keys())

    print(f"Loaded {len(identity_to_images)} identities from {data_directory}.")

    test_generator = DataProcessor(
        identity_to_images=identity_to_images,
        identities=identities,
        image_size=image_size,
        batch_size=batch_size,
        num_pairs_per_identity=num_pairs_per_identity,
        augment=False,
        shuffle=False,
        mode='test'
    )

    face_recognizer = FaceRecognition(
        input_shape=image_size + (3,),
        learning_rate=0.00005,
        dropout_rate=0.2,
        file_path=model_weights_path
    )

    face_recognizer.model.load_weights(model_weights_path)
    print(f"Weights loaded from {model_weights_path}")

    face_recognizer.evaluate(test_generator)
    print("Evaluation on test data complete.")


if __name__ == "__main__":
    main()
