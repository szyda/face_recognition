import tensorflow_datasets as tfds
from data_processor import DataProcessor
from face_recognizer import FaceRecognizer
import time
import tensorflow as tf

def main():
    lfw_dataset, info = tfds.load('lfw', split='train', with_info=True)
    input_shape = (224, 224, 3)
    batch_size = 32

    data_processor = DataProcessor(lfw_dataset, image_size=input_shape[:2], batch_size=batch_size)

    print("Loading training and validation data...")
    train_data = data_processor.get_train_data()
    val_data = data_processor.get_validation_data()

    num_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
    num_val = val_data.reduce(0, lambda x, _: x + 1).numpy()
    print(f"Number of training photos: {num_train}")
    print(f"Number of validation photos: {num_val}")


    print("Generating image pairs...")
    train_pairs, train_labels = data_processor.generate_image_pairs(train_data)
    print(f"Number of training pairs: {len(train_pairs)}")

    print("Generating validation image pairs...")
    val_pairs, val_labels = data_processor.generate_image_pairs(val_data)
    print(f"Number of validation pairs: {len(val_pairs)}")

    # Something went wrong - need to check why the pairs are empty
    if train_pairs.size == 0 or val_pairs.size == 0:
        print("Error: Generated pairs are empty. Exiting.")
        return

    print("Preparing TensorFlow datasets...")
    train_dataset = tf.data.Dataset.from_tensor_slices(((train_pairs[:, 0], train_pairs[:, 1]), train_labels)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices(((val_pairs[:, 0], val_pairs[:, 1]), val_labels)).batch(batch_size)

    face_recognizer = FaceRecognizer(input_shape=input_shape)

    print("Training the model...")
    history = face_recognizer.train(train_dataset, val_dataset, epochs=10)

if __name__ == "__main__":
    main()
