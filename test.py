import os
import random
import numpy as np
import cv2
import tensorflow as tf
from face_recognizer import FaceRecognition
from data_processor import DataProcessor
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_test_images(test_dir):
    positive_dir = os.path.join(test_dir, 'positive')
    negative_dir = os.path.join(test_dir, 'negative')

    positive_images = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    negative_images = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    return positive_images, negative_images

def compute_distance(embedding1, embedding2):
    """Computes L1 distance (absolute difference) between two embeddings."""
    return np.sum(np.abs(embedding1 - embedding2))

def main():
    print("Building model...")
    image_size = (224, 224)
    face_recognizer = FaceRecognition(input_shape=image_size + (3,))
    model = face_recognizer.build_model()
    model.load_weights("celebs.weights.h5")

    known_embeddings = np.load('embeddings/celebs_embeddings.npy')
    known_labels = np.load('embeddings/celebs_labels.npy', allow_pickle=True)

    known_embeddings = known_embeddings / np.linalg.norm(known_embeddings, axis=1, keepdims=True)

    print("Loading test images...")
    positive_images, negative_images = load_test_images('test')

    positive_images = random.sample(positive_images, min(5, len(positive_images)))
    negative_images = random.sample(negative_images, min(5, len(negative_images)))

    test_images = positive_images + negative_images
    test_labels = [1] * len(positive_images) + [0] * len(negative_images)

    test_embeddings = []
    for img_path in test_images:
        img = DataProcessor.preprocess_image(img_path)
        img = np.expand_dims(img, axis=0)
        embedding = face_recognizer.feature_extractor.predict(img)
        embedding = embedding / np.linalg.norm(embedding)
        test_embeddings.append(embedding.flatten())

    test_embeddings = np.array(test_embeddings)

    if test_embeddings.size == 0:
        raise ValueError("Test embeddings are empty. Please check the test images.")

    threshold = 7.0
    y_pred_labels = []

    for test_embedding in test_embeddings:
        distances = [compute_distance(test_embedding, known_emb) for known_emb in known_embeddings]
        min_distance = np.min(distances)
        print("Min distance: ", min_distance)
        predicted_label = 1 if min_distance < threshold else 0
        y_pred_labels.append(predicted_label)

    accuracy = accuracy_score(test_labels, y_pred_labels)
    precision = precision_score(test_labels, y_pred_labels)
    recall = recall_score(test_labels, y_pred_labels)
    conf_matrix = confusion_matrix(test_labels, y_pred_labels)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('test/confusion_matrix.png')
    plt.close()

    num_tests = len(test_labels)
    num_passed = 0

    for i, (img_path, true_label, predicted_label) in enumerate(zip(test_images, test_labels, y_pred_labels)):
        if true_label == predicted_label:
            num_passed += 1
        else:
            print(f"Test failed: Image '{img_path}' was incorrectly recognized.")
            print(f"True label: {true_label}, Predicted label: {predicted_label}")
            incorrect_img = cv2.imread(img_path)
            incorrect_img = cv2.cvtColor(incorrect_img, cv2.COLOR_BGR2RGB)
            plt.figure()
            plt.imshow(incorrect_img)
            plt.title(f"Incorrect Prediction: True label {true_label}, Predicted label {predicted_label}")
            plt.axis('off')
            plt.savefig(f'test/incorrect_prediction_{i}.png')
            plt.close()

    print(f"{num_passed}/{num_tests} tests passed.")
    print("All tests completed.")

if __name__ == "__main__":
    main()
