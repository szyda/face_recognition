from flask import Flask, request, jsonify
from flask_cors import CORS
from backend.face_recognizer import FaceRecognition
from backend.data_processor import DataProcessor
import base64
import numpy as np
import cv2
import os
import datetime
from pymongo import MongoClient
from bson.binary import Binary
import json
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

# Initialize face recognizer and load weights
face_recognizer = FaceRecognition(
    input_shape=(224, 224, 3),
    learning_rate=0.00005,
    dropout_rate=0.2,
    file_path='../backend/model.weights.h5'
)
face_recognizer.model.load_weights('../backend/model.weights.h5')

# MongoDB setup
load_dotenv(dotenv_path='../backend/config.env')
mongo_uri = os.getenv("MONGO_URI")
db_name = os.getenv("DB_NAME")
collection_name = os.getenv("COLLECTION_NAME")

client = MongoClient(mongo_uri)
db = client[db_name]
collection = db[collection_name]

def crop_and_preprocess_face(image):
    print(f"Original image shape: {image.shape}")  # Log original image shape

    # Crop face and preprocess
    cropped_face = DataProcessor.crop_face(image)
    if cropped_face is None:
        raise ValueError("No face detected")

    print(f"Cropped face shape: {cropped_face.shape}")  # Log shape after cropping

    # Preprocess the image to the desired size (224, 224)
    preprocessed_face = DataProcessor.preprocess_image(cropped_face, image_size=(224, 224))

    print(f"Preprocessed face shape (before adding batch dimension): {preprocessed_face.shape}")  # Log shape before batch dim

    # preprocessed_face shape should be (224, 224, 3)
    # Add batch dimension here to make it (1, 224, 224, 3)
    preprocessed_face = np.expand_dims(preprocessed_face, axis=0)

    print(f"Preprocessed face shape (after adding batch dimension): {preprocessed_face.shape}")  # Log shape after adding batch dim

    return cropped_face, preprocessed_face


@app.route('/add_identity', methods=['POST'])
def add_identity():
    try:
        data = request.get_json()
        image_data = data.get('image', None)
        name = data.get('name', None)

        if not image_data or not name:
            return jsonify({'status': 'error', 'message': 'Image and name required'}), 400

        # Decode image from base64
        header, encoded = image_data.split(",", 1)
        decoded_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded_bytes, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process image and get embedding
        _, preprocessed_face = crop_and_preprocess_face(img_array)
        embedding = face_recognizer.extract_embedding(preprocessed_face)

        # Store identity in MongoDB
        collection.insert_one({
            'name': name,
            'embedding': embedding.tolist()
        })

        return jsonify({'status': 'success', 'message': 'Identity added'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/verify', methods=['POST'])
def verify():
    try:
        data = request.get_json()
        image_data = data.get('image', None)
        if not image_data:
            return jsonify({'status': 'error', 'message': 'No image data provided'}), 400

        # Decode image from base64
        header, encoded = image_data.split(",", 1)
        decoded_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded_bytes, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        print(f"Decoded image shape: {img_array.shape}")  # Log shape after decoding

        # Process image (preprocessed_face will already have the correct shape)
        cropped_face, preprocessed_face = crop_and_preprocess_face(img_array)

        print(f"Cropped face shape in verify route: {cropped_face.shape}")  # Log shape of cropped face
        print(f"Preprocessed face shape in verify route: {preprocessed_face.shape}")  # Log shape of preprocessed face

        # Now preprocessed_face has shape (1, 224, 224, 3) as expected by the model
        input_embedding = face_recognizer.extract_embedding(preprocessed_face)
        print(f"After extract embedding {input_embedding.shape}")
        max_score = 0
        best_match = None
        threshold = 0.7

        # Compare with stored embeddings
        for row in collection.find():
            current_embedding = np.array(row['embedding'], dtype=np.float32)
            score = np.dot(input_embedding, current_embedding) / (np.linalg.norm(input_embedding) * np.linalg.norm(current_embedding))

            if score > max_score:
                max_score = score
                best_match = row['name']

            if score >= threshold:
                return jsonify({'status': 'authorized', 'max_score': float(max_score), 'best_match': best_match}), 200

        return jsonify({'status': 'unauthorized', 'max_score': float(max_score), 'best_match': best_match}), 401

    except ValueError as e:
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400

    except Exception as e:
        print(f"Unhandled exception: {e}")
        return jsonify({'status': 'error', 'message': 'An internal error occurred'}), 500



if __name__ == '__main__':
    app.run(debug=True)
