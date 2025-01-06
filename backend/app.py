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

load_dotenv(dotenv_path='../backend/config.env')
mongo_uri = os.getenv("MONGO_URI")
db_name = os.getenv("DB_NAME")
collection_name = os.getenv("COLLECTION_NAME")

client = MongoClient(mongo_uri)
db = client[db_name]
collection = db[collection_name]

print("Initializing model ...")
face_recognizer = FaceRecognition(
    input_shape=(224, 224, 3),
    learning_rate=0.00005,
    dropout_rate=0.2,
    file_path='../backend/model.weights.h5'
)

def preprocess_with_data_processor(image):
    cropped_face = DataProcessor.crop_face(image)
    if cropped_face is None:
        raise ValueError("No face detected")

    preprocessed_face = DataProcessor.preprocess_image(cropped_face, image_size=(224, 224))
    preprocessed_face = np.expand_dims(preprocessed_face, axis=0)

    return cropped_face, preprocessed_face


@app.route('/add_identity', methods=['POST'])
def add_identity():
    try:
        data = request.get_json()
        image_data = data.get('image', None)
        name = data.get('name', None)

        if not image_data or not name:
            return jsonify({'status': 'error', 'message': 'Image and name required'}), 400

        header, encoded = image_data.split(",", 1)
        decoded_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded_bytes, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        _, preprocessed_face = preprocess_with_data_processor(img_array)
        embedding = face_recognizer.feature_extractor.predict(preprocessed_face)

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

        header, encoded = image_data.split(",", 1)
        decoded_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded_bytes, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        _, preprocessed_face = preprocess_with_data_processor(img_array)

        query_embedding = face_recognizer.feature_extractor.predict(preprocessed_face)

        max_score = 0
        best_match = None
        threshold = 0.7

        dense_layer = face_recognizer.model.layers[-1]
        dense_weights, dense_bias = dense_layer.get_weights()

        for row in collection.find({}, {"name": 1, "embedding": 1}).limit(5):
            stored_embedding = np.array(row.get("embedding"), dtype=np.float32)
            l1_distance = np.abs(query_embedding - stored_embedding)
            similarity_score = 1 / (1 + np.exp(-(np.dot(l1_distance, dense_weights) + dense_bias)))

            if similarity_score > max_score:
                max_score = similarity_score
                best_match = row.get("name")

        print(f"Max score: {max_score}, best match: {best_match}")

        if max_score >= threshold:
            return jsonify({'status': 'authorized', 'max_score': float(max_score), 'best_match': best_match}), 200

        return jsonify({'status': 'unauthorized', 'max_score': float(max_score), 'best_match': best_match}), 200

    except ValueError as e:
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

    except Exception as e:
        print(f"Unhandled exception: {e}")
        return jsonify({'status': 'error', 'message': 'An internal error occurred'}), 500


if __name__ == '__main__':
    app.run(debug=True)
