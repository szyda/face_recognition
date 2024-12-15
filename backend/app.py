from flask import Flask, request, jsonify
from flask_cors import CORS
from face_recognizer import FaceRecognition
from data_processor import DataProcessor
import base64
import numpy as np
import cv2
import os
import datetime
from pymongo import MongoClient
from bson.binary import Binary
from dotenv import load_dotenv
import tensorflow as tf

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "static"))
CORS(app)

aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

mongo_uri = os.getenv("MONGO_URI")
db_name = os.getenv("DB_NAME")
collection_name = os.getenv("COLLECTION_NAME")

if not mongo_uri or not db_name or not collection_name:
    raise ValueError("MongoDB environment variables are not properly set.")

mongo_client = MongoClient(mongo_uri)
db = mongo_client[db_name]
collection = db[collection_name]

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

face_recognizer = None

def preprocess_with_data_processor(image):
    cropped_face = DataProcessor.crop_face(image)
    if cropped_face is None:
        raise ValueError("No face detected")

    cropped_face = cv2.resize(cropped_face, (224, 224))
    cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    cropped_face = tf.keras.applications.vgg16.preprocess_input(cropped_face)
    preprocessed = np.expand_dims(cropped_face, axis=0)
    del cropped_face

    return preprocessed

def get_face_recognizer():
    print("Initializing model ...")
    return FaceRecognition(
        input_shape=(224, 224, 3),
        learning_rate=0.00005,
        dropout_rate=0.2,
        file_path='./model.weights.h5'
    )


@app.route('/')
def serve_index():
    return app.send_static_file('index.html')


@app.route('/add_identity', methods=['POST'])
def add_identity():
    try:
        face_recognizer = get_face_recognizer()

        data = request.get_json()
        image_data = data.get('image')
        name = data.get('name')

        if not image_data or not name:
            return jsonify({'status': 'error', 'message': 'Image and name are required'}), 400

        if "," not in image_data:
            return jsonify({'status': 'error', 'message': 'Invalid image data format'}), 400

        header, encoded = image_data.split(",", 1)
        print(f"Header: {header}")
        print(f"Encoded data length: {len(encoded)}")

        try:
            # Decode the image
            decoded_bytes = base64.b64decode(encoded)
            print(f"Decoded bytes length: {len(decoded_bytes)}")

            # Convert to numpy array
            nparr = np.frombuffer(decoded_bytes, np.uint8)
            print(f"Numpy array shape: {nparr.shape}")

            # Decode the image into an array
            img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_array is None:
                raise ValueError("Failed to decode image using OpenCV")

            print(f"Image shape: {img_array.shape}")

        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Failed to process image: {e}'}), 400

        # Preprocess and predict
        try:
            preprocessed_face = preprocess_with_data_processor(img_array)
            embedding = face_recognizer.feature_extractor.predict(preprocessed_face)
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Failed during prediction: {e}'}), 500

        # Insert into MongoDB
        try:
            collection.insert_one({
                'name': name,
                'embedding': embedding.tolist()
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Failed to insert into database: {e}'}), 500

        return jsonify({'status': 'success', 'message': 'Identity added'}), 200

    except Exception as e:
        print(f"Unhandled error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/verify', methods=['POST'])
def verify():
    try:
        face_recognizer = get_face_recognizer()
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

        if max_score >= threshold:
            return jsonify({'status': 'authorized', 'max_score': max_score, 'best_match': best_match}), 200

        return jsonify({'status': 'unauthorized', 'max_score': max_score, 'best_match': best_match}), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
