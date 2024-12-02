from flask import Flask, request, jsonify
from flask_cors import CORS
from face_recognizer import FaceRecognition
from data_processor import DataProcessor
import base64
import numpy as np
import cv2
import os

app = Flask(__name__)
CORS(app)

face_recognizer = FaceRecognition(
    input_shape=(224, 224, 3),
    learning_rate=0.00005,
    dropout_rate=0.2,
    file_path='../1-december-model.weights.h5'
)
face_recognizer.model.load_weights('/Users/sszyda/face_recognition/1-december-model.weights.h5')

REFERENCE_IMAGES_DIR = '../database'

def load_reference_images():
    reference_images = []
    reference_names = []
    for filename in os.listdir(REFERENCE_IMAGES_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(REFERENCE_IMAGES_DIR, filename)
            try:
                ref_img = DataProcessor.preprocess_image(filepath, image_size=(224, 224))
                ref_img = np.expand_dims(ref_img, axis=0)
                reference_images.append(ref_img)
                reference_names.append(os.path.splitext(filename)[0])
            except ValueError as e:
                print(f"Error processing reference image {filepath}: {e}")
    return reference_images, reference_names

reference_images, reference_names = load_reference_images()

@app.route('/verify', methods=['POST'])
def verify():
    print("Received /verify POST request")
    try:
        data = request.get_json()
        print("Data received")
        image_data = data.get('image', None)
        if image_data is None:
            print("No image data provided")
            return jsonify({'status': 'error', 'message': 'No image data provided'}), 400
        # cv2.imshow(image_data)

        try:
            header, encoded = image_data.split(",", 1)
            decoded_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(decoded_bytes, np.uint8)
            img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            print("Image decoded successfully")
        except Exception as e:
            print(f"Error decoding image: {e}")
            return jsonify({'status': 'error', 'message': 'Invalid image data'}), 400

        try:
            img = DataProcessor.preprocess_image(img_array, image_size=(224, 224))
            img = np.expand_dims(img, axis=0)
            print("Captured image preprocessed")
        except ValueError as e:
            print(f"Error preprocessing image: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 400

        max_score = 0
        best_match = None
        threshold = 0.8

        if not reference_images:
            print("No reference images loaded")
            return jsonify({'status': 'error', 'message': 'No reference images available'}), 500

        for ref_img, name in zip(reference_images, reference_names):
            prediction = face_recognizer.model.predict([img, ref_img])[0][0]
            print(f"Compared with {name}, prediction score: {prediction}")
            if prediction > max_score:
                max_score = prediction
                best_match = name

            if prediction >= threshold:
                result = {'status': 'authorized', 'name': name, 'score': float(prediction)}
                print(f"Authorized: {result}")
                return jsonify(result)

        result = {'status': 'unauthorized', 'max_score': float(max_score), 'best_match': best_match}
        print(f"Unauthorized: {result}")
        return jsonify(result)

    except Exception as e:
        print(f"Unhandled exception: {e}")
        return jsonify({'status': 'error', 'message': 'An internal error occurred'}), 500


if __name__ == '__main__':
    app.run(debug=True)
