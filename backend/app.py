from flask import Flask, request, jsonify
from flask_cors import CORS
from backend.face_recognizer import FaceRecognition
from backend.data_processor import DataProcessor
import base64
import numpy as np
import cv2
import os
import datetime

app = Flask(__name__)
CORS(app)

face_recognizer = FaceRecognition(
    input_shape=(224, 224, 3),
    learning_rate=0.00005,
    dropout_rate=0.2,
    file_path='../backend/model.weights.h5'
)
face_recognizer.model.load_weights('../backend/model.weights.h5')

DATABASE = '../database'
ENTRY_LOGS = '../entry_logs'

def crop_and_preprocess_face(image):
    cropped_face = DataProcessor.crop_face(image)
    if cropped_face is None:
        raise ValueError("No face detected")
    preprocessed_face = DataProcessor.preprocess_image(cropped_face, image_size=(224, 224))

    return cropped_face, np.expand_dims(preprocessed_face, axis=0)

def load_database():
    reference_images = []
    reference_names = []
    for filename in os.listdir(DATABASE):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(DATABASE, filename)
            try:
                img_array = cv2.imread(filepath)
                if img_array is None:
                    raise ValueError(f"Unable to read image: {filepath}")

                _, ref_img = crop_and_preprocess_face(img_array)
                reference_images.append(ref_img)
                reference_names.append(os.path.splitext(filename)[0])
            except ValueError as e:
                print(f"Error processing reference image {filepath}: {e}")
    return reference_images, reference_names

reference_images, reference_names = load_database()

@app.route('/verify', methods=['POST'])
def verify():
    try:
        data = request.get_json()
        image_data = data.get('image', None)
        if not image_data:
            return jsonify({'status': 'error', 'message': 'No image data provided'}), 400

        # decode
        header, encoded = image_data.split(",", 1)
        decoded_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded_bytes, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # preprocess
        cropped_face, preprocessed_face = crop_and_preprocess_face(img_array)
        print("Face cropped and preprocessed")

        # entry log
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(ENTRY_LOGS, f"log_{timestamp}.jpg")
        cv2.imwrite(log_filename, cropped_face)
        print(f"Cropped face saved to {log_filename}")

        max_score = 0
        best_match = None
        threshold = 0.7

        if not reference_images:
            print("No reference images loaded")
            return jsonify({'status': 'error', 'message': 'No reference images available'}), 500

        for ref_img, name in zip(reference_images, reference_names):
            prediction = face_recognizer.model.predict([preprocessed_face, ref_img])[0][0]
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

    except ValueError as e:
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400

    except Exception as e:
        print(f"Unhandled exception: {e}")
        return jsonify({'status': 'error', 'message': 'An internal error occurred'}), 500


if __name__ == '__main__':
    app.run(debug=True)
