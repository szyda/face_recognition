from flask import Flask, request, jsonify
from face_recognizer import FaceRecognition
from data_processor import DataProcessor
import base64
import numpy as np
import cv2

app = Flask(__name__)

face_recognizer = FaceRecognition(
    input_shape=(224, 224, 3),
    learning_rate=0.00005,
    dropout_rate=0.2,
    file_path='model.weights.h5'
)
face_recognizer.model.load_weights('model.weights.h5')

@app.route('/verify', methods=['POST'])
def verify():
    data = request.get_json()
    image_data = data['image']

    header, encoded = image_data.split(",", 1)
    decoded_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(decoded_bytes, np.uint8)
    img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = DataProcessor.preprocess_image(img_array, image_size=(224, 224))
    img = np.expand_dims(img, axis=0)

    reference_image_path = 'reference.jpg'
    ref_img = DataProcessor.preprocess_image(reference_image_path, image_size=(224, 224))
    ref_img = np.expand_dims(ref_img, axis=0)

    prediction = face_recognizer.model.predict([img, ref_img])[0][0]

    threshold = 0.65
    if prediction >= threshold:
        result = {'status': 'authorized'}
    else:
        result = {'status': 'unauthorized'}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
