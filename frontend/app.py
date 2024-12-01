from flask import Flask, request, jsonify
from face_recognizer import FaceRecognition
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
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    reference_image_path = 'reference.jpg'
    ref_img = cv2.imread(reference_image_path)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    ref_img = cv2.resize(ref_img, (224, 224))
    ref_img = np.expand_dims(ref_img, axis=0)
    ref_img = ref_img / 255.0

    prediction = face_recognizer.model.predict([img, ref_img])[0][0]

    threshold = 0.7
    if prediction >= threshold:
        result = {'status': 'authorized'}
    else:
        result = {'status': 'unauthorized'}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
