from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import face_recognition
import os
import cv2
import numpy as np
from datetime import datetime

class DataProcessor:
    def __init__(self, data_path, image_size=(224, 224), batch_size=32):
        self.data_path = data_path
        self.image_size = image_size
        self.batch_size = batch_size


    def initialize_data(self, augument=True):
        if augument:
            data = ImageDataGenerator(
                rescale=1. / 255,
                shuffle=True,
                rotation_range=45,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest',
                validation_split=0.3
            )
        else:
            data = ImageDataGenerator(
                rescale=1. / 255,
                validation_split=0.3
            )
        return data

    def load_data(self, subset):
        data_path = self.data_path
        data = self.initialize_data(augument=(subset == 'train'))

        return data.flow_from_directory(
            data_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset=subset
        )

    def get_train_data(self):
        return self.load_data('training')

    def get_validation_data(self):
        return self.load_data('validation')

    def crop_faces(self, directory="./data", overwrite=False):
        for subdir, dirs, files in os.walk(directory):
            for file in files:
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                image_path = os.path.join(subdir, file)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load image. Skipping: {image_path}")
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = face_recognition.face_locations(image)
                if faces:
                    top, right, bottom, left = faces[0]
                    face_image = image[top:bottom, left:right]
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                    if overwrite:
                        cv2.imwrite(image_path, face_image)
                        print(f"Overwritten image at {image_path} with cropped face.")
                    else:
                        new_filename = f"{os.path.splitext(image_path)[0]}_face{os.path.splitext(image_path)[1]}"
                        cv2.imwrite(new_filename, face_image)
                        print(f"Saved cropped face image as {new_filename}")
                else:
                    print(f"No faces found in {image_path}. Skipping cropping.")

        @staticmethod
        def detect_faces(image):
            face_locations = face_recognition.face_locations(image)
            if face_locations:
                return face_locations[0]
            return None