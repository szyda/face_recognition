from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Lambda, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.backend as K
import numpy as np
import time
import datetime
import tensorflow as tf
import cv2


class FaceRecognizer:
    def __init__(self, input_shape=(224, 224, 3), learning_rate=0.0001, dropout_rate=0.3):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.model = self.build_model()

    def base_model(self):
        base = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        model = Sequential()
        model.add(base)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(self.dropout_rate))

        return model

    def siamese_model(self):
        base = self.base_model()

        first_input = Input(shape=self.input_shape)
        second_input = Input(shape=self.input_shape)

        first_processed = base(first_input)
        second_processed = base(second_input)

        distance = Lambda(lambda tensors: K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=1, keepdims=True)))(
            [first_processed, second_processed])
        output = Dense(1, activation='sigmoid')(distance)

        model = Model(inputs=[first_input, second_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def build_model(self):
        return self.siamese_model()

    def train(self, train_data, val_data, epochs=10, steps_per_epoch=None, validation_steps=None):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='500,520')

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'model.weights.h5',
            monitor='val_accuracy',
            save_weights_only=True,
            verbose=1
        )

        start = time.time()
        history = self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            validation_steps=validation_steps,
            callbacks=[checkpoint, tensorboard_callback],
            verbose=1
        )

        end = time.time()
        training_time = end - start
        print(f"Training completed in {training_time:.2f} seconds.")

        return history

    def save_weights(self, filepath='model.weights.h5'):
        self.model.save_weights(filepath)

    def load_weights(self, filepath='model.weights.h5'):
        self.model.load_weights(filepath)

    def preprocess_image(self, image):
        image = cv2.resize(image, self.input_shape[:2])
        image = np.expand_dims(image, axis=0)
        return image / 255.0

    def predict(self, image1, image2):
        img1 = self.preprocess_image(image1)
        img2 = self.preprocess_image(image2)

        return self.model.predict([img1, img2])

    def evaluate(self, test_data):
        loss, accuracy = self.model.evaluate(test_data)
        print(f"Test loss: {loss:.4f}\nTest accuracy: {accuracy:.4f}")