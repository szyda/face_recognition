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

class FaceRecognizer:
    def __init__(self, input_shape=(224, 224, 3), dropout_rate=0.3, learning_rate=0.0001):
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self._build_model(input_shape)

    @staticmethod
    def base_model(input_shape, dropout_rate=0.3):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(dropout_rate))
        return model

    @staticmethod
    def siamese_model(input_shape, dropout_rate, learning_rate):
        base_model = FaceRecognizer.base_model(input_shape, dropout_rate)
        first_photo = Input(shape=input_shape)
        second_photo = Input(shape=input_shape)

        processed_first_photo = base_model(first_photo)
        processed_second_photo = base_model(second_photo)

        distance = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([processed_first_photo, processed_second_photo])
        output = Dense(1, activation='sigmoid')(distance)

        model = Model(inputs=[first_photo, second_photo], outputs=output)
        model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def _build_model(self, input_shape):
        return self.siamese_model(input_shape, self.dropout_rate, self.learning_rate)

    def train(self, train_data, val_data, epochs=10):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'model_checkpoint.weights.h5',
            monitor='val_accuracy',
            save_weights_only=True,
            verbose=1
        )

        start_time = time.time()
        history = self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=[checkpoint, tensorboard_callback]
        )

        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training completed in {training_time:.2f} seconds.")

        return history

    def load_model(self):
        dummy_input = [np.zeros((1, *self.input_shape)), np.zeros((1, *self.input_shape))]
        _ = self.model.predict(dummy_input)

        try:
            self.model.load_weights('model_checkpoint.weights.h5')
            print("Weights are loaded.")
        except IOError:
            print("Error: Model weights not found.")
        return self.model
